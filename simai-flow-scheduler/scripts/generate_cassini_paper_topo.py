"""
Generate a configurable Cassini Figure 10-style topology.

Default shape follows the figure in the Cassini NSDI'24 paper:
  - 24 GPUs.
  - 6 layer-1 switches, each serving 4 GPUs.
  - 4 layer-2 switches, split into two groups of two.
  - 3 layer-3 switches.
  - L1-L2 edges: each L1 group connects fully to its L2 group.
  - L2-L3 edges: every L2 switch connects to every L3 switch.

With the defaults and no NVSwitch layer, the topology has:
  24 GPU-L1 links + 12 L1-L2 links + 12 L2-L3 links = 48 file links.
TopologyLoader treats each file link as bidirectional.

Optional NVSwitch mode inserts one NVSwitch per N GPUs, default N=4:
  GPU -> NVSwitch -> L1 -> L2 -> L3

File format follows src/static_analysis/passes/topology_loader.py:
  line 1: total_nodes gpus_per_server nv_switch_count other_switch_count total_links gpu_type
  line 2: switch ids, with NVSwitch ids first when enabled
  line 3+: src dst bandwidth latency error_rate
"""

from __future__ import annotations

import argparse
from math import ceil, gcd
from pathlib import Path


DEFAULT_GPU_COUNT = 24
DEFAULT_L1_SWITCHES = 6
DEFAULT_L2_SWITCHES = 4
DEFAULT_L3_SWITCHES = 3
DEFAULT_GPUS_PER_NVSWITCH = 4
DEFAULT_NET_BW_GBPS = 200.0
DEFAULT_NVLINK_BW_GBPS = 2880.0
DEFAULT_NET_LATENCY_MS = "0.0005"
DEFAULT_NVLINK_LATENCY_MS = "0.000025"
DEFAULT_GPU_TYPE = "A100"


def _fmt_bw(value: float) -> str:
    return f"{int(value)}Gbps" if value.is_integer() else f"{value}Gbps"


def _link(src: int, dst: int, bw_gbps: float, latency_ms: str) -> str:
    return f"{src} {dst} {_fmt_bw(bw_gbps)} {latency_ms}ms 0"


def _partition_even(items: list[int], num_groups: int) -> list[list[int]]:
    """Split items into num_groups contiguous groups with sizes differing by at most 1."""
    if num_groups <= 0:
        raise ValueError("num_groups must be positive")
    groups: list[list[int]] = []
    n = len(items)
    start = 0
    for i in range(num_groups):
        size = n // num_groups + (1 if i < n % num_groups else 0)
        groups.append(items[start:start + size])
        start += size
    return groups


def _owner_by_even_partition(index: int, item_count: int, owner_ids: list[int]) -> int:
    """Map a zero-based item index to an owner using contiguous even partitions."""
    if not owner_ids:
        raise ValueError("owner_ids must not be empty")
    owner_idx = min(index * len(owner_ids) // item_count, len(owner_ids) - 1)
    return owner_ids[owner_idx]


def build_topology(
    gpu_count: int = DEFAULT_GPU_COUNT,
    l1_switches: int = DEFAULT_L1_SWITCHES,
    l2_switches: int = DEFAULT_L2_SWITCHES,
    l3_switches: int = DEFAULT_L3_SWITCHES,
    with_nvswitch: bool = False,
    gpus_per_nvswitch: int = DEFAULT_GPUS_PER_NVSWITCH,
    net_bw_gbps: float = DEFAULT_NET_BW_GBPS,
    nvlink_bw_gbps: float = DEFAULT_NVLINK_BW_GBPS,
    net_latency_ms: str = DEFAULT_NET_LATENCY_MS,
    nvlink_latency_ms: str = DEFAULT_NVLINK_LATENCY_MS,
    gpu_type: str = DEFAULT_GPU_TYPE,
) -> str:
    if gpu_count <= 0:
        raise ValueError("gpu_count must be positive")
    if min(l1_switches, l2_switches, l3_switches) <= 0:
        raise ValueError("switch counts must be positive")
    if gpus_per_nvswitch <= 0:
        raise ValueError("gpus_per_nvswitch must be positive")

    nv_count = ceil(gpu_count / gpus_per_nvswitch) if with_nvswitch else 0

    nv_start = gpu_count
    l1_start = nv_start + nv_count
    l2_start = l1_start + l1_switches
    l3_start = l2_start + l2_switches
    total_nodes = l3_start + l3_switches

    nv_ids = list(range(nv_start, nv_start + nv_count))
    l1_ids = list(range(l1_start, l1_start + l1_switches))
    l2_ids = list(range(l2_start, l2_start + l2_switches))
    l3_ids = list(range(l3_start, l3_start + l3_switches))
    switch_ids = nv_ids + l1_ids + l2_ids + l3_ids

    links: list[str] = []

    if with_nvswitch:
        for gpu_id in range(gpu_count):
            nv_id = nv_ids[gpu_id // gpus_per_nvswitch]
            links.append(_link(gpu_id, nv_id, nvlink_bw_gbps, nvlink_latency_ms))

        # One uplink per NVSwitch to the corresponding L1 switch. With defaults,
        # each 4-GPU NVSwitch maps one-to-one to the 6 L1 switches in Figure 10.
        for group_idx, nv_id in enumerate(nv_ids):
            l1_id = _owner_by_even_partition(group_idx, nv_count, l1_ids)
            links.append(_link(nv_id, l1_id, net_bw_gbps, net_latency_ms))
    else:
        for gpu_id in range(gpu_count):
            l1_id = _owner_by_even_partition(gpu_id, gpu_count, l1_ids)
            links.append(_link(gpu_id, l1_id, net_bw_gbps, net_latency_ms))

    # Figure 10 has two L1/L2 clusters: 3 L1 switches connect to 2 L2 switches
    # on the left, and the same on the right. Generalize by using gcd(l1, l2)
    # clusters, then full-bipartite wiring inside each cluster.
    groups = gcd(l1_switches, l2_switches)
    l1_groups = _partition_even(l1_ids, groups)
    l2_groups = _partition_even(l2_ids, groups)
    for group_l1, group_l2 in zip(l1_groups, l2_groups):
        for l1_id in group_l1:
            for l2_id in group_l2:
                links.append(_link(l1_id, l2_id, net_bw_gbps, net_latency_ms))

    # Top layer: each L2 switch connects to every L3 switch.
    for l2_id in l2_ids:
        for l3_id in l3_ids:
            links.append(_link(l2_id, l3_id, net_bw_gbps, net_latency_ms))

    gpus_per_server_header = (
        gpus_per_nvswitch if with_nvswitch else ceil(gpu_count / l1_switches)
    )
    header = (
        f"{total_nodes} {gpus_per_server_header} {nv_count} "
        f"{l1_switches + l2_switches + l3_switches} {len(links)} {gpu_type}"
    )
    return "\n".join([header, " ".join(str(s) for s in switch_ids), *links]) + "\n"


def _default_output_name(args: argparse.Namespace) -> str:
    nv = f"_nv{args.gpus_per_nvswitch}" if args.with_nvswitch else ""
    return (
        f"Cassini_Fig10_{args.gpus}g_l1-{args.l1}_l2-{args.l2}_l3-{args.l3}"
        f"{nv}_{args.net_bw_gbps:g}Gbps_{args.gpu_type}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a configurable Cassini Figure 10-style topology.",
    )
    parser.add_argument("--gpus", type=int, default=DEFAULT_GPU_COUNT)
    parser.add_argument("--l1", type=int, default=DEFAULT_L1_SWITCHES,
                        help="Number of layer-1 switches.")
    parser.add_argument("--l2", type=int, default=DEFAULT_L2_SWITCHES,
                        help="Number of layer-2 switches.")
    parser.add_argument("--l3", type=int, default=DEFAULT_L3_SWITCHES,
                        help="Number of layer-3 switches.")
    parser.add_argument("--with-nvswitch", action="store_true",
                        help="Insert one NVSwitch per --gpus-per-nvswitch GPUs.")
    parser.add_argument("--gpus-per-nvswitch", type=int,
                        default=DEFAULT_GPUS_PER_NVSWITCH)
    parser.add_argument("--net-bw-gbps", type=float, default=DEFAULT_NET_BW_GBPS)
    parser.add_argument("--nvlink-bw-gbps", type=float, default=DEFAULT_NVLINK_BW_GBPS)
    parser.add_argument("--net-latency-ms", default=DEFAULT_NET_LATENCY_MS)
    parser.add_argument("--nvlink-latency-ms", default=DEFAULT_NVLINK_LATENCY_MS)
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/topologies"))

    args = parser.parse_args()
    output = args.output
    if output is None:
        output = args.output_dir / _default_output_name(args)

    content = build_topology(
        gpu_count=args.gpus,
        l1_switches=args.l1,
        l2_switches=args.l2,
        l3_switches=args.l3,
        with_nvswitch=args.with_nvswitch,
        gpus_per_nvswitch=args.gpus_per_nvswitch,
        net_bw_gbps=args.net_bw_gbps,
        nvlink_bw_gbps=args.nvlink_bw_gbps,
        net_latency_ms=args.net_latency_ms,
        nvlink_latency_ms=args.nvlink_latency_ms,
        gpu_type=args.gpu_type,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")

    file_links = len(content.strip().splitlines()) - 2
    nv_count = ceil(args.gpus / args.gpus_per_nvswitch) if args.with_nvswitch else 0
    print("Generated Cassini Figure 10-style topology")
    print(f"  File:       {output}")
    print(f"  GPUs:       {args.gpus}")
    print(f"  NVSwitches: {nv_count}")
    print(f"  L1/L2/L3:   {args.l1}/{args.l2}/{args.l3}")
    print(f"  File links: {file_links} (TopologyLoader creates both directions)")


if __name__ == "__main__":
    main()

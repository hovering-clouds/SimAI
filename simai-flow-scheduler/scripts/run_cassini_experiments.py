"""
Cassini experiments: compare scheduling policies with different base planners.

Runs multi-job training workloads through configurable scheduling modes:

    default           BFS + FairShare (baseline)
    puppeteer         Greedy k-path + TTE-weighted
    cassini-default   Cassini on BFS + FairShare base
    cassini-puppeteer Cassini on Greedy k-path + TTE-weighted

Two invocation modes:
  1. JSON config:  python scripts/run_cassini_experiments.py --config exp.json
  2. CLI args:     python scripts/run_cassini_experiments.py --topo ... --aicb ...
  3. Shell script: bash scripts/run_experiments.sh

JSON config schema (all fields optional, CLI args override config values):
  {
    "topology":       "path/to/topology",
    "output_dir":     "path/to/output",
    "placement":      "contiguous" | "contention-spread",
    "placement_clusters": 2,
    "gpus_per_server": 8,
    "k_paths":        4,
    "step_deg":       5,
    "no_compat":      false,
    "workloads": [
      {"aicb": "path/to/file.txt", "dp": 2, "num_jobs": 2, "num_iters": 5}
    ],
    "modes": ["default", "puppeteer", "cassini-default", "cassini-puppeteer"],
    "visualize": {
      "modes": ["verbose", "compact"],
      "detail_time_range": null,
      "show_arrows": false
    }
  }
"""

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.cassini.workload_transform import (
    merge_ga_to_one_iteration,
    patch_iteration_time_us,
    replicate_with_cross_iteration_deps,
)
from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.job_merger import JobMerger
from src.static_analysis.passes.topology_loader import NodeType, TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.static_analysis.strategies.cassini_strategy import CassiniAnalyzer
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.executor.bandwidth_allocators.tte_aware_allocator import TteAwareAllocator
from src.executor.visualizer import ChromeTraceVerbose, ChromeTraceCompact, ChromeTraceFlowDetail
from src.cassini.circle_abstraction import CircleAbstraction
from src.cassini.pair_compatibility import optimize_link_compatibility

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AICB = (
    "inputs/aicb-workload/"
    "A100-gpt_7B_ws4_pp2-world_size4-tp2-pp2-ep1-gbs16-mbs4-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
)
DEFAULT_TOPO = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
DEFAULT_OUTPUT = "outputs/cassini_experiments"

# ---------------------------------------------------------------------------
# Run functions — one per configuration
# ---------------------------------------------------------------------------


def run_default(workload, topology, **_):
    analysis = DefaultAnalyzer(topology).analyze(workload)
    policy = DefaultSchedulingPolicy(analysis=analysis)
    return AnalyticalExecutor(topology=topology, policy=policy).execute(workload)


def run_puppeteer(workload, topology, k_paths=4, **_):
    result = PuppeteerAnalyzer(topology, k_paths=k_paths).analyze(workload)
    policy = PuppeteerSchedulingPolicy(
        route_table=result.route_table,
        tte_info=result.tte_info,
        resource_dependency=result.resource_dependency,
        execution_plan=result.execution_plan,
        allocator_mode="weighted",
    )
    return AnalyticalExecutor(topology=topology, policy=policy).execute(workload)


def run_cassini_default(workload, topology, step_deg=5, **_):
    analysis = CassiniAnalyzer(topology, step_deg=step_deg).analyze(workload)
    patch_iteration_time_us(analysis, workload)
    _print_cassini_diagnostics(
        workload, topology, analysis.route_table,
        analysis.communication_patterns, analysis.time_shifts,
        step_deg, "cassini-default",
    )
    policy = CassiniSchedulingPolicy(analysis)
    return AnalyticalExecutor(topology=topology, policy=policy).execute(workload)


def run_cassini_puppeteer(workload, topology, k_paths=4, step_deg=5, **_):
    puppet = PuppeteerAnalyzer(topology, k_paths=k_paths).analyze(workload)
    cassini = CassiniAnalyzer(topology, step_deg=step_deg)
    cassini_analysis = cassini.analyze(workload, route_table=puppet.route_table)
    patch_iteration_time_us(cassini_analysis, workload)
    _print_cassini_diagnostics(
        workload, topology, cassini_analysis.route_table,
        cassini_analysis.communication_patterns, cassini_analysis.time_shifts,
        step_deg, "cassini-puppeteer",
    )
    tte_allocator = TteAwareAllocator(tte_info=puppet.tte_info, mode="weighted")
    policy = CassiniSchedulingPolicy(analysis=cassini_analysis, allocator=tte_allocator)
    return AnalyticalExecutor(topology=topology, policy=policy).execute(workload)


MODE_MAP = {
    "default": run_default,
    "puppeteer": run_puppeteer,
    "cassini-default": run_cassini_default,
    "cassini-puppeteer": run_cassini_puppeteer,
}

# ---------------------------------------------------------------------------
# Cassini iteration_time_us patching
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GPU placement strategies
# ---------------------------------------------------------------------------


def _resolve_parallelism(header, dp_override):
    tp, pp, ep = header.tp, header.pp, header.ep
    if dp_override and dp_override >= 1:
        dp = dp_override
        total_gpus = tp * dp * pp * ep
    else:
        total_gpus = header.all_gpus
        dp = total_gpus // (tp * pp * ep)
    return tp, dp, pp, ep, total_gpus


def _contiguous_gpus(job_configs):
    """Assign contiguous GPU ranges — minimal cross-server contention."""
    assignments = []
    offset = 0
    for cfg in job_configs:
        n = cfg["tp"] * cfg["dp"] * cfg["pp"] * cfg["ep"]
        assignments.append(list(range(offset, offset + n)))
        offset += n
    return assignments


def _replica_size(cfg):
    """Number of GPUs for one DP replica across PP/EP/TP dimensions."""
    return cfg["tp"] * cfg["pp"] * cfg["ep"]


def _job_size(cfg):
    return cfg["tp"] * cfg["dp"] * cfg["pp"] * cfg["ep"]


def _server_count(gpu_count, gpus_per_server):
    if gpus_per_server <= 0:
        raise ValueError("gpus_per_server must be positive")
    if gpu_count % gpus_per_server != 0:
        raise ValueError(
            f"gpu_count ({gpu_count}) must be divisible by "
            f"gpus_per_server ({gpus_per_server})"
        )
    return gpu_count // gpus_per_server


def _partition_even(items, num_groups):
    if num_groups <= 0:
        raise ValueError("placement_clusters must be positive")
    groups = []
    n = len(items)
    start = 0
    for i in range(num_groups):
        size = n // num_groups + (1 if i < n % num_groups else 0)
        groups.append(items[start:start + size])
        start += size
    return groups


def _can_fit(server_id, size, gpus_per_server, server_free):
    return server_free[server_id] + size <= (server_id + 1) * gpus_per_server


def _take_from_server(server_id, size, gpus_per_server, server_free):
    if size > gpus_per_server:
        raise RuntimeError(
            f"Replica needs {size} GPUs but one server only has {gpus_per_server}. "
            "Use smaller tp/pp/ep or a topology with more GPUs per server."
        )
    if not _can_fit(server_id, size, gpus_per_server, server_free):
        raise RuntimeError(
            f"Server {server_id} does not have {size} contiguous GPUs left "
            f"(next={server_free[server_id]}, limit={(server_id + 1) * gpus_per_server})."
        )
    base = server_free[server_id]
    server_free[server_id] += size
    return list(range(base, base + size))


def _assignment_from_dp_servers(cfg, dp_servers, gpus_per_server, server_free):
    """Build assigned_nodes in RankGrouper's [PP][DP][EP][TP] order."""
    if len(dp_servers) != cfg["dp"]:
        raise RuntimeError(
            f"Expected {cfg['dp']} DP server assignments, got {len(dp_servers)}"
        )

    blocks = []
    size = _replica_size(cfg)
    for server_id in dp_servers:
        blocks.append(_take_from_server(server_id, size, gpus_per_server, server_free))

    nodes = []
    for pp_idx in range(cfg["pp"]):
        for dp_idx in range(cfg["dp"]):
            block = blocks[dp_idx]
            for ep_idx in range(cfg["ep"]):
                for tp_idx in range(cfg["tp"]):
                    offset = pp_idx * (cfg["ep"] * cfg["tp"]) + ep_idx * cfg["tp"] + tp_idx
                    nodes.append(block[offset])

    expected = _job_size(cfg)
    if len(nodes) != expected:
        raise RuntimeError(f"Expected {expected} assigned nodes, got {len(nodes)}")
    return nodes


def _contention_spread_gpus(
    job_configs,
    gpu_count,
    gpus_per_server,
    placement_clusters=2,
    **_,
):
    """Deterministically place jobs to create cross-cluster DP contention."""
    n_servers = _server_count(gpu_count, gpus_per_server)
    server_groups = _partition_even(list(range(n_servers)), placement_clusters)
    if any(not group for group in server_groups):
        raise RuntimeError(
            f"placement_clusters={placement_clusters} is too high for "
            f"{n_servers} servers"
        )

    server_free = [s * gpus_per_server for s in range(n_servers)]
    reserved_free = list(server_free)
    cluster_cursor = [0 for _ in server_groups]
    assignments = []

    for j, cfg in enumerate(job_configs):
        replica_size = _replica_size(cfg)
        dp_servers = []
        for dp_idx in range(cfg["dp"]):
            cluster_idx = (j + dp_idx) % placement_clusters
            candidates = server_groups[cluster_idx]
            picked = None
            for attempt in range(len(candidates)):
                pos = (cluster_cursor[cluster_idx] + attempt) % len(candidates)
                server_id = candidates[pos]
                if _can_fit(server_id, replica_size, gpus_per_server, reserved_free):
                    picked = server_id
                    reserved_free[server_id] += replica_size
                    cluster_cursor[cluster_idx] = (pos + 1) % len(candidates)
                    break
            if picked is None:
                raise RuntimeError(
                    f"Cannot allocate job {j} DP replica {dp_idx}: no server in "
                    f"cluster {cluster_idx} has {replica_size} GPUs left. "
                    "Reduce num_jobs/dp/tp/pp/ep or increase topology size."
                )
            dp_servers.append(picked)

        assignments.append(
            _assignment_from_dp_servers(cfg, dp_servers, gpus_per_server, server_free)
        )

    return assignments


PLACEMENT_MAP = {
    "contiguous": _contiguous_gpus,
    "contention-spread": _contention_spread_gpus,
}

# ---------------------------------------------------------------------------
# Workload construction
# ---------------------------------------------------------------------------




def build_workload(
    workload_entries,
    topology,
    placement,
    gpus_per_server,
    placement_clusters=2,
):
    """Build a merged P2PWorkload from multiple workload entries.

    Each entry: {"aicb": path, "dp": int, "num_jobs": int, "num_iters": int}

    Each job is independently replicated to its target iteration count *before*
    merging, so different workload entries can have different iteration counts.
    """
    parser = AicbParser()

    # Expand entries → flat list of (aicb_path, dp, num_iters), one per job
    job_specs = []  # list of (aicb_path, dp, num_iters)
    for entry in workload_entries:
        for _ in range(entry["num_jobs"]):
            job_specs.append((entry["aicb"], entry["dp"], entry["num_iters"]))

    # Build configs for placement
    job_configs = []
    for aicb_path, dp_val, _ in job_specs:
        header, _ = parser.parse(aicb_path)
        tp, dp_resolved, pp, ep, _ = _resolve_parallelism(header, dp_val)
        job_configs.append({
            "tp": tp, "dp": dp_resolved, "pp": pp, "ep": ep,
            "header": header, "file": aicb_path,
        })

    placement_fn = PLACEMENT_MAP[placement]
    if placement == "contiguous":
        assignments = placement_fn(job_configs)
    elif placement == "contention-spread":
        assignments = placement_fn(
            job_configs,
            topology.gpu_count,
            gpus_per_server,
            placement_clusters=placement_clusters,
        )

    total = sum(len(a) for a in assignments)
    if total > topology.gpu_count:
        print(f"  WARNING: jobs need {total} GPUs but topology only has {topology.gpu_count}")

    workloads = []
    for idx, cfg in enumerate(job_configs):
        header, items = parser.parse(cfg["file"])
        job = Job(
            job_id=idx,
            name=Path(cfg["file"]).stem,
            model=Path(cfg["file"]).stem,
            assigned_nodes=assignments[idx],
            parallelism=ParallelismConfig(
                tp=cfg["tp"], dp=cfg["dp"], pp=cfg["pp"], ep=cfg["ep"]
            ),
        )
        wl = WorkloadBuilder().build_from_aicb(header, items, job, comm_algo="ring")
        wl = merge_ga_to_one_iteration(wl, header.ga)
        workloads.append(wl)

    # Merge all jobs, then replicate iterations on the merged workload
    if len(workloads) == 1:
        merged = workloads[0]
    else:
        merged = JobMerger().merge(workloads).merged_workload

    num_iters = job_specs[0][2] if job_specs else 1
    if num_iters > 1:
        merged = replicate_with_cross_iteration_deps(merged, num_iters)

    return merged


# ---------------------------------------------------------------------------
# Placement diagnostics
# ---------------------------------------------------------------------------


def _print_placement_report(workload, topology, gpus_per_server, placement_clusters=2):
    print("\nGPU Placement Report")
    print("-" * 40)

    job_servers = {}
    n_servers = max(1, topology.gpu_count // gpus_per_server)
    for job in workload.jobs:
        nodes = sorted(job.assigned_nodes)
        servers = sorted(set(g // gpus_per_server for g in nodes))
        clusters = sorted(set(
            min(s * placement_clusters // n_servers, placement_clusters - 1)
            for s in servers
        ))
        job_servers[job.job_id] = servers

        ranges = []
        start = nodes[0]
        end = nodes[0]
        for g in nodes[1:]:
            if g == end + 1:
                end = g
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = g
        ranges.append(f"{start}-{end}" if start != end else str(start))
        gpu_str = ",".join(ranges)

        tag = "cross-server" if len(servers) > 1 else "single-server"
        server_str = ",".join(str(s) for s in servers)
        cluster_str = ",".join(str(c) for c in clusters)
        print(f"  Job {job.job_id}: GPUs [{gpu_str}] -> "
              f"server(s) [{server_str}], cluster(s) [{cluster_str}] "
              f"({tag}, {len(servers)} server(s))")

    n_cross = sum(1 for s in job_servers.values() if len(s) > 1)
    all_servers = set().union(*job_servers.values())

    if n_cross >= 2:
        print(f"  -> {n_cross} jobs span multiple servers: "
              f"DP traffic will cross shared spine (natural contention)")
    elif n_cross == 0:
        print(f"  -> All jobs single-server: no spine traffic, no contention")
        print(f"     Hint: use larger model, higher DP, or contention-spread placement")
    else:
        print(f"  -> Only {n_cross} job(s) cross servers: limited contention")
        print(f"     Hint: ensure all jobs span 2+ servers for natural contention")
    print()


# ---------------------------------------------------------------------------
# Metrics & reporting
# ---------------------------------------------------------------------------


def extract_metrics(result, mode_name, elapsed_s=0.0):
    flow_times = []
    compute_times = []
    for tid, timing in result.per_task.items():
        t = timing.end_time_us - timing.start_time_us
        (flow_times if timing.task_type == "flow" else compute_times).append(t)

    sf = sorted(flow_times) if flow_times else [0]
    n = len(sf)
    metrics = {
        "mode": mode_name,
        "makespan_us": result.makespan_us,
        "total_time_us": result.total_time_us,
        "num_tasks": len(result.per_task),
        "num_flows": len(flow_times),
        "num_jobs": len(result.job_iteration_times),
        "avg_flow_us": sum(flow_times) / len(flow_times) if flow_times else 0,
        "median_flow_us": sf[n // 2],
        "p99_flow_us": sf[max(0, int(n * 0.99) - 1)],
        "max_flow_us": max(flow_times) if flow_times else 0,
        "min_flow_us": min(flow_times) if flow_times else 0,
        "avg_compute_us": sum(compute_times) / len(compute_times) if compute_times else 0,
        "elapsed_s": elapsed_s,
    }
    for jid, jtime in result.job_iteration_times.items():
        metrics[f"job_{jid}_iter_us"] = jtime
    return metrics


def print_comparison(all_metrics):
    headers = ["Mode", "Makespan(ms)", "Total(ms)", "AvgFlow(us)", "MedFlow(us)",
               "P99Flow(us)", "MaxFlow(us)", "Jobs", "Tasks"]
    widths = [18, 13, 11, 12, 12, 12, 12, 6, 8]
    sep = " | ".join(h.ljust(w) for h, w in zip(headers, widths))

    print("\nCassini Experiment Results")
    print("=" * len(sep))
    print(sep)
    print("-" * len(sep))

    baseline_ms = next((m["makespan_us"] / 1000 for m in all_metrics if m["mode"] == "default"), None)
    for m in all_metrics:
        speedup = ""
        if baseline_ms and m["mode"] != "default":
            ratio = baseline_ms / max(m["makespan_us"] / 1000, 0.001)
            speedup = f" ({ratio:.2f}x)"
        print(f"{m['mode'] + speedup:<18s} | {m['makespan_us'] / 1000:>10.2f} | "
              f"{m['total_time_us'] / 1000:>8.2f} | {m['avg_flow_us']:>9.1f} | "
              f"{m['median_flow_us']:>9.1f} | {m['p99_flow_us']:>9.1f} | "
              f"{m['max_flow_us']:>9.1f} | {m['num_jobs']:>4d} | {m['num_tasks']:>6d}")
    print("=" * len(sep) + "\n")


def print_job_times(all_metrics):
    job_ids = sorted({k for m in all_metrics for k in m
                      if k.startswith("job_") and k.endswith("_iter_us")})
    if not job_ids:
        return
    print("Per-Job Iteration Times (ms)")
    print("-" * 50)
    header = f"{'Mode':<18s}" + "".join(f" | {j:<14s}" for j in job_ids)
    print(header + "\n" + "-" * len(header))
    for m in all_metrics:
        row = f"{m['mode']:<18s}"
        for j in job_ids:
            row += f" | {m.get(j, 0) / 1000:>12.2f} "
        print(row)
    print()


def print_speedup(all_metrics):
    baseline = next((m for m in all_metrics if m["mode"] == "default"), None)
    if not baseline:
        return
    print("Speedup vs Default (makespan):")
    for m in all_metrics:
        if m["mode"] == "default":
            continue
        ratio = baseline["makespan_us"] / max(m["makespan_us"], 1)
        print(f"  {m['mode']:<22s}: {ratio:.2f}x")

    print("\nFlow Time Tail Comparison (p99 / avg):")
    for m in all_metrics:
        ratio = m["p99_flow_us"] / max(m["avg_flow_us"], 1)
        print(f"  {m['mode']:<22s}: p99={m['p99_flow_us']:.0f} us, "
              f"avg={m['avg_flow_us']:.0f} us, ratio={ratio:.2f}")


def _print_cassini_diagnostics(workload, topology, route_table, patterns, time_shifts, step_deg, label):
    """Print per-link Cassini before/after scores and applied time-shifts."""
    from collections import defaultdict
    from src.cassini.circle_abstraction import CircleAbstraction
    from src.cassini.pair_compatibility import compute_score

    job_links: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for task in workload.tasks:
        if not task.is_flow():
            continue
        try:
            path = route_table.get_path(task)
        except (KeyError, ValueError):
            continue
        for i in range(len(path) - 1):
            job_links[task.job_id].add((path[i], path[i + 1]))

    link_jobs: dict[tuple[int, int], list[int]] = defaultdict(list)
    link_caps: dict[tuple[int, int], float] = {}
    for jid, links in job_links.items():
        for lid in links:
            link_jobs[lid].append(jid)
            if lid not in link_caps:
                link = topology.get_link(lid[0], lid[1])
                link_caps[lid] = link.bandwidth_gbps if link else 100.0

    contended = [(lid, sorted(set(jids)))
                 for lid, jids in link_jobs.items() if len(set(jids)) >= 2]

    if not contended:
        print(f"  [{label}] No shared links — Cassini has nothing to optimize")
        return

    print(f"\n  [{label}] Per-Link Optimization")
    print(f"  {'Link':<18s} {'Jobs':<18s} {'Cap(Gbps)':<10s} {'Before':<10s} {'After':<10s} "
          f"{'Delta':<10s} {'Time-shifts(us)':<36s}")
    print(f"  " + "-" * 112)

    for lid, jids in sorted(contended):
        cap = link_caps.get(lid, 100.0)
        link_patterns = [patterns[jid] for jid in jids
                         if jid in patterns and lid in patterns[jid].link_demands]

        perimeters = {p.iteration_time_us for p in link_patterns}
        if len(perimeters) > 1:
            _, circles = CircleAbstraction.build_unified(link_patterns, lid)
        else:
            circles = [CircleAbstraction.from_pattern(p, lid) for p in link_patterns]
        if not circles or len(circles) < 2:
            continue

        n = len(circles)
        before = compute_score(circles, [0] * n, cap)

        after_shifts_deg = []
        for c, p in zip(circles, link_patterns):
            us = time_shifts.get(p.job_id, 0)
            after_shifts_deg.append(round(us * 360 / c.perimeter) if c.perimeter > 0 else 0)
        after = compute_score(circles, after_shifts_deg, cap)

        link_str = f"{lid[0]}-{lid[1]}"
        jobs_str = ",".join(f"J{j}" for j in jids)
        shifts_str = ", ".join(f"J{j}={time_shifts.get(j, 0):>6d}us" for j in jids)
        print(f"  {link_str:<18s} {jobs_str:<18s} {cap:<10.0f} {before:<10.4f} "
              f"{after:<10.4f} {after - before:<+10.4f} {shifts_str:<36s}")
    print()


def print_compatibility_report(workload, topology, step_deg=5):
    cassini = CassiniAnalyzer(topology, step_deg=step_deg)
    analysis = cassini.analyze(workload)
    if len(analysis.communication_patterns) < 2:
        print("  (single job — no pairwise compatibility to report)")
        return

    job_links = cassini._build_job_links(workload, analysis.route_table)
    link_capacities = cassini._build_link_capacities(job_links)
    patterns = analysis.communication_patterns
    job_ids = sorted(patterns.keys())

    print("\nPairwise Job Compatibility (Cassini)")
    print("-" * 45)
    print(f"{'':>8s}", end="")
    for j2 in job_ids:
        print(f" | job_{j2:<5d}", end="")
    print("\n" + "-" * (8 + 14 * len(job_ids)))

    for j1 in job_ids:
        print(f"job_{j1:<5d}", end="")
        for j2 in job_ids:
            if j1 >= j2:
                print(f" | {'--':>10s}", end="")
                continue
            shared = job_links.get(j1, set()) & job_links.get(j2, set())
            if not shared:
                print(f" | {'N/A':>10s}", end="")
                continue
            link = next(iter(shared))
            c1 = CircleAbstraction.from_pattern(patterns[j1], link)
            c2 = CircleAbstraction.from_pattern(patterns[j2], link)
            if max(c1.bw_demand.values()) == 0.0 or max(c2.bw_demand.values()) == 0.0:
                print(f" | {'no_comm':>10s}", end="")
                continue
            score = optimize_link_compatibility(
                {0: c1, 1: c2}, link_capacities[link], step_deg=step_deg
            ).score
            print(f" | {score:>9.3f}", end="")
        print()
    print("-" * (8 + 14 * len(job_ids)))
    print("  (Score 1.0 = perfectly compatible, < 0.6 = avoid placement)\n")


def save_report(all_metrics, config):
    report = {
        "experiment": "cassini_base_planner_comparison",
        "topology": config["topology"],
        "config": {
            "k_paths": config["k_paths"],
            "step_deg": config["step_deg"],
            "placement": config["placement"],
            "placement_clusters": config.get("placement_clusters"),
            "gpus_per_server": config.get("gpus_per_server"),
        },
        "results": all_metrics,
    }
    path = os.path.join(config["output_dir"], "comparison.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {path}")


def save_visualizations(workload, result, mode_name, output_dir, viz_config):
    """Generate Chrome Trace visualizations for a result."""
    for viz_mode in viz_config.get("modes", []):
        if viz_mode == "verbose":
            viz = ChromeTraceVerbose(workload)
            suffix = "verbose"
        elif viz_mode == "compact":
            show_arrows = viz_config.get("show_arrows", False)
            viz = ChromeTraceCompact(workload, show_arrows=show_arrows)
            suffix = "compact"
        elif viz_mode == "detail":
            time_range = viz_config.get("detail_time_range")
            if time_range and len(time_range) == 2:
                start_us, end_us = time_range
            else:
                start_us, end_us = 0, result.makespan_us
            viz = ChromeTraceFlowDetail(workload, start_us, end_us)
            suffix = f"detail_{start_us}_{end_us}"
        else:
            print(f"  Unknown viz mode: {viz_mode}, skipping")
            continue

        trace_path = os.path.join(
            output_dir, f"result_{mode_name}_timeline_{suffix}.json"
        )
        viz.export(result, trace_path)
        print(f"  Trace: {trace_path}")


# ---------------------------------------------------------------------------
# Config file parsing
# ---------------------------------------------------------------------------


def _deep_update(base, override):
    """Recursively update base dict with override values (in-place)."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value


def load_config(config_path: str) -> dict:
    """Load experiment configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cassini experiments — compare scheduling policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_cassini_experiments.py
  python scripts/run_cassini_experiments.py --config experiments.json
  python scripts/run_cassini_experiments.py --topo <path> --aicb <path> --dp 4 --num-jobs 4
  python scripts/run_cassini_experiments.py --modes default cassini-default
  python scripts/run_cassini_experiments.py --placement contention-spread --placement-clusters 2
  bash scripts/run_experiments.sh
        """,
    )

    # --- Top-level config ---
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="JSON config file (CLI args override config values)")

    # --- Core parameters ---
    parser.add_argument("--topo", type=str, default=None,
                        help="Topology file path")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--placement", type=str,
                        choices=["contiguous", "contention-spread"],
                        default=None,
                        help="GPU placement strategy (default: contiguous)")
    parser.add_argument("--gpus-per-server", type=int, default=None,
                        help="GPUs per NVSwitch server (default: inferred from topology)")
    parser.add_argument("--placement-clusters", type=int, default=None,
                        help="Number of fabric clusters for contention-spread")

    # --- Workload definition ---
    parser.add_argument("--aicb", nargs="+", default=None,
                        help="AICB workload file(s)")
    parser.add_argument("--dp", type=int, default=None,
                        help="DP degree override per workload")
    parser.add_argument("--num-jobs", "-n", type=int, default=None,
                        help="Number of jobs per workload")
    parser.add_argument("--num-iters", type=int, default=None,
                        help="Training iterations per job")

    # --- Modes ---
    parser.add_argument("--modes", nargs="+", default=None,
                        help="Modes to run (default, puppeteer, cassini-default, cassini-puppeteer)")

    # --- Tuning knobs ---
    parser.add_argument("--k-paths", type=int, default=None)
    parser.add_argument("--step-deg", type=int, default=None)
    parser.add_argument("--no-compat", action="store_true", default=None,
                        help="Skip pairwise compatibility analysis")

    # --- Visualization ---
    parser.add_argument("--viz-modes", nargs="+", default=None,
                        choices=["verbose", "compact", "detail"],
                        help="Visualization modes to generate")
    parser.add_argument("--viz-detail-range", nargs=2, type=int, default=None,
                        metavar=("START_US", "END_US"),
                        help="Time range for detail viz mode")
    parser.add_argument("--viz-show-arrows", action="store_true", default=None,
                        help="Show dependency arrows in compact viz")

    args = parser.parse_args()

    # --- Resolve configuration (defaults → config file → CLI) ---
    config = {
        "topology": DEFAULT_TOPO,
        "output_dir": DEFAULT_OUTPUT,
        "placement": "contiguous",
        "placement_clusters": 2,
        "gpus_per_server": None,
        "k_paths": 4,
        "step_deg": 5,
        "no_compat": False,
        "workloads": [
            {"aicb": DEFAULT_AICB, "dp": 2, "num_jobs": 2, "num_iters": 1}
        ],
        "modes": ["default", "puppeteer", "cassini-default", "cassini-puppeteer"],
        "visualize": {
            "modes": [],
            "detail_time_range": None,
            "show_arrows": False,
        },
    }

    # Layer 1: apply config file
    if args.config:
        file_config = load_config(args.config)
        _deep_update(config, file_config)

    # Layer 2: apply CLI overrides (non-None values only)
    cli_overrides = {
        "topology": args.topo,
        "output_dir": args.output,
        "placement": args.placement,
        "placement_clusters": args.placement_clusters,
        "gpus_per_server": args.gpus_per_server,
        "k_paths": args.k_paths,
        "step_deg": args.step_deg,
    }
    for key, val in cli_overrides.items():
        if val is not None:
            config[key] = val

    if args.no_compat is not None:
        config["no_compat"] = args.no_compat

    # Workload CLI overrides
    if args.aicb is not None:
        dp_val = args.dp if args.dp is not None else config["workloads"][0]["dp"]
        nj_val = args.num_jobs if args.num_jobs is not None else config["workloads"][0]["num_jobs"]
        ni_val = args.num_iters if args.num_iters is not None else config["workloads"][0]["num_iters"]
        if len(args.aicb) == 1 and args.num_jobs is not None:
            # One file replicated n times
            config["workloads"] = [{
                "aicb": args.aicb[0], "dp": dp_val,
                "num_jobs": nj_val, "num_iters": ni_val,
            }]
        else:
            # Each file = one workload entry (1 job each unless overridden)
            nj_per = args.num_jobs if args.num_jobs is not None else 1
            config["workloads"] = [{
                "aicb": f, "dp": dp_val,
                "num_jobs": nj_per, "num_iters": ni_val,
            } for f in args.aicb]
    else:
        # Apply scalar CLI overrides to existing workload entries
        if args.dp is not None:
            for w in config["workloads"]:
                w["dp"] = args.dp
        if args.num_jobs is not None:
            for w in config["workloads"]:
                w["num_jobs"] = args.num_jobs
        if args.num_iters is not None:
            for w in config["workloads"]:
                w["num_iters"] = args.num_iters

    if args.modes is not None:
        config["modes"] = args.modes

    # Visualization CLI overrides
    if args.viz_modes is not None:
        config["visualize"]["modes"] = args.viz_modes
    if args.viz_detail_range is not None:
        config["visualize"]["detail_time_range"] = tuple(args.viz_detail_range)
    if args.viz_show_arrows is not None:
        config["visualize"]["show_arrows"] = args.viz_show_arrows

    # --- Normalize paths ---
    topo = config["topology"]
    output_dir = config["output_dir"]
    placement = config["placement"]
    placement_clusters = config["placement_clusters"]
    k_paths = config["k_paths"]
    step_deg = config["step_deg"]
    viz_config = config["visualize"]
    workloads_cfg = config["workloads"]

    os.makedirs(output_dir, exist_ok=True)

    # --- Topology ---
    print("=" * 60 + "\nCassini Experiments\n" + "=" * 60)
    print(f"Topology: {topo}")
    topology = TopologyLoader().load(topo)

    # Infer gpus_per_server
    nv_switches = sum(1 for t in topology.node_types.values() if t == NodeType.NV_SWITCH)
    gpus_per_server = config["gpus_per_server"]
    if gpus_per_server is None:
        gpus_per_server = topology.gpu_count // nv_switches if nv_switches else 8

    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, "
          f"Switches: {topology.switch_count}, GPUs/server: {gpus_per_server})")

    # --- Workload ---
    print("\nBuilding workload...")
    print(f"  Placement: {placement}")
    if placement == "contention-spread":
        print(f"  Placement clusters: {placement_clusters}")
    for i, w in enumerate(workloads_cfg):
        print(f"  [{i}] {Path(w['aicb']).name}: dp={w['dp']}, "
              f"jobs={w['num_jobs']}, iters={w['num_iters']}")

    workload = build_workload(
        workloads_cfg,
        topology,
        placement,
        gpus_per_server,
        placement_clusters=placement_clusters,
    )

    print(f"  Total tasks: {len(workload.tasks)} "
          f"(compute: {len(workload.get_compute_tasks())}, "
          f"flow: {len(workload.get_flow_tasks())})")
    print(f"  Jobs: {len(workload.jobs)}")
    for job in workload.jobs:
        print(f"    Job {job.job_id} ({job.name}): {len(job.assigned_nodes)} GPUs, "
              f"tp={job.parallelism.tp}, dp={job.parallelism.dp}, "
              f"pp={job.parallelism.pp}, ep={job.parallelism.ep}")

    WorkloadWriter().write(workload, os.path.join(output_dir, "workload.json"))

    # --- Placement ---
    _print_placement_report(
        workload,
        topology,
        gpus_per_server,
        placement_clusters=placement_clusters,
    )

    # --- Compatibility ---
    if not config["no_compat"] and len(workload.jobs) > 1:
        print_compatibility_report(workload, topology, step_deg)

    # --- Run modes ---
    run_modes = [m for m in config["modes"] if m in MODE_MAP]
    if set(config["modes"]) - set(run_modes):
        unknown = set(config["modes"]) - set(run_modes)
        print(f"  Skipping unknown mode(s): {', '.join(sorted(unknown))}")

    extra = {"k_paths": k_paths, "step_deg": step_deg}
    all_metrics = []

    for mode in run_modes:
        print(f"\n{'=' * 60}\nRunning: {mode}\n{'=' * 60}")
        t0 = time.time()
        try:
            result = MODE_MAP[mode](workload, topology, **extra)
            elapsed = time.time() - t0

            result_path = os.path.join(output_dir, f"result_{mode}.json")
            result.to_json(result_path)
            print(f"  Saved: {result_path}")

            if viz_config["modes"]:
                save_visualizations(workload, result, mode, output_dir, viz_config)

            metrics = extract_metrics(result, mode, elapsed)
            all_metrics.append(metrics)
            print(f"  Makespan: {result.makespan_us / 1000:.2f} ms")
            print(f"  Completed in {elapsed:.2f}s")
            for jid, jtime in result.job_iteration_times.items():
                print(f"    job_{jid} iteration: {jtime / 1000:.2f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()

    if all_metrics:
        print_comparison(all_metrics)
        print_job_times(all_metrics)
        print_speedup(all_metrics)
        save_report(all_metrics, config)
    else:
        print("No modes completed successfully.")


if __name__ == "__main__":
    main()

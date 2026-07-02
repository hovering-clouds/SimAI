"""
Cassini dynamic-mode experiments — on-demand job expansion with one-shot analysis.

Two-phase execution:
  Phase 1 (one-shot): Build a representative single-iteration workload per logical
    job, run full Cassini analysis (routing, CPM, pattern extraction, time-shifts).
  Phase 2 (dynamic):   Create a TrainingJobSlicer DAG with all iterations, then
    execute via DynamicExecutor + CassiniSchedulingPolicy. Each iteration is
    expanded on-demand, only registering its compute_order into the policy.

Compared to the static path (run_cassini_e2e.py), this avoids replicating all
iterations upfront — memory scales with O(active jobs) not O(total iterations).

Usage:
    uv run python scripts/run_cassini_dynamic_e2e.py
    uv run python scripts/run_cassini_dynamic_e2e.py --config experiments.json
    uv run python scripts/run_cassini_dynamic_e2e.py \
        --topo <path> --aicb job_A.txt job_B.txt --dp 2 1 --num-iters 10 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.cassini.iteration_expansion import (
    merge_ga_to_one_iteration,
    patch_iteration_time_us,
)
from src.cassini.job_placement import (
    resolve_parallelism,
    PLACEMENT_MAP,
)
from src.executor.dynamic_executor import DynamicExecutor
from src.executor.job_expander import JobExpander
from src.executor.job_policy import FifoJobPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.static_analysis.strategies.cassini_strategy import CassiniAnalyzer
from src.static_analysis.strategies.default_strategy import LightweightAnalyzer
from src.static_analysis.passes.topology_loader import NodeType, TopologyLoader
from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.compact_workload import (
    CompactWorkload, JobDAG, TaskIdAllocator,
    merge_compact_workloads,
)
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.job_merger import JobMerger
from src.workload_generator.job_slicer import TrainingJobSlicer

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AICB = (
    "inputs/aicb-workload/"
    "A100-gpt_7B_ws4_pp2-world_size4-tp2-pp2-ep1-gbs16-mbs4-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
)
DEFAULT_TOPO = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
DEFAULT_OUTPUT = "outputs/cassini_dynamic"

# ---------------------------------------------------------------------------
# Phase 1: Build representative workload + Cassini analysis
# ---------------------------------------------------------------------------


def build_representative_workload(workload_entries, topology, placement,
                                  gpus_per_server, placement_clusters=2):
    """Build a single-iteration-per-job workload for Cassini analysis.

    Each logical job contributes exactly one iteration (with GA steps merged).
    No cross-iteration replication — this is the representative for analysis.
    """
    parser = AicbParser()
    job_specs = []
    for entry in workload_entries:
        for _ in range(entry["num_jobs"]):
            job_specs.append((entry["aicb"], entry["dp"], entry["num_iters"]))

    job_configs = []
    for aicb_path, dp_val, _ in job_specs:
        header, _ = parser.parse(aicb_path)
        tp, dp_resolved, pp, ep, _ = resolve_parallelism(header, dp_val)
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

    if len(workloads) == 1:
        merged = workloads[0]
    else:
        merged = JobMerger().merge(workloads).merged_workload

    return merged, assignments


# ---------------------------------------------------------------------------
# Phase 2: Build dynamic DAG from workload entries
# ---------------------------------------------------------------------------


def build_dynamic_dag(workload_entries, assignments) -> CompactWorkload:
    """Create a merged CompactWorkload with all iterations as independent Jobs.

    Each logical job's iterations become a chain of sequentially-dependent
    Jobs in the DAG, connected via TrainingJobSlicer's repeat parameter.
    """
    compact_list = []

    for idx, entry in enumerate(workload_entries):
        num_jobs = entry["num_jobs"]
        num_iters = entry["num_iters"]
        aicb_path = entry["aicb"]

        for job_idx in range(num_jobs):
            logical_idx = idx * num_jobs + job_idx
            job_nodes = list(assignments[logical_idx])

            slicer = TrainingJobSlicer()
            compact_wl = slicer.slice_trace(
                trace_path=aicb_path,
                assigned_nodes=job_nodes,
                repeat=num_iters,
                job_group_id=logical_idx,
            )
            compact_list.append(compact_wl)

    return merge_compact_workloads(compact_list)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cassini dynamic mode — on-demand iteration expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--config", "-c", type=str, default=None,
                        help="JSON config file")
    parser.add_argument("--topo", type=str, default=None,
                        help="Topology file path")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory")
    parser.add_argument("--placement", type=str,
                        choices=["contiguous", "contention-spread"], default=None)
    parser.add_argument("--gpus-per-server", type=int, default=None)
    parser.add_argument("--placement-clusters", type=int, default=None)
    parser.add_argument("--aicb", nargs="+", default=None,
                        help="AICB workload file(s)")
    parser.add_argument("--dp", nargs="+", type=int, default=None,
                        help="DP degree per workload")
    parser.add_argument("--num-jobs", nargs="+", type=int, default=None,
                        help="Job count per workload")
    parser.add_argument("--num-iters", nargs="+", type=int, default=None,
                        help="Iterations per job")
    parser.add_argument("--step-deg", type=int, default=None)

    args = parser.parse_args()

    # ── Resolve config ──
    config = {
        "topology": DEFAULT_TOPO,
        "output_dir": DEFAULT_OUTPUT,
        "placement": "contiguous",
        "placement_clusters": 2,
        "gpus_per_server": None,
        "step_deg": 5,
        "workloads": [
            {"aicb": DEFAULT_AICB, "dp": 2, "num_jobs": 2, "num_iters": 5}
        ],
    }

    if args.config:
        with open(args.config) as f:
            file_config = json.load(f)
        for key, val in file_config.items():
            if key != "workloads":
                config[key] = val
            else:
                config["workloads"] = val

    for attr, cli_key in [("topology", "topo"), ("output_dir", "output"),
                          ("placement", "placement"),
                          ("placement_clusters", "placement_clusters"),
                          ("gpus_per_server", "gpus_per_server"),
                          ("step_deg", "step_deg")]:
        val = getattr(args, cli_key, None)
        if val is not None:
            config[attr] = val

    if args.aicb is not None:
        n = len(args.aicb)
        dp_vals = args.dp or [config["workloads"][0]["dp"]] * n
        nj_vals = args.num_jobs or [config["workloads"][0].get("num_jobs", 1)] * n
        ni_vals = args.num_iters or [config["workloads"][0].get("num_iters", 1)] * n
        if len(dp_vals) == 1:
            dp_vals = dp_vals * n
        if len(nj_vals) == 1:
            nj_vals = nj_vals * n
        if len(ni_vals) == 1:
            ni_vals = ni_vals * n
        config["workloads"] = [
            {"aicb": f, "dp": dp_vals[i], "num_jobs": nj_vals[i], "num_iters": ni_vals[i]}
            for i, f in enumerate(args.aicb)
        ]

    topo_path = config["topology"]
    output_dir = config["output_dir"]
    placement = config["placement"]
    placement_clusters = config["placement_clusters"]
    gpus_per_server = config["gpus_per_server"]
    step_deg = config["step_deg"]
    workloads_cfg = config["workloads"]

    os.makedirs(output_dir, exist_ok=True)

    # ── Topology ──
    print("=" * 60)
    print("Cassini Dynamic Mode")
    print("=" * 60)
    print(f"Topology: {topo_path}")
    topology = TopologyLoader().load(topo_path)

    nv_count = sum(1 for t in topology.node_types.values() if t == NodeType.NV_SWITCH)
    if gpus_per_server is None:
        gpus_per_server = topology.gpu_count // nv_count if nv_count else 8

    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, "
          f"GPUs/server: {gpus_per_server})")

    # ── Phase 1: Build representative workload ──
    print("\n" + "=" * 60)
    print("Phase 1: Build representative workload (one iter per job)")
    print("=" * 60)
    for w in workloads_cfg:
        print(f"  {Path(w['aicb']).name}: dp={w['dp']}, "
              f"jobs={w['num_jobs']}, iters={w['num_iters']}")

    t0 = time.time()
    rep_workload, assignments = build_representative_workload(
        workloads_cfg, topology, placement, gpus_per_server,
        placement_clusters=placement_clusters,
    )
    t1 = time.time()
    print(f"  Tasks: {len(rep_workload.tasks)} "
          f"(compute: {len(rep_workload.get_compute_tasks())}, "
          f"flow: {len(rep_workload.get_flow_tasks())})")
    print(f"  Jobs: {len(rep_workload.jobs)}")
    print(f"  Build time: {t1 - t0:.2f}s")

    # ── Phase 2: Cassini analysis (one-shot) ──
    print("\n" + "=" * 60)
    print("Phase 2: Cassini analysis on representative workload")
    print("=" * 60)
    t0 = time.time()
    cassini_result = CassiniAnalyzer(topology, step_deg=step_deg).analyze(rep_workload)
    patch_iteration_time_us(cassini_result, rep_workload)
    t1 = time.time()

    for jid in sorted(cassini_result.communication_patterns.keys()):
        shift = cassini_result.time_shifts.get(jid, 0)
        print(f"  Job {jid}: time_shift={shift / 1000:.1f}ms")
    print(f"  Analysis time: {t1 - t0:.2f}s")

    # ── Phase 3: Build dynamic DAG ──
    print("\n" + "=" * 60)
    print("Phase 3: Build dynamic execution DAG")
    print("=" * 60)
    t0 = time.time()
    compact_wl = build_dynamic_dag(workloads_cfg, assignments)
    job_dag = JobDAG.from_compact(compact_wl)
    t1 = time.time()

    total_iters = len(compact_wl.jobs)
    print(f"  Total iteration jobs: {total_iters}")
    print(f"  Root jobs: {len(job_dag.root_jobs)}")
    print(f"  Build time: {t1 - t0:.2f}s")
    group_display = sorted(set(
        info.job_group_id for info in compact_wl.job_expansion_info.values()
    ))
    print(f"  Job group IDs: {group_display}")

    # ── Phase 4: Dynamic execution ──
    print("\n" + "=" * 60)
    print("Phase 4: Dynamic execution")
    print("=" * 60)

    policy = CassiniSchedulingPolicy(cassini_result)
    policy.initialize(rep_workload, topology)
    # 清除 compute_order — 动态模式的 compute_order 完全由 update_analysis() 填充，
    # 代表 workload 的 task_ids 会阻塞动态展开的 task 被调度。
    policy.compute_order = {}

    executor = DynamicExecutor(
        topology=topology,
        policy=policy,
        analyzer=LightweightAnalyzer(),
    )

    job_expander = JobExpander(
        task_id_allocator=TaskIdAllocator(),
        profile_store=None,
    )

    t_exec_start = time.time()
    result = executor.execute_dynamic(
        job_dag=job_dag,
        job_expansion_info=compact_wl.job_expansion_info,
        job_policy=FifoJobPolicy(),
        job_expander=job_expander,
    )
    t_exec_end = time.time()

    print(f"\n  Makespan: {result.makespan_us / 1000:.2f} ms  "
          f"({result.makespan_us / 1e6:.3f} s)")
    print(f"  Tasks completed: {len(result.per_task)}")
    print(f"  Execution time: {t_exec_end - t_exec_start:.2f}s")

    # ── Output ──
    result_path = os.path.join(output_dir, "result.json")
    result.to_json(result_path)
    print(f"\n  Saved: {result_path}")

    meta_path = os.path.join(output_dir, "task_meta.json")
    with open(meta_path, "w") as f:
        json.dump({str(k): v for k, v in executor._task_meta.items()}, f)
    print(f"  Saved: {meta_path}")

    # ── Chrome Trace 可视化（pid = job group, tid = node×2 + compute(0)/flow(1)）──
    trace_path = os.path.join(output_dir, "trace.json")
    events: list[dict] = []
    seen: set[tuple[int, int]] = set()  # (pid, tid) 组合已发过 metadata 标记

    for task_id, timing in result.per_task.items():
        meta = executor._task_meta.get(task_id, {})
        job_id = meta.get("job_id", 0)
        info = compact_wl.job_expansion_info.get(job_id)
        pid = info.job_group_id if info else 0
        node = timing.node
        type_bit = 0 if timing.task_type == "compute" else 1
        thread_tid = node * 2 + type_bit

        # process_name（每个 pid 一次）
        if (pid, -1) not in seen:
            seen.add((pid, -1))
            events.append({
                "name": "process_name", "ph": "M",
                "pid": pid, "tid": 0,
                "args": {"name": f"Job Group {pid}"},
            })
        # thread_name（每个 (pid, tid) 一次）
        if (pid, thread_tid) not in seen:
            seen.add((pid, thread_tid))
            events.append({
                "name": "thread_name", "ph": "M",
                "pid": pid, "tid": thread_tid,
                "args": {"name": f"Node {node} {'Compute' if type_bit == 0 else 'Comm'}"},
            })

        # 事件标签
        if timing.task_type == "compute":
            phase = meta.get("phase", "")
            layer = meta.get("layer_id", None)
            label = f"{phase} L{layer}" if phase and layer is not None else f"compute_{task_id}"
        else:
            comm = meta.get("comm_type", "")
            src = meta.get("src", None)
            dst = meta.get("dst", None)
            label = f"{comm} {src}→{dst}" if comm and src is not None and dst is not None else f"flow_{task_id}"

        events.append({
            "name": label,
            "cat": timing.task_type,
            "ph": "X",
            "ts": timing.start_time_us,
            "dur": max(timing.end_time_us - timing.start_time_us, 1),
            "pid": pid,
            "tid": thread_tid,
        })

    with open(trace_path, "w") as f:
        json.dump({"traceEvents": events}, f)
    print(f"  Saved: {trace_path}")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Done")
    print("=" * 60)
    print(f"  Output: {output_dir}/")
    print(f"  result.json — ExecutionResult with per-task timing")
    print(f"  task_meta.json — Task metadata for visualization")
    print(f"  trace.json — Open in chrome://tracing to view")


if __name__ == "__main__":
    main()

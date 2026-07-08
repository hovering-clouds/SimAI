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
from src.executor.job_policy import DelayByJobPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.static_analysis.strategies.cassini_strategy import CassiniAnalyzer
from src.static_analysis.strategies.default_strategy import LightweightAnalyzer
from src.static_analysis.passes.routing import EcmpStrategy
from src.static_analysis.passes.topology_loader import NodeType, TopologyLoader
from src.workload_format.schema import Job, Meta, P2PWorkload, ParallelismConfig
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
# 可视化辅助
# ---------------------------------------------------------------------------

def _abbrev_phase(phase) -> str:
    from src.workload_format.schema import Phase
    m = {Phase.FORWARD: "fwd", Phase.BACKWARD_INPUT: "bwd_i",
         Phase.BACKWARD_WEIGHT: "bwd_w", Phase.OPTIMIZER: "opt"}
    return m.get(phase, str(phase.value))


def _abbrev_comm(comm) -> str:
    from src.workload_format.schema import CommType
    m = {CommType.TP_ALLREDUCE_RING: "tp_ar", CommType.TP_ALLGATHER_RING: "tp_ag",
         CommType.TP_REDUCESCATTER_RING: "tp_rs", CommType.TP_ALLTOALL: "tp_a2a",
         CommType.EP_ALLTOALL: "ep_a2a", CommType.PP_SEND: "pp_snd",
         CommType.PP_RECV: "pp_rcv"}
    return m.get(comm, str(comm.value))


def generate_chrome_trace(result, executor, compact_wl, output_dir, min_dur_us: int = 0, suffix: str = ""):
    """生成实际执行 trace（pid = job_group, tid = node×2 + compute(0)/flow(1)）。

    Args:
        min_dur_us: 过滤掉 duration <= 该值的 task（默认 0 表示不过滤）。
        suffix: 文件名后缀（如 "_cassini", "_manual"），为空则输出 trace.json。
    """
    events = []
    seen: set[tuple[int, int]] = set()

    for task_id, timing in result.per_task.items():
        dur = timing.end_time_us - timing.start_time_us
        if dur <= min_dur_us:
            continue
        meta = executor._task_meta.get(task_id, {})
        job_id = meta.get("job_id", 0)
        info = compact_wl.job_expansion_info.get(job_id)
        pid = info.job_group_id if info else 0
        node = timing.node
        type_bit = 0 if timing.task_type == "compute" else 1
        thread_tid = node * 2 + type_bit

        if (pid, -1) not in seen:
            seen.add((pid, -1))
            events.append({"name": "process_name", "ph": "M", "pid": pid, "tid": 0,
                           "args": {"name": f"Job Group {pid}"}})
        if (pid, thread_tid) not in seen:
            seen.add((pid, thread_tid))
            events.append({"name": "thread_name", "ph": "M", "pid": pid, "tid": thread_tid,
                           "args": {"name": f"Node {node} {'Compute' if type_bit == 0 else 'Comm'}"}})

        if timing.task_type == "compute":
            phase = meta.get("phase", "")
            layer = meta.get("layer_id", None)
            label = f"{phase} L{layer}" if phase and layer is not None else f"compute_{task_id}"
        else:
            comm, src, dst = meta.get("comm_type", ""), meta.get("src", None), meta.get("dst", None)
            label = f"{comm} {src}→{dst}" if comm and src is not None and dst is not None else f"flow_{task_id}"

        events.append({"name": label, "cat": timing.task_type, "ph": "X",
                       "ts": timing.start_time_us, "dur": max(timing.end_time_us - timing.start_time_us, 1),
                       "pid": pid, "tid": thread_tid})

    filename = f"trace{suffix}.json" if suffix else "trace.json"
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)
    print(f"  Saved: {path}")


def generate_cpm_comparison_trace(result, executor, compact_wl, cassini_result, rep_workload, output_dir, min_dur_us: int = 0):
    """生成 CPM 理想 vs 实际执行的对比 trace。

    Args:
        min_dur_us: 过滤掉 duration <= 该值的 task（默认 0 表示不过滤）。
    """
    import os, json
    from collections import defaultdict

    iter_time = {jid: p.iteration_time_us for jid, p in cassini_result.communication_patterns.items()}
    num_iters = {}
    for info in compact_wl.job_expansion_info.values():
        num_iters[info.job_group_id] = num_iters.get(info.job_group_id, 0) + 1
    max_iters = max(num_iters.values()) if num_iters else 1
    num_groups = len(set(i.job_group_id for i in compact_wl.job_expansion_info.values()))
    events = []
    seen = set()

    # ── Actual 行: 遍历执行结果 ──
    for task_id, timing in result.per_task.items():
        dur = timing.end_time_us - timing.start_time_us
        if dur <= min_dur_us:
            continue
        meta = executor._task_meta.get(task_id, {})
        info = compact_wl.job_expansion_info.get(meta.get("job_id", 0))
        if info is None:
            continue
        gid = info.job_group_id
        node = timing.node
        type_bit = 0 if timing.task_type == "compute" else 1
        thread_tid = node * 2 + type_bit

        if (gid, -1) not in seen:
            seen.add((gid, -1))
            events.append({"name": "process_name", "ph": "M", "pid": gid, "tid": 0,
                           "args": {"name": f"Job Group {gid} (Actual)"}})
        if (gid, thread_tid) not in seen:
            seen.add((gid, thread_tid))
            events.append({"name": "thread_name", "ph": "M", "pid": gid, "tid": thread_tid,
                           "args": {"name": f"Node {node} {'Compute' if type_bit == 0 else 'Comm'}"}})

        if timing.task_type == "compute":
            label = f"{meta.get('phase', '')} L{meta.get('layer_id', '')}" if meta.get('phase') else f"c{task_id}"
        else:
            c, s, d = meta.get("comm_type", ""), meta.get("src"), meta.get("dst")
            label = f"{c} {s}->{d}" if c and s is not None else f"f{task_id}"
        events.append({"name": label, "cat": timing.task_type, "ph": "X",
                       "ts": timing.start_time_us, "dur": max(timing.end_time_us - timing.start_time_us, 1),
                       "pid": gid, "tid": thread_tid})

    # ── CPM Ideal 行: 从 CPM baseline 独立绘制 ──
    rep_task_map = {t.task_id: t for t in rep_workload.tasks}
    cpm_tasks = []
    for tid, tinfo in cassini_result.critical_path.task_timings.items():
        task = rep_task_map.get(tid)
        if task is not None:
            cpm_tasks.append((task, tinfo.earliest_start_us, tinfo.earliest_finish_us))

    for task, cpm_start, cpm_finish in cpm_tasks:
        cpm_dur = cpm_finish - cpm_start
        if cpm_dur <= min_dur_us:
            continue
        gid = task.job_id
        ideal_pid = gid + num_groups
        node = task.node if task.is_compute() else (task.src or 0)
        type_bit = 0 if task.is_compute() else 1
        thread_tid = node * 2 + type_bit

        if (ideal_pid, -1) not in seen:
            seen.add((ideal_pid, -1))
            events.append({"name": "process_name", "ph": "M", "pid": ideal_pid, "tid": 0,
                           "args": {"name": f"Job Group {gid} (CPM Ideal)"}})
        if (ideal_pid, thread_tid) not in seen:
            seen.add((ideal_pid, thread_tid))
            events.append({"name": "thread_name", "ph": "M", "pid": ideal_pid, "tid": thread_tid,
                           "args": {"name": f"Node {node} {'Compute' if type_bit == 0 else 'Comm'} (Ideal)"}})

        label = f"{_abbrev_phase(task.phase)} L{task.layer_id}" if task.is_compute() else f"{_abbrev_comm(task.comm_type)} {task.src}->{task.dst}"
        dur = max(cpm_dur, 1)
        base_shift = cassini_result.time_shifts.get(gid, 0)
        for it in range(max_iters):
            shift = base_shift + it * iter_time.get(gid, 100000)
            events.append({"name": label, "cat": "cpm_ideal", "ph": "X",
                           "ts": cpm_start + shift, "dur": int(dur),
                           "pid": ideal_pid, "tid": thread_tid})

    path = os.path.join(output_dir, "trace_cpm_comparison.json")
    with open(path, "w") as f:
        json.dump({"traceEvents": events}, f)
    print(f"  Saved: {path}")


def _find_contended_links(rep_workload, route_table):
    from collections import defaultdict
    job_links = defaultdict(set)
    for task in rep_workload.tasks:
        if not task.is_flow(): continue
        try: path = route_table.get_path(task)
        except: continue
        for i in range(len(path)-1): job_links[task.job_id].add((path[i], path[i+1]))
    link_jobs = defaultdict(set)
    for jid, links in job_links.items():
        for lid in links: link_jobs[lid].add(jid)
    return {lid for lid, jids in link_jobs.items() if len(jids) >= 2}, job_links


def _path_to_links(p):
    return [(p[i], p[i+1]) for i in range(len(p)-1)]


def save_link_contention_cpm(cassini_result, rep_workload, route_table, compact_wl, output_dir):
    """保存 CPM 理想的 link contention 数据（所有模式共享，只存一份）。"""
    from collections import defaultdict
    contended, _ = _find_contended_links(rep_workload, route_table)
    max_iters = max(
        len(set(jid for jid in compact_wl.job_expansion_info
                if compact_wl.job_expansion_info[jid].job_group_id == gid))
        for gid in set(i.job_group_id for i in compact_wl.job_expansion_info.values())
    ) if compact_wl.job_expansion_info else 1
    iter_time_map = {jid: p.iteration_time_us for jid, p in cassini_result.communication_patterns.items()}

    cpm: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    rep_task_map = {t.task_id: t for t in rep_workload.tasks}
    for tid, tinfo in cassini_result.critical_path.task_timings.items():
        task = rep_task_map.get(tid)
        if task is None or not task.is_flow(): continue
        s, d = tinfo.earliest_start_us, max(tinfo.earliest_finish_us - tinfo.earliest_start_us, 1)
        try: path = route_table.get_path(task)
        except: continue
        gid = str(task.job_id)
        base = cassini_result.time_shifts.get(task.job_id, 0)
        for link in _path_to_links(path):
            if link not in contended: continue
            for it in range(max_iters):
                shift = base + it * iter_time_map.get(task.job_id, 100000)
                cpm[link][gid].append((s + shift, s + shift + d))
    out = {}
    for link in sorted(cpm.keys(), key=lambda l: (l[0], l[1])):
        out[f"{link[0]}->{link[1]}"] = {gid: sorted(ivs) for gid, ivs in cpm[link].items()}
    path = os.path.join(output_dir, "link_contention_cpm.json")
    with open(path, "w") as f: json.dump(out, f)
    print(f"  Saved: {path}  (links={len(out)})")


def save_link_contention_actual(result, executor, compact_wl, route_table, contended, output_dir, mode_label):
    """保存单个模式的 actual link contention 数据。"""
    from collections import defaultdict
    actual: dict[tuple, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for task_id, timing in result.per_task.items():
        meta = executor._task_meta.get(task_id, {})
        if timing.task_type != "flow": continue
        if timing.end_time_us - timing.start_time_us <= 0: continue
        src, dst = meta.get("src"), meta.get("dst")
        if src is None or dst is None: continue
        try: path = route_table.get_path_by_endpoints(src, dst)
        except: continue
        dyn_job_id = meta.get("job_id", 0)
        info = compact_wl.job_expansion_info.get(dyn_job_id)
        gid = str(info.job_group_id if info else 0)
        for link in _path_to_links(path):
            if link in contended:
                actual[link][gid].append((timing.start_time_us, timing.end_time_us))
    out = {}
    for link in sorted(actual.keys(), key=lambda l: (l[0], l[1])):
        out[f"{link[0]}->{link[1]}"] = {gid: sorted(ivs) for gid, ivs in actual[link].items()}
    path = os.path.join(output_dir, f"link_contention_actual_{mode_label}.json")
    with open(path, "w") as f: json.dump(out, f)
    n = sum(len(ivs) for v in out.values() for ivs in v.values())
    print(f"  Saved: {path}  (links={len(out)} flows={n})")


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
# Shift mode parsing
# ---------------------------------------------------------------------------


def parse_mode_shifts(mode_str: str, cassini_shifts: dict, num_groups: int) -> tuple[str, dict[int, int]]:
    """Parse a mode string into (label, group_id → shift_us) mapping.

    Supported formats:
        cassini        → use Cassini-computed shifts
        none           → all zeros
        manual:0,50,100,150  → comma-separated shifts in ms (one per group)
    """
    if mode_str == "cassini":
        return "cassini", dict(cassini_shifts)
    if mode_str == "none":
        return "none", {g: 0 for g in range(num_groups)}
    if mode_str.startswith("manual:"):
        parts = mode_str[len("manual:"):].split(",")
        shifts = {}
        for g, val_ms in enumerate(parts):
            shifts[g] = int(float(val_ms) * 1000)
        return mode_str, shifts
    raise ValueError(f"Unknown mode: {mode_str}")


def run_single_mode(mode_label, time_shifts_by_group, cassini_result, rep_workload, topology,
                    compact_wl, output_dir, ecmp_routes):
    """Run dynamic execution with the given time-shift config, generate outputs."""
    # 每个 mode 重建 job_dag——JobManager.mark_completed() 会修改 dep_count，
    # 模式之间不能共用同一个 DAG 实例
    job_dag = JobDAG.from_compact(compact_wl)
    policy = CassiniSchedulingPolicy(cassini_result)
    empty_wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
    policy.initialize(empty_wl, topology)
    policy.compute_order = {}

    # Build delay_by_job: only first iter of each group gets the shift
    delay_by_job: dict[int, int] = {}
    groups_seen: set[int] = set()
    for jid, info in sorted(compact_wl.job_expansion_info.items()):
        if info.job_group_id not in groups_seen:
            groups_seen.add(info.job_group_id)
            delay_by_job[jid] = time_shifts_by_group.get(info.job_group_id, 0)

    executor = DynamicExecutor(topology=topology, policy=policy, analyzer=LightweightAnalyzer())
    job_expander = JobExpander(task_id_allocator=TaskIdAllocator(), profile_store=None)

    t_start = time.time()
    result = executor.execute_dynamic(
        job_dag=job_dag,
        job_expansion_info=compact_wl.job_expansion_info,
        job_policy=DelayByJobPolicy(delay_by_job),
        job_expander=job_expander,
    )
    elapsed = time.time() - t_start

    print(f"\n  [{mode_label}] Makespan: {result.makespan_us / 1000:.2f} ms  "
          f"({result.makespan_us / 1e6:.3f} s)")
    print(f"  [{mode_label}] Tasks completed: {len(result.per_task)}")
    print(f"  [{mode_label}] Execution time: {elapsed:.2f}s")

    # Output files with mode suffix
    suffix = f"_{mode_label}" if mode_label else ""
    result_path = os.path.join(output_dir, f"result{suffix}.json")
    result.to_json(result_path)
    meta_path = os.path.join(output_dir, f"task_meta{suffix}.json")
    with open(meta_path, "w") as f:
        json.dump({str(k): v for k, v in executor._task_meta.items()}, f)
    print(f"  Saved: {result_path}")

    # Save per-mode link contention data
    contended, _ = _find_contended_links(rep_workload, ecmp_routes)
    save_link_contention_actual(result, executor, compact_wl, ecmp_routes, contended, output_dir, mode_label)

    return result


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
    parser.add_argument("--min-trace-dur", type=int, default=0,
                        help="Filter tasks with duration <= N us from trace output (default 0 = no filter)")
    parser.add_argument("--modes", nargs="+", default=["cassini"],
                        help="Shift modes: cassini, none, manual:0,50,100,150 (ms)")

    args = parser.parse_args()

    # ── Resolve config ──
    config = {
        "topology": DEFAULT_TOPO,
        "output_dir": DEFAULT_OUTPUT,
        "placement": "contiguous",
        "placement_clusters": 2,
        "gpus_per_server": None,
        "step_deg": 5,
        "modes": ["cassini"],
        "min_trace_dur": 0,
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

    # CLI overrides for config-level fields
    if args.modes is not None and args.modes != ["cassini"]:
        config["modes"] = args.modes
    if args.min_trace_dur != 0:
        config["min_trace_dur"] = args.min_trace_dur

    min_trace_dur = config["min_trace_dur"]

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
    ecmp_routes = EcmpStrategy().compute_routes(rep_workload, topology)
    cassini_result = CassiniAnalyzer(topology, step_deg=step_deg).analyze(
        rep_workload, route_table=ecmp_routes)
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

    # ── Phase 4: Dynamic execution (one run per mode) ──
    num_groups = len(set(i.job_group_id for i in compact_wl.job_expansion_info.values()))
    results = {}

    for mode_str in config["modes"]:
        mode_label, shifts = parse_mode_shifts(mode_str, cassini_result.time_shifts, num_groups)
        print("\n" + "=" * 60)
        print(f"Phase 4 [{mode_label}]: shifts={ {g: f'{s/1000:.1f}ms' for g,s in shifts.items()} }")
        print("=" * 60)

        result = run_single_mode(
            mode_label, shifts, cassini_result, rep_workload, topology,
            compact_wl, output_dir, ecmp_routes,
        )
        results[mode_label] = result

    # CPM link contention (所有模式共享，只存一份)
    save_link_contention_cpm(cassini_result, rep_workload, ecmp_routes, compact_wl, output_dir)

    # 为每个模式生成 Chrome Trace；仅 cassini 模式额外生成 CPM 对比 trace
    print("\n" + "=" * 60)
    print("Generating traces")
    print("=" * 60)
    for mode_label, result in results.items():
        meta_file = os.path.join(output_dir, f"task_meta_{mode_label}.json")
        with open(meta_file) as f:
            _meta = {int(k): v for k, v in json.load(f).items()}
        _fake_exec = type('_E', (), {'_task_meta': _meta})()
        suffix = f"_{mode_label}" if mode_label else ""
        generate_chrome_trace(result, _fake_exec, compact_wl, output_dir, min_dur_us=min_trace_dur, suffix=suffix)
        if mode_label == "cassini":
            generate_cpm_comparison_trace(result, _fake_exec, compact_wl, cassini_result, rep_workload, output_dir, min_dur_us=min_trace_dur)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for mode_label, result in results.items():
        print(f"  [{mode_label}] makespan: {result.makespan_us / 1000:.2f} ms  "
              f"tasks: {len(result.per_task)}")
    print("\nDone")
    print(f"  Output: {output_dir}/")
    print(f"  result.json — ExecutionResult with per-task timing")
    print(f"  task_meta.json — Task metadata for visualization")
    print(f"  trace.json — Open in chrome://tracing to view")


if __name__ == "__main__":
    main()

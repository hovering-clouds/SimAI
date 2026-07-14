"""
Puppeteer CPM Comparison — verify whether the pre-computed critical path
(CPM / TTE) matches the actual critical path under execution contention.

Runs the full Puppeteer strategy (greedy routing + TTE-aware allocation
+ co-start coordination), then compares static vs actual critical paths
via console report, Chrome Trace, and matplotlib plots.

Usage:
    uv run python scripts/run_puppeteer_cpm_comparison.py
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.static_analysis.passes.critical_path import analyze_cpm, CriticalPathInfo, TaskTimingInfo
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy


# ── Default configs ─────────────────────────────────────────────

_CFG = {
    "cpp": {
        "aicb": "inputs/aicb-workload/A100-gpt_7B_ws4_pp1-world_size4-tp4-pp1-ep1-gbs2-mbs2-seq4096-MOE-False-GEMM-False-flash_attn-True.txt",
        "topo": "inputs/topologies/Cassini_Fig10_24g_l1-6_l2-4_l3-3_nv4_400Gbps_A100",
        "dp": 3,  # override DP beyond header for more traffic
    },
    "1f1b": {
        "aicb": "inputs/aicb-workload/A100-gpt_7B_ws4_pp2-world_size4-tp2-pp2-ep1-gbs32-mbs4-seq4096-MOE-False-GEMM-False-flash_attn-True.txt",
        "topo": "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100",
    },
}
DEFAULT_SERIALIZER = "cpp"
K_PATHS = 4


# ── Core analysis ───────────────────────────────────────────────

def compute_actual_critical_path(workload, result):
    """Run CPM on actual execution durations from ExecutionResult.

    Returns CriticalPathInfo where slack=0 tasks are the true bottleneck chain.
    """
    task_map = {t.task_id: t for t in workload.tasks}
    timings = result.per_task

    def actual_dur(tid):
        t = timings[tid]
        return t.end_time_us - t.start_time_us

    # topological order
    sorted_tids = []
    visited = set()

    def dfs(tid):
        if tid in visited:
            return
        visited.add(tid)
        for dep in task_map[tid].deps:
            dfs(dep)
        sorted_tids.append(tid)

    for t in workload.tasks:
        dfs(t.task_id)

    # forward pass
    es, ef = {}, {}
    for tid in sorted_tids:
        pred = [ef[d] for d in task_map[tid].deps if d in ef]
        es[tid] = max(pred) if pred else 0
        ef[tid] = es[tid] + actual_dur(tid)

    makespan = max(ef.values()) if ef else 0

    # successors for backward pass
    successors = defaultdict(list)
    for t in workload.tasks:
        for dep in t.deps:
            successors[dep].append(t.task_id)

    # backward pass
    ls, lf = {}, {}
    for tid in reversed(sorted_tids):
        children = successors.get(tid, [])
        lf[tid] = makespan if not children else min(ls[c] for c in children)
        ls[tid] = lf[tid] - actual_dur(tid)

    task_timings, critical_tasks = {}, []
    for tid in sorted_tids:
        slack = float(ls[tid] - es[tid])
        is_crit = slack == 0.0
        task_timings[tid] = TaskTimingInfo(
            task_id=tid,
            earliest_start_us=es[tid],
            earliest_finish_us=ef[tid],
            latest_start_us=ls[tid],
            latest_finish_us=lf[tid],
            slack_us=slack,
            is_critical=is_crit,
        )
        if is_crit:
            critical_tasks.append(tid)

    return CriticalPathInfo(task_timings, critical_tasks, makespan, "actual_cpm")


# ── Console report ──────────────────────────────────────────────

def print_report(static_cp, actual_cp, tte_info, result):
    """Print critical-path comparison report to terminal."""
    static_set = set(static_cp.critical_tasks)
    actual_set = set(actual_cp.critical_tasks)
    overlap = static_set & actual_set

    tte_critical = {
        tid for tid, info in (tte_info or {}).items()
        if info.priority_class == "critical"
    }

    print()
    print("=" * 62)
    print("  Critical Path Comparison Report")
    print("=" * 62)
    print(f"  Makespan:               {result.makespan_us:>12,} us")
    print(f"  Static CPM critical:     {len(static_set):>8d}")
    print(f"  Actual CPM critical:     {len(actual_set):>8d}")
    print(f"  Overlap:                 {len(overlap):>8d}")
    if static_set:
        print(f"  Static-hit ratio:        {len(overlap)/len(static_set)*100:>5.1f}%")
    if actual_set:
        print(f"  False positives:         {len(static_set - overlap):>8d}")
        print(f"  False negatives:         {len(actual_set - overlap):>8d}")
    print(f"  TTE critical flows:      {len(tte_critical):>8d}")
    print(f"  TTE crit & actual crit:  {len(tte_critical & actual_set):>8d}")
    print(f"  Predicted makespan:      {static_cp.makespan_us:>12,} us")
    print(f"  CPM-on-actual makespan:  {actual_cp.makespan_us:>12,} us")
    print("=" * 62)
    print()


# ── Chrome Trace export with critical-path annotation ───────────

def export_critical_path_trace(workload, result, static_cp, actual_cp, path):
    """Chrome Trace JSON with three pids:

    - pid=0: ALL tasks (reference)
    - pid=1: Static CPM critical tasks (predicted bottleneck)
    - pid=2: Actual CPM critical tasks (true bottleneck)
    """
    task_map = {t.task_id: t for t in workload.tasks}
    timings = result.per_task
    events: list[dict] = []

    def lane(tid):
        task = task_map[tid]
        t = timings[tid]
        if task.is_compute():
            return t.node * 2 + 1
        return (task.src or 0) * 2 + 2

    # (pid, label, set_of_critical_task_ids)
    pid_configs = [
        (1, "Static CPM", set(static_cp.critical_tasks)),
        (2, "Actual CPM", set(actual_cp.critical_tasks)),
    ]

    def add_metadata(pid, label, tracks):
        events.append(dict(name="process_name", ph="M", pid=pid, tid=0,
                           args=dict(name=f"{label}")))
        for (t, node, lbl) in sorted(tracks, key=lambda x: x[0]):
            events.append(dict(name="thread_name", ph="M", pid=pid, tid=t,
                               args=dict(name=f"Node {node} ({lbl})")))

    # ── Collect track info ──
    all_tracks: set[tuple[int, int, str]] = set()
    crit_tracks: dict[int, set[tuple[int, int, str]]] = {p: set() for p, _, _ in pid_configs}

    for tid_ in timings:
        task = task_map.get(tid_)
        if task is None:
            continue
        t_ = timings[tid_]
        l = lane(tid_)
        lbl = "Compute" if task.is_compute() else "Comm"
        all_tracks.add((l, t_.node, lbl))
        for pid, _, cset in pid_configs:
            if tid_ in cset:
                crit_tracks[pid].add((l, t_.node, lbl))

    add_metadata(0, "All Tasks", all_tracks)
    for pid, label, _ in pid_configs:
        add_metadata(pid, label, crit_tracks[pid])

    # ── Task slices ──
    for tid, t in timings.items():
        task = task_map.get(tid)
        if task is None:
            continue
        dur = t.end_time_us - t.start_time_us
        if dur == 0:
            continue
        l = lane(tid)
        cat = "compute" if task.is_compute() else "flow"
        name = f"L{task.layer_id} {task.phase.value}" if task.is_compute() \
            else f"{task.comm_type.value} {task.src}->{task.dst}"

        # pid=0: all tasks
        events.append(dict(name=name, cat=cat, ph="X",
                           ts=t.start_time_us, dur=dur, pid=0, tid=l))

        # pid=1: static CPM critical, pid=2: actual CPM critical
        for pid, _, cset in pid_configs:
            if tid in cset:
                events.append(dict(name=name, cat=cat, ph="X",
                                   ts=t.start_time_us, dur=dur, pid=pid, tid=l))

    with open(path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    n_s = len(pid_configs[0][2])
    n_a = len(pid_configs[1][2])
    print(f"  [trace] {len(events)} events — "
          f"pid=0: ALL, pid=1: static({n_s}), pid=2: actual({n_a})")
    print(f"  [trace] {path}")


# ── matplotlib plots ────────────────────────────────────────────

def plot_makespan_breakdown(workload, static_cp, actual_cp, tte_info, result, path):
    """Stacked horizontal bar: static vs actual makespan, segmented by phase + type.

    Each bar shows the total makespan broken down by training phase
    (forward, backward_input, backward_weight, optimizer) and task type
    (compute vs flow).  The gap between bar widths shows total inflation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        return

    task_map = {t.task_id: t for t in workload.tasks}
    timings = result.per_task
    phases = ["forward", "backward_input", "backward_weight", "optimizer"]
    phase_labels = ["FWD", "BWD-in", "BWD-wt", "OPT"]

    def sum_duration(tids, phase, is_compute, use_actual):
        total = 0
        for tid in tids:
            t = task_map[tid]
            if t.phase.value != phase:
                continue
            if t.is_compute() != is_compute:
                continue
            if use_actual:
                tt = timings[tid]
                total += tt.end_time_us - tt.start_time_us
            else:
                st = static_cp.task_timings.get(tid)
                if st is not None:
                    total += st.earliest_finish_us - st.earliest_start_us
        return total

    all_critical = sorted(
        set(static_cp.critical_tasks) | set(actual_cp.critical_tasks),
    )

    # accumulate segments: list of (label, static_dur, actual_dur, color)
    segments = []
    # compute → pastel (mix with white), flow → saturated (mix with dark)
    shade = {"compute": 0.30, "flow": 0.70}
    mix_tint = {"compute": np.array([0.92, 0.92, 0.92]), "flow": np.array([0.10, 0.10, 0.10])}
    base_colors = {
        "forward":         (0.15, 0.60, 0.15),  # green
        "backward_input":  (0.82, 0.18, 0.14),  # red/crimson
        "backward_weight": (0.55, 0.12, 0.68),  # purple
        "optimizer":       (0.12, 0.42, 0.72),  # blue
    }

    for phase, plabel in zip(phases, phase_labels):
        for ttype, tlabel in [("compute", "C"), ("flow", "F")]:
            is_c = ttype == "compute"
            s = sum_duration(all_critical, phase, is_c, use_actual=False)
            a = sum_duration(all_critical, phase, is_c, use_actual=True)
            if s == 0 and a == 0:
                continue
            base = np.array(base_colors[phase])
            color = tuple(base * shade[ttype] + mix_tint[ttype] * (1 - shade[ttype]))
            segments.append((f"{plabel}-{tlabel}", s, a, color))

    if not segments:
        return

    # Bar heights
    fig, ax = plt.subplots(figsize=(10, 2.5))

    # Compute cumulative positions
    static_pos = [(0, s) for _, s, _, _ in segments]
    actual_pos = [(0, a) for _, _, a, _ in segments]

    y_top = 0.85
    y_bot = -0.15

    # Draw segments for static (top) and actual (bottom)
    static_left = 0
    actual_left = 0
    for label, s_dur, a_dur, color in segments:
        if s_dur > 0:
            ax.barh(y_top, s_dur, left=static_left, height=0.5,
                    color=color, ec="white", lw=0.5, alpha=0.85)
            static_left += s_dur
        if a_dur > 0:
            ax.barh(y_bot, a_dur, left=actual_left, height=0.5,
                    color=color, ec="white", lw=0.5, alpha=0.85)
            actual_left += a_dur

    # Annotate total makespan at bar ends
    ax.text(static_left, y_top, f"  {static_left:,} us",
            va="center", fontsize=9, fontweight="bold")
    ax.text(actual_left, y_bot, f"  {actual_left:,} us",
            va="center", fontsize=9, fontweight="bold")

    # Highlight inflation gap
    max_x = max(static_left, actual_left) * 1.15
    ax.plot([static_left, actual_left], [y_top, y_bot],
            color="#e74c3c", lw=2, ls="--", marker="o", markersize=6)
    gap_label = f"+{actual_left - static_left:,} us  ({((actual_left/static_left)-1)*100:+.0f}%)"
    ax.annotate(gap_label,
                xy=((static_left + actual_left) / 2, (y_top + y_bot) / 2),
                ha="center", va="center", fontsize=9, color="#c0392b",
                fontweight="bold",
                bbox=dict(facecolor="white", ec="#e74c3c", boxstyle="round,pad=0.3"))

    ax.set_yticks([y_top, y_bot])
    ax.set_yticklabels(["Static (ideal)", "Actual (exec)"], fontsize=10)
    ax.set_xlim(0, max_x)
    ax.set_xlabel("Time (us)", fontsize=10)
    ax.set_title("Critical Path: Static vs Actual Makespan Breakdown", fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Legend: per-phase + type
    legend_handles = []
    for phase, plabel in zip(phases, phase_labels):
        base = np.array(base_colors[phase])
        for ttype, tlabel in [("compute", "Compute"), ("flow", "Flow")]:
            c = tuple(base * shade[ttype] + mix_tint[ttype] * (1 - shade[ttype]))
            legend_handles.append(mpatches.Patch(color=c, label=f"{plabel} {tlabel}"))

    ax.legend(handles=legend_handles, loc="upper center", fontsize=7,
              ncol=4, bbox_to_anchor=(0.5, -0.35), framealpha=0.9)

    # Stats below
    overlap_n = len(set(static_cp.critical_tasks) & set(actual_cp.critical_tasks))
    info = (
        f"Static critical: {len(static_cp.critical_tasks)}  |  "
        f"Actual critical: {len(actual_cp.critical_tasks)}  |  "
        f"Overlap: {overlap_n}"
    )
    fig.text(0.5, -0.12, info, ha="center", fontsize=8,
             bbox=dict(facecolor="white", alpha=0.8, boxstyle="round"))

    plt.tight_layout(rect=[0, 0.18, 1, 1])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


def plot_flow_inflation(workload, static_cp, actual_cp, tte_info, flow_timing, result, path):
    """Scatter: ideal flow duration (X) vs actual (Y), coloured by TTE priority.
    Points above the diagonal were inflated by contention.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    timings = result.per_task
    if not flow_timing:
        return

    cmap = {"critical": "#e74c3c", "elastic": "#f39c12", "background": "#3498db"}
    lmap = {"critical": "TTE Critical", "elastic": "TTE Elastic", "background": "TTE Background"}

    fig, ax = plt.subplots(figsize=(7, 7))
    for cls in ("critical", "elastic", "background"):
        xs, ys = [], []
        for tid, ft in flow_timing.items():
            info = tte_info.get(tid)
            if info is None or info.priority_class != cls:
                continue
            ideal = ft.finish_time_us - ft.start_time_us
            actual = timings[tid].end_time_us - timings[tid].start_time_us
            if ideal > 0 and actual > 0:
                xs.append(ideal)
                ys.append(actual)
        if xs:
            ax.scatter(xs, ys, c=cmap[cls], label=lmap[cls], s=8, alpha=0.6, edgecolors="none")

    m = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, m], [0, m], "k--", lw=1, alpha=0.5, label="y=x (no inflation)")
    ax.set_xlim(0, m)
    ax.set_ylim(0, m)
    ax.set_xlabel("Ideal duration (us)")
    ax.set_ylabel("Actual duration (us)")
    ax.set_title("Flow Duration Inflation - Ideal vs Actual", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {path}")


# ── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Puppeteer CPM comparison: verify static vs actual critical path")
    parser.add_argument("--mode", choices=["cpp", "1f1b"], default="cpp",
                        help="1F1B uses pp=2 workload + AlibabaHPN topology")
    parser.add_argument("--aicb", default=None,
                        help="Override AICB workload file")
    parser.add_argument("--topo", default=None,
                        help="Override topology file")
    parser.add_argument("--output", "-o", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    mode = args.mode
    cfg = _CFG[mode]
    aicb_file = args.aicb or cfg["aicb"]
    topo_file = args.topo or cfg["topo"]
    output_dir = args.output or f"outputs/puppeteer_cpm"

    os.makedirs(output_dir, exist_ok=True)

    # ── Load workload ───────────────────────────────────────────
    print("=" * 60)
    print("Loading AICB workload")
    print("=" * 60)
    ap = AicbParser()
    header, items = ap.parse(aicb_file)
    tp, pp, ep = header.tp, header.pp, header.ep
    dp = cfg.get("dp") or (header.all_gpus // (tp * pp * ep))
    total_gpus = tp * dp * pp * ep
    print(f"  model: {os.path.basename(aicb_file)}")
    print(f"  tp={tp} dp={dp} pp={pp} ep={ep}  gpus={total_gpus}")

    job = Job(
        job_id=0,
        name=f"puppeteer-{mode}",
        assigned_nodes=list(range(total_gpus)),
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )
    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job, comm_algo="ring")
    print(f"  tasks: {len(workload.tasks)}  "
          f"(compute={len(workload.get_compute_tasks())}, "
          f"flow={len(workload.get_flow_tasks())})")

    # ── Load topology ───────────────────────────────────────────
    print()
    print("=" * 60)
    print("Loading topology")
    print("=" * 60)
    topology = TopologyLoader().load(topo_file)
    print(f"  nodes={topology.total_nodes}  gpus={topology.gpu_count}  "
          f"switches={topology.switch_count}")
    if total_gpus > topology.gpu_count:
        raise ValueError(f"need {total_gpus} GPUs, only {topology.gpu_count} available")

    # ── Puppeteer pipeline ─────────────────────────────────────
    print()
    print("=" * 60)
    print(f"Running: {mode} + Puppeteer (greedy routing + TTE + co-start)")
    print("=" * 60)

    t0 = time.time()
    puppet = PuppeteerAnalyzer(topology, k_paths=K_PATHS, serializer=mode)
    pa_result = puppet.analyze(workload)

    policy = PuppeteerSchedulingPolicy(
        route_table=pa_result.route_table,
        tte_info=pa_result.tte_info,
        resource_dependency=pa_result.resource_dependency,
        execution_plan=pa_result.execution_plan,
        allocator_mode="weighted",
    )
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    result = executor.execute(workload)
    elapsed = time.time() - t0

    result.to_json(os.path.join(output_dir, "result_full.json"))
    print(f"  schedule: {mode}  pp={pp}")
    print(f"  makespan: {result.makespan_us} us ({result.makespan_us/1000:.1f} ms)")
    print(f"  completed in {elapsed:.2f}s")

    # ── Critical path analysis ──────────────────────────────────
    print()
    print("=" * 60)
    print("Critical path analysis")
    print("=" * 60)

    static_cp = analyze_cpm(workload, pa_result.route_table, topology)
    actual_cp = compute_actual_critical_path(workload, result)
    print(f"  Static CPM:  {len(static_cp.critical_tasks)} critical tasks, "
          f"predicted makespan={static_cp.makespan_us} us")
    print(f"  Actual CPM:  {len(actual_cp.critical_tasks)} critical tasks, "
          f"makespan={actual_cp.makespan_us} us")

    print_report(static_cp, actual_cp, pa_result.tte_info, result)

    # ── Visualisations ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("Generating visualizations")
    print("=" * 60)

    plot_makespan_breakdown(
        workload, static_cp, actual_cp, pa_result.tte_info, result,
        os.path.join(output_dir, "makespan_breakdown.svg"),
    )
    plot_flow_inflation(
        workload, static_cp, actual_cp, pa_result.tte_info,
        pa_result.flow_timing, result,
        os.path.join(output_dir, "flow_inflation_scatter.svg"),
    )
    export_critical_path_trace(
        workload, result, static_cp, actual_cp,
        os.path.join(output_dir, "trace_critical.json"),
    )

    # ── Save report ─────────────────────────────────────────────
    metrics = {
        "makespan_us": result.makespan_us,
        "static_critical_count": len(static_cp.critical_tasks),
        "actual_critical_count": len(actual_cp.critical_tasks),
        "overlap_count": len(set(static_cp.critical_tasks) & set(actual_cp.critical_tasks)),
        "static_makespan_us": static_cp.makespan_us,
        "actual_makespan_us": actual_cp.makespan_us,
        "tte_critical_count": len(pa_result.tte_info or {}),
        "elapsed_s": elapsed,
    }
    report_path = os.path.join(output_dir, "cpm_comparison.json")
    with open(report_path, "w") as f:
        json.dump({
            "workload": aicb_file,
            "topology": topo_file,
            "config": {"mode": mode, "tp": tp, "dp": dp, "pp": pp, "k_paths": K_PATHS},
            "metrics": metrics,
        }, f, indent=2)
    print(f"\n  report -> {report_path}")
    print("Done.")


if __name__ == "__main__":
    main()

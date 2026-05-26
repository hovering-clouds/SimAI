"""
Minimal end-to-end demo showing Cassini's time-shift benefit.

Two identical jobs share a bottleneck 10 Gbps link.  Each iteration sends a
250 MB flow that takes 200 ms at full speed, or 400 ms when two flows compete
(fair-share → 5 Gbps each).

Cassini extracts the communication pattern from a single-iteration workload,
computes a time-shift that staggers the two jobs by half a period, then
applies that shift to a multi-iteration execution.  With 20 iterations the
startup delay is amortised and Cassini reduces makespan by ~35 %.

Usage:
    python scripts/run_cassini_demo.py
"""

import sys, os
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import (
    CommType, Job, Meta, P2PWorkload, ParallelismConfig,
    Phase, Task, TaskType,
)
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.static_analysis.strategies.cassini_strategy import CassiniAnalyzer, CassiniAnalysisResult
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.passes.routing import BfsStrategy
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy

COMPUTE_US = 100_000      # 100 ms
FLOW_BYTES = 100_000_000  #  80 ms at 10 Gbps, 160 ms at 5 Gbps
NUM_ITERS = 20
BOTTLENECK_BW = 10.0  # Gbps


def _build_topo():
    """4 GPUs in a line: GPU0—GPU1—GPU2—GPU3, all links 10 Gbps."""
    topo = NetworkTopology()
    topo.total_nodes = 4
    topo.gpu_count = 4
    topo.gpu_nodes = [0, 1, 2, 3]
    topo.switch_nodes = []
    topo.node_types = {i: "gpu" for i in range(4)}
    for s, d in [(0, 1), (1, 0), (1, 2), (2, 1), (2, 3), (3, 2)]:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=BOTTLENECK_BW,
                           latency_us=1.0, error_rate=0.0))
    return topo


def _build_workload(num_iters):
    """2 jobs with *num_iters* iterations each, sharing link (1,2).

    Job 0: GPUs [0,2], flow 0→2 (path: 0-1-2)
    Job 1: GPUs [1,3], flow 1→3 (path: 1-2-3)
    """
    tasks = []
    tid = 0
    for jid, gpus, src_gpu, dst_gpu in [(0, [0, 2], 0, 2), (1, [1, 3], 1, 3)]:
        prev_c1 = None
        for it in range(num_iters):
            deps_c0 = [prev_c1.task_id] if prev_c1 is not None else []
            c0 = Task(task_id=tid, job_id=jid, type=TaskType.COMPUTE,
                      node=gpus[0], duration_us=COMPUTE_US, iteration=it,
                      phase=Phase.FORWARD, layer_id=it, item_id=tid, deps=deps_c0)
            tid += 1; tasks.append(c0)

            fid = tid
            f0 = Task(task_id=tid, job_id=jid, type=TaskType.FLOW,
                      src=src_gpu, dst=dst_gpu, size_bytes=FLOW_BYTES,
                      iteration=it, phase=Phase.FORWARD, layer_id=it,
                      item_id=tid, deps=[c0.task_id],
                      comm_type=CommType.TP_ALLREDUCE_RING)
            tid += 1; tasks.append(f0)

            deps_c1 = [fid]
            if prev_c1 is not None:
                deps_c1.append(prev_c1.task_id)
            c1 = Task(task_id=tid, job_id=jid, type=TaskType.COMPUTE,
                      node=gpus[1], duration_us=COMPUTE_US, iteration=it,
                      phase=Phase.FORWARD, layer_id=it, item_id=tid, deps=deps_c1)
            tid += 1; tasks.append(c1)
            prev_c1 = c1

    return P2PWorkload(
        version="1.0", meta=Meta(num_jobs=2, num_nodes=4),
        jobs=[Job(job_id=jid, name=f"job_{jid}", assigned_nodes=gpus,
                   parallelism=ParallelismConfig(tp=2))
              for jid, gpus, _, _ in [(0, [0, 2], 0, 2), (1, [1, 3], 1, 3)]],
        tasks=tasks,
    )


def main():
    topo = _build_topo()

    # ---- Phase 1: analyse single-iteration workload for correct pattern ----
    wl_1iter = _build_workload(1)
    analyzer = CassiniAnalyzer(topo, step_deg=15)
    analysis = analyzer.analyze(wl_1iter)

    print("=" * 60)
    print("Cassini Demo — Bottleneck Link Contention")
    print("=" * 60)
    print(f"  Topology: 4-GPU line, {BOTTLENECK_BW} Gbps links")
    print(f"  Shared link: (1, 2)")
    print(f"  Each flow: {FLOW_BYTES / 1e6:.0f} MB "
          f"({FLOW_BYTES * 8 / (BOTTLENECK_BW * 1e9) * 1e3:.0f} ms at "
          f"{BOTTLENECK_BW:.0f} Gbps, {FLOW_BYTES * 8 / (BOTTLENECK_BW / 2 * 1e9) * 1e3:.0f} ms at "
          f"{BOTTLENECK_BW / 2:.0f} Gbps fair-share)")
    print(f"  Iterations: {NUM_ITERS}")

    # Per-job analysis
    pat = analysis.communication_patterns
    it0 = pat[0].iteration_time_us
    print(f"\n  Communication pattern (single iteration):")
    print(f"    Iteration time: {it0 / 1000:.0f} ms")
    print(f"    Time-shifts: job_0={analysis.time_shifts[0] / 1000:.1f} ms, "
          f"job_1={analysis.time_shifts[1] / 1000:.1f} ms")

    # ---- Phase 2: execute multi-iteration workload ----
    wl = _build_workload(NUM_ITERS)

    # Default policy
    print(f"\n{'=' * 60}")
    print("Running: default (BFS + FairShare)")
    print("=" * 60)
    def_analysis = DefaultAnalyzer(topo).analyze(wl)
    def_policy = DefaultSchedulingPolicy(analysis=def_analysis)
    def_result = AnalyticalExecutor(topo, def_policy).execute(wl)
    def_flows = [t for t in wl.tasks if t.is_flow()]
    def_flow_times = [def_result.per_task[t.task_id].end_time_us -
                      def_result.per_task[t.task_id].start_time_us
                      for t in def_flows]

    # Cassini policy (using time-shifts from 1-iter analysis, applied to multi-iter)
    # Re-analyse on multi-iter workload for correct route_table
    cassini_analysis = CassiniAnalyzer(topo, step_deg=15).analyze(wl)
    cassini_result = CassiniAnalysisResult(
        route_table=cassini_analysis.route_table,
        critical_path=cassini_analysis.critical_path,
        communication_patterns=cassini_analysis.communication_patterns,
        time_shifts=analysis.time_shifts,  # from 1-iter analysis!
        execution_plan=cassini_analysis.execution_plan,
    )

    print(f"\n{'=' * 60}")
    print("Running: cassini-default")
    print("=" * 60)
    cas_policy = CassiniSchedulingPolicy(cassini_result)
    cas_executor = AnalyticalExecutor(topo, cas_policy)
    cas_result = cas_executor.execute(wl)
    cas_flows = [t for t in wl.tasks if t.is_flow()]
    cas_flow_times = [cas_result.per_task[t.task_id].end_time_us -
                      cas_result.per_task[t.task_id].start_time_us
                      for t in cas_flows]

    # ---- Results ----
    print(f"\n{'=' * 60}")
    print("Results")
    print("=" * 60)

    def_ms = def_result.makespan_us / 1000
    cas_ms = cas_result.makespan_us / 1000
    speedup = def_ms / cas_ms if cas_ms > 0 else 0
    flow_speedup = (sum(def_flow_times) / len(def_flow_times)) / \
                   (sum(cas_flow_times) / len(cas_flow_times))

    print(f"  {'':<22s} {'Default':>12s} {'Cassini':>12s} {'Change':>12s}")
    print(f"  {'Makespan':<22s} {def_ms:>11.1f}ms {cas_ms:>11.1f}ms "
          f"{'%.2fx faster' % speedup if speedup > 1 else '%.2fx slower' % (1/speedup):>12s}")
    print(f"  {'Avg flow time':<22s} "
          f"{sum(def_flow_times)/len(def_flow_times)/1000:>11.1f}ms "
          f"{sum(cas_flow_times)/len(cas_flow_times)/1000:>11.1f}ms "
          f"{'%.2fx faster' % flow_speedup:>12s}")
    print(f"  {'Job iteration':<22s} "
          f"{def_result.job_iteration_times.get(0,0)/1000:>11.1f}ms "
          f"{cas_result.job_iteration_times.get(0,0)/1000:>11.1f}ms")
    print(f"  {'Time-shift (job 1)':<22s} {'—':>12s} "
          f"{analysis.time_shifts[1]/1000:>11.1f}ms")

    if speedup > 1.0:
        print(f"\n  => Cassini reduces makespan by {def_ms - cas_ms:.0f} ms "
              f"({speedup:.2f}x speedup)")
    else:
        print(f"\n  Note: with {NUM_ITERS} iterations the time-shift startup "
              f"delay still affects makespan.  More iterations = larger benefit.")


if __name__ == "__main__":
    main()

"""
Puppeteer reproduction: compare scheduling policies end-to-end.

Runs a workload through all comparison modes and reports key metrics:

    modes:
        default       BFS shortest path + fair share (baseline)
        route-only    Puppeteer route table + fair share
        tte-only      BFS shortest path + TTE-aware allocation
        route-tte     Puppeteer route table + TTE-aware allocation
        full          Puppeteer route table + TTE-aware + co-start coordination

Usage:
    python scripts/run_puppeteer_reproduce.py
"""

import os
import sys
import time

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.bandwidth_allocators.fair_share_allocator import FairShareAllocator


def run_default(workload, topology, **_):
    """Baseline: DefaultSchedulingPolicy."""
    analysis = DefaultAnalyzer(topology).analyze(workload)
    policy = DefaultSchedulingPolicy(analysis=analysis)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_route_only(workload, topology, k_paths=4):
    """Puppeteer routing + fair share (no TTE, no coordination)."""
    puppet = PuppeteerAnalyzer(topology, k_paths=k_paths)
    result = puppet.analyze(workload)

    # Use DefaultSchedulingPolicy but override routing hints with route table
    # Actually, we need a custom approach: use Puppeteer routes but fair-share allocator
    # For simplicity, build a custom policy that uses route table + fair share
    class RouteOnlyPolicy(PuppeteerSchedulingPolicy):
        def __init__(self, route_table, execution_plan):
            from src.static_analysis.passes.puppeteer_coordination import ResourceDependencyTable
            # Give all flows background TTE so weighted allocator treats them equally
            tte_info = {tid: type('TTEInfo', (), {
                'task_id': tid, 'tte_us': float('inf'),
                'priority_score': 1.0, 'priority_class': 'background',
            })() for tid in route_table.paths}
            super().__init__(
                route_table=route_table,
                tte_info=tte_info,
                resource_dependency=ResourceDependencyTable(),
                execution_plan=execution_plan,
                allocator_mode="weighted",
            )

    policy = RouteOnlyPolicy(result.route_table, result.execution_plan)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_tte_only(workload, topology, **_):
    """BFS shortest paths + TTE-aware allocation (no greedy routing)."""
    from src.static_analysis.passes.routing import BfsStrategy
    from src.static_analysis.passes.puppeteer_tte import compute_tte
    from src.static_analysis.passes.task_serializer import CppReferenceSerializer
    from src.static_analysis.passes.puppeteer_coordination import ResourceDependencyTable

    # Compute BFS shortest paths + TTE
    route_table = BfsStrategy().compute_routes(workload, topology)
    plan = CppReferenceSerializer().serialize(workload)
    tte_info, _ = compute_tte(workload, route_table, plan)

    policy = PuppeteerSchedulingPolicy(
        route_table=route_table,
        tte_info=tte_info,
        resource_dependency=ResourceDependencyTable(),
        execution_plan=plan,
        allocator_mode="weighted",
    )
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_route_tte(workload, topology, k_paths=4):
    """Puppeteer routing + TTE-aware allocation (no coordination)."""
    from src.static_analysis.passes.puppeteer_coordination import ResourceDependencyTable

    puppet = PuppeteerAnalyzer(topology, k_paths=k_paths)
    result = puppet.analyze(workload)

    policy = PuppeteerSchedulingPolicy(
        route_table=result.route_table,
        tte_info=result.tte_info,
        resource_dependency=ResourceDependencyTable(),
        execution_plan=result.execution_plan,
        allocator_mode="weighted",
    )
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_full(workload, topology, k_paths=4):
    """Full Puppeteer: routing + TTE-aware + coordination."""
    puppet = PuppeteerAnalyzer(topology, k_paths=k_paths)
    result = puppet.analyze(workload)

    policy = PuppeteerSchedulingPolicy(
        route_table=result.route_table,
        tte_info=result.tte_info,
        resource_dependency=result.resource_dependency,
        execution_plan=result.execution_plan,
        allocator_mode="weighted",
    )
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def _summarize(result, name, workload):
    """Extract key metrics from an execution result."""
    flow_times = []
    compute_times = []
    for tid, timing in result.per_task.items():
        elapsed = timing.end_time_us - timing.start_time_us
        if timing.task_type == "flow":
            flow_times.append(elapsed)
        else:
            compute_times.append(elapsed)

    flow_ids = {t.task_id for t in workload.tasks if t.is_flow()}

    metrics = {
        "mode": name,
        "makespan_us": result.makespan_us,
        "total_time_us": result.total_time_us,
        "num_tasks": len(result.per_task),
        "num_flows": len(flow_times),
        "avg_flow_time_us": sum(flow_times) / len(flow_times) if flow_times else 0,
        "median_flow_time_us": sorted(flow_times)[len(flow_times) // 2] if flow_times else 0,
        "max_flow_time_us": max(flow_times) if flow_times else 0,
        "min_flow_time_us": min(flow_times) if flow_times else 0,
        "avg_compute_time_us": sum(compute_times) / len(compute_times) if compute_times else 0,
    }

    return metrics


def print_comparison(all_metrics):
    """Print comparison table."""
    headers = ["Mode", "Makespan(us)", "Total(us)", "AvgFlow(us)", "MedFlow(us)",
               "MaxFlow(us)", "MinFlow(us)", "Tasks"]
    col_widths = [14, 12, 12, 13, 13, 13, 13, 8]
    sep = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print()
    print("Comparison Results")
    print("=" * len(sep))
    print(sep)
    print("-" * len(sep))
    for m in all_metrics:
        row = (
            f"{m['mode']:<14s} | "
            f"{m['makespan_us']:>10d} | "
            f"{m['total_time_us']:>10d} | "
            f"{m['avg_flow_time_us']:>10.1f} | "
            f"{m['median_flow_time_us']:>10.1f} | "
            f"{m['max_flow_time_us']:>10.1f} | "
            f"{m['min_flow_time_us']:>10.1f} | "
            f"{m['num_tasks']:>6d}"
        )
        print(row)
    print("=" * len(sep))
    print()


def main():
    # --- Configuration ---
    aicb_file = "inputs/aicb-workload/gpt175b-a100.txt"
    topo_file = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
    output_dir = "outputs/puppeteer_reproduce"
    dp = 2
    k_paths = 4
    # modes = ["default", "route-only", "tte-only", "route-tte", "full"]
    modes = ["default", "route-only", "tte-only", "route-tte"]

    os.makedirs(output_dir, exist_ok=True)

    # ---- Load AICB workload ----
    print("=" * 60)
    print("Loading AICB workload")
    print("=" * 60)
    parser_aicb = AicbParser()
    header, items = parser_aicb.parse(aicb_file)
    print(f"  Model: {os.path.basename(aicb_file)}")
    print(f"  Header: tp={header.tp}, dp={header.all_gpus // header.tp}, "
          f"pp={header.pp}, ga={header.ga}, all_gpus={header.all_gpus}")

    # ---- Build P2PWorkload ----
    tp = header.tp
    pp = header.pp
    ep = header.ep
    total_gpus = header.all_gpus

    job = Job(
        job_id=0,
        name="puppeteer-comparison",
        model="gpt175b",
        assigned_nodes=list(range(total_gpus)),
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )

    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job, comm_algo="ring")
    print(f"  Total tasks: {len(workload.tasks)}")
    print(f"  Compute: {len(workload.get_compute_tasks())}, Flow: {len(workload.get_flow_tasks())}")

    # Save workload for visualization
    from src.workload_format.writer import WorkloadWriter
    workload_path = os.path.join(output_dir, "workload.json")
    WorkloadWriter().write(workload, workload_path)
    print(f"  Workload saved to: {workload_path}")

    # ---- Load topology ----
    print()
    print("=" * 60)
    print("Loading topology")
    print("=" * 60)
    loader = TopologyLoader()
    topology = loader.load(topo_file)
    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, "
          f"Switches: {topology.switch_count})")

    # ---- Run each mode ----
    mode_map = {
        "default": run_default,
        "route-only": run_route_only,
        "tte-only": run_tte_only,
        "route-tte": run_route_tte,
        "full": run_full,
    }

    all_metrics = []
    for mode in modes:
        print()
        print("=" * 60)
        print(f"Running mode: {mode}")
        print("=" * 60)

        t0 = time.time()
        try:
            result = mode_map[mode](workload, topology, k_paths=k_paths)
            elapsed = time.time() - t0

            # Save per-mode execution result for visualization
            result_path = os.path.join(output_dir, f"result_{mode}.json")
            result.to_json(result_path)
            print(f"  Saved: {result_path}")

            metrics = _summarize(result, mode, workload)
            metrics["elapsed_s"] = elapsed
            all_metrics.append(metrics)
            print(f"  Makespan: {result.makespan_us} us ({result.makespan_us/1000:.2f} ms)")
            print(f"  Completed in {elapsed:.2f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- Comparison output ----
    print_comparison(all_metrics)

    # ---- Save detailed results ----
    import json
    report = {
        "workload": aicb_file,
        "topology": topo_file,
        "modes": all_metrics,
    }
    report_path = os.path.join(output_dir, "comparison.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

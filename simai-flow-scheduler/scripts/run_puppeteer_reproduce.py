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
    python scripts/run_puppeteer_reproduce.py [--workload W] [--topo T] [--output O]
"""

import argparse
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
from src.executor.policy import DefaultSchedulingPolicy
from src.executor.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.bandwidth import FairShareAllocator


def run_default(workload, topology):
    """Baseline: DefaultSchedulingPolicy."""
    analysis = DefaultAnalyzer(topology).analyze(workload)
    policy = DefaultSchedulingPolicy(analysis=analysis)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_route_only(workload, topology):
    """Puppeteer routing + fair share (no TTE, no coordination)."""
    puppet = PuppeteerAnalyzer(topology, k_paths=4)
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
                min_background_share=1.0,  # all equal
            )

    policy = RouteOnlyPolicy(result.route_table, result.execution_plan)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_tte_only(workload, topology):
    """BFS shortest paths + TTE-aware allocation (no greedy routing)."""
    from src.static_analysis.passes.routing_hints import compute_routing_hints
    from src.static_analysis.passes.puppeteer_tte import compute_tte
    from src.static_analysis.passes.task_serializer import CppReferenceSerializer
    from src.static_analysis.passes.puppeteer_coordination import ResourceDependencyTable

    # Compute routing hints + TTE with shortest paths
    hints = compute_routing_hints(topology, workload)
    plan = CppReferenceSerializer().serialize(workload)
    tte_info, _ = compute_tte(workload, hints, plan)

    # Build route table from shortest paths
    route_table = type('RouteTable', (), {'paths': {}, 'get_path': lambda self, tid: self.paths[tid]})()
    for t in workload.tasks:
        if t.is_flow():
            route_table.paths[t.task_id] = hints.get_path(t.src, t.dst)

    policy = PuppeteerSchedulingPolicy(
        route_table=route_table,
        tte_info=tte_info,
        resource_dependency=ResourceDependencyTable(),
        execution_plan=plan,
        allocator_mode="weighted",
    )
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_route_tte(workload, topology):
    """Puppeteer routing + TTE-aware allocation (no coordination)."""
    from src.static_analysis.passes.puppeteer_coordination import ResourceDependencyTable

    puppet = PuppeteerAnalyzer(topology, k_paths=4)
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


def run_full(workload, topology):
    """Full Puppeteer: routing + TTE-aware + coordination."""
    puppet = PuppeteerAnalyzer(topology, k_paths=4)
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
    parser = argparse.ArgumentParser(description="Puppeteer reproduction comparison")
    parser.add_argument("--workload", default="inputs/aicb-workload/gpt175b-a100.txt",
                        help="AICB workload file")
    parser.add_argument("--topo", default="inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100",
                        help="Topology file")
    parser.add_argument("--output", default="outputs/puppeteer_reproduce",
                        help="Output directory")
    parser.add_argument("--modes", nargs="+",
                        default=["default", "route-only", "tte-only", "route-tte", "full"],
                        choices=["default", "route-only", "tte-only", "route-tte", "full"],
                        help="Modes to run (default: all)")
    parser.add_argument("--dp", type=int, default=2, help="Data parallelism degree")
    parser.add_argument("--k-paths", type=int, default=4, help="Candidate paths for greedy routing")
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ---- Load AICB workload ----
    print("=" * 60)
    print("Loading AICB workload")
    print("=" * 60)
    parser_aicb = AicbParser()
    header, items = parser_aicb.parse(args.workload)
    print(f"  Model: {os.path.basename(args.workload)}")
    print(f"  Header: tp={header.tp}, dp={header.all_gpus // header.tp}, "
          f"pp={header.pp}, ga={header.ga}, all_gpus={header.all_gpus}")

    # ---- Build P2PWorkload ----
    tp = header.tp
    dp = args.dp
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

    # ---- Load topology ----
    print()
    print("=" * 60)
    print("Loading topology")
    print("=" * 60)
    loader = TopologyLoader()
    topology = loader.load(args.topo)
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
    for mode in args.modes:
        print()
        print("=" * 60)
        print(f"Running mode: {mode}")
        print("=" * 60)

        t0 = time.time()
        try:
            result = mode_map[mode](workload, topology)
            elapsed = time.time() - t0
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
        "workload": args.workload,
        "topology": args.topo,
        "modes": all_metrics,
    }
    report_path = os.path.join(output_dir, "comparison.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()

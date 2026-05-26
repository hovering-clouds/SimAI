"""
Cassini experiments: compare scheduling policies with different base planners.

Runs multi-job training workloads through four configurations:

    default           BFS + FairShare (baseline, Themis-like)
    puppeteer         Greedy k-path + TTE-weighted (Pollux-like)
    cassini-default   Cassini on BFS + FairShare base
    cassini-puppeteer Cassini on Greedy k-path + TTE-weighted base

Corresponds to paper experiments:
    - Experiment 1 (Performance Gains): multi-job makespan + iteration time
    - Experiment 2 (Congestion Reduction): flow completion time tail reduction
    - Experiment 4 (Partial Compatibility): compatibility score analysis
    - Scalability: varying number of jobs

Only training traffic is simulated.

Usage:
    python scripts/run_cassini_experiments.py
    python scripts/run_cassini_experiments.py --num-jobs 4
    python scripts/run_cassini_experiments.py --modes default cassini-default
    python scripts/run_cassini_experiments.py --multi-aicb gpt.txt bert.txt
"""

import argparse
import json
import os
import sys
import time

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.job_merger import JobMerger
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.static_analysis.strategies.cassini_strategy import CassiniAnalyzer
from src.static_analysis.passes.routing import GreedyRouteTable
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.executor.bandwidth_allocators.tte_aware_allocator import TteAwareAllocator


# ---------------------------------------------------------------------------
# Run functions — one per configuration
# ---------------------------------------------------------------------------


def run_default(workload, topology, **_):
    """Baseline: DefaultAnalyzer + DefaultSchedulingPolicy (BFS + FairShare)."""
    analysis = DefaultAnalyzer(topology).analyze(workload)
    policy = DefaultSchedulingPolicy(analysis=analysis)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_puppeteer(workload, topology, k_paths=4, **_):
    """PuppeteerAnalyzer + PuppeteerSchedulingPolicy (Greedy + TTE-weighted)."""
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


def run_cassini_default(workload, topology, step_deg=5, **_):
    """CassiniAnalyzer + CassiniSchedulingPolicy (BFS + FairShare + time-shifts)."""
    analyzer = CassiniAnalyzer(topology, step_deg=step_deg)
    analysis = analyzer.analyze(workload)
    policy = CassiniSchedulingPolicy(analysis)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


def run_cassini_puppeteer(workload, topology, k_paths=4, step_deg=5, **_):
    """Cassini time-shift gating on Puppeteer base (Greedy + TTE-weighted).

    Pipeline:
      1. PuppeteerAnalyzer → greedy route_table + TTE info
      2. CassiniAnalyzer with greedy route_table → time_shifts
      3. CassiniSchedulingPolicy with TteAwareAllocator
    """
    # Step 1: Puppeteer analysis for greedy routes and TTE info
    puppet = PuppeteerAnalyzer(topology, k_paths=k_paths)
    puppet_result = puppet.analyze(workload)

    # Step 2: Cassini analysis reusing Puppeteer's greedy route table
    cassini = CassiniAnalyzer(topology, step_deg=step_deg)
    cassini_analysis = cassini.analyze(workload, route_table=puppet_result.route_table)

    # Step 3: Hybrid policy — Cassini time-shift gating + TTE-aware allocation
    tte_allocator = TteAwareAllocator(
        tte_info=puppet_result.tte_info,
        mode="weighted",
    )
    policy = CassiniSchedulingPolicy(
        analysis=cassini_analysis,
        allocator=tte_allocator,
    )
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    return executor.execute(workload)


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------


def extract_metrics(result, mode_name, workload, elapsed_s=0.0):
    """Extract key metrics from an execution result."""
    flow_times = []
    compute_times = []
    for tid, timing in result.per_task.items():
        elapsed = timing.end_time_us - timing.start_time_us
        if timing.task_type == "flow":
            flow_times.append(elapsed)
        else:
            compute_times.append(elapsed)

    sorted_flows = sorted(flow_times) if flow_times else [0]
    n = len(sorted_flows)
    p99_idx = max(0, int(n * 0.99) - 1)

    # Per-job iteration times from the result
    job_times = {}
    for jid, jtime in result.job_iteration_times.items():
        job_times[f"job_{jid}_iter_us"] = jtime

    metrics = {
        "mode": mode_name,
        "makespan_us": result.makespan_us,
        "total_time_us": result.total_time_us,
        "num_tasks": len(result.per_task),
        "num_flows": len(flow_times),
        "num_jobs": len(result.job_iteration_times),
        "avg_flow_us": sum(flow_times) / len(flow_times) if flow_times else 0,
        "median_flow_us": sorted_flows[n // 2] if sorted_flows else 0,
        "p99_flow_us": sorted_flows[p99_idx] if sorted_flows else 0,
        "max_flow_us": max(flow_times) if flow_times else 0,
        "min_flow_us": min(flow_times) if flow_times else 0,
        "avg_compute_us": sum(compute_times) / len(compute_times) if compute_times else 0,
        "elapsed_s": elapsed_s,
    }
    metrics.update(job_times)

    return metrics


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------


def print_comparison(all_metrics):
    """Print a formatted comparison table."""
    headers = [
        "Mode", "Makespan(ms)", "Total(ms)", "AvgFlow(us)", "MedFlow(us)",
        "P99Flow(us)", "MaxFlow(us)", "Jobs", "Tasks",
    ]
    col_widths = [18, 13, 11, 12, 12, 12, 12, 6, 8]
    sep = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print()
    print("Cassini Experiment Results")
    print("=" * len(sep))
    print(sep)
    print("-" * len(sep))

    # Find baseline for speedup calculation
    baseline_ms = None
    for m in all_metrics:
        if m["mode"] == "default":
            baseline_ms = m["makespan_us"] / 1000.0
            break

    for m in all_metrics:
        speedup = ""
        if baseline_ms and baseline_ms > 0 and m["mode"] != "default":
            current_ms = m["makespan_us"] / 1000.0
            ratio = baseline_ms / current_ms if current_ms > 0 else 0
            speedup = f" ({ratio:.2f}x)"

        row = (
            f"{m['mode'] + speedup:<18s} | "
            f"{m['makespan_us'] / 1000:>10.2f} | "
            f"{m['total_time_us'] / 1000:>8.2f} | "
            f"{m['avg_flow_us']:>9.1f} | "
            f"{m['median_flow_us']:>9.1f} | "
            f"{m['p99_flow_us']:>9.1f} | "
            f"{m['max_flow_us']:>9.1f} | "
            f"{m['num_jobs']:>4d} | "
            f"{m['num_tasks']:>6d}"
        )
        print(row)
    print("=" * len(sep))
    print()


def print_job_times(all_metrics):
    """Print per-job iteration time comparison."""
    job_ids = set()
    for m in all_metrics:
        for k in m:
            if k.startswith("job_") and k.endswith("_iter_us"):
                job_ids.add(k)

    if not job_ids:
        return

    job_ids = sorted(job_ids)
    print("Per-Job Iteration Times (ms)")
    print("-" * 50)
    header = f"{'Mode':<18s}" + "".join(f" | {j:<14s}" for j in job_ids)
    print(header)
    print("-" * len(header))
    for m in all_metrics:
        row = f"{m['mode']:<18s}"
        for j in job_ids:
            val = m.get(j, 0) / 1000.0
            row += f" | {val:>12.2f} "
        print(row)
    print()


# ---------------------------------------------------------------------------
# Multi-job workload construction
# ---------------------------------------------------------------------------


def build_single_job_workload(aicb_file, job_id=0, gpu_offset=0, comm_algo="ring"):
    """Build a P2PWorkload for a single AICB file on a contiguous GPU range."""
    parser = AicbParser()
    header, items = parser.parse(aicb_file)
    total_gpus = header.all_gpus
    tp = header.tp
    pp = header.pp
    ep = header.ep
    dp = total_gpus // (tp * pp * ep)

    assigned = list(range(gpu_offset, gpu_offset + total_gpus))
    job = Job(
        job_id=job_id,
        name=os.path.basename(aicb_file).replace(".txt", ""),
        model=os.path.basename(aicb_file).replace(".txt", ""),
        assigned_nodes=assigned,
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )

    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job, comm_algo=comm_algo)
    return workload, header


def build_multi_job_workload(aicb_files, topology):
    """Build a multi-job workload from one or more AICB files.

    When a single AICB file is given, creates N copies with disjoint GPU
    assignments.  When multiple files are given, each file becomes one job
    (possibly with different models/parallelism).

    GPU ranges are allocated contiguously from 0, respecting each job's
    all_gpus requirement and the topology's gpu_count.
    """
    workloads = []
    gpu_offset = 0

    for idx, aicb_file in enumerate(aicb_files):
        wl, header = build_single_job_workload(
            aicb_file, job_id=idx, gpu_offset=gpu_offset,
        )
        workloads.append(wl)
        gpu_offset += header.all_gpus

        if gpu_offset > topology.gpu_count:
            print(f"  WARNING: requested {gpu_offset} GPUs but topology has "
                  f"{topology.gpu_count}")

    if len(workloads) == 1:
        return workloads[0]

    merger = JobMerger()
    result = merger.merge(workloads)
    return result.merged_workload


def build_multi_job_same_aicb(aicb_file, topology, num_jobs, comm_algo="ring"):
    """Build a multi-job workload by replicating the same AICB model N times.

    Each replica gets disjoint GPU ranges.  Total GPUs = num_jobs * all_gpus.
    """
    parser = AicbParser()
    header, items = parser.parse(aicb_file)
    total_gpus = header.all_gpus
    tp = header.tp
    pp = header.pp
    ep = header.ep
    dp = total_gpus // (tp * pp * ep)

    workloads = []
    for jid in range(num_jobs):
        gpu_offset = jid * total_gpus
        assigned = list(range(gpu_offset, gpu_offset + total_gpus))
        job = Job(
            job_id=jid,
            name=f"{os.path.basename(aicb_file).replace('.txt', '')}_j{jid}",
            model=os.path.basename(aicb_file).replace(".txt", ""),
            assigned_nodes=assigned,
            parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
        )
        builder = WorkloadBuilder()
        wl = builder.build_from_aicb(header, items, job, comm_algo=comm_algo)
        workloads.append(wl)

    if num_jobs == 1:
        return workloads[0]

    if gpu_offset + total_gpus > topology.gpu_count:
        print(f"  WARNING: {num_jobs} jobs need {num_jobs * total_gpus} GPUs "
              f"but topology only has {topology.gpu_count}")

    merger = JobMerger()
    result = merger.merge(workloads)
    return result.merged_workload


# ---------------------------------------------------------------------------
# Compatibility analysis (paper Experiment 4)
# ---------------------------------------------------------------------------


def print_compatibility_report(workload, topology, step_deg=5):
    """Print pairwise job compatibility scores from Cassini analysis."""
    cassini = CassiniAnalyzer(topology, step_deg=step_deg)
    analysis = cassini.analyze(workload)

    if len(analysis.communication_patterns) < 2:
        print("  (single job — no pairwise compatibility to report)")
        return

    job_links = cassini._build_job_links(workload, analysis.route_table)
    link_capacities = cassini._build_link_capacities(job_links)
    patterns = analysis.communication_patterns

    print()
    print("Pairwise Job Compatibility (Cassini)")
    print("-" * 45)
    job_ids = sorted(patterns.keys())
    print(f"{'':>8s}", end="")
    for j2 in job_ids:
        print(f" | job_{j2:<5d}", end="")
    print()
    print("-" * (8 + 14 * len(job_ids)))

    for j1 in job_ids:
        print(f"job_{j1:<5d}", end="")
        for j2 in job_ids:
            if j1 >= j2:
                print(f" | {'--':>10s}", end="")
                continue
            shared_links = job_links.get(j1, set()) & job_links.get(j2, set())
            if not shared_links:
                print(f" | {'N/A':>10s}", end="")
                continue
            # Run pairwise analysis on first shared link
            link = next(iter(shared_links))
            from src.cassini.circle_abstraction import CircleAbstraction
            from src.cassini.pair_compatibility import optimize_link_compatibility

            circle1 = CircleAbstraction.from_pattern(
                patterns[j1], link,
            )
            circle2 = CircleAbstraction.from_pattern(
                patterns[j2], link,
            )
            if max(circle1.bw_demand.values()) == 0.0 or max(circle2.bw_demand.values()) == 0.0:
                print(f" | {'no_comm':>10s}", end="")
                continue
            result = optimize_link_compatibility(
                {0: circle1, 1: circle2},
                link_capacities[link],
                step_deg=step_deg,
            )
            score = result.score
            print(f" | {score:>9.3f}", end="")
        print()
    print("-" * (8 + 14 * len(job_ids)))
    print(f"  (Score 1.0 = perfectly compatible, < 0.6 = avoid placement)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cassini experiments — compare base planners",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_cassini_experiments.py
  python scripts/run_cassini_experiments.py --num-jobs 4
  python scripts/run_cassini_experiments.py --modes default cassini-default
  python scripts/run_cassini_experiments.py --multi-aicb gpt.txt bert.txt
        """,
    )
    parser.add_argument(
        "--aicb",
        nargs="+",
        default=["inputs/aicb-workload/gpt175b-a100.txt"],
        help="AICB workload file(s) (default: gpt175b-a100.txt)",
    )
    parser.add_argument(
        "--topo",
        default="inputs/topologies/Spectrum-X_16g_8gps_400Gbps_H100",
        help="Topology file (default: AlibabaHPN 16-GPU)",
    )
    parser.add_argument(
        "--output", "-o",
        default="outputs/cassini_experiments",
        help="Output directory (default: outputs/cassini_experiments)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["default", "puppeteer", "cassini-default", "cassini-puppeteer"],
        help="Modes to run (default: all four)",
    )
    parser.add_argument(
        "--num-jobs", "-n",
        type=int,
        default=2,
        help="Number of jobs when using --same-aicb (default: 2)",
    )
    parser.add_argument(
        "--same-aicb",
        action="store_true",
        default=True,
        help="Replicate the same AICB model N times with disjoint GPU ranges",
    )
    parser.add_argument(
        "--multi-aicb",
        action="store_true",
        default=False,
        help="Treat each --aicb file as a separate model/job",
    )
    parser.add_argument(
        "--k-paths", type=int, default=4,
        help="Candidate paths for greedy routing (default: 4)",
    )
    parser.add_argument(
        "--step-deg", type=int, default=5,
        help="Cassini angle discretization step in degrees (default: 5)",
    )
    parser.add_argument(
        "--no-compat", action="store_true",
        help="Skip pairwise compatibility analysis",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- Load topology ----
    print("=" * 60)
    print("Cassini Experiments")
    print("=" * 60)
    print(f"Topology: {args.topo}")
    loader = TopologyLoader()
    topology = loader.load(args.topo)
    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, "
          f"Switches: {topology.switch_count})")

    # ---- Build workload ----
    print()
    print("Building workload...")

    if args.multi_aicb:
        workload = build_multi_job_workload(args.aicb, topology)
        print(f"  Multi-model: {len(args.aicb)} AICB files")
    elif args.same_aicb and args.num_jobs > 1:
        workload = build_multi_job_same_aicb(
            args.aicb[0], topology, args.num_jobs,
        )
        print(f"  Same AICB × {args.num_jobs} jobs")
    else:
        workload, _ = build_single_job_workload(args.aicb[0], job_id=0)
        print(f"  Single job from: {args.aicb[0]}")

    print(f"  Total tasks: {len(workload.tasks)} "
          f"(compute: {len(workload.get_compute_tasks())}, "
          f"flow: {len(workload.get_flow_tasks())})")
    print(f"  Jobs: {len(workload.jobs)}")

    # Save workload for visualization
    workload_path = os.path.join(args.output, "workload.json")
    WorkloadWriter().write(workload, workload_path)
    print(f"  Workload saved to: {workload_path}")

    # ---- Compatibility analysis (paper Experiment 4) ----
    if not args.no_compat and len(workload.jobs) > 1:
        print_compatibility_report(workload, topology, args.step_deg)

    # ---- Run each mode ----
    mode_map = {
        "default": run_default,
        "puppeteer": run_puppeteer,
        "cassini-default": run_cassini_default,
        "cassini-puppeteer": run_cassini_puppeteer,
    }

    # Filter to requested modes, preserving order
    run_modes = [m for m in args.modes if m in mode_map]
    extra_kwargs = {
        "k_paths": args.k_paths,
        "step_deg": args.step_deg,
    }

    all_metrics = []
    for mode in run_modes:
        print()
        print("=" * 60)
        print(f"Running: {mode}")
        print("=" * 60)

        t0 = time.time()
        try:
            result = mode_map[mode](workload, topology, **extra_kwargs)
            elapsed = time.time() - t0

            result_path = os.path.join(args.output, f"result_{mode}.json")
            result.to_json(result_path)
            print(f"  Saved: {result_path}")

            metrics = extract_metrics(result, mode, workload, elapsed)
            all_metrics.append(metrics)
            print(f"  Makespan: {result.makespan_us / 1000:.2f} ms")
            print(f"  Completed in {elapsed:.2f}s")

            if result.job_iteration_times:
                for jid, jtime in result.job_iteration_times.items():
                    print(f"    job_{jid} iteration: {jtime / 1000:.2f} ms")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    # ---- Comparison output ----
    if all_metrics:
        print_comparison(all_metrics)
        print_job_times(all_metrics)

        # Per-mode speedup vs default
        baseline = None
        for m in all_metrics:
            if m["mode"] == "default":
                baseline = m
                break
        if baseline:
            print("Speedup vs Default (makespan):")
            for m in all_metrics:
                if m["mode"] == "default":
                    continue
                ratio = baseline["makespan_us"] / max(m["makespan_us"], 1)
                print(f"  {m['mode']:<22s}: {ratio:.2f}x")

            # Flow time tail comparison (paper Experiment 2)
            print()
            print("Flow Time Tail Comparison (p99 / avg):")
            for m in all_metrics:
                ratio = m["p99_flow_us"] / max(m["avg_flow_us"], 1)
                print(f"  {m['mode']:<22s}: p99={m['p99_flow_us']:.0f} us, "
                      f"avg={m['avg_flow_us']:.0f} us, ratio={ratio:.2f}")

        # Save report
        report = {
            "experiment": "cassini_base_planner_comparison",
            "aicb_files": args.aicb,
            "topology": args.topo,
            "num_jobs": args.num_jobs,
            "config": {
                "k_paths": args.k_paths,
                "step_deg": args.step_deg,
            },
            "results": all_metrics,
        }
        report_path = os.path.join(args.output, "comparison.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to: {report_path}")
    else:
        print("No modes completed successfully.")


if __name__ == "__main__":
    main()

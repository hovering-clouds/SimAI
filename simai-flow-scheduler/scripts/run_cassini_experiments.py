"""
Cassini experiments: compare scheduling policies with different base planners.

Runs multi-job training workloads through four configurations:

    default           BFS + FairShare (baseline)
    puppeteer         Greedy k-path + TTE-weighted
    cassini-default   Cassini on BFS + FairShare base
    cassini-puppeteer Cassini on Greedy k-path + TTE-weighted

Usage:
    python scripts/run_cassini_experiments.py
    python scripts/run_cassini_experiments.py --dp 4 --num-jobs 4
    python scripts/run_cassini_experiments.py --aicb gpt13b.txt gpt7b.txt --interleave
    python scripts/run_cassini_experiments.py --num-iters 10 --modes default cassini-default
"""

import argparse
import json
import os
import sys
import time
import traceback

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.job_merger import JobMerger
from src.workload_generator.iteration_replicator import replicate_iterations
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.static_analysis.strategies.cassini_strategy import CassiniAnalyzer
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.executor.bandwidth_allocators.tte_aware_allocator import TteAwareAllocator
from src.executor.visualizer import ChromeTraceVerbose, ChromeTraceCompact
from src.cassini.circle_abstraction import CircleAbstraction
from src.cassini.pair_compatibility import optimize_link_compatibility

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AICB = (
    "inputs/aicb-workload/"
    "A100-gpt_13B_ws4_pp1-world_size4-tp4-pp1-ep1-gbs2-mbs1-seq4096"
    "-MOE-False-GEMM-False-flash_attn-True.txt"
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
    policy = CassiniSchedulingPolicy(analysis)
    return AnalyticalExecutor(topology=topology, policy=policy).execute(workload)


def run_cassini_puppeteer(workload, topology, k_paths=4, step_deg=5, **_):
    puppet = PuppeteerAnalyzer(topology, k_paths=k_paths).analyze(workload)
    cassini = CassiniAnalyzer(topology, step_deg=step_deg)
    cassini_analysis = cassini.analyze(workload, route_table=puppet.route_table)
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
# Workload construction
# ---------------------------------------------------------------------------


def _resolve_parallelism(header, dp_override):
    """Return (tp, dp, pp, ep, total_gpus) from an AICB header and optional DP override."""
    tp, pp, ep = header.tp, header.pp, header.ep
    if dp_override and dp_override >= 1:
        dp = dp_override
        total_gpus = tp * dp * pp * ep
    else:
        total_gpus = header.all_gpus
        dp = total_gpus // (tp * pp * ep)
    return tp, dp, pp, ep, total_gpus


def _interleave_gpus(job_configs, gpu_count):
    """Assign GPUs so each job's DP replicas span different ASW domains.

    Each job gets one replica in a "low" domain and one in a "high" domain,
    forcing DP traffic through the shared spine fabric.
    """
    replica_sizes = [c["tp"] * c["pp"] * c["ep"] for c in job_configs]
    max_size = max(replica_sizes)
    domains = [list(range(i, i + max_size))
               for i in range(0, gpu_count, max_size) if i + max_size <= gpu_count]
    nd = len(domains)

    assignments = [[] for _ in job_configs]
    for j, cfg in enumerate(job_configs):
        size = replica_sizes[j]
        for k in range(cfg["dp"]):
            # Replica 0 → domain j, replica 1 → mirror domain nd-1-j
            d = j % nd if k == 0 else (nd - 1 - j) % nd
            assignments[j].extend(domains[d][:size])
    return assignments


def _contiguous_gpus(job_configs):
    """Assign contiguous GPU ranges to each job."""
    assignments = []
    offset = 0
    for cfg in job_configs:
        n = cfg["tp"] * cfg["dp"] * cfg["pp"] * cfg["ep"]
        assignments.append(list(range(offset, offset + n)))
        offset += n
    return assignments


def build_workload(aicb_files, topology, num_jobs, dp, interleave=False):
    """Build a P2PWorkload from one or more AICB files.

    When *aicb_files* has one entry, replicates the same model *num_jobs* times.
    When *aicb_files* has multiple entries, each becomes a distinct job.
    """
    parser = AicbParser()
    files = list(aicb_files)
    if len(files) == 1:
        files = files * num_jobs  # replicate same model

    job_configs = []
    for f in files:
        header, _ = parser.parse(f)
        tp, dp_val, pp, ep, _ = _resolve_parallelism(header, dp)
        job_configs.append({"tp": tp, "dp": dp_val, "pp": pp, "ep": ep, "header": header, "file": f})

    assignments = _interleave_gpus(job_configs, topology.gpu_count) if interleave else _contiguous_gpus(job_configs)

    total = sum(len(a) for a in assignments)
    if total > topology.gpu_count:
        print(f"  WARNING: jobs need {total} GPUs but topology only has {topology.gpu_count}")

    workloads = []
    for idx, cfg in enumerate(job_configs):
        header, items = parser.parse(cfg["file"])
        tp, dp_val, pp, ep, _ = _resolve_parallelism(header, dp)
        job = Job(
            job_id=idx,
            name=os.path.basename(cfg["file"]).replace(".txt", ""),
            model=os.path.basename(cfg["file"]).replace(".txt", ""),
            assigned_nodes=assignments[idx],
            parallelism=ParallelismConfig(tp=tp, dp=dp_val, pp=pp, ep=ep),
        )
        wl = WorkloadBuilder().build_from_aicb(header, items, job, comm_algo="ring")
        workloads.append(wl)

    if len(workloads) == 1:
        return workloads[0]

    return JobMerger().merge(workloads).merged_workload


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
            score = optimize_link_compatibility({0: c1, 1: c2}, link_capacities[link], step_deg=step_deg).score
            print(f" | {score:>9.3f}", end="")
        print()
    print("-" * (8 + 14 * len(job_ids)))
    print("  (Score 1.0 = perfectly compatible, < 0.6 = avoid placement)\n")


def save_report(all_metrics, args):
    report = {
        "experiment": "cassini_base_planner_comparison",
        "aicb_files": args.aicb,
        "topology": args.topo,
        "num_jobs": args.num_jobs,
        "config": {"k_paths": args.k_paths, "step_deg": args.step_deg},
        "results": all_metrics,
    }
    path = os.path.join(args.output, "comparison.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Cassini experiments — compare base planners",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  python scripts/run_cassini_experiments.py\n"
               "  python scripts/run_cassini_experiments.py --num-jobs 4\n"
               "  python scripts/run_cassini_experiments.py --aicb a.txt b.txt --interleave\n"
               "  python scripts/run_cassini_experiments.py --modes default cassini-default",
    )
    parser.add_argument("--aicb", nargs="+", default=[DEFAULT_AICB],
                        help="AICB file(s); one file → replicated, multiple → distinct jobs")
    parser.add_argument("--topo", default=DEFAULT_TOPO)
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT)
    parser.add_argument("--modes", nargs="+",
                        default=["default", "puppeteer", "cassini-default", "cassini-puppeteer"])
    parser.add_argument("--num-jobs", "-n", type=int, default=2)
    parser.add_argument("--dp", type=int, default=2,
                        help="DP degree override (default 2; set 1 to disable)")
    parser.add_argument("--num-iters", type=int, default=1,
                        help="Training iterations per job (default 1)")
    parser.add_argument("--interleave", action="store_true",
                        help="Cross-domain GPU assignment so jobs share spine links")
    parser.add_argument("--k-paths", type=int, default=4)
    parser.add_argument("--step-deg", type=int, default=5)
    parser.add_argument("--no-compat", action="store_true",
                        help="Skip pairwise compatibility analysis")
    parser.add_argument("--trace", choices=["verbose", "compact"], default=None)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ---- Topology ----
    print("=" * 60 + "\nCassini Experiments\n" + "=" * 60)
    print(f"Topology: {args.topo}")
    topology = TopologyLoader().load(args.topo)
    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, "
          f"Switches: {topology.switch_count})")

    # ---- Workload ----
    print("\nBuilding workload...")
    workload = build_workload(args.aicb, topology, args.num_jobs, args.dp, args.interleave)

    if args.num_iters > 1:
        workload = replicate_iterations(workload, args.num_iters)
        print(f"  Replicated to {args.num_iters} iterations per job")

    tag = "interleaved" if args.interleave else "contiguous"
    n_files = len(args.aicb)
    if n_files > 1:
        print(f"  Multi-model ({tag}): {n_files} AICB files")
    else:
        print(f"  Same AICB × {args.num_jobs} jobs ({tag})")

    print(f"  Total tasks: {len(workload.tasks)} "
          f"(compute: {len(workload.get_compute_tasks())}, "
          f"flow: {len(workload.get_flow_tasks())})")
    print(f"  Jobs: {len(workload.jobs)}")
    if workload.jobs:
        j0 = workload.jobs[0]
        print(f"  Per-job parallelism: tp={j0.parallelism.tp}, dp={j0.parallelism.dp}, "
              f"pp={j0.parallelism.pp}, ep={j0.parallelism.ep} "
              f"({len(j0.assigned_nodes)} GPUs)")
    if args.dp > 1:
        print(f"  DP override: {args.dp}")

    WorkloadWriter().write(workload, os.path.join(args.output, "workload.json"))

    # ---- Compatibility ----
    if not args.no_compat and len(workload.jobs) > 1:
        print_compatibility_report(workload, topology, args.step_deg)

    # ---- Run modes ----
    run_modes = [m for m in args.modes if m in MODE_MAP]
    extra = {"k_paths": args.k_paths, "step_deg": args.step_deg}
    all_metrics = []

    for mode in run_modes:
        print(f"\n{'=' * 60}\nRunning: {mode}\n{'=' * 60}")
        t0 = time.time()
        try:
            result = MODE_MAP[mode](workload, topology, **extra)
            elapsed = time.time() - t0

            result_path = os.path.join(args.output, f"result_{mode}.json")
            result.to_json(result_path)
            print(f"  Saved: {result_path}")

            if args.trace:
                viz_cls = ChromeTraceVerbose if args.trace == "verbose" else ChromeTraceCompact
                viz = viz_cls(workload)
                trace_path = os.path.join(args.output, f"result_{mode}_timeline_{args.trace}.json")
                viz.export(result, trace_path)
                print(f"  Trace: {trace_path}")

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
        save_report(all_metrics, args)
    else:
        print("No modes completed successfully.")


if __name__ == "__main__":
    main()


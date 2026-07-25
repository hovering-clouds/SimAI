"""
Cassini experiments: compare scheduling policies with different base planners.

Runs multi-job training workloads through configurable scheduling modes:

    default           BFS + FairShare (baseline)
    puppeteer         Greedy k-path + TTE-weighted
    cassini-default   Cassini on BFS + FairShare base
    cassini-puppeteer Cassini on Greedy k-path + TTE-weighted

Two invocation modes:
  1. JSON config:  python scripts/run_cassini_e2e.py --config exp.json
  2. CLI args:     python scripts/run_cassini_e2e.py --topo ... --aicb ...

JSON config schema (all fields optional, CLI args override config values):
  {
    "topology":       "path/to/topology",
    "output_dir":     "path/to/output",
    "placement":      "contiguous" | "contention-spread",
    "placement_clusters": 2,
    "gpus_per_server": 8,
    "k_paths":        4,
    "step_deg":       5,
    "puppeteer_bw_change_threshold_pct": 5.0,
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

from scripts.utils.diagnostics import (
    extract_metrics,
    print_comparison,
    print_job_times,
    print_speedup,
    save_text_report,
    save_json_report,
    print_cassini_diagnostics,
    print_compatibility_report,
    save_visualizations,
)
from scripts.utils.iteration_expansion import (
    merge_ga_to_one_iteration,
    patch_iteration_time_us,
    replicate_with_cross_iteration_deps,
)
from scripts.utils.job_placement import (
    resolve_parallelism,
    PLACEMENT_MAP,
    print_placement_report,
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
import src.executor.analytical as analytical_executor
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.executor.bandwidth_allocators.tte_aware_allocator import TteAwareAllocator

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_AICB = (
    "inputs/aicb-workload/"
    "A100-gpt_7B_ws4_pp2-world_size4-tp2-pp2-ep1-gbs16-mbs4-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
)
DEFAULT_TOPO = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
DEFAULT_OUTPUT = "outputs/cassini_experiments"
PUPPETEER_MODES = {"puppeteer", "cassini-puppeteer"}

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
    print_cassini_diagnostics(
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
    print_cassini_diagnostics(
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


def apply_executor_tuning(mode: str, puppeteer_bw_change_threshold_pct: float) -> None:
    """Reduce completion-event churn for TTE-weighted Puppeteer modes."""
    threshold = puppeteer_bw_change_threshold_pct if mode in PUPPETEER_MODES else 0.0
    analytical_executor.BW_CHANGE_THRESHOLD_PCT = threshold
    print(f"  Executor BW change threshold: {threshold:.2f}%")

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
    job_specs = []
    for entry in workload_entries:
        for _ in range(entry["num_jobs"]):
            job_specs.append((entry["aicb"], entry["dp"], entry["num_iters"]))

    # Build configs for placement
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
  python scripts/run_cassini_e2e.py
  python scripts/run_cassini_e2e.py --config experiments.json
  python scripts/run_cassini_e2e.py --topo <path> --aicb <path> --dp 4 --num-jobs 4
  python scripts/run_cassini_e2e.py --modes default cassini-default
  python scripts/run_cassini_e2e.py --placement contention-spread --placement-clusters 2
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
    parser.add_argument("--puppeteer-bw-change-threshold-pct", type=float, default=None,
                        help="BW change threshold for Puppeteer modes; reduces TTE event churn")
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
        "puppeteer_bw_change_threshold_pct": 5.0,
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
        "puppeteer_bw_change_threshold_pct": args.puppeteer_bw_change_threshold_pct,
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
    puppeteer_bw_change_threshold_pct = config["puppeteer_bw_change_threshold_pct"]
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
    print_placement_report(
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
        apply_executor_tuning(mode, puppeteer_bw_change_threshold_pct)
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
        save_text_report(all_metrics, output_dir)
        save_json_report(all_metrics, config)
    else:
        print("No modes completed successfully.")


if __name__ == "__main__":
    main()

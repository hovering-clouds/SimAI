"""Experiment diagnostics and report generation for Cassini experiments.

Provides formatted comparison tables, per-link optimisation diagnostics,
pairwise compatibility matrices, and report saving (JSON + text).
"""

import json
import os
from collections import defaultdict

from .circle_abstraction import CircleAbstraction
from .pair_compatibility import CompatibilityResult, compute_score, optimize_link_compatibility


def extract_metrics(result, mode_name, elapsed_s=0.0):
    """Extract a flat dict of performance metrics from an execution result."""
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


# ---------------------------------------------------------------------------
# Formatted text report
# ---------------------------------------------------------------------------


def _format_comparison_table(all_metrics) -> str:
    """Return the comparison table as a string."""
    headers = ["Mode", "Makespan(ms)", "Total(ms)", "AvgFlow(us)", "MedFlow(us)",
               "P99Flow(us)", "MaxFlow(us)", "Jobs", "Tasks"]
    widths = [18, 13, 11, 12, 12, 12, 12, 6, 8]
    sep = " | ".join(h.ljust(w) for h, w in zip(headers, widths))

    lines = ["Cassini Experiment Results",
             "=" * len(sep),
             sep,
             "-" * len(sep)]

    baseline_ms = next((m["makespan_us"] / 1000 for m in all_metrics if m["mode"] == "default"), None)
    for m in all_metrics:
        speedup = ""
        if baseline_ms and m["mode"] != "default":
            ratio = baseline_ms / max(m["makespan_us"] / 1000, 0.001)
            speedup = f" ({ratio:.2f}x)"
        lines.append(
            f"{m['mode'] + speedup:<18s} | {m['makespan_us'] / 1000:>10.2f} | "
            f"{m['total_time_us'] / 1000:>8.2f} | {m['avg_flow_us']:>9.1f} | "
            f"{m['median_flow_us']:>9.1f} | {m['p99_flow_us']:>9.1f} | "
            f"{m['max_flow_us']:>9.1f} | {m['num_jobs']:>4d} | {m['num_tasks']:>6d}"
        )
    lines.append("=" * len(sep) + "\n")
    return "\n".join(lines)


def _format_job_times(all_metrics) -> str:
    """Return per-job iteration times as a string."""
    job_ids = sorted({k for m in all_metrics for k in m
                      if k.startswith("job_") and k.endswith("_iter_us")})
    if not job_ids:
        return ""
    lines = ["Per-Job Iteration Times (ms)", "-" * 50]
    header = f"{'Mode':<18s}" + "".join(f" | {j:<14s}" for j in job_ids)
    lines.append(header + "\n" + "-" * len(header))
    for m in all_metrics:
        row = f"{m['mode']:<18s}"
        for j in job_ids:
            row += f" | {m.get(j, 0) / 1000:>12.2f} "
        lines.append(row)
    lines.append("")
    return "\n".join(lines)


def _format_speedup(all_metrics) -> str:
    """Return speedup analysis as a string."""
    lines = []
    baseline = next((m for m in all_metrics if m["mode"] == "default"), None)
    if not baseline:
        return ""
    lines.append("Speedup vs Default (makespan):")
    for m in all_metrics:
        if m["mode"] == "default":
            continue
        ratio = baseline["makespan_us"] / max(m["makespan_us"], 1)
        lines.append(f"  {m['mode']:<22s}: {ratio:.2f}x")

    lines.append("")
    lines.append("Flow Time Tail Comparison (p99 / avg):")
    for m in all_metrics:
        ratio = m["p99_flow_us"] / max(m["avg_flow_us"], 1)
        lines.append(f"  {m['mode']:<22s}: p99={m['p99_flow_us']:.0f} us, "
                     f"avg={m['avg_flow_us']:.0f} us, ratio={ratio:.2f}")
    return "\n".join(lines)


def print_comparison(all_metrics):
    """Print formatted comparison table to stdout."""
    print(_format_comparison_table(all_metrics))


def print_job_times(all_metrics):
    """Print per-job iteration times to stdout."""
    text = _format_job_times(all_metrics)
    if text:
        print(text)


def print_speedup(all_metrics):
    """Print speedup analysis to stdout."""
    print(_format_speedup(all_metrics))


def save_text_report(all_metrics, output_dir):
    """Save formatted report tables to a text file."""
    sections = [
        _format_comparison_table(all_metrics),
        _format_job_times(all_metrics),
        _format_speedup(all_metrics),
    ]
    text = "\n".join(s for s in sections if s)
    path = os.path.join(output_dir, "comparison.txt")
    with open(path, "w") as f:
        f.write(text)
    print(f"Text report saved to: {path}")


def save_json_report(all_metrics, config):
    """Save raw metrics as JSON."""
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
    print(f"JSON report saved to: {path}")


# ---------------------------------------------------------------------------
# Cassini per-link diagnostics
# ---------------------------------------------------------------------------


def print_cassini_diagnostics(
    workload, topology, route_table, patterns, time_shifts, step_deg, label,
):
    """Print per-link Cassini before/after scores and applied time-shifts."""
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
            circles = CircleAbstraction.build_unified(link_patterns, lid)
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
    """Print pairwise job compatibility matrix."""
    from ..static_analysis.strategies.cassini_strategy import CassiniAnalyzer

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


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def save_visualizations(workload, result, mode_name, output_dir, viz_config):
    """Generate Chrome Trace visualizations for a result."""
    from ..executor.visualizer import (
        ChromeTraceCompact,
        ChromeTraceFlowDetail,
        ChromeTraceVerbose,
    )

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

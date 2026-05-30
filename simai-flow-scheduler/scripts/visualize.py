"""
Convert execution results to Chrome Trace JSON for visualization.

Loads a pre-computed ExecutionResult (from run_e2e.py) and a P2PWorkload,
then generates Chrome Trace files in various modes.

Usage:
    python scripts/visualize.py
"""

import os
import sys

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.writer import WorkloadReader
from src.executor.result import ExecutionResult
from src.executor.visualizer import (
    ChromeTraceVerbose,
    ChromeTraceCompact,
    ChromeTraceFlowDetail,
)

# ── Configuration ──────────────────────────────────────────────

# Output directory from run_e2e.py (must contain workload.json and execution_result.json)
OUTPUT_DIR = "outputs/mfs_e2e_2p2d"

# workload file name (default: workload.json; for puppeteer: workload_route-tte.json, etc.)
WORKLOAD_FILE = "workload_2p2d.json"

# Result file name (default: execution_result.json; for puppeteer: result_route-tte.json, etc.)
RESULT_FILE = "mfs_execution_result_2p2d.json"

# Visualization mode: "verbose" | "compact" | "detail"
MODE = "detail"

# Show dependency arrows (compact mode only)
SHOW_ARROWS = False

# Time window for detail mode (microseconds) — only used when MODE = "detail"
DETAIL_TIME_RANGE = (5625400, 5903810)

# Output file path (None = auto-generate from mode name)
OUTPUT_FILE = "mfs_execution_timeline_verbose_2p2d.json"

# ───────────────────────────────────────────────────────────────


def main():
    workload_path = os.path.join(OUTPUT_DIR, WORKLOAD_FILE)
    result_path = os.path.join(OUTPUT_DIR, RESULT_FILE)

    if not os.path.exists(workload_path):
        print(f"Error: {workload_path} not found. Run run_e2e.py first.")
        sys.exit(1)
    if not os.path.exists(result_path):
        print(f"Error: {result_path} not found. Run run_e2e.py first.")
        sys.exit(1)

    print(f"Loading workload from: {workload_path}")
    workload = WorkloadReader().read(workload_path)
    print(f"  Tasks: {len(workload.tasks)}")

    print(f"Loading result from: {result_path}")
    result = ExecutionResult.from_json(result_path)
    print(f"  Makespan: {result.makespan_us} us ({result.makespan_us / 1000:.2f} ms)")
    print(f"  Tasks: {len(result.per_task)}")

    # Build visualizer
    if MODE == "verbose":
        viz = ChromeTraceVerbose(workload)
        suffix = "verbose"
    elif MODE == "compact":
        viz = ChromeTraceCompact(workload, show_arrows=SHOW_ARROWS)
        suffix = "compact"
    elif MODE == "detail":
        start_us, end_us = DETAIL_TIME_RANGE
        viz = ChromeTraceFlowDetail(workload, start_us, end_us)
        suffix = f"detail_{start_us}_{end_us}"
    else:
        print(f"Error: unknown mode '{MODE}'")
        sys.exit(1)

    # Export
    if OUTPUT_FILE is None:
        out_path = os.path.join(OUTPUT_DIR, f"execution_timeline_{suffix}.json")
    else:
        out_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

    viz.export(result, out_path)
    print(f"Trace saved to: {out_path}")
    print("Open in chrome://tracing to view.")


if __name__ == "__main__":
    main()

"""
Convert execution results to Chrome Trace JSON for visualization.

Loads a pre-computed ExecutionResult (from run_e2e.py) and a P2PWorkload,
then generates Chrome Trace files in various modes.

Usage:
    python scripts/visualize.py                                   # defaults
    python scripts/visualize.py --output-dir outputs/inference_e2e/
    python scripts/visualize.py -w out/wl.json -r out/res.json -o out/tl.json
    python scripts/visualize.py --output-dir out/ --mode compact --show-arrows
    python scripts/visualize.py --output-dir out/ --mode detail --time-range 1000,2000
"""

import argparse
import os
import sys

project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.writer import WorkloadReader
from src.executor.result import ExecutionResult
from src.executor.visualizer import (
    ChromeTraceVerbose,
    ChromeTraceCompact,
    ChromeTraceFlowDetail,
)

# ── Defaults (overridable via CLI) ──────────────────────────────

DEFAULT_OUTPUT_DIR = "outputs/e2e_gpipe/ep2/"
DEFAULT_WORKLOAD_FILE = "workload.json"
DEFAULT_RESULT_FILE = "execution_result.json"
DEFAULT_MODE = "verbose"
DEFAULT_OUTPUT_FILE = "execution_timeline.json"
DEFAULT_TIME_RANGE = (5625400, 5903810)


def main():
    parser = argparse.ArgumentParser(
        description="Convert execution results to Chrome Trace JSON.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory containing workload.json and execution_result.json.",
    )
    parser.add_argument(
        "--workload", "-w", default=None,
        help="Path to workload.json (overrides --output-dir).",
    )
    parser.add_argument(
        "--result", "-r", default=None,
        help="Path to execution_result.json (overrides --output-dir).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output trace file path (default: <output-dir>/execution_timeline.json).",
    )
    parser.add_argument(
        "--mode", "-m", default=DEFAULT_MODE,
        choices=["verbose", "compact", "detail"],
        help="Visualization mode.",
    )
    parser.add_argument(
        "--show-arrows", action="store_true",
        help="Show dependency arrows (compact mode only).",
    )
    parser.add_argument(
        "--time-range", default=None,
        help="Detail mode time window as START,END in microseconds.",
    )
    args = parser.parse_args()

    workload_path = args.workload or os.path.join(args.output_dir, DEFAULT_WORKLOAD_FILE)
    result_path = args.result or os.path.join(args.output_dir, DEFAULT_RESULT_FILE)
    output_path = args.output or os.path.join(args.output_dir, DEFAULT_OUTPUT_FILE)

    if not os.path.exists(workload_path):
        print(f"Error: {workload_path} not found. Run the e2e script first.")
        sys.exit(1)
    if not os.path.exists(result_path):
        print(f"Error: {result_path} not found. Run the e2e script first.")
        sys.exit(1)

    print(f"Loading workload from: {workload_path}")
    workload = WorkloadReader().read(workload_path)
    print(f"  Tasks: {len(workload.tasks)}")

    print(f"Loading result from: {result_path}")
    result = ExecutionResult.from_json(result_path)
    print(f"  Makespan: {result.makespan_us} us ({result.makespan_us / 1000:.2f} ms)")
    print(f"  Tasks: {len(result.per_task)}")

    # Build visualizer
    if args.mode == "verbose":
        viz = ChromeTraceVerbose(workload)
    elif args.mode == "compact":
        viz = ChromeTraceCompact(workload, show_arrows=args.show_arrows)
    else:  # detail
        if args.time_range:
            start_us, end_us = (int(x) for x in args.time_range.split(","))
        else:
            start_us, end_us = DEFAULT_TIME_RANGE
        viz = ChromeTraceFlowDetail(workload, start_us, end_us)

    viz.export(result, output_path)
    print(f"Trace saved to: {output_path}")
    print("Open in chrome://tracing to view.")


if __name__ == "__main__":
    main()

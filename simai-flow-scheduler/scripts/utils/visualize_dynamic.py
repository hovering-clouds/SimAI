"""
Generate Chrome Trace from ExecutionResult (with optional task metadata).

Usage:
    uv run python scripts/visualize_dynamic.py                                  # defaults
    uv run python scripts/visualize_dynamic.py --output-dir outputs/dynamic_e2e/
    uv run python scripts/visualize_dynamic.py -r out/res.json -m out/meta.json -o out/trace.json
    uv run python scripts/visualize_dynamic.py --min-dur 1                       # skip zero-duration
"""

import argparse
import os
import sys
import json

project_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

# ── Defaults (overridable via CLI) ──────────────────────────────

DEFAULT_OUTPUT_DIR = "outputs/dynamic_e2e"
DEFAULT_RESULT_FILE = "execution_result.json"
DEFAULT_META_FILE = "task_meta.json"
DEFAULT_OUTPUT_FILE = "trace.json"
DEFAULT_MIN_DUR_US = 0

# ───────────────────────────────────────────────────────────────


def _label(tid: int, timing: dict, task_meta: dict) -> str:
    """Build human-readable label from task metadata."""
    meta = task_meta.get(tid, {})
    if timing["task_type"] == "compute":
        phase = meta.get("phase", "")
        layer = meta.get("layer_id", None)
        if phase:
            phase_short = (phase.replace("backward_input", "bwd_i")
                           .replace("backward_weight", "bwd_w")
                           .replace("forward", "fwd")
                           .replace("prefill", "pre")
                           .replace("decode", "dec"))
            return f"{phase_short} L{layer}" if layer is not None else phase_short
        return f"compute_{tid}"
    else:
        comm = meta.get("comm_type", "")
        src = meta.get("src", None)
        dst = meta.get("dst", None)
        if comm and src is not None and dst is not None:
            comm_short = (comm.replace("tp_allreduce_ring", "tp_ar")
                          .replace("tp_allgather_ring", "tp_ag")
                          .replace("tp_reducescatter_ring", "tp_rs")
                          .replace("ep_alltoall", "ep_a2a")
                          .replace("pp_send", "pp")
                          .replace("kv_cache_transfer", "kv_xfer")
                          .replace("kv_cache_reuse", "kv_use"))
            return f"{comm_short} {src}→{dst}"
        return f"flow_{tid}"


def export_trace(result_path: str, meta_path: str | None, output_path: str,
                 min_dur_us: int = 0) -> int:
    """Write a Chrome Trace for a dynamic result and return its event count.

    Args:
        result_path: Path to execution_result.json.
        meta_path: Path to task_meta.json, or None.
        output_path: Output path for trace.json.
        min_dur_us: Skip events with duration <= this value (e.g. 1 to filter
                    zero-duration tasks that clutter the visualisation).
    """

    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Execution result not found: {result_path}")

    # Load execution result
    with open(result_path) as f:
        data = json.load(f)
    # Load optional task metadata
    task_meta = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path) as f:
            task_meta = {int(k): v for k, v in json.load(f).items()}

    # Build Chrome Trace events
    events = []
    for tid_str, timing in data["per_task"].items():
        tid = int(tid_str)
        dur = max(timing["end_time_us"] - timing["start_time_us"], 1)
        if dur <= min_dur_us:
            continue
        events.append({
            "name": _label(tid, timing, task_meta),
            "cat": timing["task_type"],
            "ph": "X",
            "ts": timing["start_time_us"],
            "dur": dur,
            "pid": 0,
            "tid": timing["node"],
            "args": {"task_id": tid, **task_meta.get(tid, {})},
        })

    events.sort(key=lambda e: e["ts"])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"traceEvents": events}, f)
    return len(events)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Chrome Trace from a dynamic-mode ExecutionResult.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help="Directory containing execution_result.json and task_meta.json.",
    )
    parser.add_argument(
        "--result", "-r", default=None,
        help="Path to execution_result.json (overrides --output-dir).",
    )
    parser.add_argument(
        "--meta", "-m", default=None,
        help="Path to task_meta.json (overrides --output-dir).",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output trace file path (default: <output-dir>/trace.json).",
    )
    parser.add_argument(
        "--min-dur", type=int, default=DEFAULT_MIN_DUR_US,
        help="Skip events with duration <= this value (e.g. 1 to filter "
             "zero-duration tasks that clutter the visualization).",
    )
    args = parser.parse_args()

    result_path = args.result or os.path.join(args.output_dir, DEFAULT_RESULT_FILE)
    meta_path = args.meta or os.path.join(args.output_dir, DEFAULT_META_FILE)
    output_path = args.output or os.path.join(args.output_dir, DEFAULT_OUTPUT_FILE)

    try:
        event_count = export_trace(result_path, meta_path, output_path,
                                   min_dur_us=args.min_dur)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Result: {result_path}")
    print(f"  Events: {event_count}")
    print(f"  Saved: {output_path}")
    print("  Open in chrome://tracing to view.")


if __name__ == "__main__":
    main()

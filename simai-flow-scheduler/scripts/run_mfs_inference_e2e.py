"""
MFS Inference E2E — compare default policy vs MFS policy on the same trace.

Usage:
    uv run python scripts/run_mfs_inference_e2e.py
"""
import sys
import os
import json

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_generator.inference_profile import InferenceProfileStore
from src.workload_generator.inference_trace_expander import InferenceTraceExpander
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.strategies.mfs_strategy import MfsAnalyzer
from src.workload_format.writer import WorkloadWriter
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.mfs_policy import MfsSchedulingPolicy
from src.executor.bandwidth_allocators.mfs_allocator import MfsAllocatorConfig

# ── Paths ─────────────────────────────────────────────────────────────────────

INFERENCE_TRACE = "inputs/traces/inference_trace_stage1_pp2.json"
TOPO_FILE       = "inputs/topologies/AlibabaHPN_32g_8gps_DualToR_DualPlane_200Gbps_A100_stage1"
PROFILE_DIR     = "inputs/vidur-csv/deepseek-tp2-pp1-ep4"
OUTPUT_DIR      = "outputs/mfs_reproduce"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _sep(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


def _has_slo(trace):
    """Check if the trace contains ttft_slo_us fields."""
    for req_info in trace.get("requests", {}).values():
        if req_info.get("ttft_slo_us") is not None:
            return True
    return False


def _run_policy(policy_name, policy, workload, topo):
    """Run executor with given policy, return ExecutionResult."""
    executor = AnalyticalExecutor(topology=topo, policy=policy)
    result = executor.execute(workload)
    return result


def _compute_qos(trace, batch_task_map, result, workload):
    """Compute per-request QoS metrics from execution result."""
    task_map = {t.task_id: t for t in workload.tasks}
    qos_records = []

    # Pre-build P2D task set for filtering collective flows
    p2d_tid_set = set()
    for bid, binfo in batch_task_map.items():
        if binfo["type"] == "kv_transfer":
            p2d_tid_set.update(binfo["task_ids"])

    for req_id_str, req_info in trace["requests"].items():
        req_id = int(req_id_str)

        # Request start time = earliest task start across all this request's batches
        request_start_us = None
        for bid, binfo in batch_task_map.items():
            if req_id in binfo.get("request_ids", []):
                for tid in binfo["task_ids"]:
                    if tid in result.per_task:
                        st = result.per_task[tid].start_time_us
                        if request_start_us is None or st < request_start_us:
                            request_start_us = st
        if request_start_us is None:
            continue

        decode_batches = []
        for bid, binfo in batch_task_map.items():
            if binfo["type"] == "decode" and req_id in binfo["request_ids"]:
                end_times = [
                    result.per_task[tid].end_time_us
                    for tid in binfo["task_ids"]
                    if tid in result.per_task
                ]
                if end_times:
                    decode_batches.append({
                        "batch_id": bid,
                        "end_time_us": max(end_times),
                    })

        decode_batches.sort(key=lambda b: b["end_time_us"])
        if not decode_batches:
            continue

        # TTFT = time from request's first task start to first decode batch completion
        ttft_us = decode_batches[0]["end_time_us"] - request_start_us
        tbt_list = [
            decode_batches[i + 1]["end_time_us"] - decode_batches[i]["end_time_us"]
            for i in range(len(decode_batches) - 1)
        ]
        e2e_us = decode_batches[-1]["end_time_us"]

        # P2D flow completion times
        p2d_times = []
        for bid, binfo in batch_task_map.items():
            if binfo["type"] == "kv_transfer" and req_id in binfo.get("request_ids", []):
                for tid in binfo["task_ids"]:
                    if tid in result.per_task:
                        p2d_times.append(result.per_task[tid].end_time_us)

        # Collective flow completion times
        collective_times = []
        for bid, binfo in batch_task_map.items():
            if binfo["type"] in ("prefill", "decode") and req_id in binfo.get("request_ids", []):
                for tid in binfo["task_ids"]:
                    if tid in result.per_task and task_map[tid].is_flow() and tid not in p2d_tid_set:
                        collective_times.append(result.per_task[tid].end_time_us)

        record = {
            "request_id": req_id,
            "num_prefill_tokens": req_info["num_prefill_tokens"],
            "num_decode_tokens": req_info["num_decode_tokens"],
            "ttft_us": ttft_us,
            "ttft_ms": round(ttft_us / 1000, 2),
            "avg_tbt_us": round(sum(tbt_list) / len(tbt_list), 1) if tbt_list else None,
            "e2e_us": e2e_us,
            "e2e_ms": round(e2e_us / 1000, 2),
            "p2d_completion_us": sorted(p2d_times) if p2d_times else [],
            "collective_completion_us": sorted(collective_times) if collective_times else [],
        }

        # Deadline metrics (relative to request_start_us)
        ttft_slo_us = req_info.get("ttft_slo_us")
        if ttft_slo_us is not None:
            record["ttft_slo_us"] = ttft_slo_us
            deadline_us = request_start_us + ttft_slo_us
            record["deadline_us"] = deadline_us
            record["deadline_met"] = ttft_us <= deadline_us
            record["deadline_miss_us"] = ttft_us - deadline_us
            if p2d_times:
                last_p2d = max(p2d_times)
                record["p2d_earliness_us"] = deadline_us - last_p2d

        qos_records.append(record)

    return qos_records


# ── Step 1: Expand inference trace ────────────────────────────────────────────

_sep("Step 1: Expand inference trace")

with open(INFERENCE_TRACE) as f:
    trace = json.load(f)

infer_tp = trace["parallelism"]["tp"]
infer_ep = trace["parallelism"]["ep"]
infer_pp = trace["parallelism"]["pp"]
num_batches = len(trace["batches"])
num_requests = len(trace["requests"])

print(f"  Model: {trace['model']}  tp={infer_tp} ep={infer_ep} pp={infer_pp}")
print(f"  Requests: {num_requests}  Batches: {num_batches}")

has_slo = _has_slo(trace)
print(f"  TTFT SLO configured: {has_slo}")

store = InferenceProfileStore(tp=infer_tp, ep=infer_ep, pp=infer_pp)
loaded = store.load_directory(PROFILE_DIR)
print(f"  Loaded {loaded} profiles")

expander = InferenceTraceExpander(store, tp=infer_tp, ep=infer_ep, pp=infer_pp)
storage_node_ids = [172]  # storage node in the stage1 topology
inference_wl, batch_task_map = expander.expand(
    trace, job_id=0, storage_node_ids=storage_node_ids,
)
print(f"  Tasks: {len(inference_wl.tasks)}  "
      f"(compute={len(inference_wl.get_compute_tasks())}, "
      f"flow={len(inference_wl.get_flow_tasks())})")

workload_path = os.path.join(OUTPUT_DIR, "workload.json")
WorkloadWriter().write(inference_wl, workload_path)
print(f"  Saved: {workload_path}")


# ── Step 2: Load topology ─────────────────────────────────────────────────────

_sep("Step 2: Load topology")

topology = TopologyLoader().load(TOPO_FILE)
print(f"  Nodes: {topology.total_nodes}  Links: {len(topology.links)}")


# ── Step 3: Run default policy ────────────────────────────────────────────────

_sep("Step 3: Run default policy")

default_analysis = DefaultAnalyzer(topology).analyze(inference_wl)
default_policy = DefaultSchedulingPolicy(analysis=default_analysis)
default_result = _run_policy("default", default_policy, inference_wl, topology)

print(f"  Makespan: {default_result.makespan_us} us  ({default_result.makespan_us/1e6:.3f} s)")
print(f"  Tasks completed: {len(default_result.per_task)}")

default_result_path = os.path.join(OUTPUT_DIR, "default_execution_result.json")
default_result.to_json(default_result_path)
print(f"  Saved: {default_result_path}")


# ── Step 4: Run MFS policy ────────────────────────────────────────────────────

_sep("Step 4: Run MFS policy")

mfs_analysis = MfsAnalyzer(topology).analyze(inference_wl, batch_task_map, trace=trace)
mfs_config = MfsAllocatorConfig()
mfs_policy = MfsSchedulingPolicy(analysis=mfs_analysis, allocator_config=mfs_config)
mfs_result = _run_policy("mfs", mfs_policy, inference_wl, topology)

print(f"  Makespan: {mfs_result.makespan_us} us  ({mfs_result.makespan_us/1e6:.3f} s)")
print(f"  Tasks completed: {len(mfs_result.per_task)}")

mfs_result_path = os.path.join(OUTPUT_DIR, "mfs_execution_result.json")
mfs_result.to_json(mfs_result_path)
print(f"  Saved: {mfs_result_path}")


# ── Step 5: Comparison report ─────────────────────────────────────────────────

_sep("Step 5: Comparison report")

default_qos = _compute_qos(trace, batch_task_map, default_result, inference_wl)
mfs_qos = _compute_qos(trace, batch_task_map, mfs_result, inference_wl)

default_qos_path = os.path.join(OUTPUT_DIR, "default_qos_report.json")
with open(default_qos_path, "w") as f:
    json.dump(default_qos, f, indent=2)
print(f"  Saved: {default_qos_path}")

mfs_qos_path = os.path.join(OUTPUT_DIR, "mfs_qos_report.json")
with open(mfs_qos_path, "w") as f:
    json.dump(mfs_qos, f, indent=2)
print(f"  Saved: {mfs_qos_path}")

# Summary comparison
comparison = {
    "default": {
        "makespan_us": default_result.makespan_us,
        "makespan_ms": round(default_result.makespan_us / 1000, 2),
    },
    "mfs": {
        "makespan_us": mfs_result.makespan_us,
        "makespan_ms": round(mfs_result.makespan_us / 1000, 2),
    },
}

# Per-request TTFT comparison (with deadline metrics when available)
req_comp = []
for d_rec in default_qos:
    m_rec = next((r for r in mfs_qos if r["request_id"] == d_rec["request_id"]), None)
    if m_rec:
        entry = {
            "request_id": d_rec["request_id"],
            "default_ttft_us": d_rec["ttft_us"],
            "mfs_ttft_us": m_rec["ttft_us"],
            "ttft_diff_us": m_rec["ttft_us"] - d_rec["ttft_us"],
        }
        if d_rec["p2d_completion_us"]:
            entry["default_last_p2d_us"] = d_rec["p2d_completion_us"][-1]
        if m_rec["p2d_completion_us"]:
            entry["mfs_last_p2d_us"] = m_rec["p2d_completion_us"][-1]
        if d_rec["collective_completion_us"]:
            entry["default_last_collective_us"] = d_rec["collective_completion_us"][-1]
        if m_rec["collective_completion_us"]:
            entry["mfs_last_collective_us"] = m_rec["collective_completion_us"][-1]

        # Deadline metrics
        if "deadline_us" in d_rec:
            entry["deadline_us"] = d_rec["deadline_us"]
            entry["default_deadline_met"] = d_rec["deadline_met"]
            entry["mfs_deadline_met"] = m_rec["deadline_met"]
            entry["default_deadline_miss_us"] = d_rec["deadline_miss_us"]
            entry["mfs_deadline_miss_us"] = m_rec["deadline_miss_us"]
        if "p2d_earliness_us" in m_rec:
            entry["mfs_p2d_earliness_us"] = m_rec["p2d_earliness_us"]
        if "p2d_earliness_us" in d_rec:
            entry["default_p2d_earliness_us"] = d_rec["p2d_earliness_us"]

        req_comp.append(entry)

comparison["per_request"] = req_comp

comp_path = os.path.join(OUTPUT_DIR, "comparison_report.json")
with open(comp_path, "w") as f:
    json.dump(comparison, f, indent=2)

print(f"\n  Default makespan: {comparison['default']['makespan_ms']} ms")
print(f"  MFS makespan:     {comparison['mfs']['makespan_ms']} ms")
print(f"\n  Saved: {comp_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

_sep("Done")
print(f"  Output directory: {OUTPUT_DIR}/")
print(f"  workload.json                  — inference P2PWorkload")
print(f"  default_execution_result.json  — default policy per-task timing")
print(f"  mfs_execution_result.json      — MFS policy per-task timing")
print(f"  default_qos_report.json        — default policy per-request metrics")
print(f"  mfs_qos_report.json            — MFS policy per-request metrics")
print(f"  comparison_report.json         — side-by-side comparison")

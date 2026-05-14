"""
Inference-only simulation:
  Vidur inference trace → P2PWorkload → AnalyticalExecutor → ExecutionResult + QoS report

Usage:
    uv run python scripts/run_inference_e2e.py

GPU assignment (default):
  P-replica (replica 0): GPUs 0-7   (server 0)
  D-replica (replica 1): GPUs 8-15  (server 1)
  KV transfer crosses servers via PSwitches.
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
from src.static_analysis.task_serializer import TaskSerializer
from src.workload_format.writer import WorkloadWriter
from src.executor.analytical import AnalyticalExecutor
from src.executor.policy import DefaultSchedulingPolicy


# ── Paths ─────────────────────────────────────────────────────────────────────

INFERENCE_TRACE = "inputs/traces/inference_trace.json"
TOPO_FILE       = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
PROFILE_DIR     = "inputs/vidur-csv/deepseek-tp2-pp1-ep4"
OUTPUT_DIR      = "outputs/inference_e2e"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _sep(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


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

store = InferenceProfileStore(tp=infer_tp, ep=infer_ep, pp=infer_pp)
loaded = store.load_directory(PROFILE_DIR)
print(f"  Loaded {loaded} profiles")

expander = InferenceTraceExpander(store, tp=infer_tp, ep=infer_ep, pp=infer_pp)
inference_wl, batch_task_map = expander.expand(trace, job_id=0)
print(f"  Tasks: {len(inference_wl.tasks)}  "
      f"(compute={len(inference_wl.get_compute_tasks())}, "
      f"flow={len(inference_wl.get_flow_tasks())})")
print(f"  Nodes: {inference_wl.jobs[0].assigned_nodes}")

workload_path = os.path.join(OUTPUT_DIR, "workload.json")
WorkloadWriter().write(inference_wl, workload_path)
print(f"  Saved: {workload_path}")


# ── Step 2: Topology + static analysis ────────────────────────────────────────

_sep("Step 2: Load topology & static analysis")

topology = TopologyLoader().load(TOPO_FILE)
print(f"  Nodes: {topology.total_nodes}  Links: {len(topology.links)}")

analysis = DefaultAnalyzer(topology).analyze(inference_wl)
print(f"  Critical path: {analysis.critical_path.makespan_us} us")


# ── Step 3: Serialize ─────────────────────────────────────────────────────────

_sep("Step 3: Serialize execution plan")

plan = TaskSerializer().serialize(inference_wl, analysis)
plan_path = os.path.join(OUTPUT_DIR, "execution_plan.json")
plan.to_json(plan_path)
print(f"  Nodes with compute tasks: {len(plan.compute_order)}")
print(f"  Saved: {plan_path}")


# ── Step 4: Execute ───────────────────────────────────────────────────────────

_sep("Step 4: Run analytical executor")

policy = DefaultSchedulingPolicy(routing_hints=analysis.routing_hints)
executor = AnalyticalExecutor(topology=topology, policy=policy)
result = executor.execute(inference_wl)

print(f"  Makespan: {result.makespan_us} us  ({result.makespan_us/1e6:.3f} s)")
print(f"  Tasks completed: {len(result.per_task)}")

result_path = os.path.join(OUTPUT_DIR, "execution_result.json")
result.to_json(result_path)
print(f"  Saved: {result_path}")


# ── Step 5: QoS report ────────────────────────────────────────────────────────

_sep("Step 5: QoS analysis (per-request inference metrics)")

task_map = {t.task_id: t for t in inference_wl.tasks}

qos_records = []
for req_id_str, req_info in trace["requests"].items():
    req_id = int(req_id_str)

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

    ttft_us = decode_batches[0]["end_time_us"]
    tbt_list = [
        decode_batches[i + 1]["end_time_us"] - decode_batches[i]["end_time_us"]
        for i in range(len(decode_batches) - 1)
    ]
    e2e_us = decode_batches[-1]["end_time_us"]

    qos_records.append({
        "request_id": req_id,
        "num_prefill_tokens": req_info["num_prefill_tokens"],
        "num_decode_tokens": req_info["num_decode_tokens"],
        "ttft_us": ttft_us,
        "ttft_ms": round(ttft_us / 1000, 2),
        "avg_tbt_us": round(sum(tbt_list) / len(tbt_list), 1) if tbt_list else None,
        "e2e_us": e2e_us,
        "e2e_ms": round(e2e_us / 1000, 2),
    })

    print(f"  Request {req_id}: "
          f"prefill={req_info['num_prefill_tokens']}tok  "
          f"decode={req_info['num_decode_tokens']}tok  "
          f"TTFT={ttft_us/1000:.1f}ms  "
          f"avg_TBT={qos_records[-1]['avg_tbt_us']}us  "
          f"E2E={e2e_us/1000:.1f}ms")

qos_path = os.path.join(OUTPUT_DIR, "qos_report.json")
with open(qos_path, "w") as f:
    json.dump(qos_records, f, indent=2)
print(f"  Saved: {qos_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

_sep("Done")
print(f"  Output directory: {OUTPUT_DIR}/")
print(f"  workload.json        — inference P2PWorkload")
print(f"  execution_plan.json  — serialized compute order")
print(f"  execution_result.json— per-task timing")
print(f"  qos_report.json      — per-request TTFT / TBT / E2E")

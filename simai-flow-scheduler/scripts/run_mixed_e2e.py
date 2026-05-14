"""
Mixed training + inference simulation:
  AICB training workload + Vidur inference trace → merged P2PWorkload
  → StaticAnalysis → ExecutionPlan → AnalyticalExecutor → ExecutionResult + QoS report

Usage:
    uv run python scripts/run_mixed_e2e.py

GPU assignment (default):
  Training job : GPUs 0-15  (tp=8, dp=2, servers 0-1)
  Inference job: GPUs 16-31 (tp=2, ep=4, pp=1, 2 replicas, servers 2-3)
    P-replica (replica 0): GPUs 16-23 (server 2)
    D-replica (replica 1): GPUs 24-31 (server 3)
  Both jobs share spine switches — traffic collision on PSwitches.
"""

import sys
import os
import json

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.inference_profile import InferenceProfileStore
from src.workload_generator.inference_trace_expander import InferenceTraceExpander
from src.workload_generator.job_merger import JobMerger
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.passes.task_serializer import CppReferenceSerializer
from src.workload_format.writer import WorkloadWriter
from src.executor.analytical import AnalyticalExecutor
from src.executor.policy import DefaultSchedulingPolicy


# ── Paths ─────────────────────────────────────────────────────────────────────

TRAINING_AICB   = "inputs/aicb-workload/gpt175b-a100.txt"
TOPO_FILE       = "inputs/topologies/AlibabaHPN_32g_8gps_DualToR_DualPlane_200Gbps_A100"
INFERENCE_TRACE = "inputs/traces/inference_trace.json"
OUTPUT_DIR      = "outputs/mixed_e2e"

# Profile directory: all matching CSV files will be auto-loaded
PROFILE_DIR = "inputs/vidur-csv/deepseek-tp2-pp1-ep4"

# Inference GPU node assignment: separate from training (GPUs 0-15)
INFER_NODES = list(range(16, 32))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _sep(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ── Step 1: Training workload ─────────────────────────────────────────────────

_sep("Step 1: Parse training AICB workload")

parser = AicbParser()
header, items = parser.parse(TRAINING_AICB)

tp   = header.tp
dp   = header.all_gpus // header.tp
pp   = header.pp
ep   = header.ep
total_gpus = header.all_gpus

print(f"  Model: GPT-175B  tp={tp} dp={dp} pp={pp} ep={ep}  GPUs={total_gpus}")
print(f"  Work items: {len(items)}")

training_job = Job(
    job_id=0,
    name="gpt175b-training",
    model="gpt175b",
    assigned_nodes=list(range(total_gpus)),
    parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
)

training_wl = WorkloadBuilder().build_from_aicb(header, items, training_job, comm_algo="ring")
print(f"  Tasks: {len(training_wl.tasks)}  "
      f"(compute={len(training_wl.get_compute_tasks())}, "
      f"flow={len(training_wl.get_flow_tasks())})")


# ── Step 2: Inference workload ────────────────────────────────────────────────

_sep("Step 2: Expand inference trace")

with open(INFERENCE_TRACE) as f:
    trace = json.load(f)

infer_tp = trace["parallelism"]["tp"]
infer_ep = trace["parallelism"]["ep"]
infer_pp = trace["parallelism"]["pp"]
num_batches = len(trace["batches"])
num_requests = len(trace["requests"])

print(f"  Model: {trace['model']}  tp={infer_tp} ep={infer_ep} pp={infer_pp}")
print(f"  Requests: {num_requests}  Batches: {num_batches}")

# Load profiles matching the trace's parallelism config
store = InferenceProfileStore(tp=infer_tp, ep=infer_ep, pp=infer_pp)
loaded = store.load_directory(PROFILE_DIR)
print(f"  Loaded {loaded} profiles: {store.list_profiles()}")

expander = InferenceTraceExpander(
    store, tp=infer_tp, ep=infer_ep, pp=infer_pp,
    assigned_nodes=INFER_NODES,
)
inference_wl, batch_task_map = expander.expand(trace, job_id=1)
print(f"  Tasks: {len(inference_wl.tasks)}  "
      f"(compute={len(inference_wl.get_compute_tasks())}, "
      f"flow={len(inference_wl.get_flow_tasks())})")
print(f"  Inference nodes: {inference_wl.jobs[0].assigned_nodes}")


# ── Step 3: Merge ─────────────────────────────────────────────────────────────

_sep("Step 3: Merge training + inference workloads")

merger = JobMerger()
merge_result = merger.merge([training_wl, inference_wl])
merged_wl = merge_result.merged_workload

print(f"  Total tasks: {len(merged_wl.tasks)}")
print(f"  Total nodes: {merged_wl.meta.num_nodes}")
print(f"  Job ID remapping: {merge_result.job_mapping}")

workload_path = os.path.join(OUTPUT_DIR, "workload.json")
WorkloadWriter().write(merged_wl, workload_path)
print(f"  Saved: {workload_path}")


# ── Step 4: Topology + static analysis ───────────────────────────────────────

_sep("Step 4: Load topology & static analysis")

topology = TopologyLoader().load(TOPO_FILE)
print(f"  Nodes: {topology.total_nodes}  Links: {len(topology.links)}")

analysis = DefaultAnalyzer(topology).analyze(merged_wl)
print(f"  Critical path: {analysis.critical_path.makespan_us} us")


# ── Step 5: Serialize ─────────────────────────────────────────────────────────

_sep("Step 5: Serialize execution plan")

plan = CppReferenceSerializer().serialize(merged_wl)
plan_path = os.path.join(OUTPUT_DIR, "execution_plan.json")
plan.to_json(plan_path)
print(f"  Nodes with compute tasks: {len(plan.compute_order)}")
print(f"  Saved: {plan_path}")


# ── Step 6: Execute ───────────────────────────────────────────────────────────

_sep("Step 6: Run analytical executor")

policy = DefaultSchedulingPolicy(analysis=analysis)
executor = AnalyticalExecutor(topology=topology, policy=policy)
result = executor.execute(merged_wl)

print(f"  Makespan: {result.makespan_us} us  ({result.makespan_us/1e6:.3f} s)")
print(f"  Tasks completed: {len(result.per_task)}")

result_path = os.path.join(OUTPUT_DIR, "execution_result.json")
result.to_json(result_path)
print(f"  Saved: {result_path}")


# ── Step 7: QoS report ────────────────────────────────────────────────────────

_sep("Step 7: QoS analysis (per-request inference metrics)")

task_map = {t.task_id: t for t in merged_wl.tasks}

# task_id_mapping[1] maps inference workload's old task IDs → new merged task IDs
infer_tid_map = merge_result.task_id_mapping.get(1, {})

def remap_tids(tids):
    return [infer_tid_map.get(tid, tid) for tid in tids]

qos_records = []
for req_id_str, req_info in trace["requests"].items():
    req_id = int(req_id_str)

    # Collect decode batches for this request in order
    decode_batches = []
    for bid, binfo in batch_task_map.items():
        if binfo["type"] == "decode" and req_id in binfo["request_ids"]:
            remapped = remap_tids(binfo["task_ids"])
            end_times = [
                result.per_task[tid].end_time_us
                for tid in remapped
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
print(f"  workload.json        — merged P2PWorkload")
print(f"  execution_plan.json  — serialized compute order")
print(f"  execution_result.json— per-task timing")
print(f"  qos_report.json      — per-request TTFT / TBT / E2E")

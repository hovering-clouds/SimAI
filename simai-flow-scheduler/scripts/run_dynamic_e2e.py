"""
Dynamic-mode inference simulation:
  Vidur inference trace → InferenceJobSlicer → CompactWorkload
  → DynamicExecutor → ExecutionResult

Usage:
    uv run python scripts/run_dynamic_e2e.py

Compared to the static path (run_inference_e2e.py), this version expands
each trace batch on demand, avoiding the full workload.json (1.6 GB for
16-request traces).
"""

import sys
import os
import json
import time as time_module

project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_generator.job_slicer import InferenceJobSlicer
from src.workload_generator.inference_profile import InferenceProfileStore
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer, DefaultAnalysisResult
from src.static_analysis.passes.routing import BfsRouteTable
from src.static_analysis.passes.task_serializer import ExecutionPlan
from src.workload_format.compact_workload import JobDAG, TaskIdAllocator
from src.executor.dynamic_executor import DynamicExecutor
from src.executor.job_expander import JobExpander
from src.executor.job_policy import FifoJobPolicy
from src.executor.policies.default_policy import DefaultSchedulingPolicy


# ── Paths ─────────────────────────────────────────────────────────────────────

INFERENCE_TRACE = "inputs/traces/inference_trace_stage1_pp1_4096.json"
TOPO_FILE       = "inputs/topologies/AlibabaHPN_64g_8gps_DualToR_DualPlane_200Gbps_H100_stage1"
PROFILE_DIR     = "inputs/vidur-csv/deepseek-tp2-pp1-ep8"
OUTPUT_DIR      = "outputs/dynamic_e2e"
STORAGE_NODES   = [208]  # storage node in 64g stage1 topology

os.makedirs(OUTPUT_DIR, exist_ok=True)


def _sep(title: str):
    print()
    print("=" * 60)
    print(title)
    print("=" * 60)


# ── Step 1: Build CompactWorkload from trace (no task expansion) ──────────────

_sep("Step 1: Build CompactWorkload")

t0 = time_module.time()

slicer = InferenceJobSlicer()
compact_wl = slicer.slice_trace(INFERENCE_TRACE)
job_dag = JobDAG.from_compact(compact_wl)

t1 = time_module.time()
print(f"  Trace: {INFERENCE_TRACE}")
print(f"  Jobs: {len(compact_wl.jobs)}")
print(f"  Root jobs: {len(job_dag.root_jobs)}")
print(f"  Builder time: {t1 - t0:.2f}s")


# ── Step 2: Topology + profile store ──────────────────────────────────────────

_sep("Step 2: Load topology & profiles")

topology = TopologyLoader().load(TOPO_FILE)
print(f"  Nodes: {topology.total_nodes}  Links: {len(topology.links)}")

p = compact_wl.jobs[0].parallelism
print(f"  Parallelism: tp={p.tp} ep={p.ep} pp={p.pp}")
store = InferenceProfileStore(tp=p.tp, ep=p.ep, pp=p.pp)
loaded = store.load_directory(PROFILE_DIR)
print(f"  Profiles loaded: {loaded}")


# ── Step 3: Dynamic executor setup ────────────────────────────────────────────

_sep("Step 3: Set up dynamic executor")

analyzer = DefaultAnalyzer(topology)

# Create an empty analysis result — policy will be populated incrementally
# via update_analysis() as each Job is expanded
empty_analysis = DefaultAnalysisResult(
    route_table=BfsRouteTable(topology),
    execution_plan=ExecutionPlan(),
)
policy = DefaultSchedulingPolicy(analysis=empty_analysis)
executor = DynamicExecutor(topology=topology, policy=policy, analyzer=analyzer)

job_policy = FifoJobPolicy()
job_expander = JobExpander(
    task_id_allocator=TaskIdAllocator(),
    profile_store=store,
    storage_node_ids=STORAGE_NODES,
)

print(f"  Policy: {type(policy).__name__}")
print(f"  Job policy: {type(job_policy).__name__}")


# ── Step 4: Execute ───────────────────────────────────────────────────────────

_sep("Step 4: Run dynamic executor")

t_exec_start = time_module.time()
result = executor.execute_dynamic(
    job_dag=job_dag,
    job_expansion_info=compact_wl.job_expansion_info,
    job_policy=job_policy,
    job_expander=job_expander,
)
t_exec_end = time_module.time()

print(f"  Makespan: {result.makespan_us} us  ({result.makespan_us/1e6:.3f} s)")
print(f"  Tasks completed: {len(result.per_task)}")
print(f"  Execution time: {t_exec_end - t_exec_start:.2f}s")

result_path = os.path.join(OUTPUT_DIR, "execution_result.json")
result.to_json(result_path)
print(f"  Saved: {result_path}")

meta_path = os.path.join(OUTPUT_DIR, "task_meta.json")
with open(meta_path, "w") as f:
    json.dump({str(k): v for k, v in executor._task_meta.items()}, f)
print(f"  Saved: {meta_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

_sep("Done")
print(f"  Output directory: {OUTPUT_DIR}/")
print(f"  execution_result.json — per-task timing")
print(f"  task_meta.json — task metadata for visualization")

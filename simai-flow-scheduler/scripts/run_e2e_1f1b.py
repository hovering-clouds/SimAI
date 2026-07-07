"""
1F1B end-to-end simulation: AICB workload → P2PWorkload → 1F1B compute_order → AnalyticalExecutor

Replaces the GPipe-style compute_order (all-F → all-B) with 1F1B ordering
where each GA step's backward follows its forward, reducing pipeline bubble.

Usage:
    uv run python scripts/run_e2e_1f1b.py
"""

import sys
import os

# Ensure project root is on path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.rank_grouper import RankGrouper
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.passes.routing import BfsStrategy
from src.static_analysis.strategies.default_strategy import (
    DefaultAnalysisResult, OneFOneBAnalyzer,
)
from src.workload_format.writer import WorkloadWriter
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy


def main():
    # --- Paths ---
    aicb_file = "inputs/aicb-workload/A100-gpt_7B_ws1_pp1-world_size1-tp1-pp1-ep1-gbs1-mbs1-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
    topo_file = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
    output_dir = "outputs/A100-gpt_7B_ws1_pp1-world_size1-tp1-pp1-ep1-gbs1-mbs1-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Step 1: Parse AICB workload
    # ============================================================
    print("=" * 60)
    print("Step 1: Parse AICB workload")
    print("=" * 60)
    parser = AicbParser()
    header, items = parser.parse(aicb_file)
    print(f"  Model: GPT-175B")
    print(f"  Header: tp={header.tp}, dp={header.all_gpus // header.tp}, "
          f"pp={header.pp}, ep={header.ep}, "
          f"ga={header.ga}, vpp={header.vpp}, all_gpus={header.all_gpus}")
    print(f"  Work items: {len(items)}")

    # ============================================================
    # Step 2: Build P2PWorkload
    # ============================================================
    print()
    print("=" * 60)
    print("Step 2: Build P2PWorkload")
    print("=" * 60)

    tp = header.tp
    dp = header.all_gpus // tp
    pp = header.pp
    ep = header.ep
    total_gpus = header.all_gpus

    job = Job(
        job_id=0,
        name="gpt175b-a100-dp2",
        model="gpt175b",
        assigned_nodes=list(range(total_gpus)),
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )

    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job, comm_algo="ring")

    print(f"  Total tasks: {len(workload.tasks)}")
    print(f"  Compute tasks: {len(workload.get_compute_tasks())}")
    print(f"  Flow tasks: {len(workload.get_flow_tasks())}")

    # Save P2PWorkload
    workload_path = os.path.join(output_dir, "workload.json")
    WorkloadWriter().write(workload, workload_path)
    print(f"  Workload saved to: {workload_path}")

    # ============================================================
    # Step 3: Load topology & run static analysis
    # ============================================================
    print()
    print("=" * 60)
    print("Step 3: Load topology & run 1F1B static analysis")
    print("=" * 60)

    loader = TopologyLoader()
    topology = loader.load(topo_file)
    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, Switches: {topology.switch_count})")
    print(f"  Links: {len(topology.links)}")
    print(f"  GPU type: {topology.gpu_type}")

    # Build node → PP stage mapping
    grouper = RankGrouper(job.assigned_nodes, job.parallelism)
    stage_size = grouper.dp * grouper.ep * grouper.tp
    node_to_stage: dict[int, int] = {}
    for stage_id in range(grouper.pp):
        for i in range(stage_size):
            node = grouper.nodes[stage_id * stage_size + i]
            node_to_stage[node] = stage_id

    # Compute routes (BFS) + 1F1B compute order
    route_table = BfsStrategy().compute_routes(workload, topology)
    plan = OneFOneBAnalyzer(pp=grouper.pp, node_to_stage=node_to_stage).analyze(workload)
    analysis = DefaultAnalysisResult(route_table=route_table, execution_plan=plan.execution_plan)

    print(f"  Nodes with compute tasks: {len(analysis.execution_plan.compute_order)}")
    print(f"  Schedule: 1F1B (pp={grouper.pp}, ga={header.ga})")

    # ============================================================
    # Step 4: Run analytical executor
    # ============================================================
    print()
    print("=" * 60)
    print("Step 4: Run analytical executor")
    print("=" * 60)

    policy = DefaultSchedulingPolicy(analysis=analysis)
    executor = AnalyticalExecutor(topology=topology, policy=policy)
    result = executor.execute(workload)

    print(f"  Total time: {result.total_time_us} us ({result.total_time_us / 1000:.2f} ms)")
    print(f"  Makespan: {result.makespan_us} us ({result.makespan_us / 1000:.2f} ms)")
    print(f"  Tasks completed: {len(result.per_task)}")

    # Per-task summary
    compute_times = [t.end_time_us - t.start_time_us for t in result.per_task.values() if t.task_type == "compute"]
    flow_times = [t.end_time_us - t.start_time_us for t in result.per_task.values() if t.task_type == "flow"]
    if compute_times:
        print(f"  Compute time range: {min(compute_times)} - {max(compute_times)} us")
    if flow_times:
        print(f"  Flow time range: {min(flow_times)} - {max(flow_times)} us")

    # Save ExecutionResult
    result_path = os.path.join(output_dir, "execution_result.json")
    result.to_json(result_path)
    print(f"  Result saved to: {result_path}")

    print()
    print("=" * 60)
    print("1F1B simulation complete!")
    print(f"  Output directory: {output_dir}/")
    print(f"  Visualize with:  python scripts/visualize.py {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

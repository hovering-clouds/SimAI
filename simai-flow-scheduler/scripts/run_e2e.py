"""
End-to-end simulation: AICB workload → P2PWorkload → Static Analysis → ExecutionPlan → AnalyticalExecutor

Usage:
    python scripts/run_e2e.py
"""

import argparse
import sys
import os
import json

# Ensure project root is on path
project_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, project_root)
os.chdir(project_root)

from src.workload_format.schema import Job, ParallelismConfig
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.workload_format.writer import WorkloadWriter
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy

DEFAULT_AICB = "inputs/aicb-workload/A100-gpt_7B_ws4_pp2-world_size4-tp2-pp2-ep1-gbs32-mbs4-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
DEFAULT_TOPO = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"


def main():
    parser = argparse.ArgumentParser(description="GPipe end-to-end simulation")
    parser.add_argument("--aicb", default=DEFAULT_AICB, help="AICB workload file")
    parser.add_argument("--topo", default=DEFAULT_TOPO, help="Topology file")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    args = parser.parse_args()

    aicb_file = args.aicb
    topo_file = args.topo
    output_dir = args.output or os.path.join(
        "outputs", os.path.splitext(os.path.basename(aicb_file))[0] + "_gpipe")
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================
    # Step 1: Parse AICB workload
    # ============================================================
    print("=" * 60)
    print("Step 1: Parse AICB workload")
    print("=" * 60)
    parser = AicbParser()
    header, items = parser.parse(aicb_file)
    dp = header.all_gpus // (header.tp * header.pp * header.ep)
    print(f"  Header: tp={header.tp}, dp={dp}, pp={header.pp}, ep={header.ep}, "
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
    dp = header.all_gpus // (header.tp * header.pp * header.ep)
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
    print("Step 3: Load topology & run static analysis")
    print("=" * 60)

    loader = TopologyLoader()
    topology = loader.load(topo_file)
    print(f"  Nodes: {topology.total_nodes} (GPUs: {topology.gpu_count}, Switches: {topology.switch_count})")
    print(f"  Links: {len(topology.links)}")
    print(f"  GPU type: {topology.gpu_type}")

    analysis = DefaultAnalyzer(topology).analyze(workload)
    print(f"  Nodes with compute tasks: {len(analysis.execution_plan.compute_order)}")

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
    print("Simulation complete!")
    print(f"  Output directory: {output_dir}/")
    print(f"  Visualize with:  python scripts/visualize.py {result_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

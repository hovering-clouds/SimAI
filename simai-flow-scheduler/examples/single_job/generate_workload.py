#!/usr/bin/env python3
"""
Single Job Example - Convert AICB workload to P2P Workload.

This example demonstrates how to:
1. Parse an AICB workload file
2. Define a job with parallelism configuration
3. Build a P2P Workload using WorkloadBuilder
4. Validate and write the workload to JSON

Usage:
    python examples/single_job/generate_workload.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.rank_grouper import RankGrouper
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_format.validator import WorkloadValidator


def main():
    # Step 1: Parse AICB workload - use the sample workload in the same directory
    aicb_file = Path(__file__).parent / "sample_workload.txt"
    print(f"Parsing AICB workload: {aicb_file}")

    parser = AicbParser()
    header, items = parser.parse(str(aicb_file))

    print(f"  Header: tp={header.tp}, ep={header.ep}, pp={header.pp}, "
          f"vpp={header.vpp}, ga={header.ga}, all_gpus={header.all_gpus}")
    print(f"  Items: {len(items)}")

    # Step 2: Define job
    # Calculate DP from world_size: dp = all_gpus / (tp * pp * ep)
    dp_size = header.all_gpus // (header.tp * header.pp * header.ep)
    job = Job(
        job_id=0,
        name="sample-training",
        model="test-model",
        assigned_nodes=list(range(header.all_gpus)),
        parallelism=ParallelismConfig(
            tp=header.tp,
            dp=dp_size,
            pp=header.pp,
            ep=header.ep,
        ),
    )
    print(f"  Job: {job.name}, nodes={job.assigned_nodes}")

    # Step 3: Build P2P Workload
    print("Building P2P Workload...")
    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job)

    print(f"  Generated {len(workload.tasks)} tasks")

    # Step 4: Validate
    print("Validating workload...")
    errors = workload.validate()
    if errors:
        print(f"  Validation FAILED:")
        for err in errors:
            print(f"    - {err}")
        return 1
    else:
        print("  Validation PASSED")

    # Step 5: Write to file
    output_dir = Path(__file__).parent
    output_file = output_dir / "single_job_workload.json"
    print(f"Writing workload to: {output_file}")

    writer = WorkloadWriter()
    writer.write(workload, str(output_file))

    # Print summary
    print("\n=== Summary ===")
    print(f"Version: {workload.version}")
    print(f"Jobs: {workload.meta.num_jobs}")
    print(f"Nodes: {workload.meta.num_nodes}")
    print(f"Tasks: {len(workload.tasks)}")

    # Count task types
    compute_tasks = [t for t in workload.tasks if t.is_compute()]
    flow_tasks = [t for t in workload.tasks if t.is_flow()]
    print(f"  Compute tasks: {len(compute_tasks)}")
    print(f"  Flow tasks: {len(flow_tasks)}")

    # Show first few tasks
    print("\nFirst 5 tasks:")
    for task in workload.tasks[:5]:
        if task.is_compute():
            print(f"  Task {task.task_id}: COMPUTE phase={task.phase.value}, "
                  f"layer={task.layer_id}, node={task.node}, duration={task.duration_us}us")
        else:
            print(f"  Task {task.task_id}: FLOW phase={task.phase.value}, "
                  f"src={task.src} -> dst={task.dst}, size={task.size_bytes} bytes, "
                  f"comm_type={task.comm_type.value}")

    print(f"\nWorkload written to: {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

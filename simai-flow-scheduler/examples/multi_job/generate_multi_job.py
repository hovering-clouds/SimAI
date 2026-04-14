#!/usr/bin/env python3
"""
Multi-Job Example - Merge multiple P2P Workloads into one.

This example demonstrates how to:
1. Parse multiple AICB workload files (simulating different training jobs)
2. Build individual P2P Workloads for each job
3. Merge them using JobMerger into a single multi-job workload
4. Validate and write the merged workload

Usage:
    python examples/multi_job/generate_multi_job.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.job_merger import JobMerger
from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_format.validator import WorkloadValidator


def main():
    example_dir = Path(__file__).parent

    # ===== Job 1: attention model, TP=2, DP=2, 4 GPUs =====
    print("=== Job 1: Attention Model (TP=2, DP=2) ===")
    parser = AicbParser()
    header1, items1 = parser.parse(str(example_dir / "job1_workload.txt"))
    print(f"  Header: tp={header1.tp}, dp=2, all_gpus={header1.all_gpus}")
    print(f"  Items: {len(items1)}")

    dp1 = header1.all_gpus // (header1.tp * header1.pp * header1.ep)
    job1 = Job(
        job_id=0,
        name="attention-model",
        model="attention-test",
        assigned_nodes=list(range(header1.all_gpus)),
        parallelism=ParallelismConfig(tp=header1.tp, dp=dp1, pp=header1.pp, ep=header1.ep),
    )

    builder = WorkloadBuilder()
    w1 = builder.build_from_aicb(header1, items1, job1)
    print(f"  Generated {len(w1.tasks)} tasks")

    # ===== Job 2: embedding model, TP=4, 4 GPUs =====
    print("\n=== Job 2: Embedding Model (TP=4) ===")
    header2, items2 = parser.parse(str(example_dir / "job2_workload.txt"))
    print(f"  Header: tp={header2.tp}, all_gpus={header2.all_gpus}")
    print(f"  Items: {len(items2)}")

    dp2 = header2.all_gpus // (header2.tp * header2.pp * header2.ep)
    job2 = Job(
        job_id=1,
        name="embedding-model",
        model="embedding-test",
        # Job 2 uses different GPU nodes [4..7] to avoid overlap
        assigned_nodes=list(range(header2.all_gpus, header2.all_gpus * 2)),
        parallelism=ParallelismConfig(tp=header2.tp, dp=dp2, pp=header2.pp, ep=header2.ep),
    )

    w2 = builder.build_from_aicb(header2, items2, job2)
    print(f"  Generated {len(w2.tasks)} tasks")

    # ===== Merge =====
    print("\n=== Merging Workloads ===")
    merger = JobMerger()
    result = merger.merge([w1, w2], topology_file="topologies/spectrum-x-8g.json")
    merged = result.merged_workload

    print(f"  Merged: {merged.meta.num_jobs} jobs, {len(merged.tasks)} tasks, "
          f"{merged.meta.num_nodes} nodes")

    # Show task ID mapping
    print(f"  Task ID mapping:")
    for w_idx, mapping in result.task_id_mapping.items():
        print(f"    Workload {w_idx}: {len(mapping)} tasks remapped")

    # ===== Validate =====
    print("\n=== Validation ===")
    errors = merged.validate()
    if errors:
        print(f"  Validation FAILED:")
        for err in errors:
            print(f"    - {err}")
        return 1
    print("  Validation PASSED")

    # ===== Write =====
    output_file = example_dir / "multi_job_workload.json"
    writer = WorkloadWriter()
    writer.write(merged, str(output_file))
    print(f"\n  Written to: {output_file}")

    # ===== Summary =====
    print("\n=== Summary ===")
    print(f"Version: {merged.version}")
    print(f"Jobs: {merged.meta.num_jobs}")
    print(f"Nodes: {merged.meta.num_nodes}")
    print(f"Total tasks: {len(merged.tasks)}")

    for job in merged.jobs:
        job_tasks = [t for t in merged.tasks if t.job_id == job.job_id]
        compute = [t for t in job_tasks if t.is_compute()]
        flows = [t for t in job_tasks if t.is_flow()]
        print(f"\n  Job {job.job_id} ({job.name}):")
        print(f"    Nodes: {job.assigned_nodes}")
        print(f"    Parallelism: TP={job.parallelism.tp}, DP={job.parallelism.dp}, "
              f"PP={job.parallelism.pp}, EP={job.parallelism.ep}")
        print(f"    Tasks: {len(job_tasks)} ({len(compute)} compute + {len(flows)} flow)")

    return 0


if __name__ == "__main__":
    sys.exit(main())

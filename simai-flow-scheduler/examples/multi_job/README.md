# Multi-Job Example

This example demonstrates how to build multiple P2P Workloads from different AICB training jobs and merge them into a single multi-job workload.

## Files

- `job1_workload.txt` - AICB workload for Job 1 (attention model, TP=2, DP=2, 4 GPUs)
- `job2_workload.txt` - AICB workload for Job 2 (embedding model, TP=4, 4 GPUs)
- `generate_multi_job.py` - Python script to generate and merge P2P Workloads
- `multi_job_workload.json` - Generated multi-job P2P Workload (output)

## Usage

```bash
cd simai-flow-scheduler
python examples/multi_job/generate_multi_job.py
```

## What it does

1. **Job 1**: Parse `job1_workload.txt`, build P2P Workload with TP=2, DP=2 (4 GPUs, nodes [0,1,2,3])
2. **Job 2**: Parse `job2_workload.txt`, build P2P Workload with TP=4 (4 GPUs, nodes [4,5,6,7])
3. **Merge**: Use `JobMerger` to combine into one workload:
   - Remap task_ids globally (Job 1: 0..39, Job 2: 40..159)
   - Update dependency references with new task_ids
   - Merge meta, network, jobs, and tasks
4. **Validate** the merged workload structure
5. **Write** to JSON file

## Output Structure

The generated `multi_job_workload.json` contains:
- 2 jobs sharing the same network topology
- ~160 tasks (40 from Job 1 + 120 from Job 2)
- Each job has independent dependency chains (no cross-job deps)
- Global task_id remapping ensures no conflicts

## Key Concepts

### Task ID Remapping

When merging, `JobMerger` assigns contiguous global task_ids:
```
Workload 0: task 0 → 0, task 1 → 1, ..., task N-1 → N-1
Workload 1: task 0 → N, task 1 → N+1, ..., task M-1 → N+M-1
```

Dependencies within each workload are updated to reference the new global IDs.

### Node Assignment

Jobs use non-overlapping node ranges:
- Job 1: nodes [0, 1, 2, 3]
- Job 2: nodes [4, 5, 6, 7]

This simulates two training tasks running on different GPUs in the same cluster, competing for shared network links.

## Customization

To add more jobs or use different workloads:
1. Add more `.txt` files following the AICB format
2. Update `generate_multi_job.py` to parse and build each new job
3. Pass all workloads to `merger.merge([...])`

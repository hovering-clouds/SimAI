# Workload Format Specification

This document describes the P2P Workload format used by SimAI Flow Scheduler.

## Overview

The P2P Workload format is an intermediate representation (IR) that describes:
- Compute tasks (with duration estimates)
- Flow tasks (point-to-point communications with dependencies)
- Job configurations (node assignments, parallelism)
- Network topology references

## File Structure

```json
{
  "version": "1.0",
  "meta": {
    "num_jobs": 2,
    "num_nodes": 16,
    "generated_at": "2026-04-08T10:30:00Z",
    "generator_version": "0.1.0"
  },
  "network": {
    "topology_file": "topologies/spectrum-x-16g.json",
    "bandwidth_gbps": 100,
    "latency_us": 1.5
  },
  "jobs": [
    {
      "job_id": 0,
      "name": "llama-70b",
      "model": "llama-70B",
      "assigned_nodes": [0, 1, 2, 3, 4, 5, 6, 7],
      "parallelism": {
        "tp": 8,
        "dp": 1,
        "pp": 1,
        "ep": 1
      }
    }
  ],
  "tasks": [
    {
      "task_id": 0,
      "job_id": 0,
      "iteration": 0,
      "phase": "forward",
      "layer_id": 0,
      "type": "compute",
      "node": 2,
      "duration_us": 1500,
      "deps": []
    },
    {
      "task_id": 1,
      "job_id": 0,
      "iteration": 0,
      "phase": "forward",
      "layer_id": 0,
      "type": "flow",
      "src": 2,
      "dst": 3,
      "size_bytes": 134217728,
      "comm_type": "tp_allreduce_ring",
      "chunk_id": 0,
      "num_chunks": 8,
      "deps": [0]
    }
  ]
}
```

## Schema Reference

### Root Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | Schema version (e.g., "1.0") |
| `meta` | object | Yes | Metadata |
| `network` | object | No | Network topology reference |
| `jobs` | array | No | Job configurations |
| `tasks` | array | Yes | Task list |

### Meta Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `num_jobs` | int | Yes | Number of jobs |
| `num_nodes` | int | Yes | Total number of nodes |
| `generated_at` | string | No | ISO 8601 timestamp |
| `generator_version` | string | No | Generator version |
| `description` | string | No | Free-form description |

### Task Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | int | Yes | Unique task ID |
| `job_id` | int | Yes | Parent job ID |
| `type` | string | Yes | `compute` or `flow` |
| `iteration` | int | No | GA step index (pre: -1, layer: 0..ga-1, post: ga) |
| `phase` | string | No | `forward`, `backward_input`, `backward_weight`, `optimizer` |
| `layer_id` | int | No | Logical layer index within iteration |
| `item_id` | int | No | Global AICB workload item index (0-based) |
| `node` | int | Conditional | Compute task node |
| `duration_us` | int | Conditional | Compute duration in microseconds |
| `src` | int | Conditional | Flow source node |
| `dst` | int | Conditional | Flow destination node |
| `size_bytes` | int | Conditional | Flow size in bytes |
| `comm_type` | string | No | Communication type |
| `chunk_id` | int | No | Chunk index for collective decomposition |
| `num_chunks` | int | No | Total number of chunks |
| `deps` | int[] | No | Dependency task IDs |

**Scheduling hints**: The scheduler can use `(iteration, layer_id, phase_order)` tuple to reproduce C++ reference execution order:
- Pre items: `iteration=-1`, sorted by `layer_id` ascending
- Layer items: `iteration=0..ga-1`, forward phase in ascending layer order (0→N-1), backward phase in descending order (N-1→0)
- Post items: `iteration=ga`, sorted by `layer_id` ascending

**Design principle**: Only true data dependencies are encoded as hard edges in the `deps` field. Cross-GA execution ordering is left to the scheduler using the hint fields above. This allows flexible scheduling strategies such as overlapping WG communication with the next GA's forward computation.

### Communication Types

- `tp_allreduce_ring` - Tensor Parallel AllReduce (Ring)
- `tp_allreduce_tree` - Tensor Parallel AllReduce (Tree)
- `tp_allgather_ring` - Tensor Parallel AllGather (Ring)
- `tp_reducescatter_ring` - Tensor Parallel ReduceScatter (Ring)
- `tp_alltoall` - Tensor Parallel AlltoAll
- `dp_allreduce` - Data Parallel AllReduce
- `ep_alltoall` - Expert Parallel AlltoAll
- `pp_send` - Pipeline Parallel Send
- `pp_recv` - Pipeline Parallel Receive

## Validation Rules

1. **Task ID uniqueness**: All `task_id` values must be unique
2. **Dependency existence**: All `deps` must reference existing task IDs
3. **DAG integrity**: No cycles in dependency graph
4. **Node range**: `node`, `src`, `dst` must be in `[0, num_nodes-1]`
5. **Compute task fields**: `node` and `duration_us` required for compute tasks
6. **Flow task fields**: `src`, `dst`, `size_bytes` required for flow tasks
7. **Scheduling field consistency**: `iteration` should follow the pattern: pre items (-1), layer items (0 to ga-1), post items (ga)

## Example: Ring AllReduce

For a 4-rank Ring AllReduce with 1GB data (TP=4, single layer):

```json
{
  "version": "1.0",
  "meta": {"num_jobs": 1, "num_nodes": 4},
  "jobs": [{"job_id": 0, "assigned_nodes": [0, 1, 2, 3], "parallelism": {"tp": 4}}],
  "tasks": [
    {"task_id": 0, "job_id": 0, "type": "flow", "src": 0, "dst": 1, "size_bytes": 268435456, "comm_type": "tp_allreduce_ring", "chunk_id": 0, "num_chunks": 4, "deps": []},
    {"task_id": 1, "job_id": 0, "type": "flow", "src": 1, "dst": 2, "size_bytes": 268435456, "comm_type": "tp_allreduce_ring", "chunk_id": 1, "num_chunks": 4, "deps": []},
    {"task_id": 2, "job_id": 0, "type": "flow", "src": 2, "dst": 3, "size_bytes": 268435456, "comm_type": "tp_allreduce_ring", "chunk_id": 2, "num_chunks": 4, "deps": []},
    {"task_id": 3, "job_id": 0, "type": "flow", "src": 3, "dst": 0, "size_bytes": 268435456, "comm_type": "tp_allreduce_ring", "chunk_id": 3, "num_chunks": 4, "deps": []},
    ...
  ]
}
```

**Note**: In the actual workload generated by `WorkloadBuilder`, each rank would have its own compute task, and flows would have proper dependencies based on the ring algorithm. The example above is simplified for illustration.

## Python API

```python
from simai_flow_scheduler import (
    P2PWorkload,
    Task,
    TaskType,
    WorkloadWriter,
    WorkloadValidator,
)

# Create workload
workload = P2PWorkload(
    version="1.0",
    meta={"num_jobs": 1, "num_nodes": 4},
    tasks=[
        Task(
            task_id=0,
            job_id=0,
            type=TaskType.COMPUTE,
            node=0,
            duration_us=1000,
            iteration=0,
            phase="forward",
            layer_id=0,
        ),
        Task(
            task_id=1,
            job_id=0,
            type=TaskType.FLOW,
            src=0,
            dst=1,
            size_bytes=1024,
            iteration=0,
            phase="forward",
            layer_id=0,
            deps=[0],  # depends on compute task
        ),
    ],
)

# Validate
errors = workload.validate()
if errors:
    print("Validation errors:", errors)

# Write
writer = WorkloadWriter()
writer.write(workload, "output.json")
```

### Scheduling Hints Usage

The scheduler can use `(iteration, layer_id, phase_order)` to reproduce C++ reference order:

```python
phase_order = {
    "forward": 0,
    "backward_input": 1,
    "backward_weight": 2,
    "optimizer": 3,
}

sorted_tasks = sorted(
    workload.tasks,
    key=lambda t: (t.iteration, t.layer_id, phase_order.get(t.phase, 0))
)
```

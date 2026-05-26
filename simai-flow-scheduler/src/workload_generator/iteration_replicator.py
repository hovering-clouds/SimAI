"""Replicate a single-iteration workload into N iterations for multi-iteration simulation."""

from ..workload_format.schema import P2PWorkload, Task


def replicate_iterations(workload: P2PWorkload, num_iterations: int) -> P2PWorkload:
    """Clone workload tasks *num_iterations* times with shifted iteration indices.

    Cross-iteration dependency chains are added so iteration k+1 begins only
    after iteration k completes on each node.  Multi-iteration workloads let
    Cassini's periodic pattern model amortise the startup delay over N iterations.
    """
    tasks = workload.tasks
    n_orig = len(tasks)

    orig_iterations = sorted({t.iteration for t in tasks})
    if not orig_iterations:
        return workload

    iter_stride = max(orig_iterations) - min(orig_iterations) + 1

    new_tasks: list[Task] = []
    task_id_offset = 0

    for copy_idx in range(num_iterations):
        iter_shift = copy_idx * iter_stride
        old_to_new: dict[int, int] = {}

        for t in tasks:
            new_id = t.task_id + task_id_offset
            old_to_new[t.task_id] = new_id

            new_t = Task(
                task_id=new_id,
                job_id=t.job_id,
                type=t.type,
                iteration=t.iteration + iter_shift,
                phase=t.phase,
                layer_id=t.layer_id,
                item_id=t.item_id,
                node=t.node,
                duration_us=t.duration_us,
                src=t.src,
                dst=t.dst,
                size_bytes=t.size_bytes,
                comm_type=t.comm_type,
                chunk_id=t.chunk_id,
                num_chunks=t.num_chunks,
                deps=[old_to_new.get(d, d + task_id_offset) for d in t.deps],
            )
            new_tasks.append(new_t)

        task_id_offset += n_orig

    return P2PWorkload(
        version=workload.version,
        meta=workload.meta,
        network=workload.network,
        jobs=list(workload.jobs),
        tasks=new_tasks,
    )

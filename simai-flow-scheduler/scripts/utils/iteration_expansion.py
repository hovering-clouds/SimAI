"""Iteration expansion utilities for Cassini multi-iteration experiments.

Provides:
    - merge_ga_to_one_iteration: Consolidate multiple GA steps into one periodic
      iteration label, fixing the semantic mismatch between AICB's GA-as-iteration
      and Cassini's iteration model.
    - replicate_with_cross_iteration_deps: Replicate a workload N times with
      proper cross-iteration dependency chains, enabling correct sequential
      multi-iteration simulation.
    - patch_iteration_time_us: Override Cassini's built-in iteration time
      estimation with a CPM-timing-based median that works reliably for any
      number of replicas.
"""

from collections import defaultdict
from dataclasses import replace
from statistics import median

from src.static_analysis.passes.cassini_task_serializer_patch import install_cassini_task_serializer_patch


install_cassini_task_serializer_patch()


def task_nodes(task):
    """Return the set of GPU ranks that a task touches."""
    if task.node is not None:
        return {task.node}
    return {n for n in (task.src, task.dst) if n is not None}


def _copy_iter_bounds(copy_idx, stride):
    """Return inclusive iteration-label bounds for one replicated copy."""
    return copy_idx * stride - 1, copy_idx * stride + 1


def _find_boundary_tasks(tasks, copy_ids, task_by_id):
    """Find source (no predecessor) and sink (no successor) tasks per job+node.

    Returns a dict: (job_id, node) → {"sources": set[task_id], "sinks": set[task_id]}.
    """
    successors = {tid: set() for tid in copy_ids}
    for task in tasks:
        for dep in task.deps:
            if dep in copy_ids:
                successors[dep].add(task.task_id)

    boundary = {}
    for task in tasks:
        for node in task_nodes(task):
            key = (task.job_id, node)
            entry = boundary.setdefault(key, {"sources": set(), "sinks": set()})
            has_node_pred = any(
                dep in copy_ids
                and node in task_nodes(task_by_id[dep])
                and task_by_id[dep].job_id == task.job_id
                for dep in task.deps
            )
            has_node_succ = any(
                node in task_nodes(task_by_id[succ])
                and task_by_id[succ].job_id == task.job_id
                for succ in successors[task.task_id]
            )
            if not has_node_pred:
                entry["sources"].add(task.task_id)
            if not has_node_succ:
                entry["sinks"].add(task.task_id)
    return boundary


def merge_ga_to_one_iteration(workload, ga):
    """Merge multiple GA-step iterations into a single periodic iteration.

    AICB files with ``ga > 1`` assign each GA step its own iteration label
    (0 .. ga-1).  Cassini should see all GA steps as ONE periodic iteration.
    After merging the workload has exactly three iteration labels::

        -1  pre-stage
         0  periodic (all GA steps merged)
         1  post-stage

    No-op when ``ga <= 1``.
    """
    if ga <= 1:
        return workload

    def _remap(it):
        if it == -1:
            return -1
        elif it < ga:
            return 0
        elif it == ga:
            return 1
        return it

    tasks = [replace(t, iteration=_remap(t.iteration)) for t in workload.tasks]
    return replace(workload, tasks=tasks)


def replicate_with_cross_iteration_deps(workload, num_iters):
    """Replicate all tasks N times with cross-iteration dependencies.

    Each copy spans iterations ``[-1, 0, 1]`` (pre, periodic, post) with stride 3.
    Cross-iteration deps connect ``post_k → pre_{k+1}`` on each GPU rank,
    creating a sequential chain::

        pre0 → periodic0 → post0 → pre1 → periodic1 → post1 → ...

    The input workload must already have GA steps merged (three iteration labels).
    No-op when ``num_iters <= 1``.
    """
    if num_iters <= 1:
        return workload

    stride = 3
    tasks = workload.tasks
    new_tasks = []
    id_stride = max((t.task_id for t in tasks), default=-1) + 1

    for copy_idx in range(num_iters):
        iter_shift = copy_idx * stride
        task_id_offset = copy_idx * id_stride
        old_to_new = {t.task_id: t.task_id + task_id_offset for t in tasks}

        for t in tasks:
            new_id = old_to_new[t.task_id]

            new_t = replace(
                t,
                task_id=new_id,
                iteration=t.iteration + iter_shift,
                deps=[old_to_new.get(d, d + task_id_offset) for d in t.deps],
            )
            new_tasks.append(new_t)

    # Cross-iteration deps: post_k → pre_{k+1} on each node
    task_by_id = {t.task_id: t for t in new_tasks}

    def _tasks_in_copy(copy_idx):
        lo, hi = _copy_iter_bounds(copy_idx, stride)
        return [t for t in new_tasks if lo <= t.iteration <= hi]

    for copy_idx in range(num_iters - 1):
        current_tasks = _tasks_in_copy(copy_idx)
        next_tasks = _tasks_in_copy(copy_idx + 1)
        current_boundary = _find_boundary_tasks(
            current_tasks, {t.task_id for t in current_tasks}, task_by_id,
        )
        next_boundary = _find_boundary_tasks(
            next_tasks, {t.task_id for t in next_tasks}, task_by_id,
        )

        for key, next_entry in next_boundary.items():
            current_entry = current_boundary.get(key)
            if current_entry is None:
                continue
            for source_tid in sorted(next_entry["sources"]):
                source_task = task_by_id[source_tid]
                for sink_tid in sorted(current_entry["sinks"]):
                    if sink_tid not in source_task.deps:
                        source_task.deps.append(sink_tid)

    return replace(workload, tasks=new_tasks)


def patch_iteration_time_us(analysis, workload):
    """Override Cassini's ``iteration_time_us`` with CPM-timing-based median.

    Cassini's built-in ``_estimate_iteration_time()`` skips iteration 0 as a
    warmup heuristic, producing unreliable results when few replicas exist.
    This function computes the median span of **periodic** iterations directly
    from the critical-path task timings and writes the result back to each
    job's ``CommunicationPattern``.

    ``analysis`` must be a ``CassiniAnalysisResult`` (has ``critical_path``
    and ``communication_patterns``).  ``workload`` is the ``P2PWorkload`` that
    was analysed.
    """
    cp = analysis.critical_path
    task_timings = cp.task_timings

    for job_id, pattern in analysis.communication_patterns.items():
        has_pre_stage = any(t.job_id == job_id and t.iteration < 0 for t in workload.tasks)
        spans_by_iter = defaultdict(list)
        for t in workload.tasks:
            if t.job_id != job_id:
                continue
            timing = task_timings.get(t.task_id)
            if timing is not None:
                if has_pre_stage:
                    iter_key = (t.iteration + 1) // 3
                else:
                    iter_key = t.iteration
                spans_by_iter[iter_key].append(timing)

        spans = []
        for timings in spans_by_iter.values():
            start = min(t.earliest_start_us for t in timings)
            finish = max(t.earliest_finish_us for t in timings)
            spans.append(finish - start)

        if spans:
            pattern.iteration_time_us = int(median(sorted(spans)))

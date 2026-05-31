"""Tests for Cassini iteration expansion (GA merge + multi-iteration replication)."""

from src.cassini.iteration_expansion import (
    merge_ga_to_one_iteration,
    replicate_with_cross_iteration_deps,
)
from src.workload_format.schema import (
    CommType,
    Job,
    Meta,
    P2PWorkload,
    ParallelismConfig,
    Phase,
    Task,
    TaskType,
)


def _compute(tid, iteration, node=0, deps=None):
    return Task(
        task_id=tid,
        job_id=0,
        type=TaskType.COMPUTE,
        iteration=iteration,
        phase=Phase.FORWARD,
        layer_id=max(iteration, 0),
        item_id=tid,
        node=node,
        duration_us=10,
        deps=deps or [],
    )


def _workload(tasks):
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        jobs=[
            Job(
                job_id=0,
                assigned_nodes=[0, 1],
                parallelism=ParallelismConfig(tp=2),
            )
        ],
        tasks=tasks,
    )


# --- merge_ga_to_one_iteration ---


def test_merge_ga_noop_when_ga_is_one():
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
    ]
    workload = _workload(tasks)
    merged = merge_ga_to_one_iteration(workload, ga=1)
    assert merged is workload


def test_merge_ga_merges_steps_into_one():
    # ga=2: pre=-1, GA0=0, GA1=1, post=2
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
        _compute(3, 2, deps=[2]),
    ]
    merged = merge_ga_to_one_iteration(_workload(tasks), ga=2)
    iterations = sorted({t.iteration for t in merged.tasks})
    assert iterations == [-1, 0, 1], f"Expected [-1, 0, 1], got {iterations}"
    # Both GA steps (0 and 1) become iteration 0
    assert sum(1 for t in merged.tasks if t.iteration == 0) == 2


def test_merge_ga_preserves_deps():
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
        _compute(3, 2, deps=[2]),
    ]
    merged = merge_ga_to_one_iteration(_workload(tasks), ga=2)
    t2 = next(t for t in merged.tasks if t.task_id == 2)
    t3 = next(t for t in merged.tasks if t.task_id == 3)
    assert t2.iteration == 0  # was GA step 1
    assert t3.iteration == 1  # was post-stage (ga=2)
    assert 1 in t2.deps  # dep preserved


# --- replicate_with_cross_iteration_deps ---


def test_replication_noop_when_one_iteration():
    workload = _workload([_compute(0, -1), _compute(1, 0), _compute(2, 1)])
    result = replicate_with_cross_iteration_deps(workload, 1)
    assert result is workload


def test_replication_creates_correct_number_of_copies():
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
    ]
    result = replicate_with_cross_iteration_deps(_workload(tasks), 3)
    # 3 copies × 3 tasks = 9 total
    assert len(result.tasks) == 9
    # 3 copies × 3 iteration labels = 9 distinct iteration values
    its = sorted({t.iteration for t in result.tasks})
    assert its == [-1, 0, 1, 2, 3, 4, 5, 6, 7]


def test_replication_preserves_unique_ids():
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
    ]
    result = replicate_with_cross_iteration_deps(_workload(tasks), 3)
    ids = [t.task_id for t in result.tasks]
    assert len(ids) == len(set(ids))  # all unique


def test_replication_valid_deps():
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
    ]
    result = replicate_with_cross_iteration_deps(_workload(tasks), 3)
    all_ids = {t.task_id for t in result.tasks}
    for task in result.tasks:
        assert all(d in all_ids for d in task.deps), f"Task {task.task_id} has invalid dep"


def test_replication_adds_cross_iteration_deps():
    """Check that post_k → pre_{k+1} deps are added on shared nodes."""
    tasks = [
        _compute(0, -1, node=0),
        _compute(1, 0, node=0, deps=[0]),
        _compute(2, 1, node=0, deps=[1]),
    ]
    result = replicate_with_cross_iteration_deps(_workload(tasks), 2)
    # Copy 0: iteration -1,0,1 → task_ids 0,1,2
    # Copy 1: iteration 2,3,4 → task_ids 3,4,5
    # Dep: first task in copy 1's pre (iteration 2) → last task in copy 0's post (iteration 1)
    # Task at iteration 2 is task_id=3
    t3 = next(t for t in result.tasks if t.task_id == 3)
    assert 2 in t3.deps, f"Expected cross-iteration dep from tid 3 → tid 2, deps={t3.deps}"


def test_replication_preserves_intra_copy_deps():
    """Check that within-copy deps are correctly remapped."""
    tasks = [
        _compute(0, -1),
        _compute(1, 0, deps=[0]),
        _compute(2, 1, deps=[1]),
    ]
    result = replicate_with_cross_iteration_deps(_workload(tasks), 2)
    # Copy 1: tasks with task_ids 3, 4, 5. Deps should be [3], [3], [4]
    t4 = next(t for t in result.tasks if t.task_id == 4)
    t5 = next(t for t in result.tasks if t.task_id == 5)
    assert t4.deps == [3], f"Expected [3], got {t4.deps}"
    assert t5.deps == [4], f"Expected [4], got {t5.deps}"

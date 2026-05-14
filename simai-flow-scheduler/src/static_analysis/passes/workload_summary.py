"""
Workload summary - overall workload characteristics and statistics.

Computes global metrics like communication/computation ratio, DAG width,
critical path statistics, and identifies hottest links.

Useful for:
- Understanding workload characteristics
- Identifying bottlenecks
- Comparing different workloads
"""

from dataclasses import dataclass, field

from ...workload_format.schema import P2PWorkload
from .contention_analysis import LinkContentionGroup
from .critical_path import CriticalPathInfo


@dataclass
class WorkloadSummary:
    total_tasks: int
    total_compute_tasks: int
    total_flow_tasks: int
    total_communication_bytes: int
    total_compute_time_us: int
    comm_compute_ratio: float
    avg_dag_width: float
    critical_path_length_us: int
    critical_path_comm_fraction: float
    hot_links: list[tuple[tuple[int, int], int, int]] = field(default_factory=list)


def compute_workload_summary(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
    contention_groups: dict[tuple[int, int], LinkContentionGroup],
) -> WorkloadSummary:
    compute_tasks = [t for t in workload.tasks if t.is_compute()]
    flow_tasks = [t for t in workload.tasks if t.is_flow()]

    total_comm_bytes = sum(t.size_bytes or 0 for t in flow_tasks)
    total_compute_time = sum(t.duration_us or 0 for t in compute_tasks)

    total_comm_time = sum(
        critical_path.task_timings[t.task_id].earliest_finish_us
        - critical_path.task_timings[t.task_id].earliest_start_us
        for t in flow_tasks
        if t.task_id in critical_path.task_timings
    )

    comm_compute_ratio = (
        total_comm_time / total_compute_time if total_compute_time > 0 else 0.0
    )

    dag_width = _compute_avg_dag_width(workload)

    cp_comm_time = sum(
        critical_path.task_timings[t.task_id].earliest_finish_us
        - critical_path.task_timings[t.task_id].earliest_start_us
        for t in flow_tasks
        if t.task_id in critical_path.critical_tasks
    )
    cp_total_time = critical_path.makespan_us
    cp_comm_fraction = cp_comm_time / cp_total_time if cp_total_time > 0 else 0.0

    hot_links = sorted(
        [(gid, g.total_data_bytes, g.num_flows) for gid, g in contention_groups.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return WorkloadSummary(
        total_tasks=len(workload.tasks),
        total_compute_tasks=len(compute_tasks),
        total_flow_tasks=len(flow_tasks),
        total_communication_bytes=total_comm_bytes,
        total_compute_time_us=total_compute_time,
        comm_compute_ratio=comm_compute_ratio,
        avg_dag_width=dag_width,
        critical_path_length_us=critical_path.makespan_us,
        critical_path_comm_fraction=cp_comm_fraction,
        hot_links=hot_links,
    )


def _compute_avg_dag_width(workload: P2PWorkload) -> float:
    if not workload.tasks:
        return 0.0

    task_map = {t.task_id: t for t in workload.tasks}

    depths: dict[int, int] = {}
    in_degree: dict[int, int] = {}
    children: dict[int, list[int]] = {}
    for task in workload.tasks:
        tid = task.task_id
        in_degree[tid] = len(task.deps)
        for dep in task.deps:
            children.setdefault(dep, []).append(tid)
        if not task.deps:
            depths[tid] = 0

    queue = [tid for tid, deg in in_degree.items() if deg == 0]
    while queue:
        next_queue = []
        for tid in queue:
            for child in children.get(tid, []):
                new_depth = depths[tid] + 1
                if child not in depths or new_depth > depths[child]:
                    depths[child] = new_depth
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    next_queue.append(child)
        queue = next_queue

    levels: dict[int, int] = {}
    for depth in depths.values():
        levels[depth] = levels.get(depth, 0) + 1

    if not levels:
        return 0.0

    return sum(levels.values()) / len(levels)

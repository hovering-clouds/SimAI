"""Feasibility analysis — per-task durations and per-request critical path totals.

Computes estimated durations for each task, then finds the critical path
for each request up to and including its P2D (KV cache transfer) tasks.
This supports the TTFT deadline feasibility check: if a request's remaining
critical path time exceeds its deadline, its flows are demoted to the
lowest priority queue.
"""
from collections import deque
from dataclasses import dataclass, field

from .critical_path import _estimate_duration
from .mfs_context import MfsContext, MfsStage
from .routing import RouteTable
from .topology_loader import NetworkTopology
from ...workload_format.schema import P2PWorkload


@dataclass
class FeasibilityInfo:
    task_duration_us: dict[int, int]
    critical_path_tasks: set[int] = field(default_factory=set)
    request_critical_tasks: dict[int, set[int]] = field(default_factory=dict)
    request_path_total_us: dict[int, int] = field(default_factory=dict)


def build_feasibility_info(
    workload: P2PWorkload,
    context: MfsContext,
    route_table: RouteTable,
    topology: NetworkTopology,
) -> FeasibilityInfo:
    # Step 1: per-task duration
    task_map = {t.task_id: t for t in workload.tasks}
    task_duration_us: dict[int, int] = {}
    for task in workload.tasks:
        task_duration_us[task.task_id] = _estimate_duration(
            task, route_table, topology,
        )

    critical_path_tasks: set[int] = set()
    request_critical_tasks: dict[int, set[int]] = {}
    request_path_total_us: dict[int, int] = {}

    # Step 2: per-request critical path (up to P2D tasks)
    for rid, rid_task_tuple in context.request_to_tasks.items():
        rid_tasks = set(rid_task_tuple) & set(task_duration_us.keys())
        if not rid_tasks:
            continue

        # Identify P2D tasks for this request
        p2d_tasks = {
            tid for tid in rid_tasks
            if tid in context.task_info
            and context.task_info[tid].mfs_stage == MfsStage.P2D
        }
        if not p2d_tasks:
            continue

        # Build sub-DAG: internal deps only
        internal_deps: dict[int, list[int]] = {tid: [] for tid in rid_tasks}
        dependents: dict[int, list[int]] = {tid: [] for tid in rid_tasks}
        for tid in rid_tasks:
            task = task_map.get(tid)
            if task is None:
                continue
            for dep in task.deps:
                if dep in rid_tasks:
                    internal_deps[tid].append(dep)
                    dependents[dep].append(tid)

        # Topological sort (Kahn's algorithm)
        in_degree = {tid: len(internal_deps[tid]) for tid in rid_tasks}
        queue = deque(tid for tid in rid_tasks if in_degree[tid] == 0)
        topo_order: list[int] = []
        while queue:
            tid = queue.popleft()
            topo_order.append(tid)
            for child in dependents[tid]:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        # Longest path (critical path)
        dist: dict[int, int] = {}
        for tid in topo_order:
            dist[tid] = max(
                (dist[dep] + task_duration_us[dep] for dep in internal_deps[tid]),
                default=0,
            )

        # Total = max over P2D terminals only
        p2d_finish = {
            tid: dist[tid] + task_duration_us[tid] for tid in p2d_tasks
        }
        if not p2d_finish:
            continue

        total = max(p2d_finish.values())
        request_path_total_us[rid] = total

        # Trace back from the terminal to find critical tasks
        terminal = max(p2d_finish, key=lambda t: p2d_finish[t])
        crit: set[int] = set()
        stack = [terminal]
        while stack:
            tid = stack.pop()
            if tid in crit:
                continue
            crit.add(tid)
            for dep in internal_deps[tid]:
                if dist[dep] + task_duration_us[dep] == dist[tid]:
                    stack.append(dep)
                    break

        request_critical_tasks[rid] = crit
        critical_path_tasks |= crit

    return FeasibilityInfo(
        task_duration_us=task_duration_us,
        critical_path_tasks=critical_path_tasks,
        request_critical_tasks=request_critical_tasks,
        request_path_total_us=request_path_total_us,
    )

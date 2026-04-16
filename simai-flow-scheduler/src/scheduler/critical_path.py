"""
Critical path analysis - compute task timing and identify critical tasks.

Implements CPM (Critical Path Method) with multi-hop delay estimation:
- Forward pass (ASAP): compute earliest start/finish times
- Backward pass (ALAP): compute latest start/finish times
- Slack = latest_start - earliest_start; critical tasks have slack == 0

Flow duration estimation:
- Transmission delay = size_bits / bottleneck_bandwidth (min bw along path)
- Propagation delay = sum of all link latencies along path
- Total duration = transmission_delay + propagation_delay
"""

from dataclasses import dataclass

from ..workload_format.schema import P2PWorkload, Task, TaskType
from .routing_hints import RoutingHints
from .topology_loader import NetworkTopology


@dataclass
class TaskTimingInfo:
    """
    Timing analysis result for a single task.

    Compatible with all analysis methods:
    - CPM:   all fields populated via forward/backward pass
    - TTE:   slack_us == TTE for flow tasks; compute tasks set to inf
    - RCPSP: earliest_* reflects resource-constrained schedule
    """

    task_id: int

    # Forward pass results (always present)
    earliest_start_us: int
    earliest_finish_us: int

    # Backward pass results (may be approximate depending on method)
    latest_start_us: int
    latest_finish_us: int

    # Slack = latest_start - earliest_start
    # float to allow inf (tasks with no downstream compute dependency)
    slack_us: float

    # Is this task on the critical path?
    is_critical: bool


@dataclass
class CriticalPathInfo:
    """
    Critical path analysis result.

    analysis_method records which algorithm was used,
    allowing callers to interpret precision accordingly.
    """

    # Per-task timing info
    task_timings: dict[int, TaskTimingInfo]

    # Critical tasks (slack == 0)
    critical_tasks: list[int]

    # Total makespan (optimistic lower bound)
    makespan_us: int

    # Algorithm used: "cpm" | "tte" | "rcpsp"
    analysis_method: str

    def get_slack(self, task_id: int) -> float:
        return self.task_timings[task_id].slack_us

    def is_critical(self, task_id: int) -> bool:
        return task_id in self.critical_tasks

    def get_earliest_start(self, task_id: int) -> int:
        return self.task_timings[task_id].earliest_start_us


def analyze_critical_path(
    workload: P2PWorkload,
    topology: NetworkTopology,
    routing_hints: RoutingHints,
) -> CriticalPathInfo:
    """
    Critical path analysis with multi-hop latency estimation (CPM).

    Flow duration = transmission_delay + propagation_delay
    - transmission_delay = size / bottleneck_bandwidth (min bw along path)
    - propagation_delay = sum of all link latencies along path

    Time complexity: O(V + E) where V = tasks, E = dependency edges.
    """
    tasks = workload.tasks

    if not tasks:
        return CriticalPathInfo(
            task_timings={},
            critical_tasks=[],
            makespan_us=0,
            analysis_method="cpm",
        )

    # Step 1: Topological sort
    sorted_tasks = _topological_sort(tasks)

    # Step 2: Forward pass (ASAP)
    earliest_start: dict[int, int] = {}
    earliest_finish: dict[int, int] = {}

    for task in sorted_tasks:
        earliest_start[task.task_id] = (
            0
            if not task.deps
            else max(earliest_finish[dep] for dep in task.deps)
        )
        earliest_finish[task.task_id] = (
            earliest_start[task.task_id]
            + _estimate_duration(task, topology, routing_hints)
        )

    # Step 3: Backward pass (ALAP)
    makespan = max(earliest_finish.values())
    latest_start: dict[int, int] = {}
    latest_finish: dict[int, int] = {}

    # Build dependents map
    dependents: dict[int, list[int]] = {t.task_id: [] for t in tasks}
    for task in tasks:
        for dep in task.deps:
            dependents[dep].append(task.task_id)

    for task in reversed(sorted_tasks):
        latest_finish[task.task_id] = (
            makespan
            if not dependents[task.task_id]
            else min(latest_start[d] for d in dependents[task.task_id])
        )
        latest_start[task.task_id] = (
            latest_finish[task.task_id]
            - _estimate_duration(task, topology, routing_hints)
        )

    # Step 4: Build output
    task_timings: dict[int, TaskTimingInfo] = {}
    for task in tasks:
        tid = task.task_id
        slack = float(latest_start[tid] - earliest_start[tid])
        task_timings[tid] = TaskTimingInfo(
            task_id=tid,
            earliest_start_us=earliest_start[tid],
            earliest_finish_us=earliest_finish[tid],
            latest_start_us=latest_start[tid],
            latest_finish_us=latest_finish[tid],
            slack_us=slack,
            is_critical=(slack == 0.0),
        )

    critical_tasks = [tid for tid, t in task_timings.items() if t.is_critical]

    return CriticalPathInfo(
        task_timings=task_timings,
        critical_tasks=critical_tasks,
        makespan_us=makespan,
        analysis_method="cpm",
    )


def _estimate_duration(
    task: Task,
    topology: NetworkTopology,
    routing_hints: RoutingHints,
) -> int:
    """
    Estimate task duration for critical path analysis.

    For compute tasks: returns task.duration_us or 0.
    For flow tasks:
        Total duration = transmission_delay + propagation_delay
        - transmission_delay = size_bits / bottleneck_bandwidth
        - propagation_delay = sum of all link latencies along path
    """
    if not task.is_flow():
        return task.duration_us or 0

    if task.src is None or task.dst is None:
        return 0

    # Get full path through topology (from routing hints)
    path = routing_hints.get_path(topology, task.src, task.dst)
    if len(path) < 2:
        return 0

    size_bits = (task.size_bytes or 0) * 8
    if size_bits == 0:
        return 0

    bottleneck_bw_gbps = float("inf")
    total_latency_us = 0.0

    for i in range(len(path) - 1):
        link = topology.get_link(path[i], path[i + 1])
        if link is None:
            return 0
        bottleneck_bw_gbps = min(bottleneck_bw_gbps, link.bandwidth_gbps)
        total_latency_us += link.latency_us

    if bottleneck_bw_gbps <= 0 or bottleneck_bw_gbps == float("inf"):
        return 0

    # tx_time on bottleneck link (in microseconds)
    tx_time_us = size_bits / (bottleneck_bw_gbps * 1e9) * 1e6

    return int(tx_time_us + total_latency_us)


def _topological_sort(tasks: list[Task]) -> list[Task]:
    """Topological sort via DFS post-order (dependencies appear before dependents)."""
    task_map = {t.task_id: t for t in tasks}
    visited: set[int] = set()
    result: list[Task] = []

    def visit(tid: int):
        if tid in visited:
            return
        visited.add(tid)
        for dep in task_map[tid].deps:
            visit(dep)
        result.append(task_map[tid])

    for task in tasks:
        visit(task.task_id)
    return result

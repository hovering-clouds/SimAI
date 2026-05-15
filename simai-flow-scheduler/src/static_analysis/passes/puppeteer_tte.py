"""TTE (Time-to-Exposed) analysis for Puppeteer flow prioritization.

TTE measures how much a flow can be delayed before it exposes compute stall.
Lower TTE means higher priority.

Formula:
    TTE(flow) = min(child.earliest_start - flow.earliest_finish)
    over all direct child tasks in the workload DAG.
"""
from dataclasses import dataclass

from ...workload_format.schema import P2PWorkload, Task
from .routing import RouteTable
from .topology_loader import NetworkTopology
from .task_serializer import ExecutionPlan


@dataclass
class TTEInfo:
    """Time-to-Exposed information for a flow task."""
    task_id: int
    tte_us: float
    priority_score: float
    priority_class: str  # "critical" | "elastic" | "background"


@dataclass
class FlowTiming:
    """Optimistic timing estimate for a flow task."""
    task_id: int
    start_time_us: int
    finish_time_us: int


def _estimate_flow_duration_on_path(
    task: Task,
    path: list[int],
    topology: NetworkTopology,
) -> int:
    """Estimate flow transmission duration at line rate over a specific path."""
    if len(path) < 2:
        return 0
    size_bits = (task.size_bytes or 0) * 8
    if size_bits == 0:
        return 0

    bottleneck_bw = float("inf")
    total_latency = 0.0

    for i in range(len(path) - 1):
        link = topology.get_link(path[i], path[i + 1])
        if link is None:
            return 0
        bottleneck_bw = min(bottleneck_bw, link.bandwidth_gbps)
        total_latency += link.latency_us

    if bottleneck_bw <= 0 or bottleneck_bw == float("inf"):
        return 0

    tx_time_us = size_bits / (bottleneck_bw * 1e9) * 1e6
    return int(tx_time_us + total_latency)


def compute_optimistic_timing(
    workload: P2PWorkload,
    route_table: RouteTable,
    topology: NetworkTopology,
    execution_plan: ExecutionPlan,
    route_paths: dict[int, list[int]] | None = None,
) -> dict[int, int]:
    """Compute optimistic ASAP finish times for all tasks.

    Includes both DAG dependency edges and compute-order implicit edges
    from ExecutionPlan.

    Args:
        workload: P2P workload
        route_table: Route table for default flow duration estimation
        topology: Network topology for delay estimation
        execution_plan: Compute ordering (implicit per-node serialization edges)
        route_paths: Optional explicit paths for flow tasks (from greedy routing)

    Returns:
        dict mapping task_id -> earliest_finish_us
    """
    tasks = {t.task_id: t for t in workload.tasks}

    # Build compute-order chain: each task has at most one successor and one predecessor
    compute_successors: dict[int, int] = {}
    compute_predecessors: dict[int, int] = {}
    for node, tids in execution_plan.compute_order.items():
        for i in range(len(tids) - 1):
            compute_successors[tids[i]] = tids[i + 1]
            compute_predecessors[tids[i + 1]] = tids[i]

    # Topological sort of the DAG
    visited: set[int] = set()
    sorted_tids: list[int] = []

    def _dfs(tid: int):
        if tid in visited:
            return
        visited.add(tid)
        for dep in tasks[tid].deps:
            _dfs(dep)
        sorted_tids.append(tid)

    for t in workload.tasks:
        _dfs(t.task_id)

    # Forward pass
    earliest_finish: dict[int, int] = {}

    for tid in sorted_tids:
        task = tasks[tid]

        # Collect predecessor finish times (DAG deps + compute-order deps)
        pred_finishes = [earliest_finish[dep] for dep in task.deps if dep in earliest_finish]
        pred = compute_predecessors.get(tid)
        if pred is not None and pred in earliest_finish:
            pred_finishes.append(earliest_finish[pred])

        start = max(pred_finishes) if pred_finishes else 0

        # Duration
        if task.is_compute():
            duration = task.duration_us or 0
        else:
            if route_paths and tid in route_paths:
                path = route_paths[tid]
            else:
                if task.src is None or task.dst is None:
                    duration = 0
                else:
                    path = route_table.get_path(task)
            duration = _estimate_flow_duration_on_path(task, path, topology)

        earliest_finish[tid] = start + duration

    return earliest_finish


def compute_tte(
    workload: P2PWorkload,
    route_table: RouteTable,
    topology: NetworkTopology,
    execution_plan: ExecutionPlan,
    route_paths: dict[int, list[int]] | None = None,
    small_threshold_us: float = 1000.0,
) -> tuple[dict[int, TTEInfo], dict[int, FlowTiming]]:
    """Compute TTE and priority for all flow tasks.

    Args:
        workload: P2P workload
        route_table: Route table for default path estimation
        topology: Network topology for delay estimation
        execution_plan: Compute ordering (includes implicit edges)
        route_paths: Optional explicit paths for flows (from greedy routing)
        small_threshold_us: Threshold for elastic vs background classification

    Returns:
        (tte_info_map, flow_timing_map):
            tte_info_map: task_id -> TTEInfo for each flow
            flow_timing_map: task_id -> FlowTiming for each flow
    """
    # Compute optimistic timing
    earliest_finish = compute_optimistic_timing(
        workload, route_table, topology, execution_plan, route_paths,
    )

    tasks = {t.task_id: t for t in workload.tasks}

    # Build child map: task_id -> list of dependent task ids
    children: dict[int, list[int]] = {}
    for t in workload.tasks:
        for dep in t.deps:
            children.setdefault(dep, []).append(t.task_id)

    # Compute TTE per flow
    tte_info_map: dict[int, TTEInfo] = {}
    flow_timing_map: dict[int, FlowTiming] = {}

    for t in workload.tasks:
        if not t.is_flow():
            continue

        tid = t.task_id

        # Determine path for duration estimation
        if route_paths and tid in route_paths:
            path = route_paths[tid]
        else:
            path = route_table.get_path(t)

        duration = _estimate_flow_duration_on_path(t, path, topology)
        finish = earliest_finish.get(tid, 0)
        start = finish - duration

        flow_timing_map[tid] = FlowTiming(
            task_id=tid,
            start_time_us=start,
            finish_time_us=finish,
        )

        # Compute TTE from children
        task_children = children.get(tid, [])

        if not task_children:
            tte = float("inf")
        else:
            tte = float("inf")
            for child_id in task_children:
                child_task = tasks[child_id]
                # child earliest start = earliest_finish[child] - child_duration
                if child_task.is_compute():
                    child_duration = child_task.duration_us or 0
                else:
                    child_duration = _estimate_flow_duration_on_path(
                        child_task,
                        route_table.get_path(child_task),
                        topology,
                    )
                child_start = earliest_finish.get(child_id, 0) - child_duration
                tte = min(tte, child_start - finish)

        # Classify priority
        if tte <= 0:
            priority_class = "critical"
            priority_score = 0.0
        elif tte <= small_threshold_us:
            priority_class = "elastic"
            priority_score = 1.0 / max(tte, 1.0)
        else:
            priority_class = "background"
            priority_score = 1.0 / max(tte, 1.0)

        tte_info_map[tid] = TTEInfo(
            task_id=tid,
            tte_us=tte,
            priority_score=priority_score,
            priority_class=priority_class,
        )

    return tte_info_map, flow_timing_map

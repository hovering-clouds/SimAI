"""
Extract periodic communication patterns from P2PWorkload + CriticalPathInfo.

Cassini represents each DNN training job's network activity as a periodic
pattern of Up (communication-heavy) and Down (computation-only) phases
that repeat every iteration.

The extraction pipeline:
    1. Group tasks by job_id and iteration
    2. Get each flow task's timing from CriticalPathInfo
    3. Get each flow task's path from RouteTable
    4. Discretize time into angular buckets and sum bandwidth per link
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..workload_format.schema import P2PWorkload

if TYPE_CHECKING:
    from ..static_analysis.passes.critical_path import CriticalPathInfo
    from ..static_analysis.passes.routing import RouteTable
    from ..static_analysis.passes.topology_loader import NetworkTopology


@dataclass
class CommunicationPattern:
    """Periodic communication pattern for a single job on all its links.

    Attributes:
        job_id: The job's identifier.
        iteration_time_us: Duration of one training iteration in microseconds.
        link_demands: Mapping from link_id (src, dst) to a 360-entry dict of
            bandwidth demands (Gbps) at each angular position.
    """

    job_id: int
    iteration_time_us: int
    link_demands: dict[tuple[int, int], dict[int, float]] = field(default_factory=dict)


def _path_to_links(path: list[int]) -> list[tuple[int, int]]:
    """Convert a node path [src, hop1, ..., dst] to a list of link tuples."""
    return [(path[i], path[i + 1]) for i in range(len(path) - 1)]


def extract_communication_patterns(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
    route_table: RouteTable,
    topology: "NetworkTopology",
    num_angles: int = 360,
) -> dict[int, CommunicationPattern]:
    """Extract periodic communication patterns for every job in the workload.

    For each job, this function:
        1. Determines the iteration time from task timings across
           multiple iterations (skipping warmup iteration 0 when possible).
        2. Iterates over all flow tasks and retrieves their timing and path.
        3. Discretizes each flow's bandwidth demand onto angular buckets
           for every link along its path, capped at each link's actual capacity.
        4. Aggregates overlapping demands within each bucket.

    Args:
        workload: The parsed multi-job workload.
        critical_path: Critical path analysis results (provides timing).
        route_table: Pre-computed routes for each flow task.
        topology: Network topology (used to look up per-link capacities).
        num_angles: Number of angular buckets for discretization (default 360).

    Returns:
        Dict mapping job_id to its CommunicationPattern.
    """
    # Group tasks by job
    tasks_by_job: dict[int, list] = defaultdict(list)
    for task in workload.tasks:
        tasks_by_job[task.job_id].append(task)

    patterns: dict[int, CommunicationPattern] = {}

    for job_id, tasks in tasks_by_job.items():
        _build_job_pattern(
            job_id, tasks, critical_path, route_table, topology,
            num_angles, patterns,
        )

    return patterns


def _estimate_iteration_time(
    tasks: list,
    critical_path: CriticalPathInfo,
) -> int:
    """Estimate steady-state iteration time from task timings.

    Uses the median span of non-warmup iterations when multiple iterations
    are available.  Falls back to iteration 0 (or the only iteration) when
    only one iteration is in the trace.  Returns 0 if no valid estimate can
    be made.
    """
    # Group tasks by iteration
    tasks_by_iter: dict[int, list] = defaultdict(list)
    for t in tasks:
        tasks_by_iter[t.iteration].append(t)

    if not tasks_by_iter:
        return 0

    def _iter_span(iter_tasks: list) -> int:
        """Compute the span (max_finish - min_start) of one iteration."""
        starts = []
        finishes = []
        for t in iter_tasks:
            timing = critical_path.task_timings.get(t.task_id)
            if timing is not None:
                starts.append(timing.earliest_start_us)
                finishes.append(timing.earliest_finish_us)
        if not starts:
            return 0
        return max(finishes) - min(starts)

    # Collect spans for all iterations that have timing data
    spans: dict[int, int] = {}
    for it, its in tasks_by_iter.items():
        s = _iter_span(its)
        if s > 0:
            spans[it] = s

    if not spans:
        return critical_path.makespan_us

    # Prefer non-zero iterations (skip warmup); fall back to all available
    non_warmup = {it: s for it, s in spans.items() if it != 0}
    candidates = non_warmup if non_warmup else spans

    # Median is more robust to outliers than mean
    sorted_spans = sorted(candidates.values())
    n = len(sorted_spans)
    if n % 2 == 1:
        return sorted_spans[n // 2]
    else:
        return (sorted_spans[n // 2 - 1] + sorted_spans[n // 2]) // 2


def _build_job_pattern(
    job_id: int,
    tasks: list,
    critical_path: CriticalPathInfo,
    route_table: RouteTable,
    topology: "NetworkTopology",
    num_angles: int,
    patterns: dict[int, CommunicationPattern],
) -> None:
    """Build a single job's communication pattern."""
    flow_tasks = [t for t in tasks if t.is_flow()]
    if not flow_tasks:
        return

    # Estimate iteration time from task timings.
    # Skip iteration 0 when possible — warmup / cache-fill / pipeline-fill
    # effects make it less representative of steady-state.
    iteration_time_us = _estimate_iteration_time(tasks, critical_path)
    if iteration_time_us <= 0:
        return

    pattern = CommunicationPattern(
        job_id=job_id,
        iteration_time_us=iteration_time_us,
    )

    for task in flow_tasks:
        _add_task_to_pattern(task, critical_path, route_table, topology,
                             iteration_time_us, num_angles, pattern)

    if pattern.link_demands:
        patterns[job_id] = pattern


def _add_task_to_pattern(
    task,
    critical_path: CriticalPathInfo,
    route_table: RouteTable,
    topology: "NetworkTopology",
    iteration_time_us: int,
    num_angles: int,
    pattern: CommunicationPattern,
) -> None:
    """Discretize a single flow task's bandwidth onto the pattern's links.

    For each link along the flow's path, the effective bandwidth is capped at
    the link's physical capacity.  On slow links (e.g. 200 Gbps ASW) the
    flow's duration is extended proportionally so that the total data
    transferred equals size_bytes.  This prevents the CPM ideal timing
    (which assumes NVLink ~2880 Gbps) from inflating bandwidth demands on
    inter-switch links.
    """
    timing = critical_path.task_timings.get(task.task_id)
    if timing is None:
        return

    size_bytes = task.size_bytes or 0
    if size_bytes <= 0:
        return

    # Resolve path into links
    try:
        path = route_table.get_path(task)
    except (KeyError, ValueError):
        return
    links = _path_to_links(path)

    cpm_start = timing.earliest_start_us
    cpm_finish = timing.earliest_finish_us
    cpm_duration = max(cpm_finish - cpm_start, 1)

    for link in links:
        # Look up the actual link capacity (default 100 Gbps if missing)
        topo_link = topology.get_link(*link)
        link_cap = topo_link.bandwidth_gbps if topo_link else 100.0
        if link_cap <= 0:
            continue

        # Minimum transmission time at this link's speed
        size_gbits = size_bytes * 8 / 1e9
        min_tx_us = int(size_gbits / link_cap * 1e6)

        # Bottleneck: slower of CPM ideal vs link capacity
        effective_duration = max(cpm_duration, min_tx_us)
        effective_bw = size_bytes * 8 / (effective_duration * 1000)

        # Extend end time so the angular spread reflects the true duration
        effective_end = cpm_start + effective_duration

        link_bw = pattern.link_demands.setdefault(link, {})
        _add_flow_to_link_buckets(
            link_bw, cpm_start, effective_end, effective_duration,
            effective_bw, iteration_time_us, num_angles,
        )


def _add_flow_to_link_buckets(
    link_bw: dict[int, float],
    task_start: int,
    task_end: int,
    duration: int,
    bw: float,
    iteration_time_us: int,
    num_angles: int,
) -> None:
    """Add a flow's bandwidth to the angular buckets of one link.

    When duration >= iteration_time_us, the flow spans the entire iteration
    and occupies all buckets.  Otherwise, the flow is folded into the
    iteration window via modulo and fills the buckets it overlaps.
    """
    if duration >= iteration_time_us:
        for a in range(num_angles):
            link_bw[a] = link_bw.get(a, 0.0) + bw
        return

    offset_start = task_start % iteration_time_us
    offset_end = task_end % iteration_time_us

    start_angle = int(offset_start * num_angles / iteration_time_us)
    end_angle = int(offset_end * num_angles / iteration_time_us)

    start_angle = max(0, min(start_angle, num_angles - 1))
    end_angle = max(0, min(end_angle, num_angles - 1))

    if end_angle > start_angle:
        for a in range(start_angle, end_angle):
            link_bw[a] = link_bw.get(a, 0.0) + bw
    elif end_angle == start_angle:
        # Flow fits within a single angle bucket
        link_bw[start_angle] = link_bw.get(start_angle, 0.0) + bw
    else:
        # Wrap-around
        for a in range(start_angle, num_angles):
            link_bw[a] = link_bw.get(a, 0.0) + bw
        for a in range(0, end_angle):
            link_bw[a] = link_bw.get(a, 0.0) + bw

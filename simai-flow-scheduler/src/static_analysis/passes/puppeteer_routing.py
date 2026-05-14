"""Offline greedy routing for Puppeteer-like flow planning.

Produces a precomputed RouteTable using a least-active-link heuristic.
This is an offline planning pass, not runtime dynamic routing.
"""
from collections import deque
from dataclasses import dataclass, field

from ...workload_format.schema import P2PWorkload
from .topology_loader import NetworkTopology


@dataclass
class RouteTable:
    """Precomputed route table for flow tasks.

    Maps flow task_id -> node path [src, hop1, ..., dst].
    Every flow task should have exactly one path.
    """
    paths: dict[int, list[int]] = field(default_factory=dict)

    def get_path(self, task_id: int) -> list[int]:
        if task_id not in self.paths:
            raise KeyError(
                f"No route found for flow task {task_id}. "
                f"This indicates a missing route planning entry."
            )
        return list(self.paths[task_id])


def k_shortest_paths(
    topology: NetworkTopology,
    src: int,
    dst: int,
    k: int = 4,
) -> list[list[int]]:
    """Find up to k simple shortest paths using BFS.

    Paths are sorted by length (shortest first), then deterministically
    by node IDs. Falls back to BFS shortest path when fewer than k paths exist.
    """
    if src == dst:
        return [[src]]

    candidates: list[list[int]] = []
    queue: deque[tuple[int, list[int]]] = deque([(src, [src])])

    while queue and len(candidates) < k:
        current, path = queue.popleft()
        for neighbor, _ in topology.get_neighbors(current):
            if neighbor in path:
                continue
            new_path = path + [neighbor]
            if neighbor == dst:
                candidates.append(new_path)
            else:
                queue.append((neighbor, new_path))

    candidates.sort(key=lambda p: (len(p), p))
    return candidates[:k]


def _estimate_flow_duration_us(
    task,
    path: list[int],
    topology: NetworkTopology,
) -> int:
    """Estimate flow transmission duration at line rate over a given path."""
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


def compute_greedy_routes(
    workload: P2PWorkload,
    topology: NetworkTopology,
    flow_timing: dict[int, tuple[int, int]],
    k: int = 4,
) -> RouteTable:
    """Compute offline greedy route table.

    Sorts flows by optimistic start time, then for each flow selects
    the least-contended path among k shortest candidates.

    Args:
        workload: P2P workload with flow tasks
        topology: Network topology
        flow_timing: Precomputed optimistic timing: task_id -> (start_us, finish_us)
        k: Number of candidate paths to consider per flow

    Returns:
        RouteTable with selected paths for all flow tasks
    """
    route_table = RouteTable()

    # Track planned link activity: link -> list of (start_us, finish_us)
    link_intervals: dict[tuple[int, int], list[tuple[int, int]]] = {}

    # Collect flow tasks and sort by optimistic start time
    flow_entries: list[tuple] = []
    for t in workload.tasks:
        if not t.is_flow():
            continue
        if t.src is None or t.dst is None:
            continue
        timing = flow_timing.get(t.task_id, (0, 0))
        flow_entries.append((timing[0], t))

    flow_entries.sort(key=lambda x: x[0])  # sort by start time

    for _start_us, task in flow_entries:
        start_us, finish_us = flow_timing.get(task.task_id, (0, 0))

        candidates = k_shortest_paths(topology, task.src, task.dst, k)
        if not candidates:
            raise ValueError(
                f"No path from node {task.src} to {task.dst} for flow task {task.task_id}"
            )

        best_path = None
        best_score = None

        for path in candidates:
            max_active = 0
            total_active = 0
            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

            for link in links:
                active_count = 0
                for (l_start, l_finish) in link_intervals.get(link, []):
                    if start_us < l_finish and l_start < finish_us:
                        active_count += 1
                max_active = max(max_active, active_count)
                total_active += active_count

            score = (max_active, total_active, len(path), path)

            if best_score is None or score < best_score:
                best_score = score
                best_path = path

        route_table.paths[task.task_id] = best_path

        # Update planned link activity
        links = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
        for link in links:
            if link not in link_intervals:
                link_intervals[link] = []
            link_intervals[link].append((start_us, finish_us))

    return route_table

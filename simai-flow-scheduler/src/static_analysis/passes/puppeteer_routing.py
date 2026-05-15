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

    # Sliding-window active-flow tracking per link (amortized O(1) per check).
    # Since flows are sorted by start time, we only need to track finish times —
    # intervals with finish <= current start are automatically expired.
    link_active: dict[tuple[int, int], deque] = {}

    # (src, dst) -> path list cache — many flows share the same endpoints
    path_cache: dict[tuple[int, int], list[list[int]]] = {}

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

        # Cache k_shortest_paths per (src, dst) — most flows share endpoints
        key = (task.src, task.dst)
        if key not in path_cache:
            path_cache[key] = k_shortest_paths(topology, task.src, task.dst, k)
        candidates = path_cache[key]
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
                dq = link_active.get(link)
                active_count = 0
                if dq is not None:
                    # Expire finished intervals (finish <= start_us)
                    while dq and dq[0] <= start_us:
                        dq.popleft()
                    active_count = len(dq)
                max_active = max(max_active, active_count)
                total_active += active_count

            score = (max_active, total_active, len(path), path)

            if best_score is None or score < best_score:
                best_score = score
                best_path = path

        route_table.paths[task.task_id] = best_path

        # Record link activity for the selected path
        links = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
        for link in links:
            if link not in link_active:
                link_active[link] = deque()
            link_active[link].append(finish_us)

    return route_table

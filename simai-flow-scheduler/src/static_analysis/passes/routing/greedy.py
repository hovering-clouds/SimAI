"""Offline greedy routing strategy (Puppeteer-style)."""
from collections import deque
from dataclasses import dataclass, field

from ....workload_format.schema import P2PWorkload, Task
from ..topology_loader import NetworkTopology
from .base import RouteStrategy, RouteTable


@dataclass
class GreedyRouteTable(RouteTable):
    """Route table keyed by task_id — each flow may have a unique path."""

    paths: dict[int, list[int]] = field(default_factory=dict)

    def get_path(self, task: Task) -> list[int]:
        if task.task_id not in self.paths:
            raise KeyError(
                f"No route found for flow task {task.task_id}. "
                f"This indicates a missing route planning entry."
            )
        return list(self.paths[task.task_id])

    def update_routes(self, other: "RouteTable") -> None:
        if isinstance(other, GreedyRouteTable):
            self.paths.update(other.paths)


class GreedyStrategy(RouteStrategy):
    """Offline greedy routing: least-active-link heuristic among k shortest paths.

    Args:
        flow_timing: Precomputed optimistic timing: task_id -> (start_us, finish_us)
        k: Number of candidate paths to consider per flow
    """

    def __init__(
        self,
        flow_timing: dict[int, tuple[int, int]],
        k: int = 4,
    ):
        self.flow_timing = flow_timing
        self.k = k

    def compute_routes(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> GreedyRouteTable:
        route_table = GreedyRouteTable()

        link_active: dict[tuple[int, int], deque] = {}
        path_cache: dict[tuple[int, int], list[list[int]]] = {}

        flow_entries: list[tuple] = []
        for t in workload.tasks:
            if not t.is_flow():
                continue
            if t.src is None or t.dst is None:
                continue
            timing = self.flow_timing.get(t.task_id, (0, 0))
            flow_entries.append((timing[0], t))

        flow_entries.sort(key=lambda x: x[0])

        for _start_us, task in flow_entries:
            start_us, finish_us = self.flow_timing.get(task.task_id, (0, 0))

            key = (task.src, task.dst)
            if key not in path_cache:
                path_cache[key] = k_shortest_paths(topology, task.src, task.dst, self.k)
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

            links = [(best_path[i], best_path[i + 1]) for i in range(len(best_path) - 1)]
            for link in links:
                if link not in link_active:
                    link_active[link] = deque()
                link_active[link].append(finish_us)

        return route_table


def _bfs_shortest_path(
    topology: NetworkTopology,
    src: int,
    dst: int,
    blocked_nodes: set[int] | None = None,
    blocked_edges: set[tuple[int, int]] | None = None,
) -> list[int] | None:
    """BFS shortest path avoiding blocked nodes/edges.

    Uses a visited set (efficient). Returns None if no path exists.
    """
    if src == dst:
        return [src]

    blocked_nodes = blocked_nodes or set()
    blocked_edges = blocked_edges or set()

    visited: set[int] = blocked_nodes.copy()
    queue: deque[tuple[int, list[int]]] = deque([(src, [src])])

    while queue:
        current, path = queue.popleft()
        for neighbor, _ in topology.get_neighbors(current):
            if neighbor in visited:
                continue
            if (current, neighbor) in blocked_edges:
                continue
            new_path = path + [neighbor]
            if neighbor == dst:
                return new_path
            visited.add(neighbor)
            queue.append((neighbor, new_path))

    return None


def k_shortest_paths(
    topology: NetworkTopology,
    src: int,
    dst: int,
    k: int = 4,
) -> list[list[int]]:
    """Find up to k shortest simple paths using Yen's algorithm.

    Yen's algorithm iteratively finds k shortest loopless paths by
    deviating from previously found paths at each node. Unlike the old
    naive BFS (which enumerated ALL simple paths and never terminated
    when fewer than k paths existed), this algorithm:
      - Uses a visited set for efficient shortest-path queries
      - Terminates gracefully when fewer than k paths exist
      - Runs in O(k * V * (V+E)) time

    Args:
        topology: Network topology
        src: Source node
        dst: Destination node
        k: Maximum number of paths to find

    Returns:
        List of up to k paths, sorted by length (shortest first)
    """
    if src == dst:
        return [[src]]

    import heapq

    # --- Step 1: first shortest path ---
    first = _bfs_shortest_path(topology, src, dst)
    if first is None:
        return []

    A: list[list[int]] = [first]  # finalised shortest paths
    B: dict[tuple[int, ...], list[int]] = {}  # candidates: path_tuple -> path

    for _ in range(1, k):
        prev = A[-1]

        for spur_idx in range(len(prev) - 1):
            root_path = prev[: spur_idx + 1]
            spur_node = prev[spur_idx]

            # Block nodes in the root path (except spur_node itself)
            # to prevent revisiting them in the spur path.
            blocked_nodes = set(root_path[:-1])

            # Block edges from spur_node that would recreate a path
            # already in A that shares this root prefix.
            blocked_edges: set[tuple[int, int]] = set()
            for ap in A:
                if len(ap) > spur_idx and ap[: spur_idx + 1] == root_path:
                    blocked_edges.add((spur_node, ap[spur_idx + 1]))

            spur = _bfs_shortest_path(
                topology, spur_node, dst,
                blocked_nodes=blocked_nodes,
                blocked_edges=blocked_edges,
            )
            if spur is not None:
                # Combine root_path and spur_path (avoiding duplicate spur_node)
                total = root_path[:-1] + spur
                key = tuple(total)
                # Only add if not already in A or B
                if key not in B and not any(tuple(ap) == key for ap in A):
                    B[key] = total

        if not B:
            break

        # Pick the shortest path from candidates
        best_key = min(B.keys(), key=lambda t: (len(t), t))
        A.append(B.pop(best_key))

    return A

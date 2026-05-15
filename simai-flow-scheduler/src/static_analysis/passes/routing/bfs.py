"""BFS shortest-path routing strategy."""
from collections import deque

from ....workload_format.schema import P2PWorkload, Task
from ..topology_loader import NetworkTopology
from .base import RouteStrategy, RouteTable


class BfsRouteTable(RouteTable):
    """Route table keyed by (src, dst) — deduplicates paths for same endpoint pair.

    Memory: O(P * L) where P = unique (src, dst) pairs, L = avg path length.
    """

    def __init__(self, topology: NetworkTopology):
        self._paths: dict[tuple[int, int], list[int]] = {}
        self.topology = topology

    def get_path(self, task: Task) -> list[int]:
        return list(self._paths[(task.src, task.dst)])

    def get_path_by_endpoints(self, src: int, dst: int) -> list[int]:
        """Direct lookup by endpoint pair (for callers that don't have a Task)."""
        return list(self._paths[(src, dst)])

    def ensure_path(self, src: int, dst: int, path: list[int]) -> None:
        """Register a path if not already cached."""
        if (src, dst) not in self._paths:
            self._paths[(src, dst)] = path


class BfsStrategy(RouteStrategy):
    """Computes BFS shortest paths for all flow tasks."""

    def compute_routes(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> BfsRouteTable:
        table = BfsRouteTable(topology)

        for task in workload.tasks:
            if not task.is_flow():
                continue
            if task.src is None or task.dst is None:
                continue
            table.ensure_path(
                task.src, task.dst,
                bfs_shortest_path(topology, task.src, task.dst),
            )

        return table


def bfs_shortest_path(
    topology: NetworkTopology,
    src: int,
    dst: int,
) -> list[int]:
    """Find shortest path by hop count using BFS.

    Raises ValueError if no path exists.
    """
    if src == dst:
        return [src]

    visited: set[int] = {src}
    queue: deque[tuple[int, list[int]]] = deque([(src, [src])])

    while queue:
        current, path = queue.popleft()
        for neighbor, _link in topology.get_neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == dst:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    raise ValueError(f"No path found from node {src} to node {dst}")

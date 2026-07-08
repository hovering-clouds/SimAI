"""ECMP routing strategy — hash-based path selection among equal-cost shortest paths.

Replaces BfsStrategy's deterministic first-neighbor selection with hash-shuffled
neighbor traversal.  Each (src, dst) pair gets a reproducible but different
path, spreading traffic across parallel L2/L3 switches without the complexity
of full k-shortest-path enumeration.
"""

from collections import deque

from ....workload_format.schema import P2PWorkload, Task
from ..topology_loader import NetworkTopology
from .base import RouteStrategy, RouteTable


class EcmpRouteTable(RouteTable):
    """Route table keyed by (src, dst) — same structure as BfsRouteTable.

    Only differ from BfsRouteTable in how paths are *chosen* by the strategy;
    the storage and lookup interface is identical.
    """

    def __init__(self, topology: NetworkTopology):
        self._paths: dict[tuple[int, int], list[int]] = {}
        self.topology = topology

    def get_path(self, task: Task) -> list[int]:
        return list(self._paths[(task.src, task.dst)])

    def get_path_by_endpoints(self, src: int, dst: int) -> list[int]:
        return list(self._paths[(src, dst)])

    def ensure_path(self, src: int, dst: int, path: list[int]) -> None:
        if (src, dst) not in self._paths:
            self._paths[(src, dst)] = path

    def update_routes(self, other: "RouteTable") -> None:
        if isinstance(other, EcmpRouteTable):
            for key, path in other._paths.items():
                self.ensure_path(*key, path)


def ecmp_shortest_path(
    topology: NetworkTopology,
    src: int,
    dst: int,
    seed: int = 0,
) -> list[int]:
    """Hash-shuffled BFS — deterministic ECMP shortest path per (src, dst).

    All equal-cost shortest paths are equally discoverable, but the hash-based
    neighbor ordering makes different (src, dst) pairs walk different branches
    of the BFS tree, spreading traffic across parallel switches.
    """
    if src == dst:
        return [src]

    visited: set[int] = {src}
    queue: deque[tuple[int, list[int]]] = deque([(src, [src])])

    while queue:
        current, path = queue.popleft()
        neighbors = [(n, link) for n, link in topology.get_neighbors(current)]
        # Deterministic per-(src,dst,node,neighbor) ordering
        neighbors.sort(key=lambda x: hash((src, dst, current, x[0], seed)))

        for neighbor, _link in neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == dst:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    raise ValueError(f"No path found from node {src} to node {dst}")


class EcmpStrategy(RouteStrategy):
    """ECMP routing — hash-shuffled BFS for every flow task.

    Args:
        seed: hash seed for reproducibility (default 0).
    """

    def __init__(self, seed: int = 0):
        self.seed = seed

    def compute_routes(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> EcmpRouteTable:
        table = EcmpRouteTable(topology)

        for task in workload.tasks:
            if not task.is_flow():
                continue
            if task.src is None or task.dst is None:
                continue
            table.ensure_path(
                task.src, task.dst,
                ecmp_shortest_path(topology, task.src, task.dst, self.seed),
            )

        return table

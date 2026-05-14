"""
Routing hints - precomputed shortest paths and link load statistics.

Computes BFS shortest paths for all flow (src, dst) pairs in the workload,
caches them for reuse by critical path analysis (Task 2) and contention
analysis (Task 3). Also aggregates per-link flow counts for hotspot detection.

Supports custom routing strategies for flexibility (e.g., bandwidth-aware,
latency-aware, ECMP). Default strategy is BFS by hop count.

Memory:  O(P * L) where P = unique (src, dst) pairs, L = avg path length
Compute: O(F * (V+E)) where F = num flow tasks, V = topology nodes, E = links
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from ...workload_format.schema import P2PWorkload, Task, TaskType
from .topology_loader import NetworkTopology

# Type alias for routing strategy functions
RoutingStrategy = Callable[[NetworkTopology, int, int], list[int] | None]


@dataclass
class RoutingHints:
    topology: NetworkTopology = field(repr=False)
    routing_strategy: RoutingStrategy = field(default=None, repr=False)
    _cached_paths: dict[tuple[int, int], list[int]] = field(
        default_factory=dict, repr=False
    )
    link_loads: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        if self.routing_strategy is None:
            self.routing_strategy = bfs_shortest_path

    def get_path(self, src: int, dst: int) -> list[int]:
        key = (src, dst)
        if key not in self._cached_paths:
            path = self.routing_strategy(self.topology, src, dst)
            if path is None:
                raise ValueError(
                    f"No path found from node {src} to node {dst}. "
                    f"This indicates a topology connectivity issue."
                )
            self._cached_paths[key] = path
        return self._cached_paths[key]

    def get_flow_links(self, task: Task) -> list[tuple[int, int]]:
        if task.src is None or task.dst is None:
            return []
        path = self.get_path(task.src, task.dst)
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def get_most_used_links(self, top_k: int = 20) -> list[tuple[tuple[int, int], int]]:
        return sorted(
            self.link_loads.items(), key=lambda x: x[1], reverse=True
        )[:top_k]


def compute_routing_hints(
    topology: NetworkTopology,
    workload: P2PWorkload,
    routing_strategy: RoutingStrategy | None = None,
) -> RoutingHints:
    if routing_strategy is None:
        routing_strategy = bfs_shortest_path

    hints = RoutingHints(topology=topology, routing_strategy=routing_strategy)

    for task in workload.tasks:
        if not task.is_flow():
            continue
        if task.src is None or task.dst is None:
            continue

        path = hints.get_path(task.src, task.dst)

        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        for link in links:
            hints.link_loads[link] = hints.link_loads.get(link, 0) + 1

    return hints


def bfs_shortest_path(
    topology: NetworkTopology,
    src: int,
    dst: int,
) -> list[int] | None:
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

    return None

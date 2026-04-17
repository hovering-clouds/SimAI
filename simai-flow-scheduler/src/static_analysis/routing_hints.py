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

from ..workload_format.schema import P2PWorkload, Task, TaskType
from .topology_loader import NetworkTopology

# Type alias for routing strategy functions
RoutingStrategy = Callable[[NetworkTopology, int, int], list[int] | None]
"""
Routing strategy function signature.

Args:
    topology: Network topology graph
    src: Source node ID
    dst: Destination node ID

Returns:
    Path as list of node IDs [src, hop1, ..., dst], or None if no path exists
"""


@dataclass
class RoutingHints:
    """
    On-demand routing hints with path caching.

    Design:
    - _cached_paths stores node-level paths [src, hop1, hop2, ..., dst]
    - When needed, convert path → links on demand
    - link_loads aggregates statistics for quick contention analysis
    - routing_strategy allows custom routing algorithms (default: BFS)

    Computed eagerly in Phase 3 Task 1 so that Task 2 (critical path)
    can use accurate multi-hop duration estimates.
    """

    # Routing strategy function (default: BFS shortest path)
    routing_strategy: RoutingStrategy = field(default=None, repr=False)

    # Shortest paths between actual (src, dst) pairs in the workload
    # Computed eagerly during initialization
    _cached_paths: dict[tuple[int, int], list[int]] = field(
        default_factory=dict, repr=False
    )

    # Link load statistics (aggregated across all flows during computation)
    # link_id → number of flows using this link
    link_loads: dict[tuple[int, int], int] = field(default_factory=dict)

    def __post_init__(self):
        """Set default routing strategy if not provided."""
        if self.routing_strategy is None:
            self.routing_strategy = bfs_shortest_path

    def get_path(self, topology: NetworkTopology, src: int, dst: int) -> list[int]:
        """
        Path lookup: compute via routing strategy on first access, cache for reuse.

        Raises:
            ValueError: If no path exists between src and dst (indicates topology issue)
        """
        key = (src, dst)
        if key not in self._cached_paths:
            path = self.routing_strategy(topology, src, dst)
            if path is None:
                raise ValueError(
                    f"No path found from node {src} to node {dst}. "
                    f"This indicates a topology connectivity issue."
                )
            self._cached_paths[key] = path
        return self._cached_paths[key]

    def get_flow_links(
        self, task: Task, topology: NetworkTopology
    ) -> list[tuple[int, int]]:
        """
        Convert cached path to physical links for a flow task.

        For multi-hop paths, converts [src, hop1, hop2, dst] →
        [(src,hop1), (hop1,hop2), (hop2,dst)]

        Raises:
            ValueError: If path cannot be found (propagated from get_path)
        """
        if task.src is None or task.dst is None:
            return []

        path = self.get_path(topology, task.src, task.dst)
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def get_most_used_links(self, top_k: int = 20) -> list[tuple[tuple[int, int], int]]:
        """Compute top-K most used links on demand from link_loads."""
        return sorted(
            self.link_loads.items(), key=lambda x: x[1], reverse=True
        )[:top_k]


def compute_routing_hints(
    topology: NetworkTopology,
    workload: P2PWorkload,
    routing_strategy: RoutingStrategy | None = None,
) -> RoutingHints:
    """
    Compute routing hints by finding shortest paths for all flows.

    Args:
        topology: Network topology graph
        workload: P2P workload with flow tasks
        routing_strategy: Custom routing function. If None, uses BFS shortest path.

    Steps:
    1. For each flow task, find shortest path via routing strategy
    2. Cache the path for later use by critical path analysis
    3. Aggregate link_loads for quick hotspot detection

    Memory: O(P * L) where P = unique (src,dst) pairs, L = avg path length
    Compute: O(F * (V+E)) - routing strategy for each flow task
    """
    if routing_strategy is None:
        routing_strategy = bfs_shortest_path

    hints = RoutingHints(routing_strategy=routing_strategy)

    # Process each flow task
    for task in workload.tasks:
        if not task.is_flow():
            continue
        if task.src is None or task.dst is None:
            continue

        # This triggers BFS and caches the path
        path = hints.get_path(topology, task.src, task.dst)

        # Aggregate link loads from this path
        links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        for link in links:
            hints.link_loads[link] = hints.link_loads.get(link, 0) + 1

    return hints


def bfs_shortest_path(
    topology: NetworkTopology,
    src: int,
    dst: int,
) -> list[int] | None:
    """
    BFS shortest path by hop count (default routing strategy).

    Args:
        topology: Network topology graph
        src: Source node ID
        dst: Destination node ID

    Returns:
        Path as list of node IDs, or None if no path exists
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

    return None

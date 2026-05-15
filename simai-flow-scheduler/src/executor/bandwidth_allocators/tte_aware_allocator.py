"""TTE-aware bandwidth allocation for Puppeteer.

Provides weighted fair sharing and strict-priority allocation modes
based on flow TTE (Time-to-Exposed) priority classification.
"""
from .base_allocator import BandwidthAllocator
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...static_analysis.passes.puppeteer_tte import TTEInfo


class TteAwareAllocator(BandwidthAllocator):
    """TTE-aware bandwidth allocator.

    Two modes:
    - weighted: weight = 1 / max(tte_us, epsilon), proportional fair share
    - strict_priority: critical > elastic > background, each tier gets remaining capacity

    Args:
        tte_info: Map of flow task_id -> TTEInfo
        mode: Allocation mode ("weighted" or "strict_priority")
        epsilon: Small constant to avoid division by zero in weight calculation
    """

    def __init__(
        self,
        tte_info: dict[int, TTEInfo],
        mode: str = "weighted",
        epsilon: float = 1.0,
    ):
        self.tte_info = tte_info
        self.mode = mode
        self.epsilon = epsilon

    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        current_time: int = 0,
    ) -> dict[int, float]:
        """Allocate bandwidth to active flows based on TTE priority.

        Args:
            active_flows: List of ActiveFlow objects
            topology: Network topology
            current_time: Current simulation time

        Returns:
            dict mapping task_id -> allocated_bw_gbps
        """
        if not active_flows:
            return {}

        if self.mode == "strict_priority":
            return self._allocate_strict_priority(active_flows, topology)
        return self._allocate_weighted(active_flows, topology)

    def _get_weight(self, tid: int) -> float:
        """Compute weight for a flow task based on TTE."""
        info = self.tte_info.get(tid)
        if info is None:
            return 1.0 / self.epsilon
        if info.tte_us == float("inf"):
            return 1.0 / self.epsilon
        return 1.0 / max(info.tte_us, self.epsilon)

    def _allocate_weighted(
        self,
        active_flows: list,
        topology: NetworkTopology,
    ) -> dict[int, float]:
        """Weighted fair sharing: bandwidth proportional to 1/TTE."""
        # Build per-link total weight (only need the sum, not per-flow lists)
        link_total_weight: dict[tuple[int, int], float] = {}

        for flow in active_flows:
            weight = self._get_weight(flow.task_id)
            path = flow.path
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_total_weight[link] = link_total_weight.get(link, 0.0) + weight

        # Compute per-flow allocation as bottleneck across path links
        flow_alloc: dict[int, float] = {}
        for flow in active_flows:
            tid = flow.task_id
            my_w = self._get_weight(tid)
            min_alloc = float("inf")
            path = flow.path
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_obj = topology.get_link(link[0], link[1])
                if link_obj is None:
                    continue
                total_w = link_total_weight.get(link, 0.0)
                if total_w > 0:
                    alloc = link_obj.bandwidth_gbps * my_w / total_w
                    min_alloc = min(min_alloc, alloc)
            flow_alloc[tid] = min_alloc if min_alloc != float("inf") else 0.0

        return flow_alloc

    def _allocate_strict_priority(
        self,
        active_flows: list,
        topology: NetworkTopology,
    ) -> dict[int, float]:
        """Strict priority: critical flows first, then elastic, then background."""
        # Group by priority
        groups: dict[str, list] = {"critical": [], "elastic": [], "background": []}
        for flow in active_flows:
            tid = flow.task_id
            info = self.tte_info.get(tid, TTEInfo(tid, float("inf"), 0, "background"))
            groups[info.priority_class].append(flow)

        # Pre-build per-tier link→flow mapping to avoid O(A) re-counting per link
        tier_link_flows: dict[str, dict[tuple[int, int], list]] = {}
        for priority, group in groups.items():
            link_map: dict[tuple[int, int], list] = {}
            for flow in group:
                path = flow.path
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    link_map.setdefault(link, []).append(flow)
            tier_link_flows[priority] = link_map

        # Track remaining capacity per link
        link_rem: dict[tuple[int, int], float] = {}
        for flow in active_flows:
            path = flow.path
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                if link not in link_rem:
                    link_obj = topology.get_link(link[0], link[1])
                    link_rem[link] = link_obj.bandwidth_gbps if link_obj else 0.0

        result: dict[int, float] = {}

        for priority in ("critical", "elastic", "background"):
            group = groups[priority]
            if not group:
                continue

            link_map = tier_link_flows[priority]

            # Fair share within this priority tier using remaining capacity
            for flow in group:
                tid = flow.task_id
                min_alloc = float("inf")
                path = flow.path
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    if link in link_rem and link_rem[link] > 0:
                        count = len(link_map.get(link, []))
                        if count > 0:
                            alloc = link_rem[link] / count
                            min_alloc = min(min_alloc, alloc)

                if min_alloc != float("inf"):
                    result[tid] = max(0.0, min_alloc)
                    # Consume capacity from all links on path
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        if link in link_rem:
                            link_rem[link] = max(0.0, link_rem[link] - result[tid])
                else:
                    result[tid] = 0.0

        # Ensure all active flows have an entry
        for flow in active_flows:
            if flow.task_id not in result:
                result[flow.task_id] = 0.0

        return result

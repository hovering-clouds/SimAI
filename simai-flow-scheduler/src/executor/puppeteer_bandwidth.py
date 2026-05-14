"""TTE-aware bandwidth allocation for Puppeteer.

Provides weighted fair sharing and strict-priority allocation modes
based on flow TTE (Time-to-Exposed) priority classification.
"""
from ..static_analysis.passes.routing_hints import RoutingHints
from ..static_analysis.passes.topology_loader import NetworkTopology
from ..static_analysis.passes.puppeteer_tte import TTEInfo


class TteAwareAllocator:
    """TTE-aware bandwidth allocator.

    Two modes:
    - weighted: weight = 1 / max(tte_us, epsilon), proportional fair share
    - strict_priority: critical > elastic > background, each tier gets remaining capacity

    Args:
        tte_info: Map of flow task_id -> TTEInfo
        mode: Allocation mode ("weighted" or "strict_priority")
        min_background_share: Minimum fraction of link capacity for background flows
        epsilon: Small constant to avoid division by zero in weight calculation
    """

    def __init__(
        self,
        tte_info: dict[int, TTEInfo],
        mode: str = "weighted",
        min_background_share: float = 0.05,
        epsilon: float = 1.0,
    ):
        self.tte_info = tte_info
        self.mode = mode
        self.min_background_share = min_background_share
        self.epsilon = epsilon

    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        routing_hints: RoutingHints | None = None,
        current_time: int = 0,
    ) -> dict[int, float]:
        """Allocate bandwidth to active flows based on TTE priority.

        Args:
            active_flows: List of ActiveFlow objects
            topology: Network topology
            routing_hints: Not used by this allocator (included for interface compatibility)
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
        # Build per-link flow lists with weights
        link_flows: dict[tuple[int, int], list[tuple[int, float]]] = {}

        for flow in active_flows:
            tid = flow.task_id
            weight = self._get_weight(tid)
            path = flow.path
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_flows.setdefault(link, []).append((tid, weight))

        # Compute per-flow allocation as bottleneck across path links
        flow_alloc: dict[int, float] = {}
        for flow in active_flows:
            tid = flow.task_id
            min_alloc = float("inf")
            path = flow.path
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                link_obj = topology.get_link(link[0], link[1])
                if link_obj is None:
                    continue
                cap = link_obj.bandwidth_gbps
                flows_on_link = link_flows.get(link, [])
                total_w = sum(w for _, w in flows_on_link)
                if total_w > 0:
                    my_w = next((w for fid, w in flows_on_link if fid == tid), 0.0)
                    alloc = cap * my_w / total_w
                    min_alloc = min(min_alloc, alloc)
            flow_alloc[tid] = min_alloc if min_alloc != float("inf") else 0.0

        # Apply minimum background share
        if self.min_background_share > 0:
            for flow in active_flows:
                tid = flow.task_id
                info = self.tte_info.get(tid)
                if info and info.priority_class == "background":
                    current = flow_alloc.get(tid, 0.0)
                    # Find minimum bottleneck capacity along path for min share
                    path = flow.path
                    for i in range(len(path) - 1):
                        link = (path[i], path[i + 1])
                        link_obj = topology.get_link(link[0], link[1])
                        if link_obj is not None:
                            min_share = link_obj.bandwidth_gbps * self.min_background_share
                            current = max(current, min_share)
                    flow_alloc[tid] = current

        return {flow.task_id: flow_alloc.get(flow.task_id, 0.0) for flow in active_flows}

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

            # Fair share within this priority tier using remaining capacity
            for flow in group:
                tid = flow.task_id
                min_alloc = float("inf")
                path = flow.path
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    if link in link_rem and link_rem[link] > 0:
                        count = sum(
                            1 for f in group
                            if any(
                                (f.path[j], f.path[j + 1]) == link
                                for j in range(len(f.path) - 1)
                            )
                        )
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

"""Strict inter-coflow priority allocator for Hermod §4.1."""
from collections.abc import Iterable

from .base_allocator import BandwidthAllocator, MIN_BW_REMAINING
from ...static_analysis.passes.hermod_priority import HermodPriorityAnalysis


class HermodAllocator(BandwidthAllocator):
    """Serve coflow tiers strictly, with max-min fairness inside each tier.

    This deliberately excludes Hermod §4.2's matching-based intra-coflow
    allocation. Flows in one §4.1 priority tier receive equal progressive
    filling regardless of their coflow identity.
    """

    def __init__(self, priority_analysis: HermodPriorityAnalysis):
        self.priority_analysis = priority_analysis

    @staticmethod
    def _allocate_tier(
        flows: Iterable,
        link_rem: dict[tuple[int, int], float],
        result: dict[int, float],
    ) -> None:
        """Allocate one tier with unweighted max-min progressive filling.

        A one-shot per-link equal split strands capacity if a flow is bottlenecked
        elsewhere. Here all unfrozen flows rise together; only flows traversing
        a saturated link freeze, so the remaining capacity is redistributed.
        """
        flow_list = list(flows)
        allocations = {flow.task_id: 0.0 for flow in flow_list}
        pending = {
            flow.task_id: flow for flow in flow_list
            if len(flow.path) >= 2
        }

        while pending:
            counts: dict[tuple[int, int], int] = {}
            for flow in pending.values():
                for index in range(len(flow.path) - 1):
                    link = (flow.path[index], flow.path[index + 1])
                    counts[link] = counts.get(link, 0) + 1

            shares = {link: link_rem[link] / count for link, count in counts.items()}
            increment = min(shares.values(), default=0.0)
            bottlenecks = {
                link for link, share in shares.items()
                if share <= increment + MIN_BW_REMAINING
            }

            if increment > MIN_BW_REMAINING:
                for task_id in pending:
                    allocations[task_id] += increment
                for link, count in counts.items():
                    link_rem[link] = max(0.0, link_rem[link] - increment * count)

            frozen = {
                task_id for task_id, flow in pending.items()
                if any(
                    (flow.path[index], flow.path[index + 1]) in bottlenecks
                    for index in range(len(flow.path) - 1)
                )
            }
            if not frozen:
                # Defensive progress guarantee for malformed or zero-capacity paths.
                frozen = set(pending)
            for task_id in frozen:
                pending.pop(task_id)

        for flow in flow_list:
            bandwidth = allocations[flow.task_id]
            result[flow.task_id] = (
                0.0 if bandwidth < MIN_BW_REMAINING else bandwidth
            )

    def allocate(self, active_flows, topology, current_time=0):
        if not active_flows:
            return {}
        link_rem = {}
        for flow in active_flows:
            for index in range(len(flow.path) - 1):
                link = (flow.path[index], flow.path[index + 1])
                if link not in link_rem:
                    obj = topology.get_link(*link)
                    link_rem[link] = obj.bandwidth_gbps if obj else 0.0
        tiers = self.priority_analysis.priority_tiers([flow.task_id for flow in active_flows])
        by_coflow = {}
        background = []
        for flow in active_flows:
            coflow_id = self.priority_analysis.task_to_coflow.get(flow.task_id)
            (background if coflow_id is None else by_coflow.setdefault(coflow_id, [])).append(flow)
        result = {}
        for coflow_ids in tiers:
            tier_flows = [flow for coflow_id in coflow_ids for flow in by_coflow[coflow_id]]
            self._allocate_tier(tier_flows, link_rem, result)
        # Out-of-scope traffic is fair-share background, after Hermod traffic.
        if background:
            self._allocate_tier(background, link_rem, result)
        return {flow.task_id: result.get(flow.task_id, 0.0) for flow in active_flows}

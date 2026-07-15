"""Strict inter-coflow priority allocator for Hermod §4.1."""
from .base_allocator import BandwidthAllocator, MIN_BW_REMAINING
from ...static_analysis.passes.hermod_priority import HermodPriorityAnalysis


class HermodAllocator(BandwidthAllocator):
    """Serve coflow tiers strictly, with equal sharing inside each tier.

    This is deliberately not Hermod §4.2: flows within a coflow are treated
    just like other flows in the same priority tier.
    """
    def __init__(self, priority_analysis: HermodPriorityAnalysis):
        self.priority_analysis = priority_analysis

    @staticmethod
    def _allocate_tier(flows, link_rem, result):
        counts = {}
        for flow in flows:
            for i in range(len(flow.path) - 1):
                link = (flow.path[i], flow.path[i + 1])
                counts[link] = counts.get(link, 0) + 1
        allocations = []
        for flow in flows:
            shares = [link_rem[(flow.path[i], flow.path[i + 1])] /
                      counts[(flow.path[i], flow.path[i + 1])]
                      for i in range(len(flow.path) - 1)]
            bw = min(shares, default=0.0)
            allocations.append((flow, 0.0 if bw < MIN_BW_REMAINING else bw))
        for flow, bw in allocations:
            result[flow.task_id] = bw
            for i in range(len(flow.path) - 1):
                link = (flow.path[i], flow.path[i + 1])
                link_rem[link] = max(0.0, link_rem[link] - bw)

    def allocate(self, active_flows, topology, current_time=0):
        if not active_flows:
            return {}
        link_rem = {}
        for flow in active_flows:
            for i in range(len(flow.path) - 1):
                link = (flow.path[i], flow.path[i + 1])
                if link not in link_rem:
                    obj = topology.get_link(*link)
                    link_rem[link] = obj.bandwidth_gbps if obj else 0.0
        tiers = self.priority_analysis.priority_order([f.task_id for f in active_flows])
        by_coflow = {}
        background = []
        for flow in active_flows:
            cid = self.priority_analysis.task_to_coflow.get(flow.task_id)
            (background if cid is None else by_coflow.setdefault(cid, [])).append(flow)
        result = {}
        for cid in tiers:
            self._allocate_tier(by_coflow[cid], link_rem, result)
        # Out-of-scope traffic is fair-share background, after Hermod traffic.
        if background:
            self._allocate_tier(background, link_rem, result)
        return {flow.task_id: result.get(flow.task_id, 0.0) for flow in active_flows}

"""Bandwidth allocation strategies."""
from abc import ABC, abstractmethod

from ..static_analysis.passes.routing_hints import RoutingHints
from ..static_analysis.passes.topology_loader import NetworkTopology


class BandwidthAllocator(ABC):
    """带宽分配策略接口。"""

    @abstractmethod
    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        current_time: int,
    ) -> dict[int, float]:
        """
        根据当前 active flow 集合和拓扑信息，计算每条流的带宽分配。

        Args:
            active_flows: 当前正在传输的流列表 (list[ActiveFlow])
            topology: 网络拓扑
            routing_hints: 路由提示（用于路径查询）
            current_time: 当前全局时间

        Returns:
            task_id → allocated_bw_gbps 的映射
        """
        pass


class FairShareAllocator(BandwidthAllocator):
    """逐链路均分策略：每条链路上 n 条流各得 bw/n，取路径瓶颈。"""

    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        current_time: int,
    ) -> dict[int, float]:
        if not active_flows:
            return {}

        # Step 1: 统计每条链路上有多少 active flow 经过
        link_flow_count: dict[tuple[int, int], int] = {}
        flow_links: dict[int, list[tuple[int, int]]] = {}

        for flow in active_flows:
            path = flow.path
            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            flow_links[flow.task_id] = links
            for link in links:
                link_flow_count[link] = link_flow_count.get(link, 0) + 1

        # Step 2: 对每条 flow，取路径上所有链路 fair share 的最小值
        result: dict[int, float] = {}
        for flow in active_flows:
            min_bw = float('inf')
            for link in flow_links[flow.task_id]:
                link_obj = topology.get_link(link[0], link[1])
                if link_obj is None:
                    continue
                fair_share = link_obj.bandwidth_gbps / link_flow_count[link]
                min_bw = min(min_bw, fair_share)
            result[flow.task_id] = min_bw if min_bw != float('inf') else 0.0

        return result

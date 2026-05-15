"""Bandwidth allocation interface."""
from abc import ABC, abstractmethod

from ...static_analysis.passes.topology_loader import NetworkTopology


class BandwidthAllocator(ABC):
    """带宽分配策略接口。"""

    @abstractmethod
    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        current_time: int,
    ) -> dict[int, float]:
        """
        根据当前 active flow 集合和拓扑信息，计算每条流的带宽分配。

        Args:
            active_flows: 当前正在传输的流列表 (list[ActiveFlow])
            topology: 网络拓扑
            current_time: 当前全局时间

        Returns:
            task_id → allocated_bw_gbps 的映射
        """
        pass

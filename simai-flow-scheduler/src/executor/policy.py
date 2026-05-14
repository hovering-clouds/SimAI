"""Scheduling policy interface and default implementation."""
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from .bandwidth import BandwidthAllocator, FairShareAllocator
from .runtime import ActiveFlow
from ..static_analysis.passes.routing_hints import RoutingHints
from ..static_analysis.passes.topology_loader import NetworkTopology
from ..static_analysis.task_serializer import CppReferenceOrdering
from ..workload_format.schema import P2PWorkload, Task


class SchedulingPolicy(ABC):
    """调度策略接口。

    Executor 拥有事件队列和 DAG 进度管理；Policy 决定准入、路径和带宽分配。
    """

    @abstractmethod
    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> None:
        """初始化策略，存储 workload / topology 引用供后续决策使用。"""
        ...

    @abstractmethod
    def emit_ready_tasks(
        self,
        current_time: int,
        ready_tasks: list[Task],
    ) -> list[int]:
        """准入控制：从当前可调度 task 中选择允许立即启动的 task ID 列表。"""
        ...

    @abstractmethod
    def get_flow_path(self, task: Task) -> list[int]:
        """返回 flow task 的传输路径 [src, hop1, ..., dst]。"""
        ...

    @abstractmethod
    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        """带宽分配：返回 task_id → allocated_bw_gbps 的映射。"""
        ...

    @abstractmethod
    def on_task_emitted(self, current_time: int, task: Task) -> None:
        """task 通过准入时的通知回调。"""
        ...

    @abstractmethod
    def on_task_completed(self, current_time: int, task: Task) -> None:
        """task 完成时的通知回调。"""
        ...


class DefaultSchedulingPolicy(SchedulingPolicy):
    """保持当前行为的默认策略：全准入、最短路径路由、均分带宽。

    内部维护 compute ordering（C++ 参考顺序），通过 emit_ready_tasks
    确保每个节点上的 compute 任务串行执行。
    """

    def __init__(
        self,
        routing_hints: RoutingHints,
        allocator: Optional[BandwidthAllocator] = None,
    ):
        self.routing_hints = routing_hints
        self.allocator = allocator or FairShareAllocator()
        self._topology: Optional[NetworkTopology] = None
        self.compute_order: dict[int, list[int]] = {}
        self.compute_position: dict[int, int] = {}
        self.compute_cursor: dict[int, int] = defaultdict(int)

    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> None:
        self._topology = topology
        ordering = CppReferenceOrdering()
        self.compute_order = ordering.order(workload)
        self.compute_position = {
            task_id: idx
            for node, ids in self.compute_order.items()
            for idx, task_id in enumerate(ids)
        }
        self.compute_cursor = defaultdict(int)

    def emit_ready_tasks(
        self,
        current_time: int,
        ready_tasks: list[Task],
    ) -> list[int]:
        emitted = []
        for task in sorted(ready_tasks, key=lambda t: t.task_id):
            if task.is_flow():
                emitted.append(task.task_id)
            elif task.is_compute() and self._is_next_compute(task):
                emitted.append(task.task_id)
        return emitted

    def _is_next_compute(self, task: Task) -> bool:
        node_id = task.node
        cursor = self.compute_cursor.get(node_id, 0)
        order = self.compute_order.get(node_id, [])
        return cursor < len(order) and order[cursor] == task.task_id

    def get_flow_path(self, task: Task) -> list[int]:
        return self.routing_hints.get_path(task.src, task.dst)

    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        return self.allocator.allocate(
            active_flows, self._topology, self.routing_hints, current_time,
        )

    def on_task_emitted(self, current_time: int, task: Task) -> None:
        pass

    def on_task_completed(self, current_time: int, task: Task) -> None:
        if task.is_compute():
            self.compute_cursor[task.node] += 1

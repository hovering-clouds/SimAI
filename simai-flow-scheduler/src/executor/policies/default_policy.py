"""Default scheduling policy — full admission, shortest-path routing, fair-share bandwidth."""
from collections import defaultdict
from typing import Optional

from .base_policy import SchedulingPolicy
from ..bandwidth_allocators.fair_share_allocator import FairShareAllocator
from ..runtime import ActiveFlow
from ...static_analysis.strategies.default_strategy import DefaultAnalysisResult
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...workload_format.schema import P2PWorkload, Task


class DefaultSchedulingPolicy(SchedulingPolicy):
    """保持当前行为的默认策略：全准入、最短路径路由、均分带宽。

    内部维护 compute ordering（C++ 参考顺序），通过 emit_ready_tasks
    确保每个节点上的 compute 任务串行执行。
    """

    def __init__(
        self,
        analysis: DefaultAnalysisResult,
    ):
        self.route_table = analysis.route_table
        self.compute_order = analysis.execution_plan.compute_order
        self.allocator = FairShareAllocator()
        self._topology: Optional[NetworkTopology] = None
        self.compute_position: dict[int, int] = {
            task_id: idx
            for node, ids in self.compute_order.items()
            for idx, task_id in enumerate(ids)
        }
        self.compute_cursor: dict[int, int] = defaultdict(int)

    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> None:
        self._topology = topology
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
        return self.route_table.get_path(task)

    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        return self.allocator.allocate(
            active_flows, self._topology, current_time,
        )

    def on_task_emitted(self, current_time: int, task: Task) -> None:
        pass

    def on_task_completed(self, current_time: int, task: Task) -> None:
        if task.is_compute():
            self.compute_cursor[task.node] += 1

    def update_analysis(self, workload: P2PWorkload, analysis_result: DefaultAnalysisResult) -> None:
        """增量合并新 Job 的分析结果。"""
        # 合并 compute ordering
        for node_id, task_ids in analysis_result.execution_plan.compute_order.items():
            self.compute_order.setdefault(node_id, []).extend(task_ids)

        # 重建 compute_position 索引
        self.compute_position = {
            task_id: idx
            for node, ids in self.compute_order.items()
            for idx, task_id in enumerate(ids)
        }

        # 合并路由表
        rt = analysis_result.route_table
        if hasattr(rt, '_paths') and hasattr(self.route_table, 'ensure_path'):
            for (src, dst), path in rt._paths.items():
                self.route_table.ensure_path(src, dst, path)

"""MFS scheduling policy — compute ordering + MFS-aware bandwidth allocation.

Preserves compute ordering via ExecutionPlan, uses default routing,
and delegates bandwidth allocation to MfsAllocator.
"""
from collections import defaultdict
from typing import Optional

from .base_policy import SchedulingPolicy
from ..bandwidth_allocators.mfs_allocator import MfsAllocator, MfsAllocatorConfig
from ..runtime import ActiveFlow
from ...static_analysis.strategies.mfs_strategy import MfsAnalysisResult
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...workload_format.schema import P2PWorkload, Task


class MfsSchedulingPolicy(SchedulingPolicy):
    def __init__(
        self,
        analysis: MfsAnalysisResult,
        allocator_config: MfsAllocatorConfig | None = None,
    ):
        self.route_table = analysis.route_table
        self.compute_order = analysis.execution_plan.compute_order
        self._analysis = analysis
        self._allocator_config = allocator_config
        self.allocator: Optional[MfsAllocator] = None
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
        self.allocator = MfsAllocator(
            context=self._analysis.mfs_context,
            rli_info=self._analysis.rli_info,
            config=self._allocator_config,
        )

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

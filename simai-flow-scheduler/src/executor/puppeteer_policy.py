"""Puppeteer-inspired scheduling policy.

Implements resource-aware admission, precomputed route lookup,
and TTE-aware bandwidth allocation through the SchedulingPolicy interface.
"""
from collections import defaultdict
from typing import Optional

from .bandwidth import BandwidthAllocator
from .policy import SchedulingPolicy
from .puppeteer_bandwidth import TteAwareAllocator
from .runtime import ActiveFlow
from ..static_analysis.passes.puppeteer_coordination import ResourceDependencyTable
from ..static_analysis.passes.puppeteer_routing import RouteTable
from ..static_analysis.passes.puppeteer_tte import TTEInfo
from ..static_analysis.passes.task_serializer import ExecutionPlan
from ..static_analysis.passes.topology_loader import NetworkTopology
from ..workload_format.schema import P2PWorkload, Task


class PuppeteerSchedulingPolicy(SchedulingPolicy):
    """Puppeteer scheduling policy.

    Features:
    - Precomputed per-flow route table for path selection
    - TTE-aware bandwidth allocation (weighted or strict priority)
    - Co-start coordination for flows sharing planned resources
    - Compute ordering preserved from ExecutionPlan

    Args:
        route_table: Precomputed per-flow route table
        tte_info: Per-flow TTE priority information
        resource_dependency: Coordination groups for co-start barriers
        execution_plan: Compute ordering per node
        allocator_mode: Bandwidth allocation mode ("weighted" or "strict_priority")
        min_background_share: Minimum bandwidth share for background flows
    """

    def __init__(
        self,
        route_table: RouteTable,
        tte_info: dict[int, TTEInfo],
        resource_dependency: ResourceDependencyTable,
        execution_plan: ExecutionPlan,
        allocator_mode: str = "weighted",
        min_background_share: float = 0.05,
    ):
        self.route_table = route_table
        self.tte_info = tte_info
        self.resource_dependency = resource_dependency
        self.execution_plan = execution_plan

        # Compute ordering state
        self.compute_position: dict[int, int] = {
            tid: idx
            for node, ids in execution_plan.compute_order.items()
            for idx, tid in enumerate(ids)
        }
        self.compute_cursor: dict[int, int] = defaultdict(int)

        # TTE-aware allocator
        self.allocator = TteAwareAllocator(
            tte_info=tte_info,
            mode=allocator_mode,
            min_background_share=min_background_share,
        )

        # Coordination state (initialized in initialize())
        self._task_to_group: dict[int, str] = {}
        self._topology: Optional[NetworkTopology] = None

    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> None:
        self._topology = topology
        self.compute_cursor = defaultdict(int)

        # Build coordination lookup table
        self._task_to_group = {}
        for gid, members in self.resource_dependency.groups.items():
            for tid in members:
                self._task_to_group[tid] = gid

    def emit_ready_tasks(
        self,
        current_time: int,
        ready_tasks: list[Task],
    ) -> list[int]:
        emitted = []
        ready_ids = {t.task_id for t in ready_tasks}

        for task in sorted(ready_tasks, key=lambda t: t.task_id):
            if task.is_compute():
                if self._is_next_compute(task):
                    emitted.append(task.task_id)
            elif task.is_flow():
                if self._can_emit_flow(task, ready_ids):
                    emitted.append(task.task_id)

        return emitted

    def _is_next_compute(self, task: Task) -> bool:
        cursor = self.compute_cursor.get(task.node, 0)
        order = self.execution_plan.compute_order.get(task.node, [])
        return cursor < len(order) and order[cursor] == task.task_id

    def _can_emit_flow(self, task: Task, ready_ids: set[int]) -> bool:
        """Check if a flow task can be emitted.

        If the task belongs to a coordination group, all group members
        must be ready (in the ready pool or already completed).
        """
        tid = task.task_id
        if tid not in self._task_to_group:
            return True

        gid = self._task_to_group[tid]
        members = self.resource_dependency.groups.get(gid, set())

        return all(m in ready_ids for m in members)

    def get_flow_path(self, task: Task) -> list[int]:
        try:
            return self.route_table.get_path(task.task_id)
        except KeyError:
            raise KeyError(
                f"No route for flow task {task.task_id} "
                f"(src={task.src}, dst={task.dst}). "
                f"Missing route planning in PuppeteerSchedulingPolicy."
            )

    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        return self.allocator.allocate(
            active_flows, self._topology, None, current_time,
        )

    def on_task_emitted(self, current_time: int, task: Task) -> None:
        pass

    def on_task_completed(self, current_time: int, task: Task) -> None:
        if task.is_compute():
            self.compute_cursor[task.node] += 1

"""MFS scheduling policy — compute ordering + MFS-aware bandwidth allocation.

Preserves compute ordering via ExecutionPlan, uses default routing,
and delegates bandwidth allocation to MfsAllocator. Tracks request start
times and updates dynamic RLI state in the allocator.
"""
from collections import defaultdict
from typing import Optional

from .base_policy import SchedulingPolicy
from ..bandwidth_allocators.mfs_allocator import MfsAllocator, MfsAllocatorConfig
from ..runtime import ActiveFlow
from ...static_analysis.strategies.mfs_strategy import MfsAnalysisResult
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...static_analysis.passes.mfs_context import MfsStage
from ...workload_format.schema import P2PWorkload, Task, Phase


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
            config=self._allocator_config,
        )
        # Initialize remaining_us from feasibility analysis
        fi = self._analysis.feasibility_info
        if fi:
            self.allocator.remaining_us = dict(fi.request_path_total_us)

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
        """Record request start time when the first task of a request is emitted.

        Also detect batch boundaries: when a new batch's EARLY flows are emitted
        while current_layer still tracks the previous batch, reset current_layer
        to 0 so RLI reflects this batch's progress.
        """
        info = self._analysis.mfs_context.task_info.get(task.task_id)
        if info is None:
            return
        for rid in info.request_ids:
            if rid not in self.allocator.request_start_time:
                self.allocator.request_start_time[rid] = current_time

        # Batch boundary detection: any EARLY flow with target_layer below
        # current_layer implies the previous batch's compute has finished
        # and a new batch is starting. Reset current_layer so RLI reflects
        # this batch's progress.
        if (info.mfs_stage == MfsStage.EARLY
                and info.target_layer < self.allocator.current_layer_by_stage.get(
                    (info.replica_id, info.stage_id), 0)):
            self.allocator.current_layer_by_stage[(info.replica_id, info.stage_id)] = 0

    def on_task_completed(self, current_time: int, task: Task) -> None:
        """Advance compute cursor and update dynamic layer tracking."""
        if task.is_compute():
            self.compute_cursor[task.node] += 1
            # Only advance current_layer for prefill-phase compute.
            # Decode compute would incorrectly advance RLI for other requests'
            # prefill EARLY flows sharing the same (job_id, stage_id).
            if task.phase == Phase.PREFILL:
                info = self._analysis.mfs_context.task_info.get(task.task_id)
                if info is not None:
                    stage_key = (info.replica_id, info.stage_id)
                    prev = self.allocator.current_layer_by_stage.get(stage_key, 0)
                    if task.layer_id + 1 > prev:
                        self.allocator.current_layer_by_stage[stage_key] = task.layer_id + 1

        # Feasibility: subtract critical path task duration from remaining_us
        fi = self._analysis.feasibility_info
        if fi and task.task_id in fi.critical_path_tasks:
            dur = fi.task_duration_us.get(task.task_id, 0)
            if dur > 0:
                info = self._analysis.mfs_context.task_info.get(task.task_id)
                if info:
                    for rid in info.request_ids:
                        if (rid in fi.request_critical_tasks
                                and task.task_id in fi.request_critical_tasks[rid]
                                and rid in self.allocator.remaining_us):
                            self.allocator.remaining_us[rid] -= dur

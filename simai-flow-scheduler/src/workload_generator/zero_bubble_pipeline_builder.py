"""Strategy-specific workload expansion for a ZB1P-style pipeline.

The common workload schema and :class:`WorkloadBuilder` remain unchanged.
This subclass makes the backward-input/weight split auditable, adds the
weight/DP-to-post barrier required by Zero Bubble, and records all
strategy-specific facts in a task-ID sidecar.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..workload_format.schema import (
    CommType,
    Job,
    P2PWorkload,
    Phase,
    Task,
)
from .aicb_parser import AicbHeader, AicbWorkItem
from .rank_grouper import RankGrouper
from .workload_builder import ItemTasks, WorkloadBuilder
from .zero_semantics import is_zero_workload


@dataclass(frozen=True)
class ZeroBubbleTaskInfo:
    """Analyzer-owned Zero Bubble metadata stored outside the common Task IR."""

    task_id: int
    job_id: int
    task_role: str
    microbatch_id: int
    physical_stage_id: int
    layer_id: int
    b_task_id: int | None
    w_task_id: int | None
    critical_path: bool
    preferred_slot: int | None = None
    final_local_order: int | None = None


class ZeroBubblePipelineWorkloadBuilder(WorkloadBuilder):
    """Build the normal PP endpoints with an explicit ZB B/W/optimizer DAG."""

    def __init__(
        self,
        task_info: dict[int, ZeroBubbleTaskInfo] | None = None,
    ):
        super().__init__()
        self.task_info = task_info if task_info is not None else {}
        self._active_grouper: RankGrouper | None = None

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        if is_zero_workload(aicb_items):
            raise ValueError(
                "zero_bubble does not support the DeepSpeed ZeRO/FSDP "
                "AICB row format"
            )
        if aicb_header.ga < 1:
            raise ValueError("zero_bubble requires ga >= 1")

        self._active_grouper = RankGrouper(
            job.assigned_nodes, job.parallelism,
        )
        if self._active_grouper.pp > 1 and aicb_header.pp_comm_size <= 0:
            raise ValueError(
                "zero_bubble requires a positive pp_comm_size when pp > 1"
            )

        workload = super().build_from_aicb(
            aicb_header, aicb_items, job, comm_algo,
        )
        self._record_task_info(workload)
        return workload

    def _wire_dependencies(
        self,
        item_tasks_list: list[ItemTasks],
        num_layer_items: int,
        num_pre_items: int,
        items_per_ga: int,
    ):
        """Reuse the exact F/B/W graph, then gate post work on every W/DP."""
        super()._wire_dependencies(
            item_tasks_list,
            num_layer_items,
            num_pre_items,
            items_per_ga,
        )

        post_start = num_pre_items + num_layer_items
        post_items = item_tasks_list[post_start:]
        if not post_items:
            return

        ga_groups = self._group_items_by_ga(
            item_tasks_list,
            num_pre_items,
            num_layer_items,
            items_per_ga,
        )
        optimizer_entry = post_items[0].fwd_computes
        for ga_group in ga_groups:
            for item in ga_group:
                # If W has DP communication, receiver/completion_index is the
                # terminal. Otherwise this helper uses the local W compute.
                self._wire_per_node_phase_transition(
                    src_result=item.wg_result,
                    src_computes=item.wg_computes,
                    dst_computes=optimizer_entry,
                )

    def _record_task_info(self, workload: P2PWorkload) -> None:
        grouper = self._require_grouper()
        node_to_stage: dict[int, int] = {}
        stage_width = grouper.dp * grouper.ep * grouper.tp
        for stage_id in range(grouper.pp):
            start = stage_id * stage_width
            for node in grouper.nodes[start:start + stage_width]:
                node_to_stage[node] = stage_id

        pairs: dict[tuple[int, int, int, int], dict[str, int]] = {}
        for task in workload.tasks:
            if not task.is_compute() or task.node is None:
                continue
            key = (task.node, task.iteration, task.layer_id, task.item_id)
            if task.phase is Phase.BACKWARD_INPUT:
                pairs.setdefault(key, {})["B"] = task.task_id
            elif task.phase is Phase.BACKWARD_WEIGHT:
                pairs.setdefault(key, {})["W"] = task.task_id

        for task in workload.tasks:
            endpoint = task.node if task.is_compute() else task.src
            stage_id = node_to_stage.get(endpoint, 0)
            key = (
                endpoint,
                task.iteration,
                task.layer_id,
                task.item_id,
            )
            pair = pairs.get(key, {})
            role = self._task_role(task)
            self.task_info[task.task_id] = ZeroBubbleTaskInfo(
                task_id=task.task_id,
                job_id=task.job_id,
                task_role=role,
                microbatch_id=task.iteration,
                physical_stage_id=stage_id,
                layer_id=task.layer_id,
                b_task_id=pair.get("B"),
                w_task_id=pair.get("W"),
                critical_path=role in {"B", "PP_GRAD"},
            )

    @staticmethod
    def _task_role(task: Task) -> str:
        if task.comm_type is CommType.PP_SEND:
            return "PP_ACT" if task.phase is Phase.FORWARD else "PP_GRAD"
        if task.phase is Phase.FORWARD:
            return "F"
        if task.phase is Phase.BACKWARD_INPUT:
            return "B"
        if task.phase is Phase.BACKWARD_WEIGHT:
            return "DP" if task.is_flow() else "W"
        if task.phase is Phase.OPTIMIZER:
            return "OPT"
        return "OTHER"

    def _require_grouper(self) -> RankGrouper:
        if self._active_grouper is None:
            raise RuntimeError("zero_bubble builder has no active rank grouper")
        return self._active_grouper

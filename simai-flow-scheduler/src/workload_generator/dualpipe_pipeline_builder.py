"""DeepSeek-V3-style DualPipe workload expansion.

The implementation keeps DualPipe metadata in a task-ID sidecar and reuses
only the existing task schema and collective expanders.  It intentionally
does not register the strategy in any baseline analyzer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..workload_format.schema import (
    CommType,
    Job,
    P2PWorkload,
    Phase,
    Task,
)
from .aicb_parser import AicbHeader, AicbWorkItem
from .bidirectional_pipeline_builder import (
    BidirectionalPipelineTaskInfo,
    BidirectionalPipelineWorkloadBuilder,
)
from .rank_grouper import RankGrouper
from .workload_builder import WorkloadBuilder
from .zero_semantics import is_zero_workload


@dataclass(frozen=True)
class DualPipeScheduleToken:
    """One operation in the official eight-step local DualPipe schedule."""

    physical_stage_id: int
    schedule_step: int
    loop_index: int
    local_phase: int
    module_replica_id: int
    direction: str
    operation: str
    microbatch_id: int
    defer_weight: bool = False
    weight_queue_index: int | None = None
    overlap_pair_id: int | None = None
    middle_rank_special_case: bool = False


@dataclass(frozen=True)
class DualPipeTaskInfo:
    """DualPipe-only metadata kept outside the common :class:`Task`."""

    task_id: int
    job_id: int
    task_role: str
    component: str
    microbatch_id: int
    physical_stage_id: int
    model_shard_id: int
    module_replica_id: int
    local_phase: int
    direction: str
    schedule_step: int | None = None
    schedule_loop_index: int | None = None
    defer_weight: bool = False
    weight_queue_index: int | None = None
    overlap_pair_id: int | None = None
    overlap_role: str | None = None
    overlap_model: str = "conservative"
    logical_boundary_id: int | None = None
    peer_physical_stage_id: int | None = None
    original_duration_us: int | None = None
    effective_duration_us: int | None = None
    preferred_slot: int | None = None
    final_local_order: int | None = None


def build_dualpipe_schedule(
    pp: int,
    microbatches: int,
    physical_stage_id: int,
) -> list[DualPipeScheduleToken]:
    """Return the official eight-step schedule for one physical PP stage.

    This follows ``deepseek-ai/DualPipe/dualpipe/dualpipe.py``.  Full
    backward is represented by ``BW``; deferred backward-input and its later
    FIFO weight-gradient are represented by separate ``B`` and ``W`` tokens.
    """
    if pp <= 0 or pp % 2:
        raise ValueError(f"DualPipe requires positive even pp, got {pp}")
    if microbatches <= 0 or microbatches % 2:
        raise ValueError(
            "DualPipe requires positive even microbatch count, "
            f"got {microbatches}"
        )
    if microbatches < 2 * pp:
        raise ValueError(
            f"DualPipe requires microbatches >= 2 * pp, got "
            f"{microbatches} < {2 * pp}"
        )
    if not 0 <= physical_stage_id < pp:
        raise ValueError(
            f"physical_stage_id must be in [0, {pp}), got {physical_stage_id}"
        )

    second_half = physical_stage_id >= pp // 2
    half_rank = min(physical_stage_id, pp - 1 - physical_stage_id)
    half_pp = pp // 2
    half_microbatches = microbatches // 2
    microbatch_ids = {
        0: list(range(half_microbatches)),
        1: list(range(half_microbatches, microbatches)),
    }
    forward_cursor = [0, 0]
    backward_cursor = [0, 0]
    weight_queue: list[tuple[int, int, int]] = []
    next_weight_queue_index = 0
    next_overlap_pair_id = 0
    result: list[DualPipeScheduleToken] = []

    def replica(local_phase: int) -> int:
        return local_phase ^ int(second_half)

    def append_token(
        step: int,
        loop_index: int,
        local_phase: int,
        operation: str,
        microbatch_id: int,
        *,
        defer_weight: bool = False,
        queue_index: int | None = None,
        overlap_pair_id: int | None = None,
        middle_special: bool = False,
    ) -> None:
        module_replica_id = replica(local_phase)
        result.append(DualPipeScheduleToken(
            physical_stage_id=physical_stage_id,
            schedule_step=step,
            loop_index=loop_index,
            local_phase=local_phase,
            module_replica_id=module_replica_id,
            direction="down" if module_replica_id == 0 else "up",
            operation=operation,
            microbatch_id=microbatch_id,
            defer_weight=defer_weight,
            weight_queue_index=queue_index,
            overlap_pair_id=overlap_pair_id,
            middle_rank_special_case=middle_special,
        ))

    def emit_forward(
        step: int,
        loop_index: int,
        local_phase: int,
        *,
        overlap_pair_id: int | None = None,
        middle_special: bool = False,
    ) -> None:
        module_replica_id = replica(local_phase)
        cursor = forward_cursor[module_replica_id]
        if cursor >= half_microbatches:
            raise RuntimeError("DualPipe forward cursor exceeded microbatch count")
        microbatch_id = microbatch_ids[module_replica_id][cursor]
        forward_cursor[module_replica_id] += 1
        append_token(
            step,
            loop_index,
            local_phase,
            "F",
            microbatch_id,
            overlap_pair_id=overlap_pair_id,
            middle_special=middle_special,
        )

    def emit_backward(
        step: int,
        loop_index: int,
        local_phase: int,
        *,
        defer_weight: bool,
        overlap_pair_id: int | None = None,
        middle_special: bool = False,
    ) -> None:
        nonlocal next_weight_queue_index
        module_replica_id = replica(local_phase)
        cursor = backward_cursor[module_replica_id]
        if cursor >= half_microbatches:
            raise RuntimeError("DualPipe backward cursor exceeded microbatch count")
        microbatch_id = microbatch_ids[module_replica_id][cursor]
        backward_cursor[module_replica_id] += 1
        queue_index = None
        if defer_weight:
            queue_index = next_weight_queue_index
            next_weight_queue_index += 1
            weight_queue.append(
                (module_replica_id, microbatch_id, queue_index)
            )
        append_token(
            step,
            loop_index,
            local_phase,
            "B" if defer_weight else "BW",
            microbatch_id,
            defer_weight=defer_weight,
            queue_index=queue_index,
            overlap_pair_id=overlap_pair_id,
            middle_special=middle_special,
        )

    def emit_weight(step: int, loop_index: int) -> None:
        if not weight_queue:
            raise RuntimeError("DualPipe weight-gradient FIFO underflow")
        module_replica_id, microbatch_id, queue_index = weight_queue.pop(0)
        local_phase = module_replica_id ^ int(second_half)
        append_token(
            step,
            loop_index,
            local_phase,
            "W",
            microbatch_id,
            queue_index=queue_index,
        )

    def emit_overlap(
        step: int,
        loop_index: int,
        forward_phase: int,
        backward_phase: int,
        *,
        enabled: bool = True,
        middle_special: bool = False,
    ) -> None:
        nonlocal next_overlap_pair_id
        pair_id = next_overlap_pair_id if enabled else None
        if enabled:
            next_overlap_pair_id += 1
        emit_forward(
            step,
            loop_index,
            forward_phase,
            overlap_pair_id=pair_id,
            middle_special=middle_special,
        )
        emit_backward(
            step,
            loop_index,
            backward_phase,
            defer_weight=False,
            overlap_pair_id=pair_id,
            middle_special=middle_special,
        )

    # Step 1: nF0
    for loop_index in range((half_pp - half_rank - 1) * 2):
        emit_forward(1, loop_index, 0)

    # Step 2: nF0F1
    for loop_index in range(half_rank + 1):
        emit_forward(2, loop_index, 0)
        emit_forward(2, loop_index, 1)

    # Step 3: nB1W1F1
    for loop_index in range(half_pp - half_rank - 1):
        emit_backward(3, loop_index, 1, defer_weight=True)
        emit_weight(3, loop_index)
        emit_forward(3, loop_index, 1)

    # Step 4: nF0B1F1B0
    middle_rank = physical_stage_id in {half_pp - 1, half_pp}
    for loop_index in range(
        half_microbatches - pp + half_rank + 1
    ):
        special = middle_rank and loop_index == 0
        emit_overlap(
            4,
            loop_index,
            0,
            1,
            enabled=not special,
            middle_special=special,
        )
        emit_overlap(4, loop_index, 1, 0)

    # Step 5: nB1F1B0
    for loop_index in range(half_pp - half_rank - 1):
        emit_backward(5, loop_index, 1, defer_weight=False)
        emit_overlap(5, loop_index, 1, 0)

    # Step 6: nB1B0, switching to deferred W at the official parity point.
    enable_zb = False
    step_6 = half_rank + 1
    for loop_index in range(step_6):
        if loop_index == step_6 // 2 and half_rank % 2 == 1:
            enable_zb = True
        emit_backward(6, loop_index, 1, defer_weight=enable_zb)
        if loop_index == step_6 // 2 and half_rank % 2 == 0:
            enable_zb = True
        emit_backward(6, loop_index, 0, defer_weight=enable_zb)

    # Step 7: nWB0
    for loop_index in range(half_pp - half_rank - 1):
        emit_weight(7, loop_index)
        emit_backward(7, loop_index, 0, defer_weight=True)

    # Step 8: nW
    for loop_index in range(half_rank + 1):
        emit_weight(8, loop_index)

    if weight_queue:
        raise RuntimeError(
            f"DualPipe weight-gradient FIFO is not empty: {weight_queue}"
        )
    if forward_cursor != [half_microbatches, half_microbatches]:
        raise RuntimeError(f"DualPipe forward count mismatch: {forward_cursor}")
    if backward_cursor != [half_microbatches, half_microbatches]:
        raise RuntimeError(f"DualPipe backward count mismatch: {backward_cursor}")
    return result


class DualPipePipelineWorkloadBuilder(BidirectionalPipelineWorkloadBuilder):
    """Build a direct two-replica DualPipe DAG with replica-aware DP sync."""

    def __init__(
        self,
        task_info: dict[int, DualPipeTaskInfo] | None = None,
        gradient_sync_bytes: int | None = None,
        overlap_model: str = "conservative",
        overlap_factor: float | None = None,
    ):
        self.dualpipe_task_info = task_info if task_info is not None else {}
        self._temporary_bidirectional_info: dict[
            int, BidirectionalPipelineTaskInfo
        ] = {}
        super().__init__(
            self._temporary_bidirectional_info,
            gradient_sync_bytes=gradient_sync_bytes,
        )
        if overlap_model not in {"conservative", "ideal", "profiled"}:
            raise ValueError(
                "overlap_model must be conservative, ideal, or profiled"
            )
        self.overlap_model = overlap_model
        if overlap_model == "profiled":
            if overlap_factor is None or not 0 < overlap_factor <= 1:
                raise ValueError(
                    "profiled overlap_model requires overlap_factor in (0, 1]"
                )
        elif overlap_factor is not None:
            raise ValueError(
                "overlap_factor is only valid with overlap_model='profiled'"
            )
        self.overlap_factor = overlap_factor

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        if is_zero_workload(aicb_items):
            raise ValueError(
                "DualPipe does not support ambiguous DeepSpeed ZeRO/FSDP rows"
            )
        grouper = RankGrouper(job.assigned_nodes, job.parallelism)
        self._validate_dualpipe_shape(grouper.pp, aicb_header.ga, job.job_id)
        if aicb_header.pp_comm_size <= 0:
            raise ValueError("DualPipe requires a positive pp_comm_size")

        ambiguous_dp = [
            item.name
            for item in aicb_items
            if item.dp_comm != "NONE"
            and item.name not in {"grad_param_comm", "grad_gather"}
        ]
        if ambiguous_dp:
            raise ValueError(
                "DualPipe cannot infer replica-aware semantics for dp_comm "
                f"rows {ambiguous_dp}; only grad_param_comm and the distinct "
                "pre-step grad_gather are supported"
            )

        sync_bytes = self._resolve_gradient_sync_bytes(aicb_items)
        sanitized_items = [
            replace(
                item,
                dp_comm="NONE",
                dp_comm_size=0,
            )
            if item.name == "grad_param_comm" and item.dp_comm != "NONE"
            else item
            for item in aicb_items
        ]

        self._temporary_bidirectional_info.clear()
        self.dualpipe_task_info.clear()
        self._active_job = job
        self._active_grouper = grouper
        self._active_ga = aicb_header.ga

        # Invoke the common two-phase generator directly. Dynamic dispatch
        # still uses the inherited bidirectional PP endpoints and wiring.
        workload = WorkloadBuilder.build_from_aicb(
            self,
            aicb_header,
            sanitized_items,
            job,
            comm_algo,
        )
        self._append_chimera_gradient_sync(
            workload,
            grouper,
            aicb_header,
            sync_bytes,
            comm_algo,
        )
        self._record_compute_info(workload)
        self._build_dualpipe_sidecar(workload, grouper, aicb_header)
        self._apply_overlap_durations(workload)
        errors = workload.validate()
        if errors:
            raise ValueError(f"DualPipe workload validation failed: {errors}")
        return workload

    @staticmethod
    def _validate_dualpipe_shape(pp: int, ga: int, job_id: int) -> None:
        if pp <= 0 or pp % 2:
            raise ValueError(
                f"Job {job_id} DualPipe requires positive even pp, got {pp}"
            )
        if ga <= 0 or ga % 2:
            raise ValueError(
                f"Job {job_id} DualPipe requires positive even "
                f"microbatch count, got {ga}"
            )
        if ga < 2 * pp:
            raise ValueError(
                f"Job {job_id} DualPipe requires microbatches >= 2 * pp, "
                f"got {ga} < {2 * pp}"
            )

    def _build_dualpipe_sidecar(
        self,
        workload: P2PWorkload,
        grouper: RankGrouper,
        header: AicbHeader,
    ) -> None:
        stage_by_node = {
            grouper.get_pp_rank(stage, dp_idx, ep_idx, tp_idx): stage
            for stage in range(grouper.pp)
            for dp_idx in range(grouper.dp)
            for ep_idx in range(grouper.ep)
            for tp_idx in range(grouper.tp)
        }
        schedules = {
            stage: build_dualpipe_schedule(grouper.pp, header.ga, stage)
            for stage in range(grouper.pp)
        }
        token_index: dict[
            tuple[int, int, int, str], DualPipeScheduleToken
        ] = {}
        for stage, tokens in schedules.items():
            for token in tokens:
                operations = (
                    ("B", "W") if token.operation == "BW"
                    else (token.operation,)
                )
                for operation in operations:
                    token_index[(
                        stage,
                        token.module_replica_id,
                        token.microbatch_id,
                        operation,
                    )] = token

        for task in workload.tasks:
            temporary = self._temporary_bidirectional_info.get(task.task_id)
            node = task.node if task.is_compute() else task.src
            stage = stage_by_node.get(node, -1)
            microbatch_id = task.iteration
            if 0 <= microbatch_id < header.ga:
                module_replica_id = 0 if microbatch_id < header.ga // 2 else 1
                direction = "down" if module_replica_id == 0 else "up"
                model_shard_id = (
                    stage
                    if module_replica_id == 0
                    else grouper.pp - 1 - stage
                )
                local_phase = module_replica_id ^ int(stage >= grouper.pp // 2)
            else:
                module_replica_id = getattr(
                    temporary, "module_replica_id", -1,
                )
                direction = "sync" if module_replica_id >= 0 else "none"
                model_shard_id = getattr(temporary, "model_shard_id", -1)
                local_phase = (
                    module_replica_id ^ int(stage >= grouper.pp // 2)
                    if module_replica_id >= 0 and stage >= 0
                    else -1
                )

            operation = self._operation(task)
            token = token_index.get((
                stage,
                module_replica_id,
                microbatch_id,
                operation,
            ))
            role, component = self._role_and_component(task, temporary)
            overlap_role = None
            if token is not None and token.overlap_pair_id is not None:
                overlap_role = "forward" if operation == "F" else "backward"
            global_overlap_pair_id = (
                stage * header.ga * 4 + token.overlap_pair_id
                if token is not None and token.overlap_pair_id is not None
                else None
            )

            self.dualpipe_task_info[task.task_id] = DualPipeTaskInfo(
                task_id=task.task_id,
                job_id=task.job_id,
                task_role=role,
                component=component,
                microbatch_id=microbatch_id,
                physical_stage_id=stage,
                model_shard_id=model_shard_id,
                module_replica_id=module_replica_id,
                local_phase=local_phase,
                direction=direction,
                schedule_step=(
                    token.schedule_step if token is not None else None
                ),
                schedule_loop_index=(
                    token.loop_index if token is not None else None
                ),
                defer_weight=(
                    token.defer_weight if token is not None else False
                ),
                weight_queue_index=(
                    token.weight_queue_index if token is not None else None
                ),
                overlap_pair_id=(
                    global_overlap_pair_id
                ),
                overlap_role=overlap_role,
                overlap_model=self.overlap_model,
                logical_boundary_id=getattr(
                    temporary, "logical_boundary_id", None,
                ),
                peer_physical_stage_id=getattr(
                    temporary, "peer_physical_stage_id", None,
                ),
                original_duration_us=(
                    task.duration_us if task.is_compute() else None
                ),
                effective_duration_us=(
                    task.duration_us if task.is_compute() else None
                ),
            )

    def _apply_overlap_durations(self, workload: P2PWorkload) -> None:
        """Calibrate each F&B envelope without changing the common executor.

        The executor serializes compute tasks on one GPU.  Reducing the sum of
        the paired tasks to the desired envelope duration makes the observable
        wall time match the selected overlap model while retaining individual
        F/B/W completion points and all network tasks.
        """
        if self.overlap_model == "conservative":
            return
        groups: dict[tuple[int, int], list[Task]] = {}
        for task in workload.get_compute_tasks():
            info = self.dualpipe_task_info.get(task.task_id)
            if info is None or info.overlap_pair_id is None:
                continue
            groups.setdefault(
                (task.node, info.overlap_pair_id), []
            ).append(task)

        for tasks in groups.values():
            forward_total = sum(
                task.duration_us or 0
                for task in tasks
                if task.phase is Phase.FORWARD
            )
            backward_total = sum(
                task.duration_us or 0
                for task in tasks
                if task.phase in {
                    Phase.BACKWARD_INPUT,
                    Phase.BACKWARD_WEIGHT,
                }
            )
            original_total = forward_total + backward_total
            if original_total <= 0:
                continue
            if self.overlap_model == "ideal":
                target_total = max(forward_total, backward_total)
            else:
                target_total = round(original_total * self.overlap_factor)

            weighted = [
                ((task.duration_us or 0) * target_total, task)
                for task in tasks
            ]
            assigned = {
                task.task_id: numerator // original_total
                for numerator, task in weighted
            }
            remainder = target_total - sum(assigned.values())
            order = sorted(
                weighted,
                key=lambda item: (
                    -(item[0] % original_total),
                    item[1].task_id,
                ),
            )
            for _, task in order[:remainder]:
                assigned[task.task_id] += 1

            for task in tasks:
                task.duration_us = assigned[task.task_id]
                info = self.dualpipe_task_info[task.task_id]
                self.dualpipe_task_info[task.task_id] = replace(
                    info,
                    effective_duration_us=task.duration_us,
                )

    @staticmethod
    def _operation(task: Task) -> str:
        if task.phase is Phase.FORWARD:
            return "F"
        if task.phase is Phase.BACKWARD_INPUT:
            return "B"
        if task.phase is Phase.BACKWARD_WEIGHT:
            return "W"
        return "other"

    @staticmethod
    def _role_and_component(
        task: Task,
        temporary: BidirectionalPipelineTaskInfo | None,
    ) -> tuple[str, str]:
        temporary_role = getattr(temporary, "task_role", "")
        if temporary_role == "chimera_gradient_sync":
            return "dualpipe_gradient_sync", "replica_gradient_sync"
        if temporary_role in {"pp_activation", "pp_gradient"}:
            return temporary_role, "pp"
        if task.is_compute():
            return {
                Phase.FORWARD: ("F", "compute_forward"),
                Phase.BACKWARD_INPUT: ("B", "compute_backward_input"),
                Phase.BACKWARD_WEIGHT: ("W", "compute_backward_weight"),
            }.get(task.phase, ("other", "compute"))
        if task.comm_type is CommType.EP_ALLTOALL:
            return "collective", "ep_alltoall"
        if task.comm_type in {
            CommType.TP_ALLREDUCE_RING,
            CommType.TP_ALLREDUCE_TREE,
            CommType.TP_ALLGATHER_RING,
            CommType.TP_ALLGATHER_TREE,
            CommType.TP_REDUCESCATTER_RING,
            CommType.TP_REDUCESCATTER_TREE,
            CommType.TP_ALLTOALL,
        }:
            return "collective", "tp"
        return "flow", "collective"

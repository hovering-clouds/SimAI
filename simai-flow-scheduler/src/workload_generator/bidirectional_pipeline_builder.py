"""Direct two-direction pipeline expansion.

Half of the microbatches traverse physical stages in ascending order and half
in descending order. Gradient traffic follows the exact reverse path. The
common workload builder and task schema are not modified.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..workload_format.schema import CommType, Job, P2PWorkload, Phase, TaskType
from .aicb_parser import AicbHeader, AicbWorkItem
from .collective_expander import FlowTask
from .rank_grouper import RankGrouper
from .workload_builder import ItemTasks, WorkloadBuilder
from .zero_semantics import is_zero_workload


@dataclass(frozen=True)
class BidirectionalPipelineTaskInfo:
    """Bidirectional-only metadata kept outside the common Task schema."""

    task_id: int
    job_id: int
    task_role: str
    microbatch_id: int
    pipeline_id: int
    direction: str
    physical_stage_id: int
    logical_stage_id: int
    logical_boundary_id: int | None = None
    peer_physical_stage_id: int | None = None
    peer_logical_stage_id: int | None = None


@dataclass(frozen=True)
class _BidirectionalBoundary:
    flow: FlowTask
    microbatch_id: int
    pipeline_id: int
    direction: str
    source_stage: int
    destination_stage: int
    logical_boundary: int


@dataclass
class BidirectionalPPFlowResult:
    all_flows: list[FlowTask] = field(default_factory=list)
    boundaries: list[_BidirectionalBoundary] = field(default_factory=list)


class BidirectionalPipelineWorkloadBuilder(WorkloadBuilder):
    """Generate opposing PP endpoints during the initial AICB expansion."""

    def __init__(
        self,
        task_info: dict[int, BidirectionalPipelineTaskInfo] | None = None,
    ):
        super().__init__()
        self.task_info = task_info if task_info is not None else {}
        self._active_job: Job | None = None
        self._active_grouper: RankGrouper | None = None
        self._active_ga = 0

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        if is_zero_workload(aicb_items):
            raise ValueError(
                "bidirectional does not support the DeepSpeed ZeRO/FSDP "
                "AICB row format"
            )
        grouper = RankGrouper(job.assigned_nodes, job.parallelism)
        self._validate_shape(grouper.pp, aicb_header.ga, job.job_id)
        if aicb_header.pp_comm_size <= 0:
            raise ValueError(
                "bidirectional requires a positive pp_comm_size"
            )
        self._active_job = job
        self._active_grouper = grouper
        self._active_ga = aicb_header.ga

        workload = super().build_from_aicb(
            aicb_header, aicb_items, job, comm_algo,
        )
        self._record_compute_info(workload)
        return workload

    def _generate_pp_flows(
        self,
        grouper: RankGrouper,
        aicb_header: AicbHeader,
        ga_groups: list[list[ItemTasks]],
        items_per_ga: int,
        job_id: int,
        task_id_counter: int,
    ) -> tuple[BidirectionalPPFlowResult, int]:
        result = BidirectionalPPFlowResult()
        half_ga = len(ga_groups) // 2

        for microbatch_id in range(len(ga_groups)):
            pipeline_id = 0 if microbatch_id < half_ga else 1
            direction = "down" if pipeline_id == 0 else "up"
            for logical_boundary in range(grouper.pp - 1):
                if pipeline_id == 0:
                    source_stage = logical_boundary
                    destination_stage = logical_boundary + 1
                else:
                    source_stage = grouper.pp - 1 - logical_boundary
                    destination_stage = grouper.pp - 2 - logical_boundary

                for dp_idx in range(grouper.dp):
                    for ep_idx in range(grouper.ep):
                        for tp_idx in range(grouper.tp):
                            source = grouper.get_pp_rank(
                                source_stage, dp_idx, ep_idx, tp_idx,
                            )
                            destination = grouper.get_pp_rank(
                                destination_stage, dp_idx, ep_idx, tp_idx,
                            )
                            activation = FlowTask(
                                task_id=task_id_counter,
                                job_id=job_id,
                                type=TaskType.FLOW,
                                src=source,
                                dst=destination,
                                size_bytes=aicb_header.pp_comm_size,
                                comm_type=CommType.PP_SEND,
                                phase=Phase.FORWARD,
                                layer_id=items_per_ga - 1,
                                iteration=microbatch_id,
                            )
                            task_id_counter += 1
                            gradient = FlowTask(
                                task_id=task_id_counter,
                                job_id=job_id,
                                type=TaskType.FLOW,
                                src=destination,
                                dst=source,
                                size_bytes=aicb_header.pp_comm_size,
                                comm_type=CommType.PP_SEND,
                                phase=Phase.BACKWARD_INPUT,
                                layer_id=0,
                                iteration=microbatch_id,
                            )
                            task_id_counter += 1
                            result.all_flows.extend((activation, gradient))
                            result.boundaries.extend((
                                _BidirectionalBoundary(
                                    activation,
                                    microbatch_id,
                                    pipeline_id,
                                    direction,
                                    source_stage,
                                    destination_stage,
                                    logical_boundary,
                                ),
                                _BidirectionalBoundary(
                                    gradient,
                                    microbatch_id,
                                    pipeline_id,
                                    direction,
                                    destination_stage,
                                    source_stage,
                                    logical_boundary,
                                ),
                            ))
        return result, task_id_counter

    def _wire_pp_dependencies(
        self,
        pp_result: BidirectionalPPFlowResult,
        ga_groups: list[list[ItemTasks]],
        grouper: RankGrouper,
    ):
        for boundary in pp_result.boundaries:
            ga_group = ga_groups[boundary.microbatch_id]
            first_item = ga_group[0]
            last_item = ga_group[-1]
            flow = boundary.flow

            if flow.phase is Phase.FORWARD:
                self._wire_to_flow_sender(
                    flow,
                    last_item.fwd_computes,
                    last_item.fwd_result,
                    flow.src,
                )
                first_item.fwd_computes[flow.dst].deps.append(flow.task_id)
                role = "pp_activation"
                logical_stage = boundary.logical_boundary
                peer_logical_stage = boundary.logical_boundary + 1
            else:
                self._wire_to_flow_sender(
                    flow,
                    first_item.ig_computes,
                    first_item.ig_result,
                    flow.src,
                )
                last_item.ig_computes[flow.dst].deps.append(flow.task_id)
                role = "pp_gradient"
                logical_stage = boundary.logical_boundary + 1
                peer_logical_stage = boundary.logical_boundary

            self.task_info[flow.task_id] = BidirectionalPipelineTaskInfo(
                task_id=flow.task_id,
                job_id=flow.job_id,
                task_role=role,
                microbatch_id=boundary.microbatch_id,
                pipeline_id=boundary.pipeline_id,
                direction=boundary.direction,
                physical_stage_id=boundary.source_stage,
                logical_stage_id=logical_stage,
                logical_boundary_id=boundary.logical_boundary,
                peer_physical_stage_id=boundary.destination_stage,
                peer_logical_stage_id=peer_logical_stage,
            )

    def _record_compute_info(self, workload: P2PWorkload) -> None:
        grouper = self._require_grouper()
        stage_width = grouper.dp * grouper.ep * grouper.tp
        node_to_stage: dict[int, int] = {}
        for stage_id in range(grouper.pp):
            start = stage_id * stage_width
            for node in grouper.nodes[start:start + stage_width]:
                node_to_stage[node] = stage_id

        half_ga = self._active_ga // 2
        for task in workload.tasks:
            if (
                not task.is_compute()
                or task.node is None
                or not 0 <= task.iteration < self._active_ga
                or task.phase not in (
                    Phase.FORWARD,
                    Phase.BACKWARD_INPUT,
                    Phase.BACKWARD_WEIGHT,
                )
            ):
                continue
            pipeline_id = 0 if task.iteration < half_ga else 1
            direction = "down" if pipeline_id == 0 else "up"
            physical_stage = node_to_stage[task.node]
            logical_stage = (
                physical_stage
                if pipeline_id == 0
                else grouper.pp - 1 - physical_stage
            )
            role = {
                Phase.FORWARD: "compute_forward",
                Phase.BACKWARD_INPUT: "compute_backward_input",
                Phase.BACKWARD_WEIGHT: "compute_backward_weight",
            }[task.phase]
            self.task_info[task.task_id] = BidirectionalPipelineTaskInfo(
                task_id=task.task_id,
                job_id=task.job_id,
                task_role=role,
                microbatch_id=task.iteration,
                pipeline_id=pipeline_id,
                direction=direction,
                physical_stage_id=physical_stage,
                logical_stage_id=logical_stage,
            )

    @staticmethod
    def _validate_shape(pp: int, ga: int, job_id: int) -> None:
        if pp % 2:
            raise ValueError(
                f"Job {job_id} bidirectional requires even pp, got {pp}"
            )
        if ga % 2:
            raise ValueError(
                f"Job {job_id} bidirectional requires even microbatch count, got {ga}"
            )
        if ga < 2 * pp:
            raise ValueError(
                f"Job {job_id} bidirectional requires at least 2 * pp "
                f"microbatches: ga={ga}, pp={pp}"
            )

    def _require_grouper(self) -> RankGrouper:
        if self._active_grouper is None:
            raise RuntimeError("bidirectional builder has no active rank grouper")
        return self._active_grouper

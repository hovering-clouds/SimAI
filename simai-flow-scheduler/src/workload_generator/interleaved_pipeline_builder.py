"""Direct AICB expansion for an interleaved virtual pipeline.

This strategy-specific builder leaves :class:`WorkloadBuilder` unchanged.  It
reuses its compute/collective expansion, but replaces local chunk wiring and PP
flow generation so the first materialized workload already contains the VPP
DAG.
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
class InterleavedPipelineTaskInfo:
    """Interleaved-only metadata kept outside the common Task schema."""

    task_id: int
    job_id: int
    task_role: str
    microbatch_id: int
    model_chunk_id: int
    physical_stage_id: int
    logical_stage_id: int
    logical_boundary_id: int | None = None
    peer_physical_stage_id: int | None = None
    peer_logical_stage_id: int | None = None
    direction: str = "local"


@dataclass(frozen=True)
class _InterleavedBoundary:
    flow: FlowTask
    microbatch_id: int
    source_chunk: int
    destination_chunk: int
    source_stage: int
    destination_stage: int
    source_layer: int
    destination_layer: int
    logical_boundary: int
    direction: str


@dataclass
class InterleavedPPFlowResult:
    """VPP flows plus the boundary information needed for dependency wiring."""

    all_flows: list[FlowTask] = field(default_factory=list)
    boundaries: list[_InterleavedBoundary] = field(default_factory=list)


class InterleavedPipelineWorkloadBuilder(WorkloadBuilder):
    """Build ``pp * vpp`` logical stages without a post-build overlay."""

    def __init__(
        self,
        virtual_pipeline_size: int = 2,
        task_info: dict[int, InterleavedPipelineTaskInfo] | None = None,
    ):
        super().__init__()
        if virtual_pipeline_size < 2:
            raise ValueError(
                "virtual_pipeline_size must be >= 2, "
                f"got {virtual_pipeline_size}"
            )
        self.virtual_pipeline_size = virtual_pipeline_size
        self.task_info = task_info if task_info is not None else {}
        self._active_job: Job | None = None
        self._active_grouper: RankGrouper | None = None
        self._active_ga = 0
        self._items_per_ga = 0
        self._layer_to_chunk: dict[int, int] = {}
        self._chunk_layers: dict[int, tuple[int, ...]] = {}

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        if is_zero_workload(aicb_items):
            raise ValueError(
                "interleaved_1f1b direct expansion does not support the "
                "DeepSpeed ZeRO/FSDP AICB row format"
            )
        if aicb_header.ga < 1:
            raise ValueError("interleaved_1f1b requires ga >= 1")

        num_pre = self._count_pre_items(aicb_items)
        num_post = self._count_post_items(aicb_items)
        num_layers = len(aicb_items) - num_pre - num_post
        if num_layers <= 0 or num_layers % aicb_header.ga:
            raise ValueError(
                "interleaved_1f1b requires a positive layer count divisible "
                f"by ga: layers={num_layers}, ga={aicb_header.ga}"
            )
        items_per_ga = num_layers // aicb_header.ga
        if items_per_ga < self.virtual_pipeline_size:
            raise ValueError(
                "interleaved_1f1b requires at least one local layer per "
                f"chunk: layers={items_per_ga}, vpp={self.virtual_pipeline_size}"
            )

        self._active_job = job
        self._active_grouper = RankGrouper(job.assigned_nodes, job.parallelism)
        if self._active_grouper.pp > 1 and aicb_header.pp_comm_size <= 0:
            raise ValueError(
                "interleaved_1f1b requires a positive pp_comm_size when pp > 1"
            )
        self._active_ga = aicb_header.ga
        self._items_per_ga = items_per_ga
        self._layer_to_chunk = {
            layer: min(
                layer * self.virtual_pipeline_size // items_per_ga,
                self.virtual_pipeline_size - 1,
            )
            for layer in range(items_per_ga)
        }
        self._chunk_layers = {
            chunk: tuple(
                layer for layer in range(items_per_ga)
                if self._layer_to_chunk[layer] == chunk
            )
            for chunk in range(self.virtual_pipeline_size)
        }

        workload = super().build_from_aicb(
            aicb_header, aicb_items, job, comm_algo,
        )
        self._record_compute_info(workload)
        return workload

    def _wire_dependencies(
        self,
        item_tasks_list: list[ItemTasks],
        num_layer_items: int,
        num_pre_items: int,
        items_per_ga: int,
    ):
        """Wire local dependencies within chunks; PP owns chunk boundaries."""
        ga_groups = self._group_items_by_ga(
            item_tasks_list, num_pre_items, num_layer_items, items_per_ga,
        )
        post_start = num_pre_items + num_layer_items
        pre_items = item_tasks_list[:num_pre_items]
        post_items = item_tasks_list[post_start:]

        self._wire_forward_chain(pre_items)
        self._wire_backward_chain(pre_items)
        self._wire_forward_chain(post_items)
        self._wire_backward_chain(post_items)

        for ga_group in ga_groups:
            for chunk in range(self.virtual_pipeline_size):
                chunk_items = [
                    ga_group[layer] for layer in self._chunk_layers[chunk]
                ]
                self._wire_forward_chain(chunk_items)
                self._wire_backward_chain(chunk_items)
                last_item = chunk_items[-1]
                self._wire_per_node_phase_transition(
                    src_result=last_item.fwd_result,
                    src_computes=last_item.fwd_computes,
                    dst_computes=last_item.ig_computes,
                )

            # With pp=1 the logical chunk boundary is local, and the base
            # build path intentionally does not invoke PP generation.
            if self._require_grouper().pp == 1:
                for chunk in range(self.virtual_pipeline_size - 1):
                    source = ga_group[self._chunk_layers[chunk][-1]]
                    destination = ga_group[self._chunk_layers[chunk + 1][0]]
                    self._wire_per_node_phase_transition(
                        source.fwd_result,
                        source.fwd_computes,
                        destination.fwd_computes,
                    )
                    self._wire_per_node_phase_transition(
                        destination.ig_result,
                        destination.ig_computes,
                        source.ig_computes,
                    )

        if post_items:
            last_post = post_items[-1]
            self._wire_per_node_phase_transition(
                last_post.fwd_result,
                last_post.fwd_computes,
                last_post.ig_computes,
            )

        if not ga_groups:
            if pre_items and post_items:
                self._wire_per_node_phase_transition(
                    pre_items[-1].fwd_result,
                    pre_items[-1].fwd_computes,
                    post_items[0].fwd_computes,
                )
                self._wire_per_node_phase_transition(
                    post_items[0].ig_result,
                    post_items[0].ig_computes,
                    pre_items[-1].ig_computes,
                )
            return

        if pre_items:
            for ga_group in ga_groups:
                first = ga_group[self._chunk_layers[0][0]]
                self._wire_per_node_phase_transition(
                    pre_items[-1].fwd_result,
                    pre_items[-1].fwd_computes,
                    first.fwd_computes,
                )
        if post_items:
            for ga_group in ga_groups:
                last = ga_group[
                    self._chunk_layers[self.virtual_pipeline_size - 1][-1]
                ]
                self._wire_per_node_phase_transition(
                    last.fwd_result,
                    last.fwd_computes,
                    post_items[0].fwd_computes,
                )
        if pre_items:
            for ga_group in ga_groups:
                first = ga_group[self._chunk_layers[0][0]]
                self._wire_per_node_phase_transition(
                    first.ig_result,
                    first.ig_computes,
                    pre_items[-1].ig_computes,
                )

    def _generate_pp_flows(
        self,
        grouper: RankGrouper,
        aicb_header: AicbHeader,
        ga_groups: list[list[ItemTasks]],
        items_per_ga: int,
        job_id: int,
        task_id_counter: int,
    ) -> tuple[InterleavedPPFlowResult, int]:
        del items_per_ga
        result = InterleavedPPFlowResult()
        logical_count = grouper.pp * self.virtual_pipeline_size

        for microbatch_id, ga_group in enumerate(ga_groups):
            for logical_boundary in range(logical_count - 1):
                source_logical = logical_boundary
                destination_logical = logical_boundary + 1
                source_chunk, source_stage = divmod(
                    source_logical, grouper.pp,
                )
                destination_chunk, destination_stage = divmod(
                    destination_logical, grouper.pp,
                )
                source_layer = self._chunk_layers[source_chunk][-1]
                destination_layer = self._chunk_layers[destination_chunk][0]

                for dp_idx in range(grouper.dp):
                    for ep_idx in range(grouper.ep):
                        for tp_idx in range(grouper.tp):
                            src = grouper.get_pp_rank(
                                source_stage, dp_idx, ep_idx, tp_idx,
                            )
                            dst = grouper.get_pp_rank(
                                destination_stage, dp_idx, ep_idx, tp_idx,
                            )
                            fwd = FlowTask(
                                task_id=task_id_counter,
                                job_id=job_id,
                                type=TaskType.FLOW,
                                src=src,
                                dst=dst,
                                size_bytes=aicb_header.pp_comm_size,
                                comm_type=CommType.PP_SEND,
                                phase=Phase.FORWARD,
                                layer_id=source_layer,
                                iteration=microbatch_id,
                                item_id=ga_group[
                                    source_layer
                                ].fwd_computes[src].item_id,
                            )
                            task_id_counter += 1
                            bwd = FlowTask(
                                task_id=task_id_counter,
                                job_id=job_id,
                                type=TaskType.FLOW,
                                src=dst,
                                dst=src,
                                size_bytes=aicb_header.pp_comm_size,
                                comm_type=CommType.PP_SEND,
                                phase=Phase.BACKWARD_INPUT,
                                layer_id=destination_layer,
                                iteration=microbatch_id,
                                item_id=ga_group[
                                    destination_layer
                                ].ig_computes[dst].item_id,
                            )
                            task_id_counter += 1
                            result.all_flows.extend((fwd, bwd))
                            result.boundaries.extend((
                                _InterleavedBoundary(
                                    fwd, microbatch_id,
                                    source_chunk, destination_chunk,
                                    source_stage, destination_stage,
                                    source_layer, destination_layer,
                                    logical_boundary, "forward",
                                ),
                                _InterleavedBoundary(
                                    bwd, microbatch_id,
                                    destination_chunk, source_chunk,
                                    destination_stage, source_stage,
                                    destination_layer, source_layer,
                                    logical_boundary, "backward",
                                ),
                            ))
        return result, task_id_counter

    def _wire_pp_dependencies(
        self,
        pp_result: InterleavedPPFlowResult,
        ga_groups: list[list[ItemTasks]],
        grouper: RankGrouper,
    ):
        del grouper
        for boundary in pp_result.boundaries:
            ga_group = ga_groups[boundary.microbatch_id]
            if boundary.direction == "forward":
                source = ga_group[boundary.source_layer]
                destination = ga_group[boundary.destination_layer]
                self._wire_to_flow_sender(
                    boundary.flow,
                    source.fwd_computes,
                    source.fwd_result,
                    boundary.flow.src,
                )
                destination.fwd_computes[
                    boundary.flow.dst
                ].deps.append(boundary.flow.task_id)
                role = "pp_activation"
            else:
                source = ga_group[boundary.source_layer]
                destination = ga_group[boundary.destination_layer]
                self._wire_to_flow_sender(
                    boundary.flow,
                    source.ig_computes,
                    source.ig_result,
                    boundary.flow.src,
                )
                destination.ig_computes[
                    boundary.flow.dst
                ].deps.append(boundary.flow.task_id)
                role = "pp_gradient"

            self.task_info[boundary.flow.task_id] = (
                InterleavedPipelineTaskInfo(
                    task_id=boundary.flow.task_id,
                    job_id=boundary.flow.job_id,
                    task_role=role,
                    microbatch_id=boundary.microbatch_id,
                    model_chunk_id=boundary.source_chunk,
                    physical_stage_id=boundary.source_stage,
                    logical_stage_id=(
                        boundary.source_chunk * self._require_grouper().pp
                        + boundary.source_stage
                    ),
                    logical_boundary_id=boundary.logical_boundary,
                    peer_physical_stage_id=boundary.destination_stage,
                    peer_logical_stage_id=(
                        boundary.destination_chunk * self._require_grouper().pp
                        + boundary.destination_stage
                    ),
                    direction=boundary.direction,
                )
            )

    def _record_compute_info(self, workload: P2PWorkload) -> None:
        grouper = self._require_grouper()
        stage_size = grouper.dp * grouper.ep * grouper.tp
        node_to_stage = {
            node: index // stage_size
            for index, node in enumerate(grouper.nodes)
        }
        for task in workload.tasks:
            if (
                not task.is_compute()
                or not 0 <= task.iteration < self._active_ga
                or task.phase not in (
                    Phase.FORWARD,
                    Phase.BACKWARD_INPUT,
                    Phase.BACKWARD_WEIGHT,
                )
            ):
                continue
            chunk = self._layer_to_chunk[task.layer_id]
            stage = node_to_stage[task.node]
            role = {
                Phase.FORWARD: "compute_forward",
                Phase.BACKWARD_INPUT: "compute_backward_input",
                Phase.BACKWARD_WEIGHT: "compute_backward_weight",
            }[task.phase]
            self.task_info[task.task_id] = InterleavedPipelineTaskInfo(
                task_id=task.task_id,
                job_id=task.job_id,
                task_role=role,
                microbatch_id=task.iteration,
                model_chunk_id=chunk,
                physical_stage_id=stage,
                logical_stage_id=chunk * grouper.pp + stage,
            )

    def _require_grouper(self) -> RankGrouper:
        if self._active_grouper is None:
            raise RuntimeError("Interleaved builder has no active job")
        return self._active_grouper

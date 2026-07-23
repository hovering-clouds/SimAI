"""Strategy-owned workload overlays for more accurate pipeline DAGs.

The common :mod:`workload_builder` intentionally emits one physical pipeline
chain.  Advanced schedules that need a different pipeline graph operate on a
deep copy here, keeping the common ``Task`` schema and the caller's workload
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass

from ..workload_format.schema import (
    CommType,
    P2PWorkload,
    Phase,
    Task,
    TaskType,
)
from .rank_grouper import RankGrouper


_TRAINING_PHASES = (Phase.FORWARD, Phase.BACKWARD_INPUT)
_PP_TYPES = (CommType.PP_SEND, CommType.PP_RECV)


@dataclass(frozen=True)
class InterleavedPipelineTaskInfo:
    """Interleaved-only metadata kept outside the common ``Task`` schema."""

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
class BidirectionalPipelineTaskInfo:
    """Bidirectional-only metadata kept outside the common ``Task`` schema."""

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
class PipelineWorkloadOverlayResult:
    """A transformed workload and its strategy-owned audit information."""

    workload: P2PWorkload
    task_info: dict[
        int, InterleavedPipelineTaskInfo | BidirectionalPipelineTaskInfo
    ]
    removed_task_ids: tuple[int, ...]
    added_task_ids: tuple[int, ...]


@dataclass(frozen=True)
class _ChunkLayout:
    layers: tuple[int, ...]
    layer_to_chunk: dict[int, int]
    first_layer: dict[int, int]
    last_layer: dict[int, int]


@dataclass(frozen=True)
class _FlowSpec:
    job_id: int
    iteration: int
    phase: Phase
    src: int
    dst: int
    size_bytes: int
    layer_id: int
    item_id: int
    deps: tuple[int, ...]
    consumer_id: int
    chunk_id: int
    physical_stage_id: int
    logical_stage_id: int
    boundary_id: int
    peer_physical_stage_id: int
    peer_logical_stage_id: int
    direction: str


class InterleavedOneFOneBWorkloadOverlay:
    """Replace a physical PP chain with a virtual-pipeline task DAG.

    Existing compute and non-PP communication tasks are retained.  For every
    microbatch and ``(dp, ep, tp)`` lane, this overlay builds ``pp * vpp``
    logical stages ordered by ``chunk_id * pp + physical_stage``.  A forward
    boundary is:

    ``source chunk output -> PP_SEND -> destination chunk input compute``.

    Backward uses the exact reverse boundary.  When both logical stages live
    on the same rank (``pp == 1``), the communication task is omitted and the
    producer is connected directly to the consumer.

    ``size_bytes`` is copied from the original physical ``PP_SEND`` tasks.  A
    multi-stage workload without such tasks is rejected because inventing a
    message size would change network results silently.
    """

    def __init__(
        self,
        virtual_pipeline_size: int = 2,
        allocate_task_ids: Callable[[int], list[int]] | None = None,
    ):
        if virtual_pipeline_size < 2:
            raise ValueError(
                "virtual_pipeline_size must be >= 2, "
                f"got {virtual_pipeline_size}"
            )
        self.virtual_pipeline_size = virtual_pipeline_size
        self._allocate_task_ids = allocate_task_ids

    def apply(self, workload: P2PWorkload) -> PipelineWorkloadOverlayResult:
        """Return an interleaved DAG copy without mutating ``workload``."""
        transformed = deepcopy(workload)
        jobs_by_id = {job.job_id: job for job in transformed.jobs}
        task_job_ids = {task.job_id for task in transformed.tasks}
        missing_jobs = sorted(task_job_ids - jobs_by_id.keys())
        if missing_jobs:
            raise ValueError(
                "Interleaved workload overlay requires Job metadata for task "
                f"job IDs: {missing_jobs}"
            )

        old_pp_tasks = [
            task for task in transformed.tasks
            if task.comm_type in _PP_TYPES and task.job_id in task_job_ids
        ]
        removed_ids = {task.task_id for task in old_pp_tasks}
        pp_size_by_job = self._infer_pp_message_sizes(old_pp_tasks, jobs_by_id)

        transformed.tasks = [
            task for task in transformed.tasks if task.task_id not in removed_ids
        ]
        for task in transformed.tasks:
            task.deps = [dep for dep in task.deps if dep not in removed_ids]

        task_info: dict[int, InterleavedPipelineTaskInfo] = {}
        reusable_ids = iter(sorted(removed_ids))
        pending_specs: list[_FlowSpec] = []
        direct_edges: list[tuple[int, tuple[int, ...]]] = []

        for job_id in sorted(task_job_ids):
            job = jobs_by_id[job_id]
            grouper = RankGrouper(job.assigned_nodes, job.parallelism)
            job_tasks = [task for task in transformed.tasks if task.job_id == job_id]
            iterations = self._training_iterations(job_tasks)
            if not iterations:
                continue
            layout = self._build_chunk_layout(job_tasks, iterations)
            compute_index = self._build_compute_index(job_tasks)
            flow_index = self._build_non_pp_flow_index(job_tasks)
            node_to_stage = self._node_to_stage(grouper)
            self._record_compute_info(
                job_tasks, iterations, layout, node_to_stage, grouper.pp, task_info,
            )
            self._remove_local_cross_chunk_edges(
                job_id, job.assigned_nodes, iterations, layout,
                compute_index, flow_index,
            )

            size_bytes = pp_size_by_job.get(job_id, 0)
            if grouper.pp > 1 and size_bytes <= 0:
                raise ValueError(
                    f"Job {job_id} uses pp={grouper.pp} but has no positive-size "
                    "PP_SEND task from which the VPP message size can be inferred"
                )

            for iteration in iterations:
                for chunk_id in range(self.virtual_pipeline_size):
                    for stage_id in range(grouper.pp):
                        logical_stage = chunk_id * grouper.pp + stage_id
                        if logical_stage == grouper.pp * self.virtual_pipeline_size - 1:
                            continue
                        next_logical_stage = logical_stage + 1
                        next_chunk = next_logical_stage // grouper.pp
                        next_stage = next_logical_stage % grouper.pp
                        for dp_idx in range(grouper.dp):
                            for ep_idx in range(grouper.ep):
                                for tp_idx in range(grouper.tp):
                                    src = grouper.get_pp_rank(
                                        stage_id, dp_idx, ep_idx, tp_idx,
                                    )
                                    dst = grouper.get_pp_rank(
                                        next_stage, dp_idx, ep_idx, tp_idx,
                                    )
                                    self._append_boundary_specs(
                                        pending_specs,
                                        direct_edges,
                                        job_id=job_id,
                                        iteration=iteration,
                                        src=src,
                                        dst=dst,
                                        size_bytes=size_bytes,
                                        chunk_id=chunk_id,
                                        next_chunk=next_chunk,
                                        stage_id=stage_id,
                                        next_stage=next_stage,
                                        logical_stage=logical_stage,
                                        next_logical_stage=next_logical_stage,
                                        layout=layout,
                                        compute_index=compute_index,
                                        flow_index=flow_index,
                                    )

        task_by_id = {task.task_id: task for task in transformed.tasks}
        for consumer_id, dependencies in direct_edges:
            consumer = task_by_id[consumer_id]
            self._replace_or_extend(consumer.deps, dependencies)

        new_ids = self._assign_flow_ids(
            len(pending_specs), reusable_ids, transformed.tasks,
        )
        new_tasks: list[Task] = []
        for task_id, spec in zip(new_ids, pending_specs, strict=True):
            flow = Task(
                task_id=task_id,
                job_id=spec.job_id,
                type=TaskType.FLOW,
                iteration=spec.iteration,
                phase=spec.phase,
                layer_id=spec.layer_id,
                item_id=spec.item_id,
                deps=list(spec.deps),
                src=spec.src,
                dst=spec.dst,
                size_bytes=spec.size_bytes,
                comm_type=CommType.PP_SEND,
            )
            new_tasks.append(flow)
            task_by_id[spec.consumer_id].deps.append(task_id)
            task_info[task_id] = InterleavedPipelineTaskInfo(
                task_id=task_id,
                job_id=spec.job_id,
                task_role=(
                    "pp_activation" if spec.phase is Phase.FORWARD
                    else "pp_gradient"
                ),
                microbatch_id=spec.iteration,
                model_chunk_id=spec.chunk_id,
                physical_stage_id=spec.physical_stage_id,
                logical_stage_id=spec.logical_stage_id,
                logical_boundary_id=spec.boundary_id,
                peer_physical_stage_id=spec.peer_physical_stage_id,
                peer_logical_stage_id=spec.peer_logical_stage_id,
                direction=spec.direction,
            )

        transformed.tasks.extend(new_tasks)
        errors = transformed.validate()
        if errors:
            raise ValueError(f"Interleaved workload overlay validation failed: {errors}")
        self._validate_new_flow_causality(transformed, new_tasks)
        return PipelineWorkloadOverlayResult(
            workload=transformed,
            task_info=task_info,
            removed_task_ids=tuple(sorted(removed_ids)),
            added_task_ids=tuple(new_ids),
        )

    def _build_chunk_layout(
        self,
        tasks: list[Task],
        iterations: list[int],
    ) -> _ChunkLayout:
        iteration_set = set(iterations)
        layers = tuple(sorted({
            task.layer_id for task in tasks
            if task.iteration in iteration_set
            and task.phase in _TRAINING_PHASES
            and task.is_compute()
        }))
        if len(layers) < self.virtual_pipeline_size:
            raise ValueError(
                "interleaved_1f1b requires at least one local layer per virtual "
                f"chunk: layers={len(layers)}, vpp={self.virtual_pipeline_size}"
            )
        layer_to_chunk = {
            layer_id: min(
                index * self.virtual_pipeline_size // len(layers),
                self.virtual_pipeline_size - 1,
            )
            for index, layer_id in enumerate(layers)
        }
        by_chunk = {
            chunk_id: tuple(
                layer for layer in layers if layer_to_chunk[layer] == chunk_id
            )
            for chunk_id in range(self.virtual_pipeline_size)
        }
        if any(not chunk_layers for chunk_layers in by_chunk.values()):
            raise ValueError("Every virtual pipeline chunk must own at least one layer")
        return _ChunkLayout(
            layers=layers,
            layer_to_chunk=layer_to_chunk,
            first_layer={chunk: values[0] for chunk, values in by_chunk.items()},
            last_layer={chunk: values[-1] for chunk, values in by_chunk.items()},
        )

    @staticmethod
    def _training_iterations(tasks: list[Task]) -> list[int]:
        """Infer GA microbatches while excluding zero-duration post rows."""
        max_iteration = max(
            (task.iteration for task in tasks if task.iteration >= 0),
            default=-1,
        )
        if max_iteration < 0:
            return []
        has_post = any(
            task.is_compute()
            and task.iteration == max_iteration
            and task.phase is Phase.FORWARD
            and task.duration_us == 0
            for task in tasks
        )
        ga = max_iteration if has_post else max_iteration + 1
        present = {
            task.iteration for task in tasks
            if task.is_compute()
            and task.phase in _TRAINING_PHASES
            and 0 <= task.iteration < ga
        }
        return sorted(present)

    @staticmethod
    def _infer_pp_message_sizes(old_pp_tasks, jobs_by_id) -> dict[int, int]:
        result: dict[int, int] = {}
        for job_id, job in jobs_by_id.items():
            if job.parallelism.pp <= 1:
                result[job_id] = 0
                continue
            sizes = {
                task.size_bytes for task in old_pp_tasks
                if task.job_id == job_id
                and task.comm_type is CommType.PP_SEND
                and task.size_bytes is not None
            }
            if len(sizes) > 1:
                raise ValueError(
                    f"Job {job_id} has inconsistent PP message sizes: {sorted(sizes)}"
                )
            if sizes:
                result[job_id] = sizes.pop()
        return result

    @staticmethod
    def _build_compute_index(tasks: list[Task]) -> dict[tuple, Task]:
        index: dict[tuple, Task] = {}
        for task in tasks:
            if not task.is_compute() or task.phase not in _TRAINING_PHASES:
                continue
            key = (task.job_id, task.node, task.iteration, task.phase, task.layer_id)
            if key in index:
                raise ValueError(f"Multiple compute tasks share pipeline key {key}")
            index[key] = task
        return index

    @staticmethod
    def _build_non_pp_flow_index(tasks: list[Task]) -> dict[tuple, list[Task]]:
        index: dict[tuple, list[Task]] = {}
        for task in tasks:
            if not task.is_flow() or task.comm_type in _PP_TYPES:
                continue
            key = (task.job_id, task.iteration, task.phase, task.layer_id)
            index.setdefault(key, []).append(task)
        return index

    @staticmethod
    def _node_to_stage(grouper: RankGrouper) -> dict[int, int]:
        stage_size = grouper.dp * grouper.ep * grouper.tp
        return {
            node: index // stage_size for index, node in enumerate(grouper.nodes)
        }

    def _record_compute_info(
        self,
        tasks: list[Task],
        iterations: list[int],
        layout: _ChunkLayout,
        node_to_stage: dict[int, int],
        pp: int,
        task_info: dict[int, InterleavedPipelineTaskInfo],
    ) -> None:
        iteration_set = set(iterations)
        for task in tasks:
            if (
                not task.is_compute()
                or task.iteration not in iteration_set
                or task.phase not in _TRAINING_PHASES
            ):
                continue
            chunk = layout.layer_to_chunk[task.layer_id]
            stage = node_to_stage[task.node]
            task_info[task.task_id] = InterleavedPipelineTaskInfo(
                task_id=task.task_id,
                job_id=task.job_id,
                task_role=(
                    "compute_forward" if task.phase is Phase.FORWARD
                    else "compute_backward"
                ),
                microbatch_id=task.iteration,
                model_chunk_id=chunk,
                physical_stage_id=stage,
                logical_stage_id=chunk * pp + stage,
            )

    def _phase_output_ids(
        self,
        compute_index: dict[tuple, Task],
        flow_index: dict[tuple, list[Task]],
        job_id: int,
        rank: int,
        iteration: int,
        phase: Phase,
        layer_id: int,
    ) -> tuple[int, ...]:
        flows = flow_index.get((job_id, iteration, phase, layer_id), [])
        if flows:
            incoming = sorted(task.task_id for task in flows if task.dst == rank)
            if incoming:
                return tuple(incoming)
            outgoing = sorted(task.task_id for task in flows if task.src == rank)
            if outgoing:
                return tuple(outgoing)
        compute = compute_index.get((job_id, rank, iteration, phase, layer_id))
        if compute is None:
            raise ValueError(
                "Missing pipeline boundary compute: "
                f"job={job_id}, rank={rank}, iteration={iteration}, "
                f"phase={phase.value}, layer={layer_id}"
            )
        return (compute.task_id,)

    def _remove_local_cross_chunk_edges(
        self,
        job_id: int,
        ranks: list[int],
        iterations: list[int],
        layout: _ChunkLayout,
        compute_index: dict[tuple, Task],
        flow_index: dict[tuple, list[Task]],
    ) -> None:
        for iteration in iterations:
            for upper_chunk in range(1, self.virtual_pipeline_size):
                lower_chunk = upper_chunk - 1
                for rank in ranks:
                    fwd_consumer = compute_index[
                        (job_id, rank, iteration, Phase.FORWARD,
                         layout.first_layer[upper_chunk])
                    ]
                    fwd_local_output = set(self._phase_output_ids(
                        compute_index, flow_index, job_id, rank, iteration,
                        Phase.FORWARD, layout.last_layer[lower_chunk],
                    ))
                    fwd_consumer.deps = [
                        dep for dep in fwd_consumer.deps if dep not in fwd_local_output
                    ]

                    bwd_consumer = compute_index[
                        (job_id, rank, iteration, Phase.BACKWARD_INPUT,
                         layout.last_layer[lower_chunk])
                    ]
                    bwd_local_output = set(self._phase_output_ids(
                        compute_index, flow_index, job_id, rank, iteration,
                        Phase.BACKWARD_INPUT, layout.first_layer[upper_chunk],
                    ))
                    bwd_consumer.deps = [
                        dep for dep in bwd_consumer.deps if dep not in bwd_local_output
                    ]

    def _append_boundary_specs(
        self,
        specs: list[_FlowSpec],
        direct_edges: list[tuple[int, tuple[int, ...]]],
        *,
        job_id: int,
        iteration: int,
        src: int,
        dst: int,
        size_bytes: int,
        chunk_id: int,
        next_chunk: int,
        stage_id: int,
        next_stage: int,
        logical_stage: int,
        next_logical_stage: int,
        layout: _ChunkLayout,
        compute_index: dict[tuple, Task],
        flow_index: dict[tuple, list[Task]],
    ) -> None:
        fwd_source_layer = layout.last_layer[chunk_id]
        fwd_consumer_layer = layout.first_layer[next_chunk]
        fwd_deps = self._phase_output_ids(
            compute_index, flow_index, job_id, src, iteration,
            Phase.FORWARD, fwd_source_layer,
        )
        fwd_consumer = compute_index[
            (job_id, dst, iteration, Phase.FORWARD, fwd_consumer_layer)
        ]

        bwd_source_layer = layout.first_layer[next_chunk]
        bwd_consumer_layer = layout.last_layer[chunk_id]
        bwd_deps = self._phase_output_ids(
            compute_index, flow_index, job_id, dst, iteration,
            Phase.BACKWARD_INPUT, bwd_source_layer,
        )
        bwd_consumer = compute_index[
            (job_id, src, iteration, Phase.BACKWARD_INPUT, bwd_consumer_layer)
        ]

        if src == dst:
            direct_edges.append((fwd_consumer.task_id, fwd_deps))
            direct_edges.append((bwd_consumer.task_id, bwd_deps))
            return

        specs.append(_FlowSpec(
            job_id=job_id,
            iteration=iteration,
            phase=Phase.FORWARD,
            src=src,
            dst=dst,
            size_bytes=size_bytes,
            layer_id=fwd_source_layer,
            item_id=compute_index[
                (job_id, src, iteration, Phase.FORWARD, fwd_source_layer)
            ].item_id,
            deps=fwd_deps,
            consumer_id=fwd_consumer.task_id,
            chunk_id=chunk_id,
            physical_stage_id=stage_id,
            logical_stage_id=logical_stage,
            boundary_id=logical_stage,
            peer_physical_stage_id=next_stage,
            peer_logical_stage_id=next_logical_stage,
            direction="forward",
        ))
        specs.append(_FlowSpec(
            job_id=job_id,
            iteration=iteration,
            phase=Phase.BACKWARD_INPUT,
            src=dst,
            dst=src,
            size_bytes=size_bytes,
            layer_id=bwd_source_layer,
            item_id=compute_index[
                (job_id, dst, iteration, Phase.BACKWARD_INPUT, bwd_source_layer)
            ].item_id,
            deps=bwd_deps,
            consumer_id=bwd_consumer.task_id,
            chunk_id=next_chunk,
            physical_stage_id=next_stage,
            logical_stage_id=next_logical_stage,
            boundary_id=logical_stage,
            peer_physical_stage_id=stage_id,
            peer_logical_stage_id=logical_stage,
            direction="backward",
        ))

    def _assign_flow_ids(
        self,
        count: int,
        reusable_ids,
        existing_tasks: list[Task],
    ) -> list[int]:
        ids: list[int] = []
        while len(ids) < count:
            try:
                ids.append(next(reusable_ids))
            except StopIteration:
                break
        remaining = count - len(ids)
        if remaining == 0:
            return ids
        if self._allocate_task_ids is not None:
            allocated = self._allocate_task_ids(remaining)
            if len(allocated) != remaining:
                raise ValueError(
                    f"Task ID allocator returned {len(allocated)} IDs for {remaining} tasks"
                )
            ids.extend(allocated)
            return ids
        next_id = max((task.task_id for task in existing_tasks), default=-1) + 1
        used = {task.task_id for task in existing_tasks} | set(ids)
        while len(ids) < count:
            if next_id not in used:
                ids.append(next_id)
                used.add(next_id)
            next_id += 1
        return ids

    @staticmethod
    def _replace_or_extend(target: list[int], dependencies: tuple[int, ...]) -> None:
        for dependency in dependencies:
            if dependency not in target:
                target.append(dependency)

    @staticmethod
    def _validate_new_flow_causality(
        workload: P2PWorkload,
        new_flows: list[Task],
    ) -> None:
        dependents: dict[int, list[Task]] = {}
        task_ids = {task.task_id for task in workload.tasks}
        for task in workload.tasks:
            for dependency in task.deps:
                dependents.setdefault(dependency, []).append(task)
        for flow in new_flows:
            if not flow.deps:
                raise ValueError(f"PP flow {flow.task_id} has no compute/communication producer")
            if any(dependency not in task_ids for dependency in flow.deps):
                raise ValueError(f"PP flow {flow.task_id} has a missing producer")
            consumers = dependents.get(flow.task_id, [])
            if not any(task.is_compute() for task in consumers):
                raise ValueError(f"PP flow {flow.task_id} does not unlock a compute task")


class BidirectionalPipelineWorkloadOverlay(InterleavedOneFOneBWorkloadOverlay):
    """Replace a one-way physical PP DAG with two opposing pipelines.

    The first half of training microbatches traverses physical stages from
    ``0`` to ``pp - 1`` (``down``); the second half traverses them from
    ``pp - 1`` to ``0`` (``up``).  Backward-input gradients follow the exact
    reverse path.  Existing compute and non-PP communication tasks are kept,
    so each physical stage's tasks project the module replica selected by the
    microbatch direction.

    PP message size is inferred from the original PP flow tasks.  The common
    task schema remains unchanged; replica ownership and direction are stored
    in :class:`BidirectionalPipelineTaskInfo`.
    """

    def __init__(
        self,
        allocate_task_ids: Callable[[int], list[int]] | None = None,
    ):
        # Deliberately bypass the interleaved VPP constructor.  This class
        # inherits only its strategy-neutral indexing/validation helpers.
        self._allocate_task_ids = allocate_task_ids

    def apply(self, workload: P2PWorkload) -> PipelineWorkloadOverlayResult:
        """Return a bidirectional DAG copy without mutating ``workload``."""
        transformed = deepcopy(workload)
        jobs_by_id = {job.job_id: job for job in transformed.jobs}
        training_job_ids = {
            task.job_id for task in transformed.tasks
            if task.is_compute() and task.phase in _TRAINING_PHASES
        }
        missing_jobs = sorted(training_job_ids - jobs_by_id.keys())
        if missing_jobs:
            raise ValueError(
                "Bidirectional workload overlay requires Job metadata for task "
                f"job IDs: {missing_jobs}"
            )

        old_pp_tasks = [
            task for task in transformed.tasks
            if task.comm_type in _PP_TYPES and task.job_id in training_job_ids
        ]
        removed_ids = {task.task_id for task in old_pp_tasks}
        target_jobs = {
            job_id: jobs_by_id[job_id] for job_id in training_job_ids
        }
        pp_size_by_job = self._infer_pp_message_sizes(old_pp_tasks, target_jobs)
        transformed.tasks = [
            task for task in transformed.tasks if task.task_id not in removed_ids
        ]
        for task in transformed.tasks:
            task.deps = [dep for dep in task.deps if dep not in removed_ids]

        task_info: dict[int, BidirectionalPipelineTaskInfo] = {}
        reusable_ids = iter(sorted(removed_ids))
        pending_specs: list[_FlowSpec] = []

        for job_id in sorted(training_job_ids):
            job = jobs_by_id[job_id]
            grouper = RankGrouper(job.assigned_nodes, job.parallelism)
            job_tasks = [task for task in transformed.tasks if task.job_id == job_id]
            iterations = self._training_iterations(job_tasks)
            if not iterations:
                continue
            ga = len(iterations)
            self._validate_shape(grouper.pp, ga, job_id)
            if iterations != list(range(ga)):
                raise ValueError(
                    f"Job {job_id} has non-contiguous training microbatches: {iterations}"
                )

            size_bytes = pp_size_by_job.get(job_id, 0)
            if size_bytes <= 0:
                raise ValueError(
                    f"Job {job_id} uses bidirectional pp={grouper.pp} but has no "
                    "positive-size PP_SEND task from which message size can be inferred"
                )

            compute_index = self._build_compute_index(job_tasks)
            flow_index = self._build_non_pp_flow_index(job_tasks)
            first_layer, last_layer = self._boundary_layers(job_tasks, iterations)
            node_to_stage = self._node_to_stage(grouper)
            half_ga = ga // 2
            self._record_bidirectional_compute_info(
                job_tasks, iterations, half_ga, node_to_stage, grouper.pp, task_info,
            )

            for iteration in iterations:
                pipeline_id = 0 if iteration < half_ga else 1
                direction = "down" if pipeline_id == 0 else "up"
                for logical_boundary in range(grouper.pp - 1):
                    if pipeline_id == 0:
                        src_stage = logical_boundary
                        dst_stage = logical_boundary + 1
                    else:
                        src_stage = grouper.pp - 1 - logical_boundary
                        dst_stage = grouper.pp - 2 - logical_boundary
                    for dp_idx in range(grouper.dp):
                        for ep_idx in range(grouper.ep):
                            for tp_idx in range(grouper.tp):
                                src = grouper.get_pp_rank(
                                    src_stage, dp_idx, ep_idx, tp_idx,
                                )
                                dst = grouper.get_pp_rank(
                                    dst_stage, dp_idx, ep_idx, tp_idx,
                                )
                                self._append_bidirectional_boundary_specs(
                                    pending_specs,
                                    job_id=job_id,
                                    iteration=iteration,
                                    pipeline_id=pipeline_id,
                                    direction=direction,
                                    src=src,
                                    dst=dst,
                                    src_stage=src_stage,
                                    dst_stage=dst_stage,
                                    logical_boundary=logical_boundary,
                                    size_bytes=size_bytes,
                                    first_layer=first_layer,
                                    last_layer=last_layer,
                                    compute_index=compute_index,
                                    flow_index=flow_index,
                                )

        task_by_id = {task.task_id: task for task in transformed.tasks}
        new_ids = self._assign_flow_ids(
            len(pending_specs), reusable_ids, transformed.tasks,
        )
        new_tasks: list[Task] = []
        for task_id, spec in zip(new_ids, pending_specs, strict=True):
            flow = Task(
                task_id=task_id,
                job_id=spec.job_id,
                type=TaskType.FLOW,
                iteration=spec.iteration,
                phase=spec.phase,
                layer_id=spec.layer_id,
                item_id=spec.item_id,
                deps=list(spec.deps),
                src=spec.src,
                dst=spec.dst,
                size_bytes=spec.size_bytes,
                comm_type=CommType.PP_SEND,
            )
            new_tasks.append(flow)
            self._replace_or_extend(task_by_id[spec.consumer_id].deps, (task_id,))
            pipeline_id = 0 if spec.direction == "down" else 1
            task_info[task_id] = BidirectionalPipelineTaskInfo(
                task_id=task_id,
                job_id=spec.job_id,
                task_role=(
                    "pp_activation" if spec.phase is Phase.FORWARD
                    else "pp_gradient"
                ),
                microbatch_id=spec.iteration,
                pipeline_id=pipeline_id,
                direction=spec.direction,
                physical_stage_id=spec.physical_stage_id,
                logical_stage_id=spec.logical_stage_id,
                logical_boundary_id=spec.boundary_id,
                peer_physical_stage_id=spec.peer_physical_stage_id,
                peer_logical_stage_id=spec.peer_logical_stage_id,
            )

        transformed.tasks.extend(new_tasks)
        errors = transformed.validate()
        if errors:
            raise ValueError(f"Bidirectional workload overlay validation failed: {errors}")
        self._validate_new_flow_causality(transformed, new_tasks)
        return PipelineWorkloadOverlayResult(
            workload=transformed,
            task_info=task_info,
            removed_task_ids=tuple(sorted(removed_ids)),
            added_task_ids=tuple(new_ids),
        )

    @staticmethod
    def _validate_shape(pp: int, ga: int, job_id: int) -> None:
        if pp % 2:
            raise ValueError(f"Job {job_id} bidirectional requires even pp, got {pp}")
        if ga % 2:
            raise ValueError(
                f"Job {job_id} bidirectional requires even microbatch count, got {ga}"
            )
        if ga < 2 * pp:
            raise ValueError(
                f"Job {job_id} bidirectional requires at least 2 * pp microbatches: "
                f"ga={ga}, pp={pp}"
            )

    @staticmethod
    def _boundary_layers(tasks: list[Task], iterations: list[int]) -> tuple[int, int]:
        iteration_set = set(iterations)
        layers = sorted({
            task.layer_id for task in tasks
            if task.is_compute()
            and task.iteration in iteration_set
            and task.phase in _TRAINING_PHASES
        })
        if not layers:
            raise ValueError("Bidirectional pipeline has no training compute layers")
        return layers[0], layers[-1]

    @staticmethod
    def _record_bidirectional_compute_info(
        tasks: list[Task],
        iterations: list[int],
        half_ga: int,
        node_to_stage: dict[int, int],
        pp: int,
        task_info: dict[int, BidirectionalPipelineTaskInfo],
    ) -> None:
        iteration_set = set(iterations)
        for task in tasks:
            if not task.is_compute() or task.iteration not in iteration_set:
                continue
            if task.phase not in (
                Phase.FORWARD, Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT,
            ):
                continue
            pipeline_id = 0 if task.iteration < half_ga else 1
            direction = "down" if pipeline_id == 0 else "up"
            stage = node_to_stage[task.node]
            logical_stage = stage if pipeline_id == 0 else pp - 1 - stage
            role = {
                Phase.FORWARD: "compute_forward",
                Phase.BACKWARD_INPUT: "compute_backward_input",
                Phase.BACKWARD_WEIGHT: "compute_backward_weight",
            }[task.phase]
            task_info[task.task_id] = BidirectionalPipelineTaskInfo(
                task_id=task.task_id,
                job_id=task.job_id,
                task_role=role,
                microbatch_id=task.iteration,
                pipeline_id=pipeline_id,
                direction=direction,
                physical_stage_id=stage,
                logical_stage_id=logical_stage,
            )

    def _append_bidirectional_boundary_specs(
        self,
        specs: list[_FlowSpec],
        *,
        job_id: int,
        iteration: int,
        pipeline_id: int,
        direction: str,
        src: int,
        dst: int,
        src_stage: int,
        dst_stage: int,
        logical_boundary: int,
        size_bytes: int,
        first_layer: int,
        last_layer: int,
        compute_index: dict[tuple, Task],
        flow_index: dict[tuple, list[Task]],
    ) -> None:
        del pipeline_id  # Encoded by direction in the shared flow spec.
        fwd_deps = self._phase_output_ids(
            compute_index, flow_index, job_id, src, iteration,
            Phase.FORWARD, last_layer,
        )
        fwd_consumer = compute_index[
            (job_id, dst, iteration, Phase.FORWARD, first_layer)
        ]
        bwd_deps = self._phase_output_ids(
            compute_index, flow_index, job_id, dst, iteration,
            Phase.BACKWARD_INPUT, first_layer,
        )
        bwd_consumer = compute_index[
            (job_id, src, iteration, Phase.BACKWARD_INPUT, last_layer)
        ]

        specs.append(_FlowSpec(
            job_id=job_id,
            iteration=iteration,
            phase=Phase.FORWARD,
            src=src,
            dst=dst,
            size_bytes=size_bytes,
            layer_id=last_layer,
            item_id=compute_index[
                (job_id, src, iteration, Phase.FORWARD, last_layer)
            ].item_id,
            deps=fwd_deps,
            consumer_id=fwd_consumer.task_id,
            chunk_id=0,
            physical_stage_id=src_stage,
            logical_stage_id=logical_boundary,
            boundary_id=logical_boundary,
            peer_physical_stage_id=dst_stage,
            peer_logical_stage_id=logical_boundary + 1,
            direction=direction,
        ))
        specs.append(_FlowSpec(
            job_id=job_id,
            iteration=iteration,
            phase=Phase.BACKWARD_INPUT,
            src=dst,
            dst=src,
            size_bytes=size_bytes,
            layer_id=first_layer,
            item_id=compute_index[
                (job_id, dst, iteration, Phase.BACKWARD_INPUT, first_layer)
            ].item_id,
            deps=bwd_deps,
            consumer_id=bwd_consumer.task_id,
            chunk_id=0,
            physical_stage_id=dst_stage,
            logical_stage_id=logical_boundary + 1,
            boundary_id=logical_boundary,
            peer_physical_stage_id=src_stage,
            peer_logical_stage_id=logical_boundary,
            direction=direction,
        ))


def apply_pipeline_workload_overlay(
    pipeline_mode: str,
    workload: P2PWorkload,
    *,
    virtual_pipeline_size: int = 2,
    allocate_task_ids: Callable[[int], list[int]] | None = None,
) -> PipelineWorkloadOverlayResult:
    """Apply the registered DAG overlay, or return an untouched deep copy."""
    if pipeline_mode == "interleaved_1f1b":
        return InterleavedOneFOneBWorkloadOverlay(
            virtual_pipeline_size,
            allocate_task_ids,
        ).apply(workload)
    if pipeline_mode == "bidirectional":
        return BidirectionalPipelineWorkloadOverlay(
            allocate_task_ids,
        ).apply(workload)
    copied = deepcopy(workload)
    return PipelineWorkloadOverlayResult(copied, {}, (), ())

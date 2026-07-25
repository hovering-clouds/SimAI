"""Non-invasive compute serializers for advanced pipeline schedules.

The serializers in this module only produce ``ExecutionPlan.compute_order``.
Strategy-specific attributes live in ``task_info`` and are never written back
to the common :class:`Task` schema.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, replace
import heapq

from .task_serializer import (
    CppReferenceSerializer,
    ExecutionPlan,
    OneFOneBSerializer,
    TaskSerializer,
)
from ...workload_format.schema import P2PWorkload, Phase, Task
from ...workload_generator.builders.dualpipe_pipeline_builder import (
    build_dualpipe_schedule,
)
from ...workload_generator.rank_grouper import RankGrouper


_BACKWARD_PHASES = (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT)


@dataclass
class PipelineTaskInfo:
    """Strategy-owned scheduling metadata for one compute task."""

    task_id: int
    job_id: int
    stage_id: int
    logical_stage_id: int
    microbatch_id: int | None
    model_chunk_id: int | None
    pipeline_id: int | None
    direction: str
    operation: str
    schedule_slot: int
    local_order: int = -1
    schedule_name: str = ""


@dataclass
class PipelineScheduleResult:
    """Execution plan paired with its strategy-specific sidecar."""

    execution_plan: ExecutionPlan
    task_info: dict[int, PipelineTaskInfo]


@dataclass(frozen=True)
class _JobLayout:
    pp: int
    node_to_stage: dict[int, int]


@dataclass(frozen=True)
class _ScheduleToken:
    operation: str
    microbatch_id: int
    model_chunk_id: int = 0
    pipeline_id: int = 0
    direction: str = "down"


class AdvancedPipelineSerializer(TaskSerializer):
    """Shared sidecar, grouping, and DAG-safety logic for pipeline schedules."""

    schedule_name = "advanced"

    def __init__(
        self,
        pp: int | None = None,
        node_to_stage: dict[int, int] | None = None,
    ):
        if pp is not None and pp < 1:
            raise ValueError(f"pp must be >= 1, got {pp}")
        self._fallback_pp = pp
        self._fallback_node_to_stage = dict(node_to_stage or {})
        self.task_info: dict[int, PipelineTaskInfo] = {}

    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        """Build the preferred schedule and legalize it against the task DAG."""
        self.task_info = {}
        layouts = self._derive_layouts(workload)
        compute_by_node_job: dict[tuple[int, int], list[Task]] = {}
        for task in workload.tasks:
            if task.is_compute():
                compute_by_node_job.setdefault((task.node, task.job_id), []).append(task)

        desired_by_node: dict[int, list[int]] = {}
        used_by_node: dict[int, set[int]] = {}
        job_first_task = {
            job_id: min(task.task_id for task in workload.tasks if task.job_id == job_id)
            for job_id in {task.job_id for task in workload.tasks}
        }
        keys = sorted(
            compute_by_node_job,
            key=lambda key: (key[0], job_first_task.get(key[1], key[1])),
        )

        for node_id, job_id in keys:
            tasks = compute_by_node_job[(node_id, job_id)]
            layout = layouts[job_id]
            stage_id = layout.node_to_stage.get(node_id, 0)
            ga = self._infer_ga(tasks)
            target = self._build_target_order(tasks, layout, stage_id, ga)
            desired_by_node.setdefault(node_id, []).extend(target)
            used_by_node.setdefault(node_id, set()).update(target)

        # Defensive completion: a strategy must never omit an unfamiliar compute phase.
        for task in sorted(workload.get_compute_tasks(), key=lambda item: item.task_id):
            if task.task_id in used_by_node.setdefault(task.node, set()):
                continue
            desired_by_node.setdefault(task.node, []).append(task.task_id)
            self._record_task(
                task,
                layouts[task.job_id],
                layouts[task.job_id].node_to_stage.get(task.node, 0),
                None,
                "other",
                len(desired_by_node[task.node]) - 1,
            )

        compute_order = self._legalize_against_dag(workload, desired_by_node)
        for node_id, task_ids in compute_order.items():
            for local_order, task_id in enumerate(task_ids):
                self.task_info[task_id].local_order = local_order

        errors = self.validate(workload, compute_order)
        if errors:
            raise ValueError(f"ExecutionPlan validation failed: {errors}")
        return ExecutionPlan(compute_order=compute_order)

    def serialize_with_metadata(self, workload: P2PWorkload) -> PipelineScheduleResult:
        """Return the normal plan together with a copy of its sidecar mapping."""
        plan = self.serialize(workload)
        return PipelineScheduleResult(plan, dict(self.task_info))

    @abstractmethod
    def _schedule_tokens(
        self,
        ga: int,
        layout: _JobLayout,
        stage_id: int,
    ) -> list[_ScheduleToken]:
        """Return the preferred operation-level sequence for one physical stage."""
        ...

    def _build_target_order(
        self,
        tasks: list[Task],
        layout: _JobLayout,
        stage_id: int,
        ga: int,
    ) -> list[int]:
        chunk_by_layer = self._chunk_by_layer(tasks, ga)
        grouped: dict[tuple[int, int, str], list[Task]] = {}
        for task in tasks:
            if not 0 <= task.iteration < ga:
                continue
            operation = self._operation(task)
            if operation not in {"F", "B", "W"}:
                continue
            chunk_id = chunk_by_layer.get(task.layer_id, 0)
            grouped.setdefault((task.iteration, chunk_id, operation), []).append(task)

        target: list[int] = []
        used: set[int] = set()
        slot = 0

        pre_forward = [
            task for task in tasks
            if task.iteration < 0 and task.phase not in _BACKWARD_PHASES
        ]
        slot = self._append_plain_tasks(
            target, used, pre_forward, layout, stage_id, "pre", slot, reverse=False
        )

        for token in self._schedule_tokens(ga, layout, stage_id):
            operation_tasks = self._tasks_for_token(grouped, token)
            new_tasks = [task for task in operation_tasks if task.task_id not in used]
            if not new_tasks:
                continue
            for task in new_tasks:
                target.append(task.task_id)
                used.add(task.task_id)
                self._record_task(task, layout, stage_id, token, token.operation, slot)
            slot += 1

        pre_backward = [
            task for task in tasks
            if task.iteration < 0 and task.phase in _BACKWARD_PHASES
        ]
        slot = self._append_plain_tasks(
            target, used, pre_backward, layout, stage_id, "pre", slot, reverse=True
        )
        post_forward = [
            task for task in tasks
            if task.iteration >= ga and task.phase not in _BACKWARD_PHASES
        ]
        slot = self._append_plain_tasks(
            target, used, post_forward, layout, stage_id, "post", slot, reverse=False
        )
        post_backward = [
            task for task in tasks
            if task.iteration >= ga and task.phase in _BACKWARD_PHASES
        ]
        slot = self._append_plain_tasks(
            target, used, post_backward, layout, stage_id, "post", slot, reverse=True
        )

        leftovers = [task for task in tasks if task.task_id not in used]
        self._append_plain_tasks(
            target, used, leftovers, layout, stage_id, "other", slot, reverse=False
        )
        return target

    def _chunk_by_layer(self, tasks: list[Task], ga: int) -> dict[int, int]:
        """Map local layer IDs to strategy model chunks."""
        del ga
        return {task.layer_id: 0 for task in tasks}

    @staticmethod
    def _operation(task: Task) -> str:
        if task.phase == Phase.FORWARD:
            return "F"
        if task.phase == Phase.BACKWARD_INPUT:
            return "B"
        if task.phase == Phase.BACKWARD_WEIGHT:
            return "W"
        if task.phase in (Phase.PREFILL, Phase.DECODE):
            return "inference"
        return "other"

    @staticmethod
    def _tasks_for_token(
        grouped: dict[tuple[int, int, str], list[Task]],
        token: _ScheduleToken,
    ) -> list[Task]:
        key = (token.microbatch_id, token.model_chunk_id)
        if token.operation == "F":
            return sorted(
                grouped.get((*key, "F"), []),
                key=lambda task: (task.layer_id, task.item_id, task.task_id),
            )
        if token.operation == "B":
            return sorted(
                grouped.get((*key, "B"), []),
                key=lambda task: (-task.layer_id, task.item_id, task.task_id),
            )
        if token.operation == "W":
            return sorted(
                grouped.get((*key, "W"), []),
                key=lambda task: (-task.layer_id, task.item_id, task.task_id),
            )
        if token.operation == "BW":
            tasks = grouped.get((*key, "B"), []) + grouped.get((*key, "W"), [])
            return sorted(
                tasks,
                key=lambda task: (
                    -task.layer_id,
                    0 if task.phase == Phase.BACKWARD_INPUT else 1,
                    task.item_id,
                    task.task_id,
                ),
            )
        raise ValueError(f"Unsupported pipeline operation: {token.operation!r}")

    def _append_plain_tasks(
        self,
        target: list[int],
        used: set[int],
        tasks: list[Task],
        layout: _JobLayout,
        stage_id: int,
        operation: str,
        slot: int,
        *,
        reverse: bool,
    ) -> int:
        ordered = sorted(
            tasks,
            key=lambda task: (
                -task.layer_id if reverse else task.layer_id,
                0 if task.phase == Phase.BACKWARD_INPUT else 1,
                task.item_id,
                task.task_id,
            ),
        )
        for task in ordered:
            if task.task_id in used:
                continue
            target.append(task.task_id)
            used.add(task.task_id)
            self._record_task(task, layout, stage_id, None, operation, slot)
            slot += 1
        return slot

    def _record_task(
        self,
        task: Task,
        layout: _JobLayout,
        stage_id: int,
        token: _ScheduleToken | None,
        operation: str,
        slot: int,
    ) -> None:
        direction = token.direction if token is not None else "none"
        logical_stage = self._logical_stage_id(
            layout, stage_id, token, direction,
        )
        self.task_info[task.task_id] = PipelineTaskInfo(
            task_id=task.task_id,
            job_id=task.job_id,
            stage_id=stage_id,
            logical_stage_id=logical_stage,
            microbatch_id=token.microbatch_id if token is not None else None,
            model_chunk_id=token.model_chunk_id if token is not None else None,
            pipeline_id=token.pipeline_id if token is not None else None,
            direction=direction,
            operation=operation,
            schedule_slot=slot,
            schedule_name=self.schedule_name,
        )

    def _logical_stage_id(
        self,
        layout: _JobLayout,
        stage_id: int,
        token: _ScheduleToken | None,
        direction: str,
    ) -> int:
        """Map a local schedule token to its strategy logical stage."""
        del token
        return layout.pp - 1 - stage_id if direction == "up" else stage_id

    def _derive_layouts(self, workload: P2PWorkload) -> dict[int, _JobLayout]:
        layouts: dict[int, _JobLayout] = {}
        for job in workload.jobs:
            grouper = RankGrouper(job.assigned_nodes, job.parallelism)
            stage_size = grouper.dp * grouper.ep * grouper.tp
            node_to_stage: dict[int, int] = {}
            for stage_id in range(grouper.pp):
                start = stage_id * stage_size
                for node in grouper.nodes[start:start + stage_size]:
                    node_to_stage[node] = stage_id
            layouts[job.job_id] = _JobLayout(grouper.pp, node_to_stage)

        task_job_ids = {task.job_id for task in workload.tasks}
        for job_id in task_job_ids - layouts.keys():
            pp = self._fallback_pp or 1
            layouts[job_id] = _JobLayout(pp, dict(self._fallback_node_to_stage))
        return layouts

    @staticmethod
    def _infer_ga(tasks: list[Task]) -> int:
        max_iteration = max((task.iteration for task in tasks if task.iteration >= 0), default=-1)
        if max_iteration < 0:
            return 0
        has_post = any(
            task.iteration == max_iteration
            and task.phase == Phase.FORWARD
            and task.duration_us == 0
            for task in tasks
        )
        return max_iteration if has_post else max_iteration + 1

    @staticmethod
    def _legalize_against_dag(
        workload: P2PWorkload,
        desired_by_node: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        """Project a preferred order through a global DAG topological order.

        Any per-node projection of one global topological order is safe to add
        as resource edges.  Ready compute tasks are selected by the strategy's
        preferred local position; ready flow tasks are released eagerly.
        """
        tasks_by_id = {task.task_id: task for task in workload.tasks}
        successors: dict[int, list[int]] = {task_id: [] for task_id in tasks_by_id}
        in_degree = {task_id: 0 for task_id in tasks_by_id}
        for task in workload.tasks:
            for dependency in task.deps:
                if dependency not in successors:
                    raise ValueError(
                        f"Task {task.task_id} depends on missing task {dependency}"
                    )
                successors[dependency].append(task.task_id)
                in_degree[task.task_id] += 1

        preferred = {
            task_id: position
            for task_ids in desired_by_node.values()
            for position, task_id in enumerate(task_ids)
        }

        def priority(task_id: int) -> tuple[int, int, int, int]:
            task = tasks_by_id[task_id]
            if task.is_flow():
                return (0, 0, task.src or -1, task_id)
            return (1, preferred.get(task_id, 10**12), task.node or -1, task_id)

        ready = [priority(task_id) for task_id, degree in in_degree.items() if degree == 0]
        heapq.heapify(ready)
        topological_order: list[int] = []
        while ready:
            *_, task_id = heapq.heappop(ready)
            topological_order.append(task_id)
            for successor in successors[task_id]:
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    heapq.heappush(ready, priority(successor))

        if len(topological_order) != len(tasks_by_id):
            cyclic = sorted(task_id for task_id, degree in in_degree.items() if degree > 0)
            raise ValueError(f"Workload DAG contains a cycle involving tasks: {cyclic}")

        compute_order: dict[int, list[int]] = {}
        for task_id in topological_order:
            task = tasks_by_id[task_id]
            if task.is_compute():
                compute_order.setdefault(task.node, []).append(task_id)
        return compute_order


class InterleavedOneFOneBSerializer(AdvancedPipelineSerializer):
    """Megatron-style interleaved 1F1B compute-order projection."""

    schedule_name = "interleaved_1f1b"

    def __init__(
        self,
        virtual_pipeline_size: int = 2,
        interleave_group_size: int | None = None,
        pp: int | None = None,
        node_to_stage: dict[int, int] | None = None,
        expansion_task_info: Mapping[int, object] | None = None,
    ):
        super().__init__(pp, node_to_stage)
        if virtual_pipeline_size < 2:
            raise ValueError(
                f"virtual_pipeline_size must be >= 2, got {virtual_pipeline_size}"
            )
        if interleave_group_size is not None and interleave_group_size < 1:
            raise ValueError(
                f"interleave_group_size must be >= 1, got {interleave_group_size}"
            )
        self.virtual_pipeline_size = virtual_pipeline_size
        self.interleave_group_size = interleave_group_size
        self.expansion_task_info = expansion_task_info or {}

    def _chunk_by_layer(self, tasks: list[Task], ga: int) -> dict[int, int]:
        sidecar_mapping: dict[int, int] = {}
        for task in tasks:
            info = self.expansion_task_info.get(task.task_id)
            chunk_id = getattr(info, "model_chunk_id", None)
            if chunk_id is None or not 0 <= task.iteration < ga:
                continue
            previous = sidecar_mapping.setdefault(task.layer_id, chunk_id)
            if previous != chunk_id:
                raise ValueError(
                    f"Layer {task.layer_id} maps to multiple interleaved chunks"
                )
        if sidecar_mapping:
            chunks = set(sidecar_mapping.values())
            expected = set(range(self.virtual_pipeline_size))
            if chunks != expected:
                raise ValueError(
                    "Interleaved expansion sidecar chunk mismatch: "
                    f"expected={sorted(expected)}, actual={sorted(chunks)}"
                )
            return sidecar_mapping

        layers = sorted({
            task.layer_id
            for task in tasks
            if 0 <= task.iteration < ga and self._operation(task) in {"F", "B", "W"}
        })
        if layers and len(layers) < self.virtual_pipeline_size:
            raise ValueError(
                "interleaved_1f1b requires at least one local layer per virtual chunk: "
                f"layers={len(layers)}, vpp={self.virtual_pipeline_size}"
            )
        return {
            layer_id: min(
                index * self.virtual_pipeline_size // len(layers),
                self.virtual_pipeline_size - 1,
            )
            for index, layer_id in enumerate(layers)
        }

    def _logical_stage_id(
        self,
        layout: _JobLayout,
        stage_id: int,
        token: _ScheduleToken | None,
        direction: str,
    ) -> int:
        del direction
        chunk_id = token.model_chunk_id if token is not None else 0
        return chunk_id * layout.pp + stage_id

    def _schedule_tokens(
        self,
        ga: int,
        layout: _JobLayout,
        stage_id: int,
    ) -> list[_ScheduleToken]:
        if ga == 0:
            return []
        group_size = min(self.interleave_group_size or layout.pp, ga)
        forward: list[_ScheduleToken] = []
        backward: list[_ScheduleToken] = []
        for group_start in range(0, ga, group_size):
            microbatches = range(group_start, min(group_start + group_size, ga))
            microbatch_ids = list(microbatches)
            for chunk_id in range(self.virtual_pipeline_size):
                forward.extend(
                    _ScheduleToken("F", microbatch_id, chunk_id)
                    for microbatch_id in microbatch_ids
                )
            for chunk_id in reversed(range(self.virtual_pipeline_size)):
                backward.extend(
                    _ScheduleToken("BW", microbatch_id, chunk_id)
                    for microbatch_id in microbatch_ids
                )

        warmup = min(
            len(forward),
            2 * (layout.pp - stage_id - 1)
            + (self.virtual_pipeline_size - 1) * group_size,
        )
        result = forward[:warmup]
        remaining = len(forward) - warmup
        for index in range(remaining):
            result.append(forward[warmup + index])
            result.append(backward[index])
        result.extend(backward[remaining:])
        return result


class ZeroBubbleSerializer(AdvancedPipelineSerializer):
    """ZB1P-style order that propagates B before scheduling deferred W."""

    schedule_name = "zero_bubble"

    def __init__(
        self,
        pp: int | None = None,
        node_to_stage: dict[int, int] | None = None,
        expansion_task_info: Mapping[int, object] | None = None,
        max_inflight_microbatches: int | None = None,
    ):
        super().__init__(pp=pp, node_to_stage=node_to_stage)
        if max_inflight_microbatches is not None and max_inflight_microbatches < 1:
            raise ValueError("max_inflight_microbatches must be >= 1")
        self.expansion_task_info = (
            expansion_task_info if expansion_task_info is not None else {}
        )
        self.max_inflight_microbatches = max_inflight_microbatches

    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        plan = super().serialize(workload)
        if isinstance(self.expansion_task_info, dict):
            for task_id, schedule_info in self.task_info.items():
                expansion_info = self.expansion_task_info.get(task_id)
                if (
                    expansion_info is None
                    or not hasattr(expansion_info, "preferred_slot")
                    or not hasattr(expansion_info, "final_local_order")
                ):
                    continue
                self.expansion_task_info[task_id] = replace(
                    expansion_info,
                    preferred_slot=schedule_info.schedule_slot,
                    final_local_order=schedule_info.local_order,
                )
        return plan

    def _operation(self, task: Task) -> str:
        info = self.expansion_task_info.get(task.task_id)
        role = getattr(info, "task_role", None)
        if role in {"F", "B", "W"}:
            return role
        return super()._operation(task)

    def _schedule_tokens(
        self,
        ga: int,
        layout: _JobLayout,
        stage_id: int,
    ) -> list[_ScheduleToken]:
        if ga == 0:
            return []
        warmup = min(ga, max(1, layout.pp - stage_id))
        if self.max_inflight_microbatches is not None:
            warmup = min(warmup, self.max_inflight_microbatches)
        result = [_ScheduleToken("F", microbatch_id) for microbatch_id in range(warmup)]

        steady_count = ga - warmup
        for microbatch_id in range(steady_count):
            result.append(_ScheduleToken("B", microbatch_id))
            result.append(_ScheduleToken("F", warmup + microbatch_id))
            result.append(_ScheduleToken("W", microbatch_id))

        cooldown_ids = list(range(steady_count, ga))
        result.extend(_ScheduleToken("B", microbatch_id) for microbatch_id in cooldown_ids)
        result.extend(_ScheduleToken("W", microbatch_id) for microbatch_id in cooldown_ids)
        return result


class BidirectionalPipelineSerializer(AdvancedPipelineSerializer):
    """Basic Chimera schedule formed by merging two mirrored 1F1B pipelines.

    Each direction owns half of the microbatches.  Its logical stages use the
    normal 1F1B warmup/steady/cooldown sequence; the two local sequences are
    then alternated, starting with the direction entering from the nearer end.
    """

    schedule_name = "chimera"

    def __init__(
        self,
        pp: int | None = None,
        node_to_stage: dict[int, int] | None = None,
        expansion_task_info: Mapping[int, object] | None = None,
    ):
        super().__init__(pp=pp, node_to_stage=node_to_stage)
        self.expansion_task_info = (
            expansion_task_info if expansion_task_info is not None else {}
        )

    def _record_task(
        self,
        task: Task,
        layout: _JobLayout,
        stage_id: int,
        token: _ScheduleToken | None,
        operation: str,
        slot: int,
    ) -> None:
        expansion_info = self.expansion_task_info.get(task.task_id)
        if token is not None and expansion_info is not None:
            microbatch_id = getattr(expansion_info, "microbatch_id", None)
            if microbatch_id != token.microbatch_id:
                raise ValueError(
                    "Bidirectional expansion/serializer microbatch mismatch "
                    f"for task {task.task_id}: expansion={microbatch_id}, "
                    f"serializer={token.microbatch_id}"
                )
            token = replace(
                token,
                pipeline_id=getattr(
                    expansion_info, "pipeline_id", token.pipeline_id,
                ),
                direction=getattr(
                    expansion_info, "direction", token.direction,
                ),
            )
        super()._record_task(
            task, layout, stage_id, token, operation, slot,
        )

    def _schedule_tokens(
        self,
        ga: int,
        layout: _JobLayout,
        stage_id: int,
    ) -> list[_ScheduleToken]:
        if layout.pp % 2:
            raise ValueError(f"bidirectional requires even pp, got {layout.pp}")
        if ga % 2:
            raise ValueError(f"bidirectional requires even microbatch count, got {ga}")
        half_ga = ga // 2
        microbatches = {
            0: list(range(half_ga)),
            1: list(range(half_ga, ga)),
        }
        sequences = {
            direction: self._one_f_one_b_sequence(
                microbatches[direction],
                (
                    stage_id
                    if direction == 0
                    else layout.pp - 1 - stage_id
                ),
                layout.pp,
                direction,
            )
            for direction in (0, 1)
        }

        # The pipeline entering from the nearer physical end receives the
        # first preference. DAG legalization remains authoritative if actual
        # communication readiness differs from this unit-time Chimera merge.
        directions = (
            (0, 1)
            if stage_id < layout.pp // 2
            else (1, 0)
        )
        result: list[_ScheduleToken] = []
        cursor = [0, 0]
        while any(cursor[d] < len(sequences[d]) for d in directions):
            for direction in directions:
                if cursor[direction] >= len(sequences[direction]):
                    continue
                result.append(sequences[direction][cursor[direction]])
                cursor[direction] += 1
        return result

    @classmethod
    def _one_f_one_b_sequence(
        cls,
        microbatches: list[int],
        logical_stage: int,
        pp: int,
        direction: int,
    ) -> list[_ScheduleToken]:
        """Return one pipeline's ordinary synchronous 1F1B local sequence."""
        count = len(microbatches)
        warmup = min(max(1, pp - logical_stage), count)
        sequence = [
            cls._direction_token("F", microbatch_id, direction)
            for microbatch_id in microbatches[:warmup]
        ]
        for index in range(count - warmup):
            sequence.append(
                cls._direction_token(
                    "BW", microbatches[index], direction,
                )
            )
            sequence.append(
                cls._direction_token(
                    "F", microbatches[warmup + index], direction,
                )
            )
        placed_backward = count - warmup
        sequence.extend(
            cls._direction_token("BW", microbatch_id, direction)
            for microbatch_id in microbatches[placed_backward:]
        )
        return sequence

    @staticmethod
    def _direction_token(
        operation: str,
        microbatch_id: int,
        direction: int,
    ) -> _ScheduleToken:
        return _ScheduleToken(
            operation=operation,
            microbatch_id=microbatch_id,
            pipeline_id=direction,
            direction="down" if direction == 0 else "up",
        )


class DualPipeSerializer(AdvancedPipelineSerializer):
    """Official eight-step DeepSeek DualPipe compute-order projection."""

    schedule_name = "dualpipe"

    def __init__(
        self,
        pp: int | None = None,
        node_to_stage: dict[int, int] | None = None,
        expansion_task_info: Mapping[int, object] | None = None,
    ):
        super().__init__(pp=pp, node_to_stage=node_to_stage)
        self.expansion_task_info = (
            expansion_task_info if expansion_task_info is not None else {}
        )

    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        plan = super().serialize(workload)
        if isinstance(self.expansion_task_info, dict):
            for task_id, schedule_info in self.task_info.items():
                expansion_info = self.expansion_task_info.get(task_id)
                if (
                    expansion_info is None
                    or not hasattr(expansion_info, "preferred_slot")
                    or not hasattr(expansion_info, "final_local_order")
                ):
                    continue
                self.expansion_task_info[task_id] = replace(
                    expansion_info,
                    preferred_slot=schedule_info.schedule_slot,
                    final_local_order=schedule_info.local_order,
                )
        return plan

    def _record_task(
        self,
        task: Task,
        layout: _JobLayout,
        stage_id: int,
        token: _ScheduleToken | None,
        operation: str,
        slot: int,
    ) -> None:
        expansion_info = self.expansion_task_info.get(task.task_id)
        if token is not None and expansion_info is not None:
            expected_microbatch = getattr(
                expansion_info, "microbatch_id", token.microbatch_id,
            )
            expected_replica = getattr(
                expansion_info, "module_replica_id", token.pipeline_id,
            )
            if expected_microbatch != token.microbatch_id:
                raise ValueError(
                    "DualPipe expansion/serializer microbatch mismatch for "
                    f"task {task.task_id}: expansion={expected_microbatch}, "
                    f"serializer={token.microbatch_id}"
                )
            if expected_replica != token.pipeline_id:
                raise ValueError(
                    "DualPipe expansion/serializer replica mismatch for "
                    f"task {task.task_id}: expansion={expected_replica}, "
                    f"serializer={token.pipeline_id}"
                )
        super()._record_task(
            task, layout, stage_id, token, operation, slot,
        )

    def _schedule_tokens(
        self,
        ga: int,
        layout: _JobLayout,
        stage_id: int,
    ) -> list[_ScheduleToken]:
        return [
            _ScheduleToken(
                operation=token.operation,
                microbatch_id=token.microbatch_id,
                pipeline_id=token.module_replica_id,
                direction=token.direction,
            )
            for token in build_dualpipe_schedule(layout.pp, ga, stage_id)
        ]


ADVANCED_PIPELINE_NAMES = frozenset({
    "interleaved_1f1b",
    "zero_bubble",
    "bidirectional",
    "dualpipe",
})

PIPELINE_NAMES = frozenset({"gpipe", "1f1b", *ADVANCED_PIPELINE_NAMES})


def build_pipeline_serializer(
    mode: str,
    workload: P2PWorkload,
    *,
    virtual_pipeline_size: int = 2,
    interleave_group_size: int | None = None,
    expansion_task_info: Mapping[int, object] | None = None,
) -> TaskSerializer:
    """Create a registered serializer without changing the common task IR."""
    if mode == "gpipe":
        return CppReferenceSerializer()

    node_to_stage: dict[int, int] = {}
    pp = 1
    for job in workload.jobs:
        grouper = RankGrouper(job.assigned_nodes, job.parallelism)
        pp = max(pp, grouper.pp)
        stage_size = grouper.dp * grouper.ep * grouper.tp
        for stage_id in range(grouper.pp):
            start = stage_id * stage_size
            for node in grouper.nodes[start:start + stage_size]:
                node_to_stage[node] = stage_id

    if mode == "1f1b":
        return OneFOneBSerializer(pp=pp, node_to_stage=node_to_stage)
    if mode == "interleaved_1f1b":
        return InterleavedOneFOneBSerializer(
            virtual_pipeline_size=virtual_pipeline_size,
            interleave_group_size=interleave_group_size,
            pp=pp,
            node_to_stage=node_to_stage,
            expansion_task_info=expansion_task_info,
        )
    if mode == "zero_bubble":
        return ZeroBubbleSerializer(
            pp=pp,
            node_to_stage=node_to_stage,
            expansion_task_info=expansion_task_info,
        )
    if mode == "bidirectional":
        return BidirectionalPipelineSerializer(
            pp=pp,
            node_to_stage=node_to_stage,
            expansion_task_info=expansion_task_info,
        )
    if mode == "dualpipe":
        return DualPipeSerializer(
            pp=pp,
            node_to_stage=node_to_stage,
            expansion_task_info=expansion_task_info,
        )
    raise ValueError(
        f"Unsupported pipeline mode {mode!r}; expected one of {sorted(PIPELINE_NAMES)}"
    )

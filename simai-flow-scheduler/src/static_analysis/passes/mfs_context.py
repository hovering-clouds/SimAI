"""MFS context metadata — maps task IDs to MFS concepts.

Builds sidecar metadata that classifies each task by MFS stage (EARLY, P2D,
BACKGROUND), recovers batch/request/stage associations, and supports RLI
priority computation.
"""
from dataclasses import dataclass, field
from enum import Enum

from ...workload_format.schema import P2PWorkload, Task, CommType, Phase, BatchTaskInfo


class MfsStage(str, Enum):
    EARLY = "early"
    P2D = "p2d"
    BACKGROUND = "background"


@dataclass
class MfsRequestInfo:
    request_id: int
    ttft_slo_us: int | None


@dataclass
class MfsTaskInfo:
    task_id: int
    batch_id: str | None
    request_ids: tuple[int, ...]
    stage_id: int
    mfs_stage: MfsStage
    target_layer: int
    replica_id: int = 0


@dataclass
class MfsContext:
    task_info: dict[int, MfsTaskInfo] = field(default_factory=dict)
    batch_to_tasks: dict[str, tuple[int, ...]] = field(default_factory=dict)
    request_to_tasks: dict[int, tuple[int, ...]] = field(default_factory=dict)
    request_info: dict[int, MfsRequestInfo] = field(default_factory=dict)


_COLLECTIVE_TYPES = frozenset({
    CommType.TP_ALLREDUCE_RING, CommType.TP_ALLREDUCE_TREE,
    CommType.TP_ALLGATHER_RING, CommType.TP_ALLGATHER_TREE,
    CommType.TP_REDUCESCATTER_RING, CommType.TP_REDUCESCATTER_TREE,
    CommType.TP_ALLTOALL,
    CommType.DP_ALLREDUCE, CommType.DP_ALLGATHER,
    CommType.DP_REDUCESCATTER, CommType.DP_ALLTOALL,
    CommType.EP_ALLTOALL,
})


def _classify_task(task: Task) -> MfsStage:
    if task.is_compute():
        return MfsStage.BACKGROUND
    ct = task.comm_type
    if ct == CommType.KV_CACHE_TRANSFER:
        return MfsStage.P2D
    if ct == CommType.KV_CACHE_REUSE:
        return MfsStage.EARLY
    if ct == CommType.PP_SEND or ct == CommType.PP_RECV:
        return MfsStage.EARLY
    if ct in _COLLECTIVE_TYPES:
        if task.phase == Phase.DECODE:
            return MfsStage.BACKGROUND
        return MfsStage.EARLY
    return MfsStage.BACKGROUND


def build_mfs_context(
    workload: P2PWorkload,
    batch_task_info: list[BatchTaskInfo],
    trace: dict,
) -> MfsContext:
    """Build MFS sidecar metadata from a workload and its batch_task_info.

    All BatchTaskInfo fields are required — no silent defaults.

    Args:
        workload: The P2PWorkload (output of InferenceTraceExpander).
        batch_task_info: List of BatchTaskInfo entries.
        trace: Raw Vidur trace dict. Used to extract ttft_slo_us per request.

    Returns:
        MfsContext with per-task info, batch grouping, request grouping,
        and optional deadline metadata.
    """
    # Build reverse index: task_id -> (batch_id, request_ids, stage_id, replica_id)
    tid_to_batch: dict[int, tuple[str, tuple[int, ...], int, int]] = {}
    for info in batch_task_info:
        task_ids = info.task_ids
        req_ids = tuple(info.request_ids)
        for tid in task_ids:
            tid_to_batch[tid] = (info.batch_id, req_ids, info.stage_id, info.replica_id)

    ctx = MfsContext()

    # Per-task classification
    for task in workload.tasks:
        mfs_stage = _classify_task(task)
        bid, req_ids, stage_id, replica_id = tid_to_batch.get(
            task.task_id, (None, (), 0, 0),
        )
        ctx.task_info[task.task_id] = MfsTaskInfo(
            task_id=task.task_id,
            batch_id=bid,
            request_ids=req_ids,
            stage_id=stage_id,
            replica_id=replica_id,
            mfs_stage=mfs_stage,
            target_layer=task.layer_id,
        )

    # Batch grouping
    for info in batch_task_info:
        task_ids = info.task_ids
        if task_ids:
            ctx.batch_to_tasks[info.batch_id] = tuple(task_ids)

    # Request grouping: collect all tasks that belong to each request
    req_tasks: dict[int, list[int]] = {}
    for info in ctx.task_info.values():
        for rid in info.request_ids:
            req_tasks.setdefault(rid, []).append(info.task_id)
    ctx.request_to_tasks = {rid: tuple(tids) for rid, tids in req_tasks.items()}

    # Parse request SLO metadata from trace
    if "requests" in trace:
        for rid_str, req_entry in trace["requests"].items():
            rid = int(rid_str)
            ctx.request_info[rid] = MfsRequestInfo(
                request_id=rid,
                ttft_slo_us=req_entry.get("ttft_slo_us"),
            )

    return ctx

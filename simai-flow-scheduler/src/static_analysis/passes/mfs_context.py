"""MFS context metadata — maps task IDs to MFS concepts.

Builds sidecar metadata that classifies each task by MFS stage (EARLY, P2D,
BACKGROUND), recovers batch/request/stage associations, and supports RLI
priority computation.
"""
from dataclasses import dataclass, field
from enum import Enum

from ...workload_format.schema import P2PWorkload, Task, CommType, Phase


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
    job_id: int
    batch_id: str | None
    request_ids: tuple[int, ...]
    stage_id: int
    mfs_stage: MfsStage
    target_layer: int
    comm_role: str


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


def _classify_task(task: Task) -> tuple[MfsStage, str]:
    if task.is_compute():
        return MfsStage.BACKGROUND, "compute"
    ct = task.comm_type
    if ct == CommType.KV_CACHE_TRANSFER:
        return MfsStage.P2D, "p2d_transfer"
    if ct == CommType.PP_SEND or ct == CommType.PP_RECV:
        return MfsStage.EARLY, "pp_send"
    if ct in _COLLECTIVE_TYPES:
        # Decode-phase collectives are not part of the prefill→P2D pipeline
        if task.phase == Phase.DECODE:
            return MfsStage.BACKGROUND, "decode_collective"
        return MfsStage.EARLY, "collective"
    return MfsStage.BACKGROUND, "unknown"


def build_mfs_context(
    workload: P2PWorkload,
    batch_task_map: dict,
    trace: dict,
) -> MfsContext:
    """Build MFS sidecar metadata from a workload and its batch_task_map.

    Args:
        workload: The P2PWorkload (output of InferenceTraceExpander).
        batch_task_map: Mapping from batch/transfer key to
            {task_ids, request_ids, type, replica_id, ...}.
        trace: Raw Vidur trace dict. Used to extract ttft_slo_us per request.

    Returns:
        MfsContext with per-task info, batch grouping, request grouping,
        and optional deadline metadata.
    """
    # Build reverse index: task_id -> (batch_id, request_ids, stage_id)
    tid_to_batch: dict[int, tuple[str, tuple[int, ...], int]] = {}
    for bid, binfo in batch_task_map.items():
        task_ids = binfo.get("task_ids", [])
        req_ids = tuple(binfo.get("request_ids", []))
        stage_id = binfo.get("stage_id", 0)
        for tid in task_ids:
            tid_to_batch[tid] = (bid, req_ids, stage_id)

    ctx = MfsContext()

    # Per-task classification
    for task in workload.tasks:
        mfs_stage, comm_role = _classify_task(task)
        bid, req_ids, stage_id = tid_to_batch.get(
            task.task_id, (None, (), 0),
        )
        ctx.task_info[task.task_id] = MfsTaskInfo(
            task_id=task.task_id,
            job_id=task.job_id,
            batch_id=bid,
            request_ids=req_ids,
            stage_id=stage_id,
            mfs_stage=mfs_stage,
            target_layer=task.layer_id,
            comm_role=comm_role,
        )

    # Batch grouping
    for bid, binfo in batch_task_map.items():
        task_ids = binfo.get("task_ids", [])
        if task_ids:
            ctx.batch_to_tasks[bid] = tuple(task_ids)

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

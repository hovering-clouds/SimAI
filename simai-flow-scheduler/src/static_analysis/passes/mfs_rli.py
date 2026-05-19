"""RLI (Relative Layer Index) metadata for MFS flow prioritization.

Computes static RLI for early-stage flows. Lower RLI means higher urgency.
P2D flows get a sentinel value so they don't compete with early-stage RLI ranking.
"""
from dataclasses import dataclass

from ...workload_format.schema import P2PWorkload
from .mfs_context import MfsContext, MfsStage

_P2D_SENTINEL_RLI = 10_000
_BG_SENTINEL_RLI = 10_000


@dataclass
class RliInfo:
    task_id: int
    target_layer: int
    base_rli: int


def compute_static_rli(
    workload: P2PWorkload,
    context: MfsContext,
    current_layer_by_stage: dict[tuple[int, int], int] | None = None,
) -> dict[int, RliInfo]:
    """Compute static RLI metadata for all tasks in the workload.

    Args:
        workload: The P2PWorkload.
        context: MFS context with per-task classification.
        current_layer_by_stage: Optional mapping of (job_id, stage_id) to the
            current compute layer. Defaults to 0 for all stages.

    Returns:
        dict mapping task_id -> RliInfo.
    """
    cur = current_layer_by_stage or {}

    result: dict[int, RliInfo] = {}
    for task in workload.tasks:
        info = context.task_info.get(task.task_id)
        if info is None:
            continue

        if info.mfs_stage == MfsStage.P2D:
            rli = _P2D_SENTINEL_RLI
        elif info.mfs_stage == MfsStage.BACKGROUND:
            rli = _BG_SENTINEL_RLI
        else:
            # EARLY: RLI = max(target_layer - current_layer, 0)
            stage_key = (task.job_id, info.stage_id)
            current = cur.get(stage_key, 0)
            rli = max(info.target_layer - current, 0)

        result[task.task_id] = RliInfo(
            task_id=task.task_id,
            target_layer=info.target_layer,
            base_rli=rli,
        )

    return result

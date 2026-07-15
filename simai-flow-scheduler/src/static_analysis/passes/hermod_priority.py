"""Hermod §4.1 coflow-priority analysis.

This pass models only inter-coflow priority.  It intentionally does not
implement Hermod §4.2's matching-based intra-coflow flow allocation.
"""
from dataclasses import dataclass
from enum import Enum

from ...workload_format.schema import CommType, P2PWorkload


class HermodCoflowType(str, Enum):
    EP = "ep"
    PP = "pp"
    DP = "dp"


class HermodScheduleVariant(str, Enum):
    CONVENTIONAL_1F1B = "conventional_1f1b"
    INTERLEAVED_1F1B = "interleaved_1f1b"


class HermodEpMode(str, Enum):
    REJECT = "reject"
    ENABLE = "enable"


@dataclass(frozen=True)
class HermodCoflowInfo:
    coflow_id: str
    task_ids: tuple[int, ...]
    microbatch_id: int
    logical_layer_id: int
    coflow_type: HermodCoflowType


def classify_coflow_type(comm_type: CommType) -> HermodCoflowType | None:
    if comm_type == CommType.EP_ALLTOALL:
        return HermodCoflowType.EP
    if comm_type in (CommType.PP_SEND, CommType.PP_RECV):
        return HermodCoflowType.PP
    if comm_type.value.startswith("dp_"):
        return HermodCoflowType.DP
    return None


class HermodPriorityAnalysis:
    """Validate Hermod metadata and rank the currently competing coflows.

    MID is the default first factor.  For the §4.1.2 cross-iteration
    DP-versus-EP/PP exception, LID becomes the first factor.  The exception is
    selected from the *active candidate set*, so unrelated, inactive coflows
    cannot perturb a decision.  The final coflow id is only a deterministic
    tie-breaker, never a paper priority factor.
    """

    def __init__(
        self, coflows: dict[str, HermodCoflowInfo],
        variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
        ep_mode: HermodEpMode = HermodEpMode.ENABLE,
    ):
        self.coflows = coflows
        self.variant = variant
        self.ep_mode = ep_mode
        self.task_to_coflow = {
            task_id: info.coflow_id
            for info in coflows.values() for task_id in info.task_ids
        }

    @classmethod
    def from_workload(
        cls, workload: P2PWorkload,
        variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
        ep_mode: HermodEpMode = HermodEpMode.ENABLE,
    ) -> "HermodPriorityAnalysis":
        grouped: dict[str, list] = {}
        for task in workload.get_flow_tasks():
            ctype = classify_coflow_type(task.comm_type)
            if ctype is None:
                continue  # TP/unknown traffic is outside the §4.1 policy.
            if ctype == HermodCoflowType.EP and ep_mode == HermodEpMode.REJECT:
                raise ValueError(
                    f"Hermod EP is disabled: task {task.task_id} is {task.comm_type.value}"
                )
            missing = [
                name for name, value in (
                    ("coflow_id", task.coflow_id),
                    ("microbatch_id", task.microbatch_id),
                    ("logical_layer_id", task.logical_layer_id),
                ) if value is None
            ]
            if missing:
                raise ValueError(
                    f"Hermod task {task.task_id} lacks required metadata: {', '.join(missing)}"
                )
            grouped.setdefault(task.coflow_id, []).append(task)

        coflows: dict[str, HermodCoflowInfo] = {}
        for coflow_id, tasks in grouped.items():
            first = tasks[0]
            signature = (first.microbatch_id, first.logical_layer_id,
                         classify_coflow_type(first.comm_type))
            for task in tasks[1:]:
                actual = (task.microbatch_id, task.logical_layer_id,
                          classify_coflow_type(task.comm_type))
                if actual != signature:
                    raise ValueError(
                        f"Hermod coflow {coflow_id!r} has inconsistent priority metadata"
                    )
            coflows[coflow_id] = HermodCoflowInfo(
                coflow_id=coflow_id,
                task_ids=tuple(sorted(task.task_id for task in tasks)),
                microbatch_id=first.microbatch_id,
                logical_layer_id=first.logical_layer_id,
                coflow_type=signature[2],
            )
        return cls(coflows, variant, ep_mode)

    def _ctype_rank(self, coflow_type: HermodCoflowType) -> int:
        if coflow_type == HermodCoflowType.EP:
            return 0
        if coflow_type == HermodCoflowType.PP:
            return 0 if self.variant == HermodScheduleVariant.CONVENTIONAL_1F1B else 1
        return 2

    def priority_order(self, active_task_ids: list[int]) -> list[str]:
        """Return active coflow ids from high to low priority."""
        active = {self.task_to_coflow[tid] for tid in active_task_ids
                  if tid in self.task_to_coflow}
        infos = [self.coflows[cid] for cid in active]
        # §4.1.2: competing DP and EP/PP from different LIDs use LID first.
        has_cross_iteration_dp_conflict = any(
            a.coflow_type == HermodCoflowType.DP
            and b.coflow_type != HermodCoflowType.DP
            and a.logical_layer_id != b.logical_layer_id
            for a in infos for b in infos
        )
        if has_cross_iteration_dp_conflict:
            key = lambda c: (c.logical_layer_id, c.microbatch_id,
                             self._ctype_rank(c.coflow_type), c.coflow_id)
        else:
            key = lambda c: (c.microbatch_id, self._ctype_rank(c.coflow_type),
                             c.logical_layer_id, c.coflow_id)
        return [c.coflow_id for c in sorted(infos, key=key)]

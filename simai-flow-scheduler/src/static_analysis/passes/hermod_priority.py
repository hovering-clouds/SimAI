"""Hermod §4.1 coflow-priority analysis.

This pass models only inter-coflow priority. It intentionally excludes §4.2's
matching-based intra-coflow flow allocation.
"""
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

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
    job_id: int
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
    """Validate Hermod metadata and rank currently competing coflows.

    The default §4.1 order is MID, CType, then LID. The §4.1.2 Case III
    exception is pair-specific, so an unrelated active coflow cannot change
    another pair's ordering.
    """

    def __init__(
        self,
        coflows: dict[str, HermodCoflowInfo],
        variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
        ep_mode: HermodEpMode = HermodEpMode.REJECT,
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
        hermod_records: dict,
        variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
        ep_mode: HermodEpMode = HermodEpMode.REJECT,
    ) -> "HermodPriorityAnalysis":
        grouped: dict[str, list] = {}
        for task in workload.get_flow_tasks():
            record = hermod_records.get(task.task_id)
            if record is None:
                continue
            coflow_id = record.coflow_id
            microbatch_id = record.microbatch_id
            logical_layer_id = record.logical_layer_id
            coflow_type = classify_coflow_type(task.comm_type)
            grouped.setdefault(coflow_id, []).append((task, microbatch_id, logical_layer_id))

        coflows: dict[str, HermodCoflowInfo] = {}
        for coflow_id, items in grouped.items():
            first_task, first_mid, first_lid = items[0]
            signature = (
                first_task.job_id,
                first_mid,
                first_lid,
                classify_coflow_type(first_task.comm_type),
            )
            for task, mid, lid in items[1:]:
                actual = (
                    task.job_id,
                    mid,
                    lid,
                    classify_coflow_type(task.comm_type),
                )
                if actual != signature:
                    raise ValueError(
                        f"Hermod coflow {coflow_id!r} has inconsistent priority metadata"
                    )
            coflows[coflow_id] = HermodCoflowInfo(
                coflow_id=coflow_id,
                task_ids=tuple(sorted(task.task_id for task, _, _ in items)),
                job_id=first_task.job_id,
                microbatch_id=first_mid,
                logical_layer_id=first_lid,
                coflow_type=signature[3],
            )
        return cls(coflows, variant, ep_mode)

    def _ctype_rank(self, coflow_type: HermodCoflowType) -> int:
        if coflow_type == HermodCoflowType.EP:
            return 0
        if coflow_type == HermodCoflowType.PP:
            return 0 if self.variant == HermodScheduleVariant.CONVENTIONAL_1F1B else 1
        return 2

    def _default_key(self, coflow: HermodCoflowInfo) -> tuple[int, int, int, int]:
        return (
            coflow.job_id,
            coflow.microbatch_id,
            self._ctype_rank(coflow.coflow_type),
            coflow.logical_layer_id,
        )

    def _stable_key(self, coflow: HermodCoflowInfo) -> tuple[int, int, int, int, str]:
        """Deterministic presentation order; coflow_id is not a priority signal."""
        return (*self._default_key(coflow), coflow.coflow_id)

    @staticmethod
    def _is_case_iii_pair(a: HermodCoflowInfo, b: HermodCoflowInfo) -> bool:
        """Identify the DP-later/MID-higher and LID-earlier Case III pair."""
        dp, non_dp = (a, b) if a.coflow_type == HermodCoflowType.DP else (b, a)
        return (
            dp.coflow_type == HermodCoflowType.DP
            and non_dp.coflow_type != HermodCoflowType.DP
            and dp.microbatch_id > non_dp.microbatch_id
            and dp.logical_layer_id < non_dp.logical_layer_id
        )

    def _pair_winner(self, a: HermodCoflowInfo, b: HermodCoflowInfo) -> str | None:
        # MID/LID are local to one training job. In a dynamic multi-job run
        # they restart at zero for every Job, so applying Case III across Jobs
        # can create a non-transitive relation. Hermod §4.1 does not define a
        # cross-job comparison; use the dynamic scheduler's stable Job order
        # at that boundary and apply the paper rules only within a Job.
        if a.job_id != b.job_id:
            return a.coflow_id if a.job_id < b.job_id else b.coflow_id
        if self._is_case_iii_pair(a, b):
            if a.logical_layer_id != b.logical_layer_id:
                return a.coflow_id if a.logical_layer_id < b.logical_layer_id else b.coflow_id
        a_key, b_key = self._default_key(a), self._default_key(b)
        if a_key == b_key:
            # §4.1 supplies no ordering signal here.  These coflows must
            # share a tier; using their opaque IDs would invent a priority.
            return None
        return a.coflow_id if a_key < b_key else b.coflow_id

    def priority_tiers(self, active_task_ids: list[int]) -> list[list[str]]:
        """Return active coflow tiers from high to low priority.

        Coflows tied on every §4.1 priority dimension are deliberately kept
        in one tier.  The allocator applies max-min sharing to that tier;
        serializing it by coflow ID would be an unsupported policy choice.
        """
        active = {
            self.task_to_coflow[task_id] for task_id in active_task_ids
            if task_id in self.task_to_coflow
        }
        infos = [self.coflows[coflow_id] for coflow_id in active]
        by_id = {info.coflow_id: info for info in infos}
        successors = {coflow_id: set() for coflow_id in by_id}
        indegree = {coflow_id: 0 for coflow_id in by_id}
        for a, b in combinations(infos, 2):
            winner = self._pair_winner(a, b)
            if winner is None:
                continue
            loser = b.coflow_id if winner == a.coflow_id else a.coflow_id
            successors[winner].add(loser)
            indegree[loser] += 1

        tiers: list[list[str]] = []
        remaining = set(by_id)
        while remaining:
            ready = sorted(
                (coflow_id for coflow_id in remaining if indegree[coflow_id] == 0),
                key=lambda coflow_id: self._stable_key(by_id[coflow_id]),
            )
            if not ready:
                raise ValueError(
                    "Hermod active coflows create a cyclic §4.1 priority relation; "
                    "the MID/LID metadata does not represent a reachable training DAG"
                )
            tiers.append(ready)
            for winner in ready:
                remaining.remove(winner)
            for winner in ready:
                for loser in successors[winner]:
                    indegree[loser] -= 1
        return tiers

    def priority_order(self, active_task_ids: list[int]) -> list[str]:
        """Return active coflow IDs from high to low priority.

        The pairwise rules are topologically sorted. A cycle means that the
        metadata describes an unreachable/ambiguous paper case and is rejected
        rather than hidden by an arbitrary global key.
        """
        return [coflow_id for tier in self.priority_tiers(active_task_ids) for coflow_id in tier]

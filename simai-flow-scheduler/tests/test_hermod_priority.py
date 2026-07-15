import pytest

from src.static_analysis.passes.hermod_priority import (
    HermodPriorityAnalysis, HermodScheduleVariant,
)
from src.workload_format.schema import CommType, Meta, P2PWorkload, Task, TaskType


def flow(tid, coflow, comm_type, mid=0, lid=0):
    return Task(task_id=tid, job_id=0, type=TaskType.FLOW, src=0, dst=1,
                size_bytes=100, comm_type=comm_type, coflow_id=coflow,
                microbatch_id=mid, logical_layer_id=lid)


def analysis(*tasks, variant=HermodScheduleVariant.CONVENTIONAL_1F1B):
    wl = P2PWorkload(version="1", meta=Meta(num_jobs=0, num_nodes=2), tasks=list(tasks))
    return HermodPriorityAnalysis.from_workload(wl, variant)


def test_case_i_mid_has_priority_across_ep_pp_coflows():
    a = analysis(flow(1, "pp-late", CommType.PP_SEND, mid=2),
                 flow(2, "ep-early", CommType.EP_ALLTOALL, mid=1))
    assert a.priority_order([1, 2]) == ["ep-early", "pp-late"]


def test_case_ii_conventional_ep_and_pp_tie_ahead_of_dp():
    a = analysis(flow(1, "ep", CommType.EP_ALLTOALL),
                 flow(2, "pp", CommType.PP_SEND),
                 flow(3, "dp", CommType.DP_ALLREDUCE))
    assert a.priority_order([3, 2, 1]) == ["ep", "pp", "dp"]


def test_case_ii_interleaved_ep_ahead_of_pp_ahead_of_dp():
    a = analysis(flow(1, "ep", CommType.EP_ALLTOALL),
                 flow(2, "pp", CommType.PP_SEND),
                 flow(3, "dp", CommType.DP_ALLREDUCE),
                 variant=HermodScheduleVariant.INTERLEAVED_1F1B)
    assert a.priority_order([3, 2, 1]) == ["ep", "pp", "dp"]


def test_case_iii_cross_iteration_dp_conflict_uses_lid_before_mid():
    a = analysis(flow(1, "dp", CommType.DP_ALLREDUCE, mid=0, lid=5),
                 flow(2, "ep", CommType.EP_ALLTOALL, mid=2, lid=1))
    assert a.priority_order([1, 2]) == ["ep", "dp"]


def test_missing_required_metadata_fails_fast():
    bad = Task(task_id=1, job_id=0, type=TaskType.FLOW, src=0, dst=1,
               size_bytes=1, comm_type=CommType.EP_ALLTOALL, coflow_id="ep")
    with pytest.raises(ValueError, match="microbatch_id"):
        analysis(bad)


def test_coflow_metadata_must_be_consistent():
    with pytest.raises(ValueError, match="inconsistent"):
        analysis(flow(1, "ep", CommType.EP_ALLTOALL, mid=0),
                 flow(2, "ep", CommType.EP_ALLTOALL, mid=1))

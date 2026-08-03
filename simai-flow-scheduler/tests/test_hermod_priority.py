import pytest

from src.static_analysis.passes.hermod_metadata import HermodMetadataRecord
from src.static_analysis.passes.hermod_priority import (
    HermodEpMode, HermodPriorityAnalysis, HermodScheduleVariant,
)
from src.workload_format.schema import CommType, Meta, P2PWorkload, Task, TaskType


def _rec(tid, coflow, comm_type, mid=0, lid=0, job_id=0):
    """Build a HermodMetadataRecord with a synthetic coflow_id and type string."""
    ctype = comm_type.value.split("_")[0]  # e.g. "tp_allreduce_ring" → "tp"
    return HermodMetadataRecord(
        task_id=tid, coflow_id=coflow, microbatch_id=mid, logical_layer_id=lid,
        coflow_type=ctype,
        provenance="test",
    )


def _task(tid, comm_type, job_id=0):
    return Task(task_id=tid, job_id=job_id, type=TaskType.FLOW, src=0, dst=1,
                size_bytes=100, comm_type=comm_type)


def analysis(*tasks, variant=HermodScheduleVariant.CONVENTIONAL_1F1B,
             ep_mode=HermodEpMode.ENABLE, records=None):
    if records is None:
        records = {}
    wl = P2PWorkload(version="1", meta=Meta(num_jobs=0, num_nodes=2), tasks=list(tasks))
    return HermodPriorityAnalysis.from_workload(wl, records, variant, ep_mode)


def test_case_i_mid_has_priority_across_ep_pp_coflows():
    records = {1: _rec(1, "pp-late", CommType.PP_SEND, mid=2),
               2: _rec(2, "ep-early", CommType.EP_ALLTOALL, mid=1)}
    a = analysis(_task(1, CommType.PP_SEND), _task(2, CommType.EP_ALLTOALL),
                 records=records)
    assert a.priority_order([1, 2]) == ["ep-early", "pp-late"]


def test_case_ii_conventional_ep_and_pp_tie_ahead_of_dp():
    records = {1: _rec(1, "ep", CommType.EP_ALLTOALL),
               2: _rec(2, "pp", CommType.PP_SEND),
               3: _rec(3, "dp", CommType.DP_ALLREDUCE)}
    a = analysis(_task(1, CommType.EP_ALLTOALL), _task(2, CommType.PP_SEND),
                 _task(3, CommType.DP_ALLREDUCE), records=records)
    assert a.priority_order([3, 2, 1]) == ["ep", "pp", "dp"]


def test_case_ii_interleaved_ep_ahead_of_pp_ahead_of_dp():
    records = {1: _rec(1, "ep", CommType.EP_ALLTOALL),
               2: _rec(2, "pp", CommType.PP_SEND),
               3: _rec(3, "dp", CommType.DP_ALLREDUCE)}
    a = analysis(_task(1, CommType.EP_ALLTOALL), _task(2, CommType.PP_SEND),
                 _task(3, CommType.DP_ALLREDUCE),
                 variant=HermodScheduleVariant.INTERLEAVED_1F1B, records=records)
    assert a.priority_order([3, 2, 1]) == ["ep", "pp", "dp"]


def test_case_iii_later_dp_with_earlier_lid_uses_lid_before_mid():
    records = {1: _rec(1, "dp", CommType.DP_ALLREDUCE, mid=2, lid=1),
               2: _rec(2, "ep", CommType.EP_ALLTOALL, mid=1, lid=5)}
    a = analysis(_task(1, CommType.DP_ALLREDUCE), _task(2, CommType.EP_ALLTOALL),
                 records=records)
    assert a.priority_order([1, 2]) == ["dp", "ep"]


def test_different_lid_without_case_iii_keeps_default_mid_order():
    records = {1: _rec(1, "dp", CommType.DP_ALLREDUCE, mid=0, lid=9),
               2: _rec(2, "pp", CommType.PP_SEND, mid=1, lid=1)}
    a = analysis(_task(1, CommType.DP_ALLREDUCE), _task(2, CommType.PP_SEND),
                 records=records)
    assert a.priority_order([1, 2]) == ["dp", "pp"]


def test_case_iii_does_not_reorder_an_unrelated_coflow_pair():
    records = {1: _rec(1, "dp", CommType.DP_ALLREDUCE, mid=2, lid=1),
               2: _rec(2, "pp-late", CommType.PP_SEND, mid=1, lid=5),
               3: _rec(3, "pp-early", CommType.PP_SEND, mid=0, lid=0)}
    a = analysis(_task(1, CommType.DP_ALLREDUCE), _task(2, CommType.PP_SEND),
                 _task(3, CommType.PP_SEND), records=records)
    assert a.priority_order([1, 2, 3]) == ["pp-early", "dp", "pp-late"]


def test_multiple_jobs_do_not_compare_job_local_mid_lid_with_case_iii():
    """Dynamic jobs restart MID/LID, so cross-job order must be stable."""
    records = {1: _rec(1, "j0-dp", CommType.DP_ALLREDUCE, mid=2, lid=1, job_id=0),
               2: _rec(2, "j0-pp", CommType.PP_SEND, mid=1, lid=5, job_id=0),
               3: _rec(3, "j1-pp", CommType.PP_SEND, mid=0, lid=0, job_id=1),
               4: _rec(4, "j1-dp", CommType.DP_ALLREDUCE, mid=2, lid=1, job_id=1)}
    a = analysis(_task(1, CommType.DP_ALLREDUCE, job_id=0),
                 _task(2, CommType.PP_SEND, job_id=0),
                 _task(3, CommType.PP_SEND, job_id=1),
                 _task(4, CommType.DP_ALLREDUCE, job_id=1),
                 records=records)
    assert a.priority_order([1, 2, 3, 4]) == ["j0-dp", "j0-pp", "j1-pp", "j1-dp"]


def test_pp_dp_case_ii_gives_pp_priority_over_dp():
    records = {1: _rec(1, "pp", CommType.PP_SEND, mid=1, lid=3),
               2: _rec(2, "dp", CommType.DP_ALLREDUCE, mid=1, lid=3)}
    a = analysis(_task(1, CommType.PP_SEND), _task(2, CommType.DP_ALLREDUCE),
                 records=records)
    assert a.priority_order([2, 1]) == ["pp", "dp"]


def test_equal_paper_priority_coflows_share_one_tier():
    records = {1: _rec(1, "dp-a", CommType.DP_ALLREDUCE, mid=1, lid=3),
               2: _rec(2, "dp-b", CommType.DP_ALLREDUCE, mid=1, lid=3)}
    a = analysis(_task(1, CommType.DP_ALLREDUCE), _task(2, CommType.DP_ALLREDUCE),
                 records=records)
    assert a.priority_tiers([2, 1]) == [["dp-a", "dp-b"]]


def test_pp_dp_case_iii_uses_lid_before_mid():
    records = {1: _rec(1, "dp", CommType.DP_ALLREDUCE, mid=2, lid=1),
               2: _rec(2, "pp", CommType.PP_SEND, mid=1, lid=5)}
    a = analysis(_task(1, CommType.DP_ALLREDUCE), _task(2, CommType.PP_SEND),
                 records=records)
    assert a.priority_order([1, 2]) == ["dp", "pp"]


def test_missing_records_entry_is_rejected():
    with pytest.raises(ValueError, match="metadata is missing"):
        analysis(_task(1, CommType.PP_SEND), records={})


def test_reject_mode_is_enforced_by_priority_analysis():
    records = {1: _rec(1, "ep", CommType.EP_ALLTOALL)}
    with pytest.raises(ValueError, match="EP is disabled"):
        analysis(
            _task(1, CommType.EP_ALLTOALL), records=records,
            ep_mode=HermodEpMode.REJECT,
        )


def test_metadata_coflow_type_must_match_task():
    records = {1: _rec(1, "ep", CommType.PP_SEND)}
    # _rec derives "pp" from the supplied type; corrupt it deliberately.
    records[1] = HermodMetadataRecord(
        task_id=1, coflow_id="ep", microbatch_id=0, logical_layer_id=0,
        coflow_type="ep", provenance="test",
    )
    with pytest.raises(ValueError, match="does not match"):
        analysis(_task(1, CommType.PP_SEND), records=records)


def test_coflow_metadata_must_be_consistent():
    records = {1: _rec(1, "ep", CommType.EP_ALLTOALL, mid=0),
               2: _rec(2, "ep", CommType.EP_ALLTOALL, mid=1)}
    with pytest.raises(ValueError, match="inconsistent"):
        analysis(_task(1, CommType.EP_ALLTOALL), _task(2, CommType.EP_ALLTOALL),
                 records=records)

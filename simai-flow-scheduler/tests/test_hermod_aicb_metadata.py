import pytest

from src.static_analysis.passes.hermod_priority import HermodEpMode, HermodPriorityAnalysis
from src.workload_format.schema import CommType, Meta, P2PWorkload, Task, TaskType
from src.workload_format.schema import Job
from src.workload_generator.aicb_parser import AicbHeader
from src.workload_generator.hermod_aicb_metadata import HermodAicbMetadataAdapter
from src.workload_generator.job_merger import JobMerger


def _flow(tid, comm_type, coflow_id, iteration, layer_id):
    return Task(tid, 0, TaskType.FLOW, src=0, dst=1, size_bytes=1,
                comm_type=comm_type, coflow_id=coflow_id,
                iteration=iteration, layer_id=layer_id)


def _header(ga=2):
    return AicbHeader(tp=1, ep=1, pp=2, vpp=2, ga=ga, all_gpus=2, pp_comm_size=1)


def test_adapter_maps_ga_step_to_mid_and_keeps_lid_separate():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.PP_SEND, "pp0", 0, 3),
        _flow(2, CommType.DP_ALLREDUCE, "dp", 1, 1),
    ])
    records = HermodAicbMetadataAdapter(_header()).apply(wl)
    assert [(r.microbatch_id, r.logical_layer_id) for r in records] == [(0, 3), (1, 1)]
    assert HermodPriorityAnalysis.from_workload(wl).coflows["pp0"].microbatch_id == 0


def test_post_ga_dp_is_assigned_to_last_microbatch():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(0, CommType.PP_SEND, "pp", 0, 0),
        _flow(1, CommType.DP_ALLREDUCE, "dp", 2, 4),
    ])
    records = HermodAicbMetadataAdapter(_header()).apply(wl)
    assert records[1].microbatch_id == 1
    assert records[1].provenance == "aicb_dp_post_ga"


def test_adapter_and_analysis_reject_ep_by_default():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.EP_ALLTOALL, "ep", 0, 0),
    ])
    with pytest.raises(ValueError, match="EP is disabled"):
        HermodAicbMetadataAdapter(_header()).apply(wl)
    wl.tasks[0].microbatch_id = 0
    wl.tasks[0].logical_layer_id = 0
    with pytest.raises(ValueError, match="EP is disabled"):
        HermodPriorityAnalysis.from_workload(wl, ep_mode=HermodEpMode.REJECT)
    assert HermodPriorityAnalysis.from_workload(wl, ep_mode=HermodEpMode.ENABLE).coflows


def test_adapter_explicitly_rejects_unimplemented_ep_enable_mode():
    with pytest.raises(NotImplementedError, match="EP metadata"):
        HermodAicbMetadataAdapter(_header(), reject_ep=False)


def test_adapter_rejects_unmapped_negative_layer_id():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.PP_SEND, "pp", 0, -1),
    ])
    with pytest.raises(ValueError, match="layer position"):
        HermodAicbMetadataAdapter(_header()).apply(wl)


def test_job_merger_preserves_and_namespaces_hermod_metadata():
    task = _flow(1, CommType.PP_SEND, "pp0", 0, 0)
    task.microbatch_id = 0
    task.logical_layer_id = 0
    wl = P2PWorkload("1", Meta(1, 2), jobs=[Job(0)], tasks=[task])
    merged = JobMerger().merge([wl]).merged_workload.tasks[0]
    assert merged.coflow_id == "job0:pp0"
    assert merged.microbatch_id == 0
    assert merged.logical_layer_id == 0

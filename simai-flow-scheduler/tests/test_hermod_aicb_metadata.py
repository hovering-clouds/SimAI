import pytest

from src.static_analysis.passes.hermod_priority import HermodEpMode, HermodPriorityAnalysis
from src.workload_format.schema import (
    CommType, Job, Meta, P2PWorkload, ParallelismConfig, Task, TaskType,
)
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.static_analysis.passes.hermod_metadata import HermodAicbMetadataAdapter
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.job_merger import JobMerger


def _flow(tid, comm_type, iteration, layer_id):
    """Create a Hermod-relevant flow task.  No coflow_id needed on Task."""
    return Task(tid, 0, TaskType.FLOW, src=0, dst=1, size_bytes=1,
                comm_type=comm_type, iteration=iteration, layer_id=layer_id)


def _header(ga=2):
    return AicbHeader(tp=1, ep=1, pp=2, vpp=2, ga=ga, all_gpus=2, pp_comm_size=1)


def test_adapter_maps_ga_step_to_mid_and_keeps_lid_separate():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.PP_SEND, 0, 3),
        _flow(2, CommType.DP_ALLREDUCE, 1, 1),
    ])
    records = HermodAicbMetadataAdapter(_header()).apply(wl)
    assert [(r.microbatch_id, r.logical_layer_id) for r in records.values()] == [(0, 3), (1, 1)]
    assert HermodPriorityAnalysis.from_workload(
        wl, hermod_records=records).coflows["j0:i0:forward:pp"].microbatch_id == 0


def test_post_ga_dp_is_assigned_to_last_microbatch():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(0, CommType.PP_SEND, 0, 0),
        _flow(1, CommType.DP_ALLREDUCE, 2, 4),
    ])
    records = HermodAicbMetadataAdapter(_header()).apply(wl)
    assert records[1].microbatch_id == 1
    assert records[1].provenance == "aicb_dp_post_ga"


def test_adapter_recovers_one_lid_for_an_attention_mlp_transformer_layer():
    items = [
        AicbWorkItem("embedding_layer", 0, "NONE", 0, 0, "NONE", 0, 0, "NONE", 0, 0),
        AicbWorkItem("attention_layer", 0, "NONE", 0, 0, "NONE", 0, 0, "NONE", 0, 0),
        AicbWorkItem("mlp_layer", 0, "NONE", 0, 0, "NONE", 0, 0, "NONE", 0, 0),
    ]
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.DP_ALLREDUCE, 0, 0),
        _flow(2, CommType.DP_ALLREDUCE, 0, 1),
        _flow(3, CommType.DP_ALLREDUCE, 0, 2),
    ])
    for task in wl.tasks:
        task.item_id = task.layer_id
    records = HermodAicbMetadataAdapter(_header(), items).apply(wl)
    by_id = {r.task_id: r for r in records.values()}
    assert [by_id[t].logical_layer_id for t in (1, 2, 3)] == [0, 1, 1]
    assert by_id[2].mapping_rule == "attention_mlp_transformer_layer"
    assert by_id[3].source_operation == "mlp_layer"


def test_adapter_and_analysis_reject_ep_by_default():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.EP_ALLTOALL, 0, 0),
    ])
    with pytest.raises(ValueError, match="EP is disabled"):
        HermodAicbMetadataAdapter(_header()).apply(wl)


def test_adapter_supports_explicit_ep_enable_mode():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.EP_ALLTOALL, 0, 0),
    ])
    records = HermodAicbMetadataAdapter(
        _header(), ep_mode=HermodEpMode.ENABLE,
    ).apply(wl)
    assert records[1].coflow_type == "ep"


def test_adapter_rejects_unmapped_negative_layer_id():
    wl = P2PWorkload("1", Meta(0, 2), tasks=[
        _flow(1, CommType.PP_SEND, 0, -1),
    ])
    with pytest.raises(ValueError, match="layer position"):
        HermodAicbMetadataAdapter(_header()).apply(wl)


def test_job_merger_removes_hermod_fields():
    """Hermod fields are no longer on Task; job_merger won't propagate them."""
    task = _flow(1, CommType.PP_SEND, 0, 0)
    wl = P2PWorkload("1", Meta(1, 2), jobs=[Job(0)], tasks=[task])
    merged = JobMerger().merge([wl]).merged_workload.tasks[0]
    assert not hasattr(merged, "coflow_id")


def test_adapter_maps_mixtral_rows_and_separates_ep_collectives():
    names = [
        "embedding_layer",
        "attention_column", "attention_row",
        "mlp_moelayer", "mlp_moelayer", "mlp_moelayer",
        "attention_column", "attention_row",
        "mlp_moelayer", "mlp_moelayer",
        "final_column",
    ]
    items = [
        AicbWorkItem(name, 0, "NONE", 0, 0, "NONE", 0, 0, "NONE", 0, 0)
        for name in names
    ]
    tasks = []
    for lid in range(len(items)):
        task = _flow(lid + 1, CommType.EP_ALLTOALL, 0, lid)
        task.item_id = lid
        tasks.append(task)
    wl = P2PWorkload("1", Meta(0, 2), tasks=tasks)
    records = HermodAicbMetadataAdapter(
        _header(), items, ep_mode=HermodEpMode.ENABLE,
    ).apply(wl)

    assert [records[tid].logical_layer_id for tid in range(1, 12)] == [
        0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
    ]
    assert records[4].mapping_rule == "mixtral_attention_moe_transformer_layer"
    assert records[4].coflow_id != records[5].coflow_id


def test_mixtral_ep_collective_reaches_priority_analysis_end_to_end():
    def item(name, fwd="NONE", bwd="NONE"):
        return AicbWorkItem(
            name, 0, fwd, 1024 if fwd != "NONE" else 0,
            0, bwd, 1024 if bwd != "NONE" else 0,
            0, "NONE", 0, 0,
        )

    items = [
        item("embedding_layer"),
        item("attention_column"),
        item("attention_row"),
        item("mlp_moelayer", "ALLTOALL_EP", "ALLTOALL_EP"),
        item("mlp_moelayer", "ALLTOALL_EP", "ALLTOALL_EP"),
        item("final_column"),
    ]
    header = AicbHeader(
        tp=1, ep=2, pp=1, vpp=1, ga=1, all_gpus=2, pp_comm_size=0,
    )
    job = Job(
        job_id=0, assigned_nodes=[0, 1],
        parallelism=ParallelismConfig(tp=1, dp=2, pp=1, ep=2),
    )
    workload = WorkloadBuilder().build_from_aicb(header, items, job)
    records = HermodAicbMetadataAdapter(
        header, items, ep_mode=HermodEpMode.ENABLE,
    ).apply(workload)
    analysis = HermodPriorityAnalysis.from_workload(
        workload, records, ep_mode=HermodEpMode.ENABLE,
    )

    ep_tasks = [task for task in workload.get_flow_tasks()
                if task.comm_type == CommType.EP_ALLTOALL]
    assert len(ep_tasks) == 8
    assert len(analysis.coflows) == 4
    assert {coflow.logical_layer_id for coflow in analysis.coflows.values()} == {1}

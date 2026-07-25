"""Golden DAG tests for direct basic-Chimera pipeline expansion."""

import pytest

from src.executor.dynamic.pipeline_job_expander import PipelineJobExpander
from src.static_analysis.passes.pipeline_task_serializers import (
    BidirectionalPipelineSerializer,
)
from src.workload_format.compact_workload import (
    JobExpansionInfo,
    TaskIdAllocator,
)
from src.workload_format.schema import CommType, Job, ParallelismConfig, Phase
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.builders.bidirectional_pipeline_builder import (
    BidirectionalPipelineWorkloadBuilder,
)
from src.workload_generator.inference_profile import InferenceProfileStore


def _item(layer: int) -> AicbWorkItem:
    return AicbWorkItem(
        name=f"layer_{layer}",
        forward_compute_time=10_000,
        forward_comm="ALLREDUCE",
        forward_comm_size=1024,
        backward_compute_time=20_000,
        backward_comm="ALLREDUCE",
        backward_comm_size=1024,
        dp_compute_time=5_000,
        dp_comm="NONE",
        dp_comm_size=0,
        process_time=100,
    )


def _build(*, pp=4, tp=2, ga=8):
    nodes = [10 + index * 3 for index in range(pp * tp)]
    header = AicbHeader(
        tp=tp,
        ep=1,
        pp=pp,
        vpp=2 * pp,
        ga=ga,
        all_gpus=len(nodes),
        pp_comm_size=4096,
    )
    items = [
        _item(layer)
        for _microbatch in range(ga)
        for layer in range(2)
    ]
    job = Job(
        job_id=0,
        assigned_nodes=nodes,
        parallelism=ParallelismConfig(tp=tp, pp=pp),
    )
    sidecar = {}
    workload = BidirectionalPipelineWorkloadBuilder(
        sidecar,
        gradient_sync_bytes=8192,
    ).build_from_aicb(header, items, job)
    return workload, sidecar, nodes


def _compute(workload, node, iteration, phase, layer):
    return next(
        task for task in workload.tasks
        if task.is_compute()
        and task.node == node
        and task.iteration == iteration
        and task.phase is phase
        and task.layer_id == layer
    )


def test_direct_builder_materializes_opposing_endpoints():
    workload, sidecar, nodes = _build()
    pp_flows = [
        task for task in workload.tasks if task.comm_type is CommType.PP_SEND
    ]
    assert len(pp_flows) == 8 * 3 * 2 * 2

    down = next(
        task for task in pp_flows
        if task.iteration == 0
        and task.phase is Phase.FORWARD
        and sidecar[task.task_id].logical_boundary_id == 0
    )
    up = next(
        task for task in pp_flows
        if task.iteration == 4
        and task.phase is Phase.FORWARD
        and sidecar[task.task_id].logical_boundary_id == 0
    )
    assert (down.src, down.dst) == (nodes[0], nodes[2])
    assert (up.src, up.dst) == (nodes[6], nodes[4])
    assert sidecar[down.task_id].direction == "down"
    assert sidecar[up.task_id].direction == "up"
    assert workload.validate() == []


def test_up_activation_and_gradient_are_compute_tp_pp_compute():
    workload, sidecar, nodes = _build()
    task_by_id = {task.task_id: task for task in workload.tasks}
    activation = next(
        task for task in workload.tasks
        if task.comm_type is CommType.PP_SEND
        and task.iteration == 4
        and task.phase is Phase.FORWARD
        and sidecar[task.task_id].logical_boundary_id == 0
        and task.src == nodes[6]
    )
    gradient = next(
        task for task in workload.tasks
        if task.comm_type is CommType.PP_SEND
        and task.iteration == 4
        and task.phase is Phase.BACKWARD_INPUT
        and sidecar[task.task_id].logical_boundary_id == 0
        and task.src == nodes[4]
    )

    assert all(
        task_by_id[dependency].comm_type is CommType.TP_ALLREDUCE_RING
        and task_by_id[dependency].dst == activation.src
        for dependency in activation.deps
    )
    assert all(
        task_by_id[dependency].comm_type is CommType.TP_ALLREDUCE_RING
        and task_by_id[dependency].dst == gradient.src
        for dependency in gradient.deps
    )
    assert activation.task_id in _compute(
        workload, nodes[4], 4, Phase.FORWARD, 0,
    ).deps
    assert gradient.task_id in _compute(
        workload, nodes[6], 4, Phase.BACKWARD_INPUT, 1,
    ).deps


def test_serializer_consumes_direction_sidecar():
    workload, sidecar, _ = _build()
    serializer = BidirectionalPipelineSerializer(
        expansion_task_info=sidecar,
    )
    plan = serializer.serialize(workload)
    assert serializer.validate(workload, plan.compute_order) == []
    for task_id, schedule_info in serializer.task_info.items():
        expansion_info = sidecar.get(task_id)
        if expansion_info is None:
            continue
        assert schedule_info.direction == expansion_info.direction
        assert schedule_info.pipeline_id == expansion_info.pipeline_id
        assert schedule_info.logical_stage_id == expansion_info.logical_stage_id


def test_dynamic_direct_builder_preserves_ids_and_directions(tmp_path):
    trace = tmp_path / "bidirectional-training.txt"
    rows = [
        f"layer_{layer} -1 10000 NONE 0 20000 NONE 0 5000 NONE 0 100"
        for _microbatch in range(4)
        for layer in range(2)
    ]
    trace.write_text(
        "\n".join([
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            "model_parallel_NPU_group: 1 ep: 1 pp: 2 vpp: 4 ga: 4 "
            "all_gpus: 2 checkpoints: 0 checkpoint_initiates: 0 pp_comm 4096",
            str(len(rows)),
            *rows,
        ]),
        encoding="utf-8",
    )
    sidecar = {}
    expander = PipelineJobExpander(
        TaskIdAllocator(start=50),
        InferenceProfileStore(),
        pipeline_mode="bidirectional",
        pipeline_task_info=sidecar,
        pipeline_gradient_sync_bytes=8192,
    )
    info = JobExpansionInfo(job_type="training", trace_src=str(trace))
    first = expander.expand_job(
        Job(0, assigned_nodes=[9, 2], parallelism=ParallelismConfig(pp=2)),
        info,
    )
    second = expander.expand_job(
        Job(1, assigned_nodes=[9, 2], parallelism=ParallelismConfig(pp=2)),
        info,
    )

    first_ids = {task.task_id for task in first.tasks}
    second_ids = {task.task_id for task in second.tasks}
    assert first_ids.isdisjoint(second_ids)
    assert min(first_ids) == 50
    assert max(first_ids) < min(second_ids)
    assert {
        metadata.direction
        for task_id, metadata in sidecar.items()
        if task_id in first_ids
        and metadata.pipeline_id >= 0
    } == {"down", "up"}
    assert len([
        task for task in first.tasks if task.comm_type is CommType.PP_SEND
    ]) == 8


@pytest.mark.parametrize(
    ("pp", "ga", "message"),
    [
        (3, 6, "even pp"),
        (4, 7, "even microbatch"),
    ],
)
def test_shape_constraints_are_explicit(pp, ga, message):
    with pytest.raises(ValueError, match=message):
        _build(pp=pp, tp=1, ga=ga)


def test_chimera_gradient_sync_joins_mirrored_replicas():
    workload, sidecar, nodes = _build(pp=4, tp=1, ga=4)
    task_by_id = {task.task_id: task for task in workload.tasks}
    sync_flows = [
        task for task in workload.tasks
        if sidecar.get(task.task_id) is not None
        and sidecar[task.task_id].task_role == "chimera_gradient_sync"
    ]

    assert sync_flows
    shard_zero = [
        task for task in sync_flows
        if sidecar[task.task_id].model_shard_id == 0
    ]
    assert {(task.src, task.dst) for task in shard_zero} == {
        (nodes[0], nodes[3]),
        (nodes[3], nodes[0]),
    }
    assert {task.size_bytes for task in shard_zero} == {4096}
    assert sum(task.size_bytes for task in shard_zero) == 16384
    assert all(
        task_by_id[dependency].phase is Phase.BACKWARD_WEIGHT
        for task in shard_zero
        for dependency in task.deps
        if task_by_id[dependency].is_compute()
    )
    assert workload.validate() == []


def test_chimera_supports_even_microbatch_count_below_pipeline_depth():
    workload, _, _ = _build(pp=4, tp=1, ga=2)
    assert workload.validate() == []

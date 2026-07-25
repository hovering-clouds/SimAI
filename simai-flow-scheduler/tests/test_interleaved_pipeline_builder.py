"""Golden DAG tests for direct Interleaved 1F1B expansion."""

from src.executor.dynamic.pipeline_job_expander import PipelineJobExpander
from src.static_analysis.passes.pipeline_task_serializers import (
    InterleavedOneFOneBSerializer,
)
from src.workload_format.compact_workload import (
    JobExpansionInfo,
    TaskIdAllocator,
)
from src.workload_format.schema import CommType, Job, ParallelismConfig, Phase
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.inference_profile import InferenceProfileStore
from src.workload_generator.builders.interleaved_pipeline_builder import (
    InterleavedPipelineWorkloadBuilder,
)


def _item(layer: int, *, with_tp: bool = True) -> AicbWorkItem:
    comm = "ALLREDUCE" if with_tp else "NONE"
    size = 1024 if with_tp else 0
    return AicbWorkItem(
        name=f"layer_{layer}",
        forward_compute_time=10_000,
        forward_comm=comm,
        forward_comm_size=size,
        backward_compute_time=20_000,
        backward_comm=comm,
        backward_comm_size=size,
        dp_compute_time=5_000,
        dp_comm="NONE",
        dp_comm_size=0,
        process_time=100,
    )


def _build(
    *,
    pp: int = 2,
    tp: int = 2,
    ga: int = 2,
    layers: int = 4,
    vpp: int = 2,
    nodes: list[int] | None = None,
    with_tp: bool = True,
):
    nodes = nodes or list(range(pp * tp))
    header = AicbHeader(
        tp=tp,
        ep=1,
        pp=pp,
        vpp=layers * pp,
        ga=ga,
        all_gpus=pp * tp,
        pp_comm_size=4096 if pp > 1 else 0,
    )
    items = [_item(layer, with_tp=with_tp)
             for _microbatch in range(ga) for layer in range(layers)]
    job = Job(
        job_id=0,
        assigned_nodes=nodes,
        parallelism=ParallelismConfig(tp=tp, pp=pp),
    )
    info = {}
    workload = InterleavedPipelineWorkloadBuilder(vpp, info).build_from_aicb(
        header, items, job,
    )
    return workload, info


def _compute(workload, node, iteration, phase, layer):
    return next(
        task for task in workload.tasks
        if task.is_compute()
        and task.node == node
        and task.iteration == iteration
        and task.phase is phase
        and task.layer_id == layer
    )


def test_direct_builder_materializes_every_logical_boundary():
    workload, info = _build()
    pp_flows = [
        task for task in workload.tasks if task.comm_type is CommType.PP_SEND
    ]

    assert len(pp_flows) == 2 * 3 * 2 * 2
    for microbatch in range(2):
        for direction in ("forward", "backward"):
            assert {
                metadata.logical_boundary_id
                for metadata in info.values()
                if metadata.microbatch_id == microbatch
                and metadata.direction == direction
            } == {0, 1, 2}
    assert workload.validate() == []


def test_direct_builder_wrap_is_compute_tp_pp_compute():
    workload, info = _build()
    task_by_id = {task.task_id: task for task in workload.tasks}
    wrap = next(
        task for task in workload.tasks
        if task.comm_type is CommType.PP_SEND
        and task.iteration == 0
        and task.phase is Phase.FORWARD
        and task.src == 2
        and task.dst == 0
        and info[task.task_id].logical_boundary_id == 1
    )

    assert wrap.layer_id == 1
    assert all(
        task_by_id[dep].comm_type is CommType.TP_ALLREDUCE_RING
        and task_by_id[dep].dst == wrap.src
        for dep in wrap.deps
    )
    receiver = _compute(workload, 0, 0, Phase.FORWARD, 2)
    assert wrap.task_id in receiver.deps
    assert not any(
        task_by_id[dep].layer_id == 1
        and task_by_id[dep].comm_type is CommType.TP_ALLREDUCE_RING
        for dep in receiver.deps
    )


def test_direct_builder_supports_noncontiguous_ranks_and_serializer():
    workload, info = _build(
        nodes=[10, 12, 20, 22],
        with_tp=False,
    )
    wrap = next(
        task for task in workload.tasks
        if task.comm_type is CommType.PP_SEND
        and task.phase is Phase.FORWARD
        and task.src == 20
        and task.dst == 10
    )
    assert info[wrap.task_id].logical_boundary_id == 1

    serializer = InterleavedOneFOneBSerializer(
        virtual_pipeline_size=2,
        interleave_group_size=2,
        expansion_task_info=info,
    )
    plan = serializer.serialize(workload)
    assert serializer.validate(workload, plan.compute_order) == []


def test_direct_builder_pp_one_uses_local_chunk_boundary():
    workload, _ = _build(
        pp=1,
        tp=1,
        ga=2,
        layers=4,
        with_tp=False,
    )
    assert not [
        task for task in workload.tasks if task.comm_type is CommType.PP_SEND
    ]
    assert _compute(
        workload, 0, 0, Phase.FORWARD, 1,
    ).task_id in _compute(
        workload, 0, 0, Phase.FORWARD, 2,
    ).deps
    assert _compute(
        workload, 0, 0, Phase.BACKWARD_INPUT, 2,
    ).task_id in _compute(
        workload, 0, 0, Phase.BACKWARD_INPUT, 1,
    ).deps


def test_dynamic_direct_builder_preserves_global_ids_and_sidecar(tmp_path):
    trace = tmp_path / "vpp-training.txt"
    rows = [
        f"layer_{layer} -1 10000 NONE 0 20000 NONE 0 5000 NONE 0 100"
        for layer in range(4)
    ]
    trace.write_text(
        "\n".join([
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            "model_parallel_NPU_group: 1 ep: 1 pp: 2 vpp: 8 ga: 1 "
            "all_gpus: 2 checkpoints: 0 checkpoint_initiates: 0 pp_comm 4096",
            str(len(rows)),
            *rows,
        ]),
        encoding="utf-8",
    )
    sidecar = {}
    expander = PipelineJobExpander(
        TaskIdAllocator(),
        InferenceProfileStore(),
        pipeline_mode="interleaved_1f1b",
        pipeline_vpp=2,
        pipeline_task_info=sidecar,
    )
    info = JobExpansionInfo(job_type="training", trace_src=str(trace))
    first = expander.expand_job(
        Job(0, assigned_nodes=[0, 1], parallelism=ParallelismConfig(pp=2)),
        info,
    )
    second = expander.expand_job(
        Job(1, assigned_nodes=[0, 1], parallelism=ParallelismConfig(pp=2)),
        info,
    )

    first_ids = {task.task_id for task in first.tasks}
    second_ids = {task.task_id for task in second.tasks}
    assert first_ids.isdisjoint(second_ids)
    assert max(first_ids) < min(second_ids)
    assert set(sidecar) <= first_ids | second_ids
    assert all(metadata.task_id == task_id for task_id, metadata in sidecar.items())
    assert len([
        task for task in first.tasks if task.comm_type is CommType.PP_SEND
    ]) == 6

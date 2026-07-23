"""Golden DAG tests for the strategy-owned interleaved pipeline overlay."""

import pytest

from src.static_analysis.passes.pipeline_task_serializers import (
    BidirectionalPipelineSerializer,
    InterleavedOneFOneBSerializer,
)
from src.static_analysis.passes.hermod_metadata import HermodAicbMetadataAdapter
from src.executor.pipeline_job_expander import PipelineJobExpander
from src.workload_format.compact_workload import JobExpansionInfo, TaskIdAllocator
from src.workload_format.schema import (
    CommType,
    Job,
    Meta,
    P2PWorkload,
    ParallelismConfig,
    Phase,
    Task,
    TaskType,
)
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.pipeline_workload_overlay import (
    BidirectionalPipelineWorkloadOverlay,
    InterleavedOneFOneBWorkloadOverlay,
)
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.inference_profile import InferenceProfileStore


def _item(index: int, *, with_tp: bool = True) -> AicbWorkItem:
    comm = "ALLREDUCE" if with_tp else "NONE"
    size = 1024 if with_tp else 0
    return AicbWorkItem(
        name=f"layer_{index}",
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
    items = [_item(i % layers, with_tp=with_tp) for i in range(ga * layers)]
    job = Job(
        job_id=0,
        assigned_nodes=nodes,
        parallelism=ParallelismConfig(tp=tp, pp=pp),
    )
    return WorkloadBuilder().build_from_aicb(header, items, job)


def _compute(workload, node, iteration, phase, layer):
    return next(
        task for task in workload.tasks
        if task.is_compute()
        and task.node == node
        and task.iteration == iteration
        and task.phase is phase
        and task.layer_id == layer
    )


def test_overlay_builds_wraparound_compute_tp_pp_compute_causality():
    original = _build()
    original_pp_ids = {
        task.task_id for task in original.tasks
        if task.comm_type is CommType.PP_SEND
    }

    result = InterleavedOneFOneBWorkloadOverlay(2).apply(original)
    workload = result.workload
    task_by_id = {task.task_id: task for task in workload.tasks}

    # pp=2, vpp=2 => three logical boundaries.  For every GA, TP lane and
    # direction there is one flow per boundary.
    pp_flows = [
        task for task in workload.tasks if task.comm_type is CommType.PP_SEND
    ]
    assert len(pp_flows) == 3 * 2 * 2 * 2
    assert set(result.removed_task_ids) == original_pp_ids
    assert not original.validate()
    assert not workload.validate()

    # Logical boundary 1 is the VPP wrap: physical stage 1/chunk 0 sends to
    # physical stage 0/chunk 1.  Lane tp=0 is rank 2 -> rank 0.
    wrap = next(
        flow for flow in pp_flows
        if flow.iteration == 0
        and flow.phase is Phase.FORWARD
        and flow.src == 2
        and flow.dst == 0
        and result.task_info[flow.task_id].logical_boundary_id == 1
    )
    assert wrap.layer_id == 1
    assert all(task_by_id[dep].is_flow() for dep in wrap.deps)
    assert all(task_by_id[dep].comm_type is CommType.TP_ALLREDUCE_RING for dep in wrap.deps)
    assert all(task_by_id[dep].dst == wrap.src for dep in wrap.deps)

    first_chunk_one = _compute(workload, 0, 0, Phase.FORWARD, 2)
    assert wrap.task_id in first_chunk_one.deps
    # The old local chunk0 -> chunk1 output is gone; only the remote VPP flow
    # supplies the cross-chunk activation.
    assert not any(
        task_by_id[dep].comm_type is CommType.TP_ALLREDUCE_RING
        and task_by_id[dep].layer_id == 1
        for dep in first_chunk_one.deps
    )

    reverse_wrap = next(
        flow for flow in pp_flows
        if flow.iteration == 0
        and flow.phase is Phase.BACKWARD_INPUT
        and flow.src == 0
        and flow.dst == 2
        and result.task_info[flow.task_id].logical_boundary_id == 1
    )
    chunk_zero_backward_entry = _compute(
        workload, 2, 0, Phase.BACKWARD_INPUT, 1,
    )
    assert reverse_wrap.task_id in chunk_zero_backward_entry.deps
    assert all(task_by_id[dep].dst == reverse_wrap.src for dep in reverse_wrap.deps)

    # The input object is not mutated.
    assert {
        task.task_id for task in original.tasks
        if task.comm_type is CommType.PP_SEND
    } == original_pp_ids


def test_overlay_preserves_within_chunk_edges_and_noncontiguous_lanes():
    workload = _build(
        pp=2,
        tp=2,
        ga=1,
        layers=5,
        nodes=[10, 12, 20, 22],
        with_tp=False,
    )
    transformed = InterleavedOneFOneBWorkloadOverlay(2).apply(workload).workload
    task_by_id = {task.task_id: task for task in transformed.tasks}

    # Five layers partition as chunk0=[0,1,2], chunk1=[3,4].
    layer_one = _compute(transformed, 10, 0, Phase.FORWARD, 1)
    assert _compute(transformed, 10, 0, Phase.FORWARD, 0).task_id in layer_one.deps
    layer_two = _compute(transformed, 10, 0, Phase.FORWARD, 2)
    assert layer_one.task_id in layer_two.deps

    wrap = next(
        task for task in transformed.tasks
        if task.comm_type is CommType.PP_SEND
        and task.phase is Phase.FORWARD
        and task.src == 20
        and task.dst == 10
    )
    assert task_by_id[wrap.deps[0]].node == 20
    assert wrap.task_id in _compute(
        transformed, 10, 0, Phase.FORWARD, 3,
    ).deps


def test_pp_one_uses_direct_virtual_chunk_edges_without_network_flow():
    original = _build(pp=1, tp=1, ga=1, layers=4, with_tp=False)
    result = InterleavedOneFOneBWorkloadOverlay(2).apply(original)
    workload = result.workload

    assert not [task for task in workload.tasks if task.comm_type is CommType.PP_SEND]
    assert _compute(workload, 0, 0, Phase.FORWARD, 1).task_id in _compute(
        workload, 0, 0, Phase.FORWARD, 2,
    ).deps
    assert _compute(workload, 0, 0, Phase.BACKWARD_INPUT, 2).task_id in _compute(
        workload, 0, 0, Phase.BACKWARD_INPUT, 1,
    ).deps


@pytest.mark.parametrize(
    ("pp", "vpp", "ga", "layers"),
    [(2, 2, 2, 4), (4, 2, 1, 4), (4, 4, 2, 8)],
)
def test_every_logical_boundary_exists_per_microbatch(pp, vpp, ga, layers):
    result = InterleavedOneFOneBWorkloadOverlay(vpp).apply(_build(
        pp=pp,
        tp=1,
        ga=ga,
        layers=layers,
        with_tp=False,
    ))
    expected_boundaries = set(range(pp * vpp - 1))

    for iteration in range(ga):
        for direction in ("forward", "backward"):
            assert {
                info.logical_boundary_id
                for info in result.task_info.values()
                if info.microbatch_id == iteration and info.direction == direction
            } == expected_boundaries


def test_overlay_dag_and_serializer_resource_edges_are_compatible():
    workload = InterleavedOneFOneBWorkloadOverlay(2).apply(_build()).workload
    serializer = InterleavedOneFOneBSerializer(
        virtual_pipeline_size=2,
        interleave_group_size=2,
    )

    plan = serializer.serialize(workload)

    assert serializer.validate(workload, plan.compute_order) == []
    assert all(
        len(order) == len([
            task for task in workload.tasks
            if task.is_compute() and task.node == node
        ])
        for node, order in plan.compute_order.items()
    )


def test_dynamic_pipeline_expander_reserves_noncolliding_extra_ids(tmp_path):
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
    allocator = TaskIdAllocator()
    expander = PipelineJobExpander(
        allocator,
        InferenceProfileStore(),
        pipeline_mode="interleaved_1f1b",
        pipeline_vpp=2,
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
    assert len([
        task for task in first.tasks if task.comm_type is CommType.PP_SEND
    ]) == 6
    assert set(first.entry_task_ids) <= first_ids
    assert set(first.terminal_task_ids) <= first_ids


def test_bidirectional_overlay_builds_opposing_compute_tp_pp_compute_dags():
    nodes = [10, 11, 20, 21, 30, 31, 40, 41]
    original = _build(pp=4, tp=2, ga=8, layers=3, nodes=nodes)
    original_pp_ids = {
        task.task_id for task in original.tasks
        if task.comm_type is CommType.PP_SEND
    }

    result = BidirectionalPipelineWorkloadOverlay().apply(original)
    workload = result.workload
    task_by_id = {task.task_id: task for task in workload.tasks}
    pp_flows = [
        task for task in workload.tasks if task.comm_type is CommType.PP_SEND
    ]

    # ga * (pp - 1) * tp lanes * (activation + gradient)
    assert len(pp_flows) == 8 * 3 * 2 * 2
    assert set(result.removed_task_ids) == original_pp_ids
    assert not workload.validate()

    down_activation = next(
        flow for flow in pp_flows
        if flow.iteration == 0
        and flow.phase is Phase.FORWARD
        and flow.src == 10
        and flow.dst == 20
    )
    down_gradient = next(
        flow for flow in pp_flows
        if flow.iteration == 0
        and flow.phase is Phase.BACKWARD_INPUT
        and flow.src == 20
        and flow.dst == 10
    )
    up_activation = next(
        flow for flow in pp_flows
        if flow.iteration == 4
        and flow.phase is Phase.FORWARD
        and flow.src == 40
        and flow.dst == 30
    )
    up_gradient = next(
        flow for flow in pp_flows
        if flow.iteration == 4
        and flow.phase is Phase.BACKWARD_INPUT
        and flow.src == 30
        and flow.dst == 40
    )

    for activation, gradient, direction in (
        (down_activation, down_gradient, "down"),
        (up_activation, up_gradient, "up"),
    ):
        assert activation.size_bytes == gradient.size_bytes == 4096
        assert result.task_info[activation.task_id].direction == direction
        assert result.task_info[gradient.task_id].direction == direction
        assert (gradient.src, gradient.dst) == (activation.dst, activation.src)
        assert all(task_by_id[dep].is_flow() for dep in activation.deps)
        assert all(
            task_by_id[dep].comm_type is CommType.TP_ALLREDUCE_RING
            and task_by_id[dep].dst == activation.src
            for dep in activation.deps
        )
        assert activation.task_id in _compute(
            workload, activation.dst, activation.iteration, Phase.FORWARD, 0,
        ).deps
        assert gradient.task_id in _compute(
            workload, gradient.dst, gradient.iteration,
            Phase.BACKWARD_INPUT, 2,
        ).deps

    up_info = result.task_info[up_activation.task_id]
    assert (
        up_info.pipeline_id,
        up_info.physical_stage_id,
        up_info.logical_stage_id,
        up_info.peer_physical_stage_id,
        up_info.peer_logical_stage_id,
    ) == (1, 3, 0, 2, 1)
    up_compute = _compute(workload, 40, 4, Phase.FORWARD, 0)
    assert result.task_info[up_compute.task_id].logical_stage_id == 0

    # The input object remains the original one-way pipeline.
    assert {
        task.task_id for task in original.tasks
        if task.comm_type is CommType.PP_SEND
    } == original_pp_ids


def test_bidirectional_overlay_has_every_boundary_for_each_lane_and_microbatch():
    result = BidirectionalPipelineWorkloadOverlay().apply(_build(
        pp=4,
        tp=2,
        ga=8,
        layers=2,
        nodes=[91, 7, 55, 3, 88, 2, 77, 1],
        with_tp=False,
    ))

    for iteration in range(8):
        expected_direction = "down" if iteration < 4 else "up"
        for role in ("pp_activation", "pp_gradient"):
            infos = [
                info for info in result.task_info.values()
                if info.microbatch_id == iteration and info.task_role == role
            ]
            assert len(infos) == 3 * 2
            assert {info.logical_boundary_id for info in infos} == {0, 1, 2}
            assert {info.direction for info in infos} == {expected_direction}


@pytest.mark.parametrize(
    ("pp", "ga", "message"),
    [(3, 6, "even pp"), (2, 5, "even microbatch"), (4, 6, "2 \\* pp")],
)
def test_bidirectional_overlay_rejects_shapes_not_supported_by_serializer(
    pp, ga, message,
):
    with pytest.raises(ValueError, match=message):
        BidirectionalPipelineWorkloadOverlay().apply(_build(
            pp=pp, tp=1, ga=ga, layers=1, with_tp=False,
        ))


def test_bidirectional_overlay_leaves_inference_pp_dag_unchanged():
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        jobs=[
            Job(0, assigned_nodes=[0, 1], parallelism=ParallelismConfig(pp=2)),
        ],
        tasks=[
            Task(
                0, 0, TaskType.COMPUTE, phase=Phase.PREFILL,
                node=0, duration_us=10,
            ),
            Task(
                1, 0, TaskType.FLOW, phase=Phase.PREFILL, deps=[0],
                src=0, dst=1, size_bytes=64, comm_type=CommType.PP_SEND,
            ),
            Task(
                2, 0, TaskType.COMPUTE, phase=Phase.PREFILL, deps=[1],
                node=1, duration_us=10,
            ),
        ],
    )

    result = BidirectionalPipelineWorkloadOverlay().apply(workload)

    assert result.removed_task_ids == result.added_task_ids == ()
    assert result.workload is not workload
    assert result.workload.tasks == workload.tasks


def test_bidirectional_overlay_serializer_and_hermod_direction_are_consistent():
    header = AicbHeader(
        tp=1, ep=1, pp=4, vpp=8, ga=8, all_gpus=4, pp_comm_size=4096,
    )
    result = BidirectionalPipelineWorkloadOverlay().apply(_build(
        pp=4,
        tp=1,
        ga=8,
        layers=2,
        nodes=[91, 7, 55, 3],
        with_tp=False,
    ))
    serializer = BidirectionalPipelineSerializer()

    plan = serializer.serialize(result.workload)
    records = HermodAicbMetadataAdapter(header).apply(
        result.workload,
        split_pp_by_direction=True,
    )

    assert serializer.validate(result.workload, plan.compute_order) == []
    assert all(
        records[task_id].coflow_id.endswith(
            ":p" + result.task_info[task_id].direction
        )
        for task_id in result.added_task_ids
    )


def test_dynamic_bidirectional_expander_preserves_global_task_id_space(tmp_path):
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
    expander = PipelineJobExpander(
        TaskIdAllocator(),
        InferenceProfileStore(),
        pipeline_mode="bidirectional",
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
    assert max(first_ids) < min(second_ids)
    assert len([
        task for task in first.tasks if task.comm_type is CommType.PP_SEND
    ]) == 8
    assert {
        info.direction for task_id, info in expander.pipeline_task_info.items()
        if task_id in first_ids
    } == {"down", "up"}

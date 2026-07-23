"""Tests for non-invasive advanced pipeline compute serializers."""

import pytest

from src.static_analysis.passes.pipeline_task_serializers import (
    BidirectionalPipelineSerializer,
    InterleavedOneFOneBSerializer,
    ZeroBubbleSerializer,
    build_pipeline_serializer,
)
from src.static_analysis.passes.task_serializer import (
    CppReferenceSerializer,
    OneFOneBSerializer,
)
from src.static_analysis.passes.topology_loader import NetworkTopology
from src.static_analysis.strategies.default_strategy import PipelineAnalyzer
from src.static_analysis.strategies.hermod_strategy import HermodAnalyzer
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.workload_format.schema import (
    Job,
    Meta,
    P2PWorkload,
    ParallelismConfig,
    Phase,
    Task,
    TaskType,
)


def _make_pipeline_workload(pp=4, ga=8, layers=4):
    tasks = []
    task_id = 0
    for node in range(pp):
        for microbatch_id in range(ga):
            for layer_id in range(layers):
                for phase in (
                    Phase.FORWARD,
                    Phase.BACKWARD_INPUT,
                    Phase.BACKWARD_WEIGHT,
                ):
                    tasks.append(Task(
                        task_id=task_id,
                        job_id=0,
                        type=TaskType.COMPUTE,
                        node=node,
                        duration_us=1,
                        iteration=microbatch_id,
                        phase=phase,
                        layer_id=layer_id,
                        item_id=layer_id,
                    ))
                    task_id += 1
    job = Job(
        job_id=0,
        assigned_nodes=list(range(pp)),
        parallelism=ParallelismConfig(pp=pp),
    )
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=pp),
        jobs=[job],
        tasks=tasks,
    )


def _operation_groups(serializer, plan, node):
    groups = []
    previous_slot = None
    for task_id in plan.compute_order[node]:
        info = serializer.task_info[task_id]
        if info.schedule_slot == previous_slot:
            continue
        previous_slot = info.schedule_slot
        groups.append((
            info.operation,
            info.microbatch_id,
            info.model_chunk_id,
            info.direction,
        ))
    return groups


def test_interleaved_1f1b_uses_vpp_chunks_and_stage_warmup():
    workload = _make_pipeline_workload(pp=4, ga=4, layers=4)
    serializer = InterleavedOneFOneBSerializer(
        virtual_pipeline_size=2,
        interleave_group_size=2,
    )

    plan = serializer.serialize(workload)

    # Last physical stage has two warmup virtual micro-steps, then alternates.
    assert _operation_groups(serializer, plan, 3)[:8] == [
        ("F", 0, 0, "down"),
        ("F", 1, 0, "down"),
        ("F", 0, 1, "down"),
        ("BW", 0, 1, "down"),
        ("F", 1, 1, "down"),
        ("BW", 1, 1, "down"),
        ("F", 2, 0, "down"),
        ("BW", 0, 0, "down"),
    ]
    chunk_by_layer = {
        task.layer_id: serializer.task_info[task.task_id].model_chunk_id
        for task in workload.tasks
        if task.node == 0 and task.iteration == 0 and task.phase == Phase.FORWARD
    }
    assert chunk_by_layer == {0: 0, 1: 0, 2: 1, 3: 1}
    chunk_one_task = next(
        task for task in workload.tasks
        if task.node == 3 and task.iteration == 0
        and task.phase == Phase.FORWARD and task.layer_id == 2
    )
    assert serializer.task_info[chunk_one_task.task_id].logical_stage_id == 7
    assert all(
        len(order) == 4 * 4 * 3 for order in plan.compute_order.values()
    )


def test_interleaved_1f1b_rejects_more_chunks_than_local_layers():
    workload = _make_pipeline_workload(pp=2, ga=2, layers=1)
    serializer = InterleavedOneFOneBSerializer(virtual_pipeline_size=2)

    with pytest.raises(ValueError, match="one local layer per virtual chunk"):
        serializer.serialize(workload)


def test_zero_bubble_splits_input_and_weight_backward():
    workload = _make_pipeline_workload(pp=4, ga=8, layers=1)
    serializer = ZeroBubbleSerializer()

    plan = serializer.serialize(workload)
    stage_zero = _operation_groups(serializer, plan, 0)

    assert stage_zero[:10] == [
        ("F", 0, 0, "down"),
        ("F", 1, 0, "down"),
        ("F", 2, 0, "down"),
        ("F", 3, 0, "down"),
        ("B", 0, 0, "down"),
        ("F", 4, 0, "down"),
        ("W", 0, 0, "down"),
        ("B", 1, 0, "down"),
        ("F", 5, 0, "down"),
        ("W", 1, 0, "down"),
    ]
    for microbatch_id in range(8):
        b_position = next(
            index for index, group in enumerate(stage_zero)
            if group[:2] == ("B", microbatch_id)
        )
        w_position = next(
            index for index, group in enumerate(stage_zero)
            if group[:2] == ("W", microbatch_id)
        )
        assert b_position < w_position


def test_bidirectional_uses_dualpipe_regions_and_mirrored_stage_sidecar():
    workload = _make_pipeline_workload(pp=4, ga=8, layers=1)
    serializer = BidirectionalPipelineSerializer()

    plan = serializer.serialize(workload)
    stage_zero = _operation_groups(serializer, plan, 0)

    assert stage_zero[:9] == [
        ("F", 0, 0, "down"),
        ("F", 1, 0, "down"),
        ("F", 2, 0, "down"),
        ("F", 4, 0, "up"),
        ("B", 4, 0, "up"),
        ("W", 4, 0, "up"),
        ("F", 5, 0, "up"),
        ("F", 3, 0, "down"),
        ("BW", 5, 0, "up"),
    ]
    up_task_id = next(
        task_id for task_id in plan.compute_order[0]
        if serializer.task_info[task_id].direction == "up"
    )
    assert serializer.task_info[up_task_id].stage_id == 0
    assert serializer.task_info[up_task_id].logical_stage_id == 3
    assert {info.pipeline_id for info in serializer.task_info.values()} == {0, 1}


@pytest.mark.parametrize(
    ("pp", "ga", "message"),
    [(3, 6, "even pp"), (4, 7, "even microbatch"), (4, 6, "2 \\* pp")],
)
def test_bidirectional_rejects_unsupported_shapes(pp, ga, message):
    workload = _make_pipeline_workload(pp=pp, ga=ga, layers=1)

    with pytest.raises(ValueError, match=message):
        BidirectionalPipelineSerializer().serialize(workload)


def test_strategy_order_is_legalized_against_the_original_dag():
    # ZB's preferred pp=1 order starts F0, B0, F1.  The explicit F1 -> B0
    # dependency forces the safe projection F0, F1, B0 without changing Task.
    tasks = [
        Task(0, 0, TaskType.COMPUTE, node=0, duration_us=1,
             iteration=0, phase=Phase.FORWARD),
        Task(1, 0, TaskType.COMPUTE, node=0, duration_us=1, deps=[2],
             iteration=0, phase=Phase.BACKWARD_INPUT),
        Task(2, 0, TaskType.COMPUTE, node=0, duration_us=1,
             iteration=1, phase=Phase.FORWARD),
        Task(3, 0, TaskType.COMPUTE, node=0, duration_us=1, deps=[1],
             iteration=0, phase=Phase.BACKWARD_WEIGHT),
    ]
    workload = P2PWorkload(
        "1.0",
        Meta(1, 1),
        jobs=[Job(0, assigned_nodes=[0], parallelism=ParallelismConfig())],
        tasks=tasks,
    )
    serializer = ZeroBubbleSerializer()

    plan = serializer.serialize(workload)

    assert plan.compute_order[0] == [0, 2, 1, 3]
    assert serializer.validate(workload, plan.compute_order) == []
    assert [serializer.task_info[task_id].local_order for task_id in plan.compute_order[0]] \
        == [0, 1, 2, 3]


def test_factory_preserves_existing_modes_and_registers_new_modes():
    workload = _make_pipeline_workload(pp=2, ga=4, layers=2)

    gpipe = build_pipeline_serializer("gpipe", workload)
    one_f_one_b = build_pipeline_serializer("1f1b", workload)
    assert isinstance(gpipe, CppReferenceSerializer)
    assert isinstance(one_f_one_b, OneFOneBSerializer)
    assert gpipe.serialize(workload) == CppReferenceSerializer().serialize(workload)
    assert one_f_one_b.serialize(workload) == OneFOneBSerializer(
        pp=2,
        node_to_stage={0: 0, 1: 1},
    ).serialize(workload)
    assert isinstance(
        build_pipeline_serializer("interleaved_1f1b", workload),
        InterleavedOneFOneBSerializer,
    )
    assert isinstance(build_pipeline_serializer("zero_bubble", workload), ZeroBubbleSerializer)
    assert isinstance(
        build_pipeline_serializer("bidirectional", workload),
        BidirectionalPipelineSerializer,
    )
    with pytest.raises(ValueError, match="Unsupported pipeline mode"):
        build_pipeline_serializer("unknown", workload)


@pytest.mark.parametrize(
    "mode",
    ["interleaved_1f1b", "zero_bubble", "bidirectional"],
)
def test_analyzers_share_advanced_pipeline_compute_order(mode):
    workload = _make_pipeline_workload(pp=4, ga=8, layers=4)
    topology = NetworkTopology()

    default_analyzer = PipelineAnalyzer(
        topology,
        mode,
        pipeline_vpp=2,
        interleave_group_size=2,
    )
    default = default_analyzer.analyze(workload)
    hermod_analyzer = HermodAnalyzer(
        topology,
        pipeline_mode=mode,
        pipeline_vpp=2,
        interleave_group_size=2,
    )
    hermod = hermod_analyzer.analyze(workload, hermod_records={})
    puppeteer_analyzer = PuppeteerAnalyzer(
        topology,
        serializer=mode,
        pipeline_vpp=2,
        interleave_group_size=2,
    )
    puppeteer = puppeteer_analyzer.analyze(workload)

    assert default.execution_plan.compute_order == hermod.execution_plan.compute_order
    assert default.execution_plan.compute_order == puppeteer.execution_plan.compute_order
    assert default_analyzer.pipeline_task_info.keys() \
        == hermod_analyzer.pipeline_task_info.keys()
    assert default_analyzer.pipeline_task_info.keys() \
        == puppeteer_analyzer.pipeline_task_info.keys()

"""Golden schedule, DAG, and traffic tests for DeepSeek-style DualPipe."""

from collections import Counter

import pytest

from src.executor.dynamic.pipeline_job_expander import PipelineJobExpander
from src.static_analysis.passes.pipeline_task_serializers import (
    DualPipeSerializer,
)
from src.workload_format.compact_workload import (
    JobExpansionInfo,
    TaskIdAllocator,
)
from src.workload_format.schema import CommType, Job, ParallelismConfig, Phase
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.builders.dualpipe_pipeline_builder import (
    DualPipePipelineWorkloadBuilder,
    build_dualpipe_schedule,
)
from src.workload_generator.inference_profile import InferenceProfileStore


def _layer(layer: int, *, ep: bool = False) -> AicbWorkItem:
    comm = "ALLTOALL_EP" if ep else "ALLREDUCE"
    return AicbWorkItem(
        name=f"layer_{layer}",
        forward_compute_time=10_000,
        forward_comm=comm,
        forward_comm_size=1024,
        backward_compute_time=20_000,
        backward_comm=comm,
        backward_comm_size=1024,
        dp_compute_time=5_000,
        dp_comm="NONE",
        dp_comm_size=0,
        process_time=100,
    )


def _gradient_row(size: int = 8192) -> AicbWorkItem:
    return AicbWorkItem(
        name="grad_param_comm",
        forward_compute_time=0,
        forward_comm="NONE",
        forward_comm_size=0,
        backward_compute_time=0,
        backward_comm="NONE",
        backward_comm_size=0,
        dp_compute_time=0,
        dp_comm="ALLREDUCE",
        dp_comm_size=size,
        process_time=100,
    )


def _build(
    *,
    pp=4,
    tp=1,
    ep=1,
    dp=1,
    ga=8,
    include_gradient=True,
    overlap_model="conservative",
    overlap_factor=None,
):
    effective_dp = dp * ep  # new model: dp includes the ep factor
    nodes = [10 + index * 3 for index in range(pp * tp * effective_dp)]
    header = AicbHeader(
        tp=tp,
        ep=ep,
        pp=pp,
        vpp=2 * pp,
        ga=ga,
        all_gpus=len(nodes),
        pp_comm_size=4096,
    )
    items = [_gradient_row()] if include_gradient else []
    items.extend(
        _layer(layer, ep=ep > 1)
        for _microbatch in range(ga)
        for layer in range(2)
    )
    job = Job(
        job_id=0,
        assigned_nodes=nodes,
        parallelism=ParallelismConfig(tp=tp, dp=effective_dp, pp=pp, ep=ep),
    )
    sidecar = {}
    workload = DualPipePipelineWorkloadBuilder(
        sidecar,
        gradient_sync_bytes=None if include_gradient else 8192,
        overlap_model=overlap_model,
        overlap_factor=overlap_factor,
    ).build_from_aicb(header, items, job)
    return workload, sidecar, nodes


@pytest.mark.parametrize(("pp", "ga"), [(2, 4), (4, 8), (4, 10), (8, 20)])
def test_official_schedule_consumes_every_microbatch_once(pp, ga):
    for stage in range(pp):
        tokens = build_dualpipe_schedule(pp, ga, stage)
        counts = Counter()
        for token in tokens:
            counts[(token.module_replica_id, "F")] += token.operation == "F"
            counts[(token.module_replica_id, "B")] += token.operation in {"B", "BW"}
            counts[(token.module_replica_id, "W")] += token.operation in {"W", "BW"}
        assert counts == Counter({
            (0, "F"): ga // 2,
            (0, "B"): ga // 2,
            (0, "W"): ga // 2,
            (1, "F"): ga // 2,
            (1, "B"): ga // 2,
            (1, "W"): ga // 2,
        })
        deferred = {
            token.weight_queue_index
            for token in tokens
            if token.operation == "B"
        }
        drained = {
            token.weight_queue_index
            for token in tokens
            if token.operation == "W"
        }
        assert deferred == drained


def test_middle_rank_uses_official_first_step_four_exception():
    tokens = build_dualpipe_schedule(4, 8, 1)
    first_step_four = [token for token in tokens if token.schedule_step == 4][:2]
    assert [token.operation for token in first_step_four] == ["F", "BW"]
    assert all(token.middle_rank_special_case for token in first_step_four)
    assert all(token.overlap_pair_id is None for token in first_step_four)


def test_p8_m20_step_lengths_match_official_formulas():
    pp = 8
    ga = 20
    for stage in range(pp):
        half_rank = min(stage, pp - 1 - stage)
        tokens = build_dualpipe_schedule(pp, ga, stage)
        by_step = Counter(token.schedule_step for token in tokens)
        expected_repeats = {
            1: 2 * (pp // 2 - half_rank - 1),
            2: half_rank + 1,
            3: pp // 2 - half_rank - 1,
            4: ga // 2 - pp + half_rank + 1,
            5: pp // 2 - half_rank - 1,
            6: half_rank + 1,
            7: pp // 2 - half_rank - 1,
            8: half_rank + 1,
        }
        operations_per_repeat = {
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 3,
            6: 2,
            7: 2,
            8: 1,
        }
        assert by_step == Counter({
            step: repeats * operations_per_repeat[step]
            for step, repeats in expected_repeats.items()
            if repeats
        })


def test_builder_materializes_bidirectional_pp_dag_and_exact_traffic():
    workload, sidecar, nodes = _build(pp=4, tp=2, ga=8)
    task_by_id = {task.task_id: task for task in workload.tasks}
    pp_flows = [
        task for task in workload.tasks if task.comm_type is CommType.PP_SEND
    ]
    assert len(pp_flows) == 2 * 8 * (4 - 1) * 2
    assert sum(task.size_bytes for task in pp_flows) == len(pp_flows) * 4096

    down = next(
        task for task in pp_flows
        if task.iteration == 0
        and task.phase is Phase.FORWARD
        and sidecar[task.task_id].logical_boundary_id == 0
        and task.src == nodes[0]
    )
    up = next(
        task for task in pp_flows
        if task.iteration == 4
        and task.phase is Phase.FORWARD
        and sidecar[task.task_id].logical_boundary_id == 0
        and task.src == nodes[6]
    )
    assert (down.src, down.dst) == (nodes[0], nodes[2])
    assert (up.src, up.dst) == (nodes[6], nodes[4])
    assert sidecar[down.task_id].direction == "down"
    assert sidecar[up.task_id].direction == "up"
    assert all(
        task_by_id[dependency].phase is Phase.BACKWARD_INPUT
        for task in pp_flows
        if task.phase is Phase.BACKWARD_INPUT
        for dependency in task.deps
        if task_by_id[dependency].is_compute()
    )
    assert workload.validate() == []


def test_pp_gradient_ancestor_closure_excludes_weight_gradient():
    workload, _, _ = _build(pp=4, tp=2, ga=8)
    task_by_id = {task.task_id: task for task in workload.tasks}

    def ancestors(task_id):
        pending = list(task_by_id[task_id].deps)
        result = set()
        while pending:
            dependency = pending.pop()
            if dependency in result:
                continue
            result.add(dependency)
            pending.extend(task_by_id[dependency].deps)
        return result

    gradients = [
        task for task in workload.get_flow_tasks()
        if task.comm_type is CommType.PP_SEND
        and task.phase is Phase.BACKWARD_INPUT
    ]
    assert gradients
    for gradient in gradients:
        assert not any(
            task_by_id[dependency].phase is Phase.BACKWARD_WEIGHT
            for dependency in ancestors(gradient.task_id)
        )


def test_gradient_row_becomes_one_replica_aware_sync_not_duplicate_dp():
    workload, sidecar, nodes = _build(pp=4, tp=1, dp=1, ga=8)
    dp_flows = [
        task for task in workload.get_flow_tasks()
        if task.comm_type is CommType.DP_ALLREDUCE
    ]
    assert dp_flows
    assert all(
        sidecar[task.task_id].task_role == "dualpipe_gradient_sync"
        for task in dp_flows
    )
    shard_zero = [
        task for task in dp_flows
        if sidecar[task.task_id].model_shard_id == 0
    ]
    assert {(task.src, task.dst) for task in shard_zero} == {
        (nodes[0], nodes[3]),
        (nodes[3], nodes[0]),
    }
    assert all(task.iteration == 8 for task in dp_flows)


def test_ep_alltoall_preserves_participants_flow_count_and_bytes():
    workload, sidecar, _ = _build(
        pp=2,
        ep=2,
        ga=4,
        include_gradient=False,
    )
    ep_flows = [
        task for task in workload.get_flow_tasks()
        if task.comm_type is CommType.EP_ALLTOALL
    ]
    # P groups * E*(E-1) direct flows * M microbatches * two local
    # layers * forward/backward-input collectives.
    assert len(ep_flows) == 2 * (2 * 1) * 4 * 2 * 2
    assert {task.size_bytes for task in ep_flows} == {512}
    assert all(sidecar[task.task_id].component == "ep_alltoall" for task in ep_flows)


def test_replica_sync_bytes_include_external_dp_once():
    gradient_bytes = 8192
    workload, sidecar, _ = _build(pp=2, dp=2, ga=4)
    sync_flows = [
        task for task in workload.get_flow_tasks()
        if sidecar[task.task_id].task_role == "dualpipe_gradient_sync"
    ]
    # P shards * 2*(2D-1)*G bytes per ring group.
    assert sum(task.size_bytes for task in sync_flows) == (
        2 * 2 * (2 * 2 - 1) * gradient_bytes
    )


def test_serializer_uses_official_tokens_and_updates_sidecar():
    workload, sidecar, _ = _build(pp=4, ga=8)
    serializer = DualPipeSerializer(expansion_task_info=sidecar)
    plan = serializer.serialize(workload)

    assert serializer.validate(workload, plan.compute_order) == []
    assert all(
        info.preferred_slot is not None and info.final_local_order is not None
        for task_id, info in sidecar.items()
        if task_id in serializer.task_info
    )
    assert {
        info.schedule_name for info in serializer.task_info.values()
    } == {"dualpipe"}
    assert {
        info.pipeline_id
        for info in serializer.task_info.values()
        if info.microbatch_id is not None
    } == {0, 1}


@pytest.mark.parametrize(
    ("model", "factor"),
    [("ideal", None), ("profiled", 0.7)],
)
def test_overlap_envelope_calibrates_wall_time_and_keeps_audit(model, factor):
    workload, sidecar, _ = _build(
        pp=2,
        ga=4,
        overlap_model=model,
        overlap_factor=factor,
    )
    task_by_id = {task.task_id: task for task in workload.tasks}
    pair_ids = sorted({
        info.overlap_pair_id
        for info in sidecar.values()
        if info.overlap_pair_id is not None
    })
    assert pair_ids
    for pair_id in pair_ids:
        task_ids = [
            task_id
            for task_id, info in sidecar.items()
            if info.overlap_pair_id == pair_id and task_by_id[task_id].is_compute()
        ]
        original_forward = sum(
            sidecar[task_id].original_duration_us or 0
            for task_id in task_ids
            if task_by_id[task_id].phase is Phase.FORWARD
        )
        original_backward = sum(
            sidecar[task_id].original_duration_us or 0
            for task_id in task_ids
            if task_by_id[task_id].phase in {
                Phase.BACKWARD_INPUT,
                Phase.BACKWARD_WEIGHT,
            }
        )
        effective = sum(task_by_id[task_id].duration_us for task_id in task_ids)
        expected = (
            max(original_forward, original_backward)
            if model == "ideal"
            else round((original_forward + original_backward) * factor)
        )
        assert effective == expected
        assert all(
            sidecar[task_id].effective_duration_us
            == task_by_id[task_id].duration_us
            for task_id in task_ids
        )


def test_profiled_overlap_requires_explicit_factor():
    with pytest.raises(ValueError, match="requires overlap_factor"):
        DualPipePipelineWorkloadBuilder(overlap_model="profiled")


def test_dynamic_expansion_remaps_dualpipe_sidecar(tmp_path):
    trace = tmp_path / "dualpipe-training.txt"
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
        pipeline_mode="dualpipe",
        pipeline_task_info=sidecar,
        pipeline_gradient_sync_bytes=8192,
    )
    expanded = expander.expand_job(
        Job(0, assigned_nodes=[9, 2], parallelism=ParallelismConfig(pp=2)),
        JobExpansionInfo(job_type="training", trace_src=str(trace)),
    )

    ids = {task.task_id for task in expanded.tasks}
    assert min(ids) == 50
    assert ids == set(sidecar)
    assert all(info.task_id == task_id for task_id, info in sidecar.items())
    assert {
        info.direction
        for info in sidecar.values()
        if info.module_replica_id >= 0
    } >= {"down", "up"}


@pytest.mark.parametrize(
    ("pp", "ga", "message"),
    [
        (3, 8, "even pp"),
        (4, 7, "even microbatch"),
        (4, 6, "microbatches >= 2"),
    ],
)
def test_dualpipe_shape_constraints(pp, ga, message):
    with pytest.raises(ValueError, match=message):
        _build(pp=pp, ga=ga)


def test_ambiguous_layer_dp_communication_is_rejected():
    item = _layer(0)
    item.dp_comm = "ALLREDUCE"
    item.dp_comm_size = 1024
    header = AicbHeader(1, 1, 2, 4, 4, 2, 4096)
    items = [item for _ in range(4)]
    job = Job(
        0,
        assigned_nodes=[0, 1],
        parallelism=ParallelismConfig(pp=2),
    )
    with pytest.raises(ValueError, match="cannot infer replica-aware"):
        DualPipePipelineWorkloadBuilder(
            gradient_sync_bytes=8192,
        ).build_from_aicb(header, items, job)

"""DAG and sidecar tests for direct Zero Bubble expansion."""

from src.static_analysis.passes.pipeline_task_serializers import (
    ZeroBubbleSerializer,
)
from src.executor.pipeline_job_expander import PipelineJobExpander
from src.workload_format.compact_workload import (
    JobExpansionInfo,
    TaskIdAllocator,
)
from src.workload_format.schema import (
    CommType,
    Job,
    ParallelismConfig,
    Phase,
)
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.inference_profile import InferenceProfileStore
from src.workload_generator.zero_bubble_pipeline_builder import (
    ZeroBubblePipelineWorkloadBuilder,
)


def _item(name: str, *, dp: bool = False) -> AicbWorkItem:
    return AicbWorkItem(
        name=name,
        forward_compute_time=10_000,
        forward_comm="ALLREDUCE",
        forward_comm_size=1024,
        backward_compute_time=20_000,
        backward_comm="ALLREDUCE",
        backward_comm_size=1024,
        dp_compute_time=5_000,
        dp_comm="ALLREDUCE" if dp else "NONE",
        dp_comm_size=2048 if dp else 0,
        process_time=100,
    )


def _build(*, pp=2, tp=2, dp=1, ga=2, with_post=True):
    nodes = [10 + index * 2 for index in range(pp * tp * dp)]
    items = [
        _item(f"layer_{layer}", dp=dp > 1)
        for _microbatch in range(ga)
        for layer in range(2)
    ]
    if with_post:
        items.append(_item("optimizer1"))
    header = AicbHeader(
        tp=tp,
        ep=1,
        pp=pp,
        vpp=2 * pp,
        ga=ga,
        all_gpus=len(nodes),
        pp_comm_size=4096 if pp > 1 else 0,
    )
    job = Job(
        job_id=0,
        assigned_nodes=nodes,
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp),
    )
    sidecar = {}
    workload = ZeroBubblePipelineWorkloadBuilder(sidecar).build_from_aicb(
        header, items, job,
    )
    return workload, sidecar


def _compute(workload, *, node, iteration, layer, phase):
    return next(
        task for task in workload.tasks
        if task.is_compute()
        and task.node == node
        and task.iteration == iteration
        and task.layer_id == layer
        and task.phase is phase
    )


def _ancestors(task_id, task_by_id):
    pending = list(task_by_id[task_id].deps)
    result = set()
    while pending:
        dependency = pending.pop()
        if dependency in result:
            continue
        result.add(dependency)
        pending.extend(task_by_id[dependency].deps)
    return result


def test_b_to_w_pair_does_not_gate_previous_layer_b():
    workload, sidecar = _build()
    node = 10
    b_last = _compute(
        workload, node=node, iteration=0, layer=1,
        phase=Phase.BACKWARD_INPUT,
    )
    w_last = _compute(
        workload, node=node, iteration=0, layer=1,
        phase=Phase.BACKWARD_WEIGHT,
    )
    b_previous = _compute(
        workload, node=node, iteration=0, layer=0,
        phase=Phase.BACKWARD_INPUT,
    )
    task_by_id = {task.task_id: task for task in workload.tasks}

    assert b_last.task_id in _ancestors(w_last.task_id, task_by_id)
    assert w_last.task_id not in _ancestors(b_previous.task_id, task_by_id)
    assert sidecar[b_last.task_id].w_task_id == w_last.task_id
    assert sidecar[w_last.task_id].b_task_id == b_last.task_id
    assert sidecar[b_last.task_id].critical_path
    assert not sidecar[w_last.task_id].critical_path


def test_backward_pp_gradient_closure_contains_b_but_no_w_or_dp():
    workload, sidecar = _build(dp=2)
    task_by_id = {task.task_id: task for task in workload.tasks}
    gradient = next(
        task for task in workload.tasks
        if task.comm_type is CommType.PP_SEND
        and task.phase is Phase.BACKWARD_INPUT
        and task.src == 18
        and task.dst == 10
    )
    ancestors = _ancestors(gradient.task_id, task_by_id)

    assert sidecar[gradient.task_id].task_role == "PP_GRAD"
    assert any(
        task_by_id[task_id].phase is Phase.BACKWARD_INPUT
        for task_id in ancestors
    )
    assert not any(
        sidecar[task_id].task_role in {"W", "DP"}
        for task_id in ancestors
    )


def test_optimizer_entry_waits_for_every_weight_or_dp_terminal():
    workload, _ = _build(dp=2)
    task_by_id = {task.task_id: task for task in workload.tasks}
    optimizer_entry = _compute(
        workload, node=10, iteration=2, layer=0, phase=Phase.FORWARD,
    )

    for microbatch in range(2):
        for layer in range(2):
            w_compute = _compute(
                workload,
                node=10,
                iteration=microbatch,
                layer=layer,
                phase=Phase.BACKWARD_WEIGHT,
            )
            dp_completion = [
                task.task_id
                for task in workload.tasks
                if task.phase is Phase.BACKWARD_WEIGHT
                and task.is_flow()
                and task.iteration == microbatch
                and task.layer_id == layer
                and task.dst == 10
            ]
            assert w_compute.task_id not in optimizer_entry.deps
            assert set(dp_completion) <= set(optimizer_entry.deps)
            assert dp_completion
    assert workload.validate() == []


def test_sidecar_driven_serializer_keeps_each_w_after_its_b():
    workload, sidecar = _build(pp=1, tp=1, ga=4, with_post=False)
    serializer = ZeroBubbleSerializer(
        expansion_task_info=sidecar,
    )
    plan = serializer.serialize(workload)
    order = {
        task_id: index
        for task_ids in plan.compute_order.values()
        for index, task_id in enumerate(task_ids)
    }
    for info in sidecar.values():
        if info.task_role != "B" or info.w_task_id is None:
            continue
        assert order[info.task_id] < order[info.w_task_id]
    assert serializer.validate(workload, plan.compute_order) == []


def test_dynamic_expansion_remaps_zero_bubble_sidecar(tmp_path):
    trace = tmp_path / "zero-bubble.txt"
    rows = [
        f"layer_{layer} -1 10000 NONE 0 20000 NONE 0 5000 NONE 0 100"
        for _microbatch in range(2)
        for layer in range(2)
    ]
    trace.write_text(
        "\n".join([
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            "model_parallel_NPU_group: 1 ep: 1 pp: 2 vpp: 4 ga: 2 "
            "all_gpus: 2 checkpoints: 0 checkpoint_initiates: 0 pp_comm 4096",
            str(len(rows)),
            *rows,
        ]),
        encoding="utf-8",
    )
    shared_info = {}
    expander = PipelineJobExpander(
        TaskIdAllocator(start=100),
        InferenceProfileStore(),
        pipeline_mode="zero_bubble",
        pipeline_task_info=shared_info,
    )
    expanded = expander.expand_job(
        Job(
            4,
            assigned_nodes=[7, 13],
            parallelism=ParallelismConfig(pp=2),
        ),
        JobExpansionInfo(job_type="training", trace_src=str(trace)),
    )

    task_ids = {task.task_id for task in expanded.tasks}
    assert min(task_ids) == 100
    assert set(shared_info) == task_ids
    assert all(info.task_id == task_id for task_id, info in shared_info.items())
    assert {
        info.task_role for info in shared_info.values()
    } >= {"F", "B", "W", "PP_ACT", "PP_GRAD"}

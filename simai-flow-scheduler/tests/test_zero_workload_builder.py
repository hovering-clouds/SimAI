"""Tests for ZeroWorkloadBuilder — DeepSpeed ZeRO/FSDP DAG semantics.

Covers:
1. Forward/bwd parameter allgather precedes compute
2. Gradient reduce-scatter follows weight-gradient compute (bucketing)
3. Step collectives and 1/2 step-chains gate post compute
4. Init broadcast gates forward chain
5. GA boundary markers support non-uniform prefetch rows
6. PP dependencies with comm-only rows
"""

from src.workload_format.schema import CommType, Job, ParallelismConfig, Phase
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.builders.zero_workload_builder import ZeroWorkloadBuilder


# ===========================================================================
# Helper functions
# ===========================================================================

def build_zero_workload(
    items: list[AicbWorkItem],
    ga: int = 1,
    dp: int = 2,
    pp: int = 1,
    pp_comm_size: int = 0,
):
    header = AicbHeader(
        tp=1, ep=1, pp=pp, vpp=len(items), ga=ga,
        all_gpus=dp * pp, pp_comm_size=pp_comm_size,
    )
    job = Job(
        job_id=0,
        assigned_nodes=list(range(dp * pp)),
        parallelism=ParallelismConfig(tp=1, dp=dp, pp=pp, ep=1),
    )
    return ZeroWorkloadBuilder().build_from_aicb(header, items, job)


def zero_comm_item(name: str, comm: str, size: int) -> AicbWorkItem:
    return AicbWorkItem(
        name=name,
        forward_compute_time=0,
        forward_comm="NONE",
        forward_comm_size=0,
        backward_compute_time=0,
        backward_comm="NONE",
        backward_comm_size=0,
        dp_compute_time=0,
        dp_comm=comm,
        dp_comm_size=size,
        process_time=0,
    )


# ===========================================================================
# Tests
# ===========================================================================

class TestZeroWorkloadBuilder:
    """Validate the ZeRO-specific DAG wiring."""

    def _task_map(self, workload):
        return {task.task_id: task for task in workload.tasks}

    def test_zero3_forward_allgather_precedes_forward_compute(self):
        items = [
            zero_comm_item("zero3_forward_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        for rank in [0, 1]:
            fwd = next(
                task for task in workload.tasks
                if task.is_compute() and task.phase == Phase.FORWARD
                and task.node == rank
            )
            dep_tasks = [task_map[dep] for dep in fwd.deps]
            assert any(
                dep.is_flow() and dep.comm_type == CommType.DP_ALLGATHER
                and dep.phase == Phase.FORWARD and dep.dst == rank
                for dep in dep_tasks
            )

    def test_zero3_backward_allgather_precedes_backward_compute(self):
        items = [
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_backward_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        for rank in [0, 1]:
            bwd = next(
                task for task in workload.tasks
                if task.is_compute() and task.phase == Phase.BACKWARD_INPUT
                and task.node == rank
            )
            dep_tasks = [task_map[dep] for dep in bwd.deps]
            assert any(
                dep.is_flow() and dep.comm_type == CommType.DP_ALLGATHER
                and dep.phase == Phase.BACKWARD_INPUT and dep.dst == rank
                for dep in dep_tasks
            )

    def test_zero3_first_backward_allgather_waits_for_forward_tail(self):
        items = [
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_backward_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        for flow in (
            task for task in workload.tasks
            if task.is_flow()
            and task.comm_type == CommType.DP_ALLGATHER
            and task.phase == Phase.BACKWARD_INPUT
        ):
            assert any(
                task_map[dep].is_compute()
                and task_map[dep].phase == Phase.FORWARD
                and task_map[dep].node == flow.src
                for dep in flow.deps
            )

    def test_zero3_reduce_scatter_follows_weight_gradient_compute(self):
        items = [
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0_weight_grad",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_step_grad_reduce_scatter", "REDUCESCATTER", 1024),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        rs_flows = [
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.DP_REDUCESCATTER
            and task.phase == Phase.BACKWARD_WEIGHT
        ]
        assert rs_flows
        for flow in rs_flows:
            dep_tasks = [task_map[dep] for dep in flow.deps]
            assert any(
                dep.is_compute() and dep.phase == Phase.BACKWARD_WEIGHT
                and dep.node == flow.src
                for dep in dep_tasks
            )

    def test_zero3_reduce_scatter_waits_for_every_weight_gradient_in_bucket(self):
        items = [
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0_weight_grad",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_1",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_1_weight_grad",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_grad_reduce_scatter", "REDUCESCATTER", 2048),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        for flow in (
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.DP_REDUCESCATTER
        ):
            weight_grad_deps = [
                task_map[dep] for dep in flow.deps
                if task_map[dep].is_compute()
                and task_map[dep].phase == Phase.BACKWARD_WEIGHT
                and task_map[dep].node == flow.src
            ]
            assert len(weight_grad_deps) == 2

    def test_zero3_compute_less_reduce_scatter_waits_for_backward_compute(self):
        items = [
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_grad_reduce_scatter", "REDUCESCATTER", 1024),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        for flow in (
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.DP_REDUCESCATTER
        ):
            assert any(
                task_map[dep].is_compute()
                and task_map[dep].phase == Phase.BACKWARD_INPUT
                and task_map[dep].node == flow.src
                for dep in flow.deps
            )

    def test_zero3_layer_allgather_precedes_layer_compute(self):
        items = [
            zero_comm_item("zero3_forward_allgather_attention_layer", "ALLGATHER", 1024),
            AicbWorkItem(
                name="attention_layer",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)
        fwd = next(
            task for task in workload.tasks
            if task.is_compute() and task.phase == Phase.FORWARD and task.node == 0
        )
        assert any(
            task_map[dep].is_flow()
            and task_map[dep].comm_type == CommType.DP_ALLGATHER
            for dep in fwd.deps
        )

    def test_zero2_grad_sync_keeps_post_weight_gradient_order(self):
        items = [
            AicbWorkItem(
                name="zero2_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero2_backward_param_0_weight_grad",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero2_grad_sync", "ALLREDUCE", 1024),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)
        sync_flows = [
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.DP_ALLREDUCE
        ]
        assert sync_flows
        for flow in sync_flows:
            assert any(
                task_map[dep].is_compute()
                and task_map[dep].phase == Phase.BACKWARD_WEIGHT
                and task_map[dep].node == flow.src
                for dep in flow.deps
            )

    def test_zero_ga_with_step_and_post_items_groups_correctly(self):
        ga_item = AicbWorkItem(
            name="zero3_forward_param_0",
            forward_compute_time=1000,
            forward_comm="NONE", forward_comm_size=0,
            backward_compute_time=0,
            backward_comm="NONE", backward_comm_size=0,
            dp_compute_time=0,
            dp_comm="NONE", dp_comm_size=0, process_time=0,
        )
        bwd_item = AicbWorkItem(
            name="zero3_backward_param_0",
            forward_compute_time=0,
            forward_comm="NONE", forward_comm_size=0,
            backward_compute_time=1000,
            backward_comm="NONE", backward_comm_size=0,
            dp_compute_time=0,
            dp_comm="NONE", dp_comm_size=0, process_time=0,
        )
        ga_items = [ga_item, bwd_item]
        items = ga_items + ga_items + [
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            zero_comm_item("zero3_grad_norm", "ALLREDUCE", 8),
            AicbWorkItem(
                name="cross_entropy1",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="cross_entropy2",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="cross_entropy3",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="optimizer1",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="optimizer2",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="optimizer3",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="optimizer4",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items, ga=2)
        assert not workload.validate()

        layer_iterations = {
            task.iteration for task in workload.tasks
            if task.is_compute() and task.item_id < 4
        }
        assert layer_iterations == {0, 1}

    def test_zero_step_collectives_and_post_compute_are_serialized(self):
        items = [
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            AicbWorkItem(
                name="zero3_backward_param_0_weight_grad",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_grad_reduce_scatter", "REDUCESCATTER", 1024),
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            zero_comm_item("zero3_grad_norm", "ALLREDUCE", 8),
            zero_comm_item("zero3_step_persistent_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="cross_entropy1",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)

        step_item_ids = [4, 5, 6]
        for index in range(len(step_item_ids) - 1):
            previous_id = step_item_ids[index]
            current_id = step_item_ids[index + 1]
            for flow in (
                task for task in workload.tasks
                if task.is_flow() and task.item_id == current_id
            ):
                assert any(task_map[dep].item_id == previous_id for dep in flow.deps)

        for compute in (
            task for task in workload.tasks
            if task.is_compute() and task.item_id == 7 and task.phase == Phase.FORWARD
        ):
            assert any(
                task_map[dep].is_flow()
                and task_map[dep].item_id == 6
                and task_map[dep].dst == compute.node
                for dep in compute.deps
            )

    def test_zero_init_broadcast_is_expanded_and_gates_forward(self):
        items = [
            zero_comm_item("zero3_init_broadcast_model", "BROADCAST", 1024),
            zero_comm_item("zero3_init_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items)
        task_map = self._task_map(workload)
        broadcast_flows = [
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.DP_BROADCAST
        ]
        assert broadcast_flows
        assert not workload.validate()

        init_allgather = [
            task for task in workload.tasks
            if task.is_flow()
            and task.comm_type == CommType.DP_ALLGATHER
            and task.item_id == 1
        ]
        assert all(
            any(task_map[dep].comm_type == CommType.DP_BROADCAST for dep in flow.deps)
            for flow in init_allgather
        )

    def test_zero3_step_flush_is_post_not_a_ga_layer_item(self):
        ga_item = AicbWorkItem(
            name="zero3_forward_param_0",
            forward_compute_time=1000,
            forward_comm="NONE", forward_comm_size=0,
            backward_compute_time=0,
            backward_comm="NONE", backward_comm_size=0,
            dp_compute_time=0,
            dp_comm="NONE", dp_comm_size=0, process_time=0,
        )
        bwd_item = AicbWorkItem(
            name="zero3_backward_param_0",
            forward_compute_time=0,
            forward_comm="NONE", forward_comm_size=0,
            backward_compute_time=1000,
            backward_comm="NONE", backward_comm_size=0,
            dp_compute_time=0,
            dp_comm="NONE", dp_comm_size=0, process_time=0,
        )
        ga_items = [ga_item, bwd_item]
        items = ga_items + ga_items + [
            zero_comm_item("zero3_step_grad_reduce_scatter", "REDUCESCATTER", 1024),
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            AicbWorkItem(
                name="cross_entropy1",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items, ga=2)
        assert not workload.validate()
        assert {
            task.iteration for task in workload.tasks
            if task.is_compute() and task.item_id < len(ga_items) * 2
        } == {0, 1}

    def test_zero_ga_boundaries_support_nonuniform_prefetch_rows(self):
        ga_item = AicbWorkItem(
            name="zero3_forward_param_0",
            forward_compute_time=1000,
            forward_comm="NONE", forward_comm_size=0,
            backward_compute_time=0,
            backward_comm="NONE", backward_comm_size=0,
            dp_compute_time=0,
            dp_comm="NONE", dp_comm_size=0, process_time=0,
        )
        bwd_item = AicbWorkItem(
            name="zero3_backward_param_0",
            forward_compute_time=0,
            forward_comm="NONE", forward_comm_size=0,
            backward_compute_time=1000,
            backward_comm="NONE", backward_comm_size=0,
            dp_compute_time=0,
            dp_comm="NONE", dp_comm_size=0, process_time=0,
        )
        ga_compute_items = [ga_item, bwd_item]
        items = ga_compute_items + [
            zero_comm_item("zero3_ga_boundary", "NONE", 0),
            *ga_compute_items,
            zero_comm_item("zero3_grad_reduce_scatter", "REDUCESCATTER", 1024),
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            AicbWorkItem(
                name="cross_entropy1",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items, ga=2)
        assert not workload.validate()
        assert {
            task.iteration for task in workload.tasks
            if task.is_compute() and task.item_id < 5
        } == {0, 1}

    def test_zero1_and_zero2_step_chains_gate_post_compute(self):
        for stage in (1, 2):
            step_items = [
                zero_comm_item(f"zero{stage}_has_overflow", "ALLREDUCE", 1),
            ]
            if stage == 2:
                step_items.append(zero_comm_item("zero2_grad_norm", "ALLREDUCE", 8))
            step_items.append(
                zero_comm_item(f"zero{stage}_param_allgather", "ALLGATHER", 1024)
            )
            items = [
                AicbWorkItem(
                    name=f"zero{stage}_forward_param_0",
                    forward_compute_time=1000,
                    forward_comm="NONE", forward_comm_size=0,
                    backward_compute_time=0,
                    backward_comm="NONE", backward_comm_size=0,
                    dp_compute_time=0,
                    dp_comm="NONE", dp_comm_size=0, process_time=0,
                ),
                AicbWorkItem(
                    name=f"zero{stage}_backward_param_0",
                    forward_compute_time=0,
                    forward_comm="NONE", forward_comm_size=0,
                    backward_compute_time=1000,
                    backward_comm="NONE", backward_comm_size=0,
                    dp_compute_time=0,
                    dp_comm="NONE", dp_comm_size=0, process_time=0,
                ),
                AicbWorkItem(
                    name=f"zero{stage}_backward_param_0_weight_grad",
                    forward_compute_time=0,
                    forward_comm="NONE", forward_comm_size=0,
                    backward_compute_time=1000,
                    backward_comm="NONE", backward_comm_size=0,
                    dp_compute_time=0,
                    dp_comm="NONE", dp_comm_size=0, process_time=0,
                ),
                zero_comm_item(f"zero{stage}_grad_sync", "ALLREDUCE", 1024),
                *step_items,
                AicbWorkItem(
                    name="cross_entropy1",
                    forward_compute_time=100,
                    forward_comm="NONE", forward_comm_size=0,
                    backward_compute_time=0,
                    backward_comm="NONE", backward_comm_size=0,
                    dp_compute_time=0,
                    dp_comm="NONE", dp_comm_size=0, process_time=0,
                ),
            ]
            workload = build_zero_workload(items)
            task_map = self._task_map(workload)
            first_post_compute = next(
                task for task in workload.tasks
                if task.is_compute() and task.item_id == len(items) - 1
                and task.phase == Phase.FORWARD and task.node == 0
            )
            final_step_item_id = len(items) - 2
            assert any(
                task_map[dep].is_flow()
                and task_map[dep].item_id == final_step_item_id
                and task_map[dep].dst == first_post_compute.node
                for dep in first_post_compute.deps
            )

    def test_zero_pp_dependencies_preserve_communication_before_compute(self):
        items = [
            zero_comm_item("zero3_forward_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="zero3_forward_param_0",
                forward_compute_time=1000,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=0,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
            zero_comm_item("zero3_backward_param_allgather", "ALLGATHER", 1024),
            AicbWorkItem(
                name="zero3_backward_param_0",
                forward_compute_time=0,
                forward_comm="NONE", forward_comm_size=0,
                backward_compute_time=1000,
                backward_comm="NONE", backward_comm_size=0,
                dp_compute_time=0,
                dp_comm="NONE", dp_comm_size=0, process_time=0,
            ),
        ]
        workload = build_zero_workload(items, dp=1, pp=2, pp_comm_size=1024)
        task_map = self._task_map(workload)
        pp_flows = [
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.PP_SEND
        ]
        assert pp_flows
        assert all(flow.deps for flow in pp_flows)

        stage_one_forward = next(
            task for task in workload.tasks
            if task.is_compute() and task.phase == Phase.FORWARD and task.node == 1
        )
        assert any(
            task_map[dep].is_flow() and task_map[dep].comm_type == CommType.PP_SEND
            for dep in stage_one_forward.deps
        )

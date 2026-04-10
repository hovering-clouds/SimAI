"""
Tests for Workload Builder.

Covers:
1. Basic single-layer workload with ALLREDUCE
2. GA iteration assignment verification
3. Dependency chain integrity
4. Rank group integration with different comm contexts (TP, DP_EP)
5. Edge cases (no communication, multiple phases)
"""

import pytest
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_format.schema import Job, ParallelismConfig, TaskType, Phase


# ===========================================================================
# Helper functions
# ===========================================================================

def create_dummy_item(
    name="test_layer",
    fwd_compute=100,
    fwd_comm="NONE",
    fwd_comm_size=0,
    bwd_compute=200,
    bwd_comm="NONE",
    bwd_comm_size=0,
    dp_compute=50,
    dp_comm="NONE",
    dp_comm_size=0,
) -> AicbWorkItem:
    """Create a dummy AICB work item."""
    return AicbWorkItem(
        name=name,
        forward_compute_time=fwd_compute,
        forward_comm=fwd_comm,
        forward_comm_size=fwd_comm_size,
        backward_compute_time=bwd_compute,
        backward_comm=bwd_comm,
        backward_comm_size=bwd_comm_size,
        dp_compute_time=dp_compute,
        dp_comm=dp_comm,
        dp_comm_size=dp_comm_size,
        process_time=100,
    )


def build_simple_workload(
    tp: int = 1, dp: int = 1, pp: int = 1, ep: int = 1,
    items: list[AicbWorkItem] = None,
) -> tuple:
    """Build a simple workload for testing."""
    if items is None:
        items = [create_dummy_item()]

    header = AicbHeader(
        tp=tp, ep=ep, pp=pp, vpp=len(items), ga=1,
        all_gpus=tp * dp * pp * ep, pp_comm_size=0,
    )
    num_gpus = tp * dp * pp * ep
    job = Job(
        job_id=0,
        assigned_nodes=list(range(num_gpus)),
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )

    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job)
    return header, items, job, builder, workload


# ===========================================================================
# TestBasicWorkload - Single layer with communication
# ===========================================================================

class TestBasicWorkload:
    """Test basic single-layer workload generation."""

    def test_single_layer_allreduce(self):
        """Test single layer with ALLREDUCE in forward and backward."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE",
            fwd_comm_size=1024,
            bwd_comm="ALLREDUCE",
            bwd_comm_size=1024,
        )]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        # Should have compute + flow tasks
        compute_tasks = [t for t in workload.tasks if t.is_compute()]
        flow_tasks = [t for t in workload.tasks if t.is_flow()]

        assert len(compute_tasks) >= 2  # forward + backward compute
        assert len(flow_tasks) > 0  # ALLREDUCE flows

        # Verify phases
        forward_computes = [t for t in compute_tasks if t.phase == Phase.FORWARD]
        backward_computes = [t for t in compute_tasks if t.phase == Phase.BACKWARD_INPUT]
        assert len(forward_computes) >= 1
        assert len(backward_computes) >= 1

    def test_no_communication(self):
        """Test layer with no communication (compute only)."""
        items = [create_dummy_item()]  # All NONE
        _, _, _, _, workload = build_simple_workload(tp=1, items=items)

        compute_tasks = [t for t in workload.tasks if t.is_compute()]
        flow_tasks = [t for t in workload.tasks if t.is_flow()]

        assert len(compute_tasks) >= 1  # At least one compute task
        assert len(flow_tasks) == 0  # No flows when all comms are NONE

    def test_task_id_uniqueness(self):
        """Verify all task IDs are unique."""
        _, _, _, _, workload = build_simple_workload()
        task_ids = [t.task_id for t in workload.tasks]
        assert len(task_ids) == len(set(task_ids))


# ===========================================================================
# TestGAIteration - Gradient Accumulation iteration assignment
# ===========================================================================

class TestGAIteration:
    """Test GA iteration field assignment."""

    def test_ga_iteration_assignment(self):
        """Verify that GA iterations are correctly assigned."""
        # Create workload with ga=2, vpp=2 → 4 layer items
        header = AicbHeader(tp=2, ep=1, pp=1, vpp=2, ga=2, all_gpus=2, pp_comm_size=0)
        items = [create_dummy_item() for _ in range(4)]

        job = Job(
            job_id=0,
            assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2, dp=1, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        # Verify iteration field
        iterations = sorted(set(t.iteration for t in workload.tasks))
        assert 0 in iterations  # First GA step
        assert 1 in iterations  # Second GA step

    def test_pre_post_items_iteration(self):
        """Pre-layer items should have iteration=0, post-layer iteration=ga."""
        # Create items: 1 pre + 2 layer + 1 post
        pre_item = create_dummy_item(name="grad_gather")
        layer_items = [create_dummy_item() for _ in range(2)]
        post_item = create_dummy_item(name="embedding_norm")
        items = [pre_item] + layer_items + [post_item]

        header = AicbHeader(tp=2, ep=1, pp=1, vpp=2, ga=1, all_gpus=2, pp_comm_size=0)
        job = Job(
            job_id=0,
            assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2, dp=1, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        # Check iterations
        for task in workload.tasks:
            if task.layer_id == 0:  # Pre item
                assert task.iteration == 0
            elif task.layer_id == len(items) - 1:  # Post item
                assert task.iteration == header.ga


# ===========================================================================
# TestDependencyChain - Dependency integrity verification
# ===========================================================================

class TestDependencyChain:
    """Test dependency chain integrity."""

    def test_dependency_chain_forward_to_backward(self):
        """Verify forward → backward dependency chain."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE",
            fwd_comm_size=1024,
            bwd_comm="ALLREDUCE",
            bwd_comm_size=1024,
        )]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        # Find forward compute, forward flows, backward compute
        forward_compute = None
        forward_flows = []
        backward_compute = None

        for task in workload.tasks:
            if task.is_compute() and task.phase == Phase.FORWARD:
                forward_compute = task
            elif task.is_flow() and task.phase == Phase.FORWARD:
                forward_flows.append(task)
            elif task.is_compute() and task.phase == Phase.BACKWARD_INPUT:
                backward_compute = task

        if forward_compute and forward_flows:
            # First forward flow should depend on forward compute
            assert forward_compute.task_id in forward_flows[0].deps

    def test_dag_no_cycles(self):
        """Verify DAG has no cycles using DFS."""
        _, _, _, _, workload = build_simple_workload()

        # Build adjacency list
        adj = {}
        task_ids = set()
        for task in workload.tasks:
            task_ids.add(task.task_id)
            adj[task.task_id] = task.deps

        # DFS cycle detection
        visited = set()
        in_stack = set()

        def has_cycle(node):
            visited.add(node)
            in_stack.add(node)
            for dep in adj.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in in_stack:
                    return True
            in_stack.discard(node)
            return False

        for task_id in task_ids:
            if task_id not in visited:
                assert not has_cycle(task_id), "DAG contains cycles"

    def test_cross_item_dependencies(self):
        """Verify dependencies between consecutive items."""
        items = [
            create_dummy_item(name="layer_0"),
            create_dummy_item(name="layer_1"),
        ]
        _, _, _, _, workload = build_simple_workload(tp=1, items=items)

        # Find tasks for each layer
        layer_0_tasks = [t for t in workload.tasks if t.layer_id == 0]
        layer_1_tasks = [t for t in workload.tasks if t.layer_id == 1]

        if layer_0_tasks and layer_1_tasks:
            first_layer_1_task = layer_1_tasks[0]
            last_layer_0_task_id = layer_0_tasks[-1].task_id

            # First task of layer 1 should depend on last task of layer 0
            assert last_layer_0_task_id in first_layer_1_task.deps


# ===========================================================================
# TestRankGroupIntegration - Rank group selection by context
# ===========================================================================

class TestRankGroupIntegration:
    """Test that correct rank groups are used for different comm contexts."""

    def test_tp_context_uses_tp_group(self):
        """ALLGATHER without suffix (TP context) should use TP group."""
        items = [create_dummy_item(fwd_comm="ALLGATHER", fwd_comm_size=1024)]
        # TP=2, DP=1 → ranks [0, 1] in TP group
        _, _, _, _, workload = build_simple_workload(tp=2, dp=1, items=items)

        flow_tasks = [t for t in workload.tasks if t.is_flow()]
        assert len(flow_tasks) > 0

        # All flows should involve ranks from the TP group [0, 1]
        tp_ranks = {0, 1}
        for flow in flow_tasks:
            assert flow.src in tp_ranks or flow.dst in tp_ranks

    def test_dp_ep_context_uses_dp_ep_group(self):
        """ALLGATHER_DP_EP should use DP×EP group."""
        items = [create_dummy_item(fwd_comm="ALLGATHER_DP_EP", fwd_comm_size=1024)]
        # TP=2, DP=2, EP=1, PP=1 → 4 GPUs
        header = AicbHeader(tp=2, ep=1, pp=1, vpp=1, ga=1, all_gpus=4, pp_comm_size=0)
        job = Job(
            job_id=0,
            assigned_nodes=[0, 1, 2, 3],
            parallelism=ParallelismConfig(tp=2, dp=2, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        flow_tasks = [t for t in workload.tasks if t.is_flow()]
        assert len(flow_tasks) > 0

        # Flows should involve ranks from DP×EP group
        # With tp=2, dp=2, ep=1: DP×EP group for pp_idx=0, tp_idx=0
        # should be [0, 2] (dp_idx varies, ep_idx=0)
        all_ranks_in_flows = set()
        for flow in flow_tasks:
            all_ranks_in_flows.add(flow.src)
            all_ranks_in_flows.add(flow.dst)

        # Should include ranks from multiple DP groups
        assert len(all_ranks_in_flows) >= 2


# ===========================================================================
# TestEdgeCases - Boundary conditions
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_gpu_no_comm(self):
        """Single GPU with no communication."""
        items = [create_dummy_item()]
        _, _, _, _, workload = build_simple_workload(tp=1, dp=1, pp=1, ep=1, items=items)

        compute_tasks = [t for t in workload.tasks if t.is_compute()]
        assert len(compute_tasks) >= 1

    def test_multiple_phases_with_comm(self):
        """Item with all three phases having communication."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE", fwd_comm_size=512,
            bwd_comm="ALLREDUCE", bwd_comm_size=512,
            dp_comm="ALLREDUCE", dp_comm_size=256,
        )]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        flow_tasks = [t for t in workload.tasks if t.is_flow()]
        assert len(flow_tasks) > 0

        # Should have flows in multiple phases
        phases_with_flows = set(t.phase for t in flow_tasks)
        assert len(phases_with_flows) >= 2

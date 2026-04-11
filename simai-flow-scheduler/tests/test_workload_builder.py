"""
Tests for Workload Builder (refactored).

Covers:
1. Per-rank compute tasks
2. Forward/Backward dependency ordering (fwd正序, IG逆序, WG同层)
3. Receiver-based dependencies (dst flows)
4. Diamond dependency (IG→WG parallel with IG→next IG)
5. GA iteration assignment
6. Rank group integration (TP, DP_EP contexts)
7. Edge cases (single GPU, multi-phase communication)
"""

import pytest
from src.workload_generator.workload_builder import (
    WorkloadBuilder, FlowGroupResult, ItemTasks,
)
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
    ga: int = 1, vpp: int = None,
    items: list[AicbWorkItem] = None,
) -> tuple:
    """Build a simple workload for testing.

    Returns (header, items, job, builder, workload).
    """
    if items is None:
        items = [create_dummy_item()]
    if vpp is None:
        vpp = len(items) if ga == 1 else len(items) // ga

    num_gpus = tp * dp * pp * ep
    header = AicbHeader(
        tp=tp, ep=ep, pp=pp, vpp=vpp, ga=ga,
        all_gpus=num_gpus, pp_comm_size=0,
    )
    job = Job(
        job_id=0,
        assigned_nodes=list(range(num_gpus)),
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )

    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job)
    return header, items, job, builder, workload


# ===========================================================================
# TestPerRankCompute - Per-rank compute task generation
# ===========================================================================

class TestPerRankCompute:
    """Test that compute tasks are created per-rank."""

    def test_all_ranks_have_compute_tasks(self):
        """Every rank should have compute tasks for each phase."""
        items = [create_dummy_item()]
        _, _, job, _, workload = build_simple_workload(tp=2, dp=2, items=items)
        # 4 ranks, 3 phases each = 12 compute tasks
        compute_tasks = [t for t in workload.tasks if t.is_compute()]
        assert len(compute_tasks) == 12  # 4 ranks × 3 phases

        # Each rank should have compute tasks in all 3 phases
        ranks_with_tasks = set(t.node for t in compute_tasks)
        assert ranks_with_tasks == {0, 1, 2, 3}

    def test_compute_task_assigned_to_correct_node(self):
        """Each compute task's node field should match its rank."""
        _, _, job, _, workload = build_simple_workload(tp=2, items=[
            create_dummy_item(fwd_comm="ALLREDUCE", fwd_comm_size=1024),
        ])
        for task in workload.tasks:
            if task.is_compute():
                assert task.node in {0, 1}

    def test_task_id_uniqueness(self):
        """All task IDs should be unique."""
        _, _, _, _, workload = build_simple_workload(tp=2, dp=2)
        task_ids = [t.task_id for t in workload.tasks]
        assert len(task_ids) == len(set(task_ids))


# ===========================================================================
# TestForwardBackwardOrdering - Dependency chain ordering
# ===========================================================================

class TestForwardBackwardOrdering:
    """Test Forward/Backward dependency ordering."""

    def test_forward_chain_per_node(self):
        """Forward chain: layer[i].fwd.last → layer[i+1].fwd_compute, per rank."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE", fwd_comm_size=1024,
        ) for _ in range(3)]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        # For each rank, verify forward chain ordering
        for rank in [0, 1]:
            fwd_computes = [
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.FORWARD and t.node == rank
            ]
            # Should have 3 forward computes (one per layer)
            assert len(fwd_computes) == 3

            task_by_layer = {t.layer_id: t for t in fwd_computes}

            # Layer 1 fwd should depend on layer 0's output
            assert len(task_by_layer[1].deps) > 0
            # Layer 2 fwd should depend on layer 1's output
            assert len(task_by_layer[2].deps) > 0

    def test_backward_reverse_order(self):
        """Backward IG goes in reverse: layer[N-1].ig → layer[N-2].ig → ... → layer[0].ig."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE", fwd_comm_size=1024,
            bwd_comm="ALLREDUCE", bwd_comm_size=1024,
        ) for _ in range(3)]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        for rank in [0, 1]:
            ig_computes = {
                t.layer_id: t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_INPUT and t.node == rank
            }
            assert len(ig_computes) == 3

            # Layer 2 IG should depend on layer 2's fwd output (bridge)
            assert len(ig_computes[2].deps) > 0

            # Layer 1 IG should depend on layer 2's IG output (reverse)
            assert len(ig_computes[1].deps) > 0
            # Verify the dep comes from BACKWARD_INPUT phase (from layer 2's IG)
            dep_tasks = {t.task_id: t for t in workload.tasks}
            ig1_deps_phases = set()
            for dep_id in ig_computes[1].deps:
                if dep_id in dep_tasks:
                    ig1_deps_phases.add(dep_tasks[dep_id].phase)
            assert Phase.BACKWARD_INPUT in ig1_deps_phases

    def test_wg_depends_on_same_layer_ig(self):
        """WG(i) depends on IG(i), not IG(i-1) or WG(i+1)."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE", fwd_comm_size=1024,
            bwd_comm="ALLREDUCE", bwd_comm_size=1024,
            dp_comm="ALLREDUCE", dp_comm_size=512,
        ) for _ in range(3)]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        for rank in [0, 1]:
            ig_computes = {
                t.layer_id: t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_INPUT and t.node == rank
            }
            wg_computes = {
                t.layer_id: t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_WEIGHT and t.node == rank
            }

            for layer_id in ig_computes:
                wg_task = wg_computes[layer_id]
                assert len(wg_task.deps) > 0, \
                    f"WG layer {layer_id} rank {rank} should have deps"


# ===========================================================================
# TestDiamondDependency - IG→WG parallel with IG→next IG
# ===========================================================================

class TestDiamondDependency:
    """Test that IG creates diamond: IG→WG and IG→next IG can run in parallel."""

    def test_ig_fanout_to_wg_and_next_ig(self):
        """IG(i) flows should fan out to both WG(i) and IG(i-1)."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE", fwd_comm_size=1024,
            bwd_comm="ALLREDUCE", bwd_comm_size=1024,
        ) for _ in range(3)]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        for rank in [0, 1]:
            # Find WG compute for layer 2 and IG compute for layer 1
            wg_compute_l2 = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_WEIGHT
                and t.layer_id == 2 and t.node == rank
            )
            ig_compute_l1 = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_INPUT
                and t.layer_id == 1 and t.node == rank
            )

            # Both should have deps
            assert len(wg_compute_l2.deps) > 0
            assert len(ig_compute_l1.deps) > 0

            # WG(l2) and IG(l1) should NOT depend on each other
            assert wg_compute_l2.task_id not in ig_compute_l1.deps
            assert ig_compute_l1.task_id not in wg_compute_l2.deps


# ===========================================================================
# TestReceiverBasedDeps - Receiver (dst) based dependencies
# ===========================================================================

class TestReceiverBasedDeps:
    """Test that cross-phase deps use receiver (dst) flows."""

    def test_compute_depends_on_received_flows(self):
        """Next phase compute should depend on flows where rank is dst (receiver)."""
        items = [create_dummy_item(fwd_comm="ALLREDUCE", fwd_comm_size=1024)]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        for rank in [0, 1]:
            ig_compute = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_INPUT and t.node == rank
            )
            # IG compute should have deps
            assert len(ig_compute.deps) > 0

            # If deps are flows, they should be where rank is dst (receiver)
            task_map = {t.task_id: t for t in workload.tasks}
            for dep_id in ig_compute.deps:
                dep_task = task_map.get(dep_id)
                if dep_task and dep_task.is_flow() and dep_task.phase == Phase.FORWARD:
                    assert dep_task.dst == rank, \
                        f"IG compute for rank {rank} depends on fwd flow " \
                        f"where dst={dep_task.dst}, expected dst={rank}"

    def test_no_comm_compute_to_compute(self):
        """When no communication, compute(R) should depend on compute(R) directly."""
        items = [create_dummy_item()]  # All NONE comm
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)

        for rank in [0, 1]:
            fwd_compute = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.FORWARD and t.node == rank
            )
            ig_compute = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_INPUT and t.node == rank
            )
            # IG compute should directly depend on fwd compute (same rank)
            assert fwd_compute.task_id in ig_compute.deps


# ===========================================================================
# TestGAIteration - GA iteration assignment
# ===========================================================================

class TestGAIteration:
    """Test GA iteration field assignment."""

    def test_ga_iteration_assignment(self):
        """Verify GA iterations are correctly assigned."""
        header = AicbHeader(tp=2, ep=1, pp=1, vpp=2, ga=2, all_gpus=2, pp_comm_size=0)
        items = [create_dummy_item() for _ in range(4)]

        job = Job(
            job_id=0, assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2, dp=1, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        iterations = sorted(set(t.iteration for t in workload.tasks))
        assert 0 in iterations
        assert 1 in iterations

    def test_ga_bridge_dependency(self):
        """Layer[0].wg(GA=0) should feed into layer[0].fwd(GA=1)."""
        header = AicbHeader(tp=2, ep=1, pp=1, vpp=1, ga=2, all_gpus=2, pp_comm_size=0)
        items = [create_dummy_item() for _ in range(2)]

        job = Job(
            job_id=0, assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2, dp=1, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        for rank in [0, 1]:
            wg_ga0 = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.BACKWARD_WEIGHT
                and t.iteration == 0 and t.node == rank
            )
            fwd_ga1 = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.FORWARD
                and t.iteration == 1 and t.node == rank
            )
            assert len(fwd_ga1.deps) > 0


# ===========================================================================
# TestRankGroupIntegration - Correct rank groups for comm contexts
# ===========================================================================

class TestRankGroupIntegration:
    """Test that correct rank groups are used for different comm contexts."""

    def test_tp_context_uses_tp_group(self):
        """ALLGATHER (TP context) should use TP group."""
        items = [create_dummy_item(fwd_comm="ALLGATHER", fwd_comm_size=1024)]
        _, _, _, _, workload = build_simple_workload(tp=2, dp=1, items=items)

        flow_tasks = [t for t in workload.tasks if t.is_flow()]
        assert len(flow_tasks) > 0

        # All flows should involve ranks from TP groups [0,1]
        tp_ranks = {0, 1}
        for flow in flow_tasks:
            assert flow.src in tp_ranks
            assert flow.dst in tp_ranks

    def test_dp_ep_context_uses_dp_ep_group(self):
        """ALLGATHER_DP_EP should use DP×EP group."""
        items = [create_dummy_item(fwd_comm="ALLGATHER_DP_EP", fwd_comm_size=1024)]
        header = AicbHeader(tp=2, ep=1, pp=1, vpp=1, ga=1, all_gpus=4, pp_comm_size=0)
        job = Job(
            job_id=0, assigned_nodes=[0, 1, 2, 3],
            parallelism=ParallelismConfig(tp=2, dp=2, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        flow_tasks = [t for t in workload.tasks if t.is_flow()]
        assert len(flow_tasks) > 0

        all_ranks_in_flows = set()
        for flow in flow_tasks:
            all_ranks_in_flows.add(flow.src)
            all_ranks_in_flows.add(flow.dst)
        assert len(all_ranks_in_flows) >= 2

    def test_all_subgroups_expanded(self):
        """When dp=2, tp=2, ALLREDUCE (tp context) should expand for both TP groups."""
        items = [create_dummy_item(fwd_comm="ALLREDUCE", fwd_comm_size=1024)]
        _, _, _, _, workload = build_simple_workload(tp=2, dp=2, items=items)

        flow_tasks = [t for t in workload.tasks if t.is_flow() and t.phase == Phase.FORWARD]
        # Should have flows in BOTH TP groups: [0,1] and [2,3]
        ranks_in_flows = set()
        for flow in flow_tasks:
            ranks_in_flows.add(flow.src)
            ranks_in_flows.add(flow.dst)
        assert 0 in ranks_in_flows
        assert 1 in ranks_in_flows
        assert 2 in ranks_in_flows
        assert 3 in ranks_in_flows


# ===========================================================================
# TestDagIntegrity - DAG structure verification
# ===========================================================================

class TestDagIntegrity:
    """Test DAG integrity."""

    def _check_no_cycles(self, workload):
        """Helper: check DAG has no cycles."""
        adj = {t.task_id: t.deps for t in workload.tasks}
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

        for task_id in adj:
            if task_id not in visited:
                assert not has_cycle(task_id), "DAG contains cycles"

    def test_dag_no_cycles_simple(self):
        """Verify DAG has no cycles (simple case)."""
        _, _, _, _, workload = build_simple_workload(tp=2, items=[
            create_dummy_item(
                fwd_comm="ALLREDUCE", fwd_comm_size=1024,
                bwd_comm="ALLREDUCE", bwd_comm_size=1024,
            )
        ])
        self._check_no_cycles(workload)

    def test_dag_no_cycles_multi_layer(self):
        """Multi-layer workload should have no cycles."""
        items = [create_dummy_item(
            fwd_comm="ALLREDUCE", fwd_comm_size=1024,
            bwd_comm="ALLREDUCE", bwd_comm_size=1024,
            dp_comm="ALLREDUCE", dp_comm_size=512,
        ) for _ in range(4)]
        _, _, _, _, workload = build_simple_workload(tp=2, items=items)
        self._check_no_cycles(workload)

    def test_all_deps_exist(self):
        """All dependency IDs should reference existing tasks."""
        _, _, _, _, workload = build_simple_workload(tp=2, items=[
            create_dummy_item(fwd_comm="ALLREDUCE", fwd_comm_size=1024),
        ])

        task_ids = {t.task_id for t in workload.tasks}
        for task in workload.tasks:
            for dep in task.deps:
                assert dep in task_ids, \
                    f"Task {task.task_id} has non-existent dep {dep}"

    def test_no_self_deps(self):
        """No task should depend on itself."""
        _, _, _, _, workload = build_simple_workload(tp=2, items=[
            create_dummy_item(fwd_comm="ALLREDUCE", fwd_comm_size=1024),
        ])

        for task in workload.tasks:
            assert task.task_id not in task.deps, \
                f"Task {task.task_id} depends on itself"


# ===========================================================================
# TestEdgeCases
# ===========================================================================

class TestEdgeCases:
    """Test edge cases."""

    def test_single_gpu_no_comm(self):
        """Single GPU with no communication."""
        items = [create_dummy_item()]
        _, _, _, _, workload = build_simple_workload(
            tp=1, dp=1, pp=1, ep=1, items=items)

        compute_tasks = [t for t in workload.tasks if t.is_compute()]
        # 1 rank × 3 phases = 3 compute tasks
        assert len(compute_tasks) == 3

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

        phases_with_flows = set(t.phase for t in flow_tasks)
        assert len(phases_with_flows) >= 2

    def test_compute_to_all_src_flows(self):
        """Verify compute connects to ALL src flows (not just first)."""
        # AlltoAll: each rank sends to all others (multiple independent src flows)
        items = [create_dummy_item(fwd_comm="ALLTOALL", fwd_comm_size=1024)]
        _, _, _, _, workload = build_simple_workload(tp=3, items=items)

        for rank in [0, 1, 2]:
            fwd_compute = next(
                t for t in workload.tasks
                if t.is_compute() and t.phase == Phase.FORWARD and t.node == rank
            )
            # Find all flows where this rank is src
            src_flows = [
                t for t in workload.tasks
                if t.is_flow() and t.src == rank and t.phase == Phase.FORWARD
            ]
            assert len(src_flows) >= 2  # AlltoAll: sends to N-1 others

            # All src flows should depend on the compute task
            for flow in src_flows:
                assert fwd_compute.task_id in flow.deps, \
                    f"Flow {flow.task_id} (src={rank}) should depend on " \
                    f"compute {fwd_compute.task_id}"

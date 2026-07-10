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

from src.workload_format.schema import CommType, Job, ParallelismConfig, Phase
from src.workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from src.workload_generator.rank_grouper import RankGrouper
from src.workload_generator.workload_builder import WorkloadBuilder

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
        ranks_with_tasks = {t.node for t in compute_tasks}
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

        iterations = sorted({t.iteration for t in workload.tasks})
        assert 0 in iterations
        assert 1 in iterations

    def test_ga_bridge_no_hard_dependency(self):
        """GA bridge is not a hard data dependency - scheduler can overlap.

        The workload only contains true data dependencies (forward chain,
        IG reverse chain, same-layer IG→WG). Cross-GA ordering is left to
        the scheduler using (iteration, layer_id, phase) hints.
        """
        header = AicbHeader(tp=2, ep=1, pp=1, vpp=1, ga=2, all_gpus=2, pp_comm_size=0)
        items = [create_dummy_item() for _ in range(2)]

        job = Job(
            job_id=0, assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2, dp=1, pp=1, ep=1),
        )

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)

        # Verify that GA[1].fwd does NOT depend on GA[0].wg
        # (this is the intended design for Route C - flexible scheduling)
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
            # No dependency: scheduler can use iteration/layer_id/phase hints
            assert wg_ga0.task_id not in fwd_ga1.deps


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

        phases_with_flows = {t.phase for t in flow_tasks}
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


# ===========================================================================
# TestPipelineParallelism - PP flow generation and wiring
# ===========================================================================

def build_pp_workload(
    pp: int = 2, tp: int = 1, dp: int = 1, ep: int = 1,
    ga: int = 1, vpp: int = 1,
    pp_comm_size: int = 1024,
    fwd_comm: str = "NONE", fwd_comm_size: int = 0,
    bwd_comm: str = "NONE", bwd_comm_size: int = 0,
):
    """Build a workload with pp > 1 for PP testing."""
    num_gpus = pp * tp * dp * ep
    items = [create_dummy_item(
        fwd_comm=fwd_comm, fwd_comm_size=fwd_comm_size,
        bwd_comm=bwd_comm, bwd_comm_size=bwd_comm_size,
    ) for _ in range(ga * vpp)]
    header = AicbHeader(
        tp=tp, ep=ep, pp=pp, vpp=vpp, ga=ga,
        all_gpus=num_gpus, pp_comm_size=pp_comm_size,
    )
    job = Job(
        job_id=0,
        assigned_nodes=list(range(num_gpus)),
        parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep),
    )
    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job)
    return workload


class TestPipelineParallelism:
    """Test PP flow generation and dependency wiring."""

    def _check_no_cycles(self, workload):
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

    def test_no_pp_flows_when_pp1(self):
        """pp=1 should generate no PP flows."""
        _, _, _, _, workload = build_simple_workload(tp=2, pp=1)
        pp_flows = [t for t in workload.tasks if t.is_flow() and
                    hasattr(t, 'comm_type') and str(t.comm_type) == "pp_send"]
        assert len(pp_flows) == 0

    def test_no_pp_flows_when_pp_comm_size_zero(self):
        """pp > 1 but pp_comm_size=0 should generate no PP flows."""
        num_gpus = 2
        items = [create_dummy_item()]
        header = AicbHeader(tp=1, ep=1, pp=2, vpp=1, ga=1,
                            all_gpus=num_gpus, pp_comm_size=0)
        job = Job(job_id=0, assigned_nodes=[0, 1],
                  parallelism=ParallelismConfig(tp=1, dp=1, pp=2, ep=1))
        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)
        from src.workload_format.schema import CommType
        pp_flows = [t for t in workload.tasks
                    if t.is_flow() and t.comm_type == CommType.PP_SEND]
        assert len(pp_flows) == 0

    def test_pp_flow_count(self):
        """PP flow count = ga × (pp-1) × (dp×ep×tp) × 2."""
        pp, tp, dp, ep, ga, vpp = 2, 2, 1, 1, 1, 1
        workload = build_pp_workload(pp=pp, tp=tp, dp=dp, ep=ep, ga=ga, vpp=vpp)
        from src.workload_format.schema import CommType
        pp_flows = [t for t in workload.tasks
                    if t.is_flow() and t.comm_type == CommType.PP_SEND]
        expected = ga * (pp - 1) * (dp * ep * tp) * 2
        assert len(pp_flows) == expected

    def test_pp_flow_count_multi_stage(self):
        """pp=3 should have 2 boundaries × 2 directions = 4 PP flows per GA."""
        pp, tp, dp, ep, ga, vpp = 3, 1, 1, 1, 1, 1
        workload = build_pp_workload(pp=pp, tp=tp, dp=dp, ep=ep, ga=ga, vpp=vpp)
        from src.workload_format.schema import CommType
        pp_flows = [t for t in workload.tasks
                    if t.is_flow() and t.comm_type == CommType.PP_SEND]
        expected = ga * (pp - 1) * (dp * ep * tp) * 2
        assert len(pp_flows) == expected

    def test_pp_flow_count_multi_ga(self):
        """Each GA step gets its own set of PP flows."""
        pp, tp, dp, ep, ga, vpp = 2, 1, 1, 1, 3, 2
        workload = build_pp_workload(pp=pp, tp=tp, dp=dp, ep=ep, ga=ga, vpp=vpp)
        from src.workload_format.schema import CommType
        pp_flows = [t for t in workload.tasks
                    if t.is_flow() and t.comm_type == CommType.PP_SEND]
        expected = ga * (pp - 1) * (dp * ep * tp) * 2
        assert len(pp_flows) == expected

    def test_forward_pp_flow_direction(self):
        """Forward PP flows go from stage k to stage k+1."""
        # pp=2, tp=1: stage0=rank0, stage1=rank1
        workload = build_pp_workload(pp=2, tp=1, ga=1, vpp=1)
        from src.workload_format.schema import CommType, Phase
        fwd_pp = [t for t in workload.tasks
                  if t.is_flow() and t.comm_type == CommType.PP_SEND
                  and t.phase == Phase.FORWARD]
        assert len(fwd_pp) == 1
        assert fwd_pp[0].src == 0  # stage 0
        assert fwd_pp[0].dst == 1  # stage 1

    def test_backward_pp_flow_direction(self):
        """Backward PP flows go from stage k+1 to stage k."""
        workload = build_pp_workload(pp=2, tp=1, ga=1, vpp=1)
        from src.workload_format.schema import CommType, Phase
        bwd_pp = [t for t in workload.tasks
                  if t.is_flow() and t.comm_type == CommType.PP_SEND
                  and t.phase == Phase.BACKWARD_INPUT]
        assert len(bwd_pp) == 1
        assert bwd_pp[0].src == 1  # stage 1
        assert bwd_pp[0].dst == 0  # stage 0

    def test_forward_pp_sender_dep(self):
        """Forward PP flow depends on stage k's last layer fwd compute."""
        # pp=2, tp=1, vpp=2: stage0=rank0, stage1=rank1; last layer = layer_id=1
        workload = build_pp_workload(pp=2, tp=1, ga=1, vpp=2)
        from src.workload_format.schema import CommType, Phase
        fwd_pp = next(t for t in workload.tasks
                      if t.is_flow() and t.comm_type == CommType.PP_SEND
                      and t.phase == Phase.FORWARD)
        task_map = {t.task_id: t for t in workload.tasks}
        # The PP flow's deps should include rank 0's last layer fwd compute
        dep_tasks = [task_map[d] for d in fwd_pp.deps if d in task_map]
        assert any(
            t.is_compute() and t.phase == Phase.FORWARD
            and t.node == 0 and t.layer_id == 1
            for t in dep_tasks
        ), "Forward PP flow should depend on stage 0's last layer fwd compute"

    def test_forward_pp_receiver_dep(self):
        """Stage k+1's first layer fwd compute depends on forward PP flow."""
        workload = build_pp_workload(pp=2, tp=1, ga=1, vpp=2)
        from src.workload_format.schema import CommType, Phase
        fwd_pp = next(t for t in workload.tasks
                      if t.is_flow() and t.comm_type == CommType.PP_SEND
                      and t.phase == Phase.FORWARD)
        # rank 1 (stage 1), layer_id=0 (first layer), fwd compute
        first_layer_fwd_rank1 = next(
            t for t in workload.tasks
            if t.is_compute() and t.phase == Phase.FORWARD
            and t.node == 1 and t.layer_id == 0
        )
        assert fwd_pp.task_id in first_layer_fwd_rank1.deps, \
            "Stage 1's first layer fwd compute should depend on forward PP flow"

    def test_backward_pp_sender_dep(self):
        """Backward PP flow depends on stage k+1's first layer ig compute."""
        workload = build_pp_workload(pp=2, tp=1, ga=1, vpp=2)
        from src.workload_format.schema import CommType, Phase
        bwd_pp = next(t for t in workload.tasks
                      if t.is_flow() and t.comm_type == CommType.PP_SEND
                      and t.phase == Phase.BACKWARD_INPUT)
        task_map = {t.task_id: t for t in workload.tasks}
        dep_tasks = [task_map[d] for d in bwd_pp.deps if d in task_map]
        # Sender is rank 1 (stage 1), first layer (layer_id=0) ig compute
        assert any(
            t.is_compute() and t.phase == Phase.BACKWARD_INPUT
            and t.node == 1 and t.layer_id == 0
            for t in dep_tasks
        ), "Backward PP flow should depend on stage 1's first layer ig compute"

    def test_backward_pp_receiver_dep(self):
        """Stage k's last layer ig compute depends on backward PP flow."""
        workload = build_pp_workload(pp=2, tp=1, ga=1, vpp=2)
        from src.workload_format.schema import CommType, Phase
        bwd_pp = next(t for t in workload.tasks
                      if t.is_flow() and t.comm_type == CommType.PP_SEND
                      and t.phase == Phase.BACKWARD_INPUT)
        # rank 0 (stage 0), last layer (layer_id=1), ig compute
        last_layer_ig_rank0 = next(
            t for t in workload.tasks
            if t.is_compute() and t.phase == Phase.BACKWARD_INPUT
            and t.node == 0 and t.layer_id == 1
        )
        assert bwd_pp.task_id in last_layer_ig_rank0.deps, \
            "Stage 0's last layer ig compute should depend on backward PP flow"

    def test_pp_with_layer_comm_sender_dep(self):
        """When layer has AllReduce, PP flow depends on receiver flows, not compute."""
        # pp=2, tp=2: stage0=[0,1], stage1=[2,3]
        workload = build_pp_workload(
            pp=2, tp=2, ga=1, vpp=1,
            fwd_comm="ALLREDUCE", fwd_comm_size=1024,
        )
        from src.workload_format.schema import CommType, Phase
        fwd_pp_flows = [t for t in workload.tasks
                        if t.is_flow() and t.comm_type == CommType.PP_SEND
                        and t.phase == Phase.FORWARD]
        task_map = {t.task_id: t for t in workload.tasks}
        for pp_flow in fwd_pp_flows:
            sender = pp_flow.src
            dep_tasks = [task_map[d] for d in pp_flow.deps if d in task_map]
            # Deps should be flows (AllReduce receiver flows), not compute
            assert all(t.is_flow() for t in dep_tasks), \
                f"PP flow from rank {sender} should depend on AllReduce flows, not compute"
            # All dep flows should have sender as dst (receiver-based)
            assert all(t.dst == sender for t in dep_tasks), \
                f"PP flow from rank {sender} should depend on flows where dst={sender}"

    def test_pp_dag_no_cycles(self):
        """Full PP workload DAG should have no cycles."""
        workload = build_pp_workload(pp=2, tp=2, dp=1, ep=1, ga=2, vpp=3)
        self._check_no_cycles(workload)

    def test_pp_all_deps_exist(self):
        """All PP flow dependency IDs should reference existing tasks."""
        workload = build_pp_workload(pp=3, tp=2, ga=2, vpp=2)
        task_ids = {t.task_id for t in workload.tasks}
        for task in workload.tasks:
            for dep in task.deps:
                assert dep in task_ids, \
                    f"Task {task.task_id} has non-existent dep {dep}"

    def test_pp_flow_size(self):
        """PP flows should use pp_comm_size from header."""
        pp_comm_size = 50331648
        workload = build_pp_workload(pp=2, tp=1, pp_comm_size=pp_comm_size)
        from src.workload_format.schema import CommType
        pp_flows = [t for t in workload.tasks
                    if t.is_flow() and t.comm_type == CommType.PP_SEND]
        assert all(t.size_bytes == pp_comm_size for t in pp_flows)


# ===========================================================================
# TestZeroWorkloadBuilder - DeepSpeed ZeRO/FSDP DAG semantics
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
    builder = WorkloadBuilder()
    return builder.build_from_aicb(header, items, job)


def zero_comm_item(name: str, comm: str, size: int) -> AicbWorkItem:
    return create_dummy_item(
        name=name,
        fwd_compute=0,
        bwd_compute=0,
        dp_compute=0,
        dp_comm=comm,
        dp_comm_size=size,
    )


class TestZeroWorkloadBuilder:
    def _task_map(self, workload):
        return {task.task_id: task for task in workload.tasks}

    def test_zero3_forward_allgather_precedes_forward_compute(self):
        items = [
            zero_comm_item("zero3_forward_param_allgather", "ALLGATHER", 1024),
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
        ]
        workload = build_zero_workload(items)

        ag_flows = [
            task for task in workload.tasks
            if task.is_flow() and task.comm_type == CommType.DP_ALLGATHER
            and task.phase == Phase.FORWARD
        ]
        assert ag_flows

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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            zero_comm_item("zero3_backward_param_allgather", "ALLGATHER", 1024),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            zero_comm_item("zero3_backward_param_allgather", "ALLGATHER", 1024),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0_weight_grad",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            zero_comm_item(
                "zero3_step_grad_reduce_scatter", "REDUCESCATTER", 1024
            ),
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0_weight_grad",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_1",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_1_weight_grad",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
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
            create_dummy_item(
                name="attention_layer",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
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
            create_dummy_item(
                name="zero2_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero2_backward_param_0_weight_grad",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
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
        ga_items = [
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
        ]
        items = ga_items + ga_items + [
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            zero_comm_item("zero3_grad_norm", "ALLREDUCE", 8),
            create_dummy_item(name="cross_entropy1"),
            create_dummy_item(name="cross_entropy2"),
            create_dummy_item(name="cross_entropy3"),
            create_dummy_item(name="optimizer1"),
            create_dummy_item(name="optimizer2"),
            create_dummy_item(name="optimizer3"),
            create_dummy_item(name="optimizer4"),
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0_weight_grad",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
            zero_comm_item("zero3_grad_reduce_scatter", "REDUCESCATTER", 1024),
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            zero_comm_item("zero3_grad_norm", "ALLREDUCE", 8),
            zero_comm_item(
                "zero3_step_persistent_param_allgather", "ALLGATHER", 1024
            ),
            create_dummy_item(name="cross_entropy1"),
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
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

    def test_broadcast_root_uses_completion_not_receiver_index(self):
        grouper = RankGrouper(
            [0, 1], ParallelismConfig(tp=1, dp=2, pp=1, ep=1)
        )
        result, _ = WorkloadBuilder()._expand_comm_all_groups(
            "BROADCAST", 1024, grouper, Phase.FORWARD,
            layer_id=0, iteration=-1, item_id=0, job_id=0,
            task_id_counter=0, default_context="dp",
        )
        assert 0 not in result.receiver_index
        assert result.completion_index[0] == [flow.task_id for flow in result.flows]
        assert result.completion_index[1] == result.receiver_index[1]

    def test_zero3_step_flush_is_post_not_a_ga_layer_item(self):
        ga_items = [
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
        ]
        items = ga_items + ga_items + [
            zero_comm_item(
                "zero3_step_grad_reduce_scatter", "REDUCESCATTER", 1024
            ),
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            create_dummy_item(name="cross_entropy1"),
        ]
        workload = build_zero_workload(items, ga=2)
        assert not workload.validate()
        assert {
            task.iteration for task in workload.tasks
            if task.is_compute() and task.item_id < len(ga_items) * 2
        } == {0, 1}

    def test_zero_ga_boundaries_support_nonuniform_prefetch_rows(self):
        ga_compute_items = [
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
            ),
        ]
        items = ga_compute_items + [
            zero_comm_item("zero3_ga_boundary", "NONE", 0),
            *ga_compute_items,
            zero_comm_item("zero3_grad_reduce_scatter", "REDUCESCATTER", 1024),
            zero_comm_item("zero3_has_overflow", "ALLREDUCE", 1),
            create_dummy_item(name="cross_entropy1"),
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
                create_dummy_item(
                    name=f"zero{stage}_forward_param_0",
                    fwd_compute=1000,
                    bwd_compute=0,
                    dp_compute=0,
                ),
                create_dummy_item(
                    name=f"zero{stage}_backward_param_0",
                    fwd_compute=0,
                    bwd_compute=1000,
                    dp_compute=0,
                ),
                create_dummy_item(
                    name=f"zero{stage}_backward_param_0_weight_grad",
                    fwd_compute=0,
                    bwd_compute=1000,
                    dp_compute=0,
                ),
                zero_comm_item(f"zero{stage}_grad_sync", "ALLREDUCE", 1024),
                *step_items,
                create_dummy_item(name="cross_entropy1"),
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
            create_dummy_item(
                name="zero3_forward_param_0",
                fwd_compute=1000,
                bwd_compute=0,
                dp_compute=0,
            ),
            zero_comm_item("zero3_backward_param_allgather", "ALLGATHER", 1024),
            create_dummy_item(
                name="zero3_backward_param_0",
                fwd_compute=0,
                bwd_compute=1000,
                dp_compute=0,
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

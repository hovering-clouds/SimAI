"""
Tests for task serializer module.

Tests ExecutionPlan, CppReferenceOrdering,
and TaskSerializer base class for compute task ordering.
"""

import pytest
import tempfile
import os

from src.static_analysis.passes.task_serializer import (
    ExecutionPlan,
    CppReferenceSerializer,
    TaskSerializer,
)
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    Phase,
    P2PWorkload,
    Task,
    TaskType,
)


# --- Helpers ---


def _make_compute(task_id, duration_us, node=0, deps=None, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0):
    return Task(
        task_id=task_id, job_id=0, type=TaskType.COMPUTE,
        node=node, duration_us=duration_us, deps=deps or [],
        iteration=iteration, phase=phase, layer_id=layer_id, item_id=item_id,
    )


def _make_flow(task_id, src, dst, size_bytes, deps=None):
    return Task(
        task_id=task_id, job_id=0, type=TaskType.FLOW,
        src=src, dst=dst, size_bytes=size_bytes,
        comm_type=CommType.TP_ALLREDUCE_RING, deps=deps or [],
    )


def _make_workload(tasks):
    max_node = 0
    for t in tasks:
        if t.is_flow() and t.src is not None and t.dst is not None:
            max_node = max(max_node, t.src, t.dst)
        elif t.is_compute() and t.node is not None:
            max_node = max(max_node, t.node)
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=max_node + 1),
        tasks=tasks,
    )


def _make_star_topo(bw=400.0, lat=0.5):
    """Star: nodes 0,1,2 connect to switch 10 (bidirectional)."""
    topo = NetworkTopology()
    for leaf in [0, 1, 2]:
        topo.add_link(Link(src=leaf, dst=10, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
        topo.add_link(Link(src=10, dst=leaf, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
    return topo


# ============================================================
# Tests for ExecutionPlan
# ============================================================


class TestExecutionPlan:
    """Tests for the ExecutionPlan dataclass."""

    def test_default_values(self):
        """ExecutionPlan has correct default values."""
        plan = ExecutionPlan()
        assert plan.version == "1.0"
        assert plan.compute_order == {}

    def test_with_values(self):
        """ExecutionPlan accepts custom values."""
        plan = ExecutionPlan(
            version="2.0",
            compute_order={0: [1, 2, 3], 1: [4, 5, 6]},
        )
        assert plan.version == "2.0"
        assert plan.compute_order[0] == [1, 2, 3]


# ============================================================
# Tests for CppReferenceOrdering
# ============================================================


class TestCppReferenceOrdering:
    """Tests for the CppReferenceOrdering strategy."""

    def test_empty_workload(self):
        """Ordering handles empty workload."""
        strategy = CppReferenceSerializer()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))

        plan = strategy.serialize(wl)

        assert plan.compute_order == {}

    def test_single_compute_task(self):
        """Ordering handles single compute task."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 1000, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        wl = _make_workload([c0])

        plan = strategy.serialize(wl)

        assert plan.compute_order[0] == [0]

    def test_order_by_iteration(self):
        """Ordering sorts by iteration (GA) first."""
        strategy = CppReferenceSerializer()
        # Two compute tasks on same node, different iterations
        c0 = _make_compute(0, 100, node=0, iteration=1, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        wl = _make_workload([c0, c1])

        plan = strategy.serialize(wl)

        # iteration=0 should come before iteration=1
        assert plan.compute_order[0] == [1, 0]

    def test_order_by_phase(self):
        """Ordering sorts by phase within same iteration."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.BACKWARD_WEIGHT, layer_id=0, item_id=0)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c2 = _make_compute(2, 100, node=0, iteration=0, phase=Phase.BACKWARD_INPUT, layer_id=0, item_id=0)
        wl = _make_workload([c0, c1, c2])

        plan = strategy.serialize(wl)

        # Phase order: FORWARD(0) -> BACKWARD_INPUT(1) -> BACKWARD_WEIGHT(2)
        assert plan.compute_order[0] == [1, 2, 0]

    def test_order_by_layer_id(self):
        """Ordering sorts by layer_id within same phase."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=2, item_id=0)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c2 = _make_compute(2, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=1, item_id=0)
        wl = _make_workload([c0, c1, c2])

        plan = strategy.serialize(wl)

        # layer_id order: 0 -> 1 -> 2
        assert plan.compute_order[0] == [1, 2, 0]

    def test_order_by_item_id(self):
        """Ordering sorts by item_id within same layer."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=2)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c2 = _make_compute(2, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1)
        wl = _make_workload([c0, c1, c2])

        plan = strategy.serialize(wl)

        # item_id order: 0 -> 1 -> 2
        assert plan.compute_order[0] == [1, 2, 0]

    def test_multiple_nodes(self):
        """Ordering handles multiple nodes independently."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c1 = _make_compute(1, 100, node=1, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c2 = _make_compute(2, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=1, item_id=0)
        wl = _make_workload([c0, c1, c2])

        plan = strategy.serialize(wl)

        assert plan.compute_order[0] == [0, 2]
        assert plan.compute_order[1] == [1]

    def test_skips_flow_tasks(self):
        """Ordering ignores flow tasks."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        f0 = _make_flow(1, 0, 1, 1000, deps=[0])
        wl = _make_workload([c0, f0])

        plan = strategy.serialize(wl)

        # Only compute task should be in the result
        assert plan.compute_order[0] == [0]
        assert 1 not in plan.compute_order  # flow task should not appear


# ============================================================
# Tests for TaskSerializer
# ============================================================


class TestTaskSerializer:
    """Tests for the TaskSerializer class."""

    def test_empty_workload(self):
        """Serializer handles empty workload."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))

        plan = serializer.serialize(wl)

        assert isinstance(plan, ExecutionPlan)
        assert plan.compute_order == {}

    def test_single_compute_task(self):
        """Serializer handles single compute task."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        c0 = _make_compute(0, 1000, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        wl = _make_workload([c0])

        plan = serializer.serialize(wl)

        assert plan.compute_order[0] == [0]

    def test_multiple_tasks_sorted(self):
        """Serializer sorts multiple compute tasks correctly."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        # Tasks out of order
        c2 = _make_compute(2, 100, node=0, iteration=1, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.BACKWARD_INPUT, layer_id=0, item_id=0)
        wl = _make_workload([c2, c0, c1])

        plan = serializer.serialize(wl)

        # Should be sorted: all FORWARD first (GA ascending), then all BACKWARD
        # c0: FORWARD iter=0, c2: FORWARD iter=1, c1: BACKWARD iter=0
        assert plan.compute_order[0] == [0, 2, 1]

    def test_validate_no_errors_on_valid_order(self):
        """Validation passes for valid ordering."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=0)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=1, item_id=0)
        wl = _make_workload([c0, c1])

        errors = serializer.validate(wl, {0: [0, 1]})

        assert errors == []

    def test_validate_detects_cycle_from_deps(self):
        """Validation detects when ordering would create a cycle."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        # Task 0 depends on task 1 - this creates a reverse dependency
        c0 = _make_compute(0, 100, node=0, deps=[1])
        c1 = _make_compute(1, 100, node=0)
        wl = _make_workload([c0, c1])

        # Trying to order as [0, 1] would mean 0 must complete before 1,
        # but 0 depends on 1 - this is already a cycle in the original DAG
        errors = serializer.validate(wl, {0: [0, 1]})

        assert len(errors) > 0
        assert "cycle" in errors[0].lower() or "circular" in errors[0].lower()

    def test_validate_detects_implicit_cycle(self):
        """Validation detects implicit cycle from ordering."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        # Original DAG: 0 -> 1 (no deps), 2 -> 3 (no deps)
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 100, node=0, deps=[0])  # 1 depends on 0
        c2 = _make_compute(2, 100, node=0)
        c3 = _make_compute(3, 100, node=0, deps=[2])  # 3 depends on 2
        wl = _make_workload([c0, c1, c2, c3])

        # Ordering as [1, 0, 3, 2]:
        # - 1 before 0: but 1 depends on 0, so this adds implicit 1 -> 0
        #   Combined with existing 0 -> 1, this creates a cycle
        errors = serializer.validate(wl, {0: [1, 0, 3, 2]})

        assert len(errors) > 0

    def test_validate_detects_multi_edge_cycle(self):
        """Validation detects cycle formed by multiple implicit edges combined.

        This is the case the old per-edge approach would miss:
        - Original deps: task 2 depends on task 0 (0 -> 2)
        - Ordering: [0, 1, 2] adds implicit edges 0->1 and 1->2
        - No single implicit edge alone creates a cycle, but together:
          0->1->2 combined with original 0->2 is fine, BUT
          if we add original dep 2->0 (task 0 depends on task 2):
          cycle is 0->1->2->0
        """
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        # task 0 depends on task 2 (2 -> 0 in execution order)
        c0 = _make_compute(0, 100, node=0, deps=[2])
        c1 = _make_compute(1, 100, node=0)
        c2 = _make_compute(2, 100, node=0)
        wl = _make_workload([c0, c1, c2])

        # Ordering [0, 1, 2] adds implicit edges: 0->1, 1->2
        # Combined with original 2->0: cycle is 0->1->2->0
        # Old per-edge check would miss this:
        #   - Adding 0->1 alone: no path from 1 back to 0 in original graph
        #   - Adding 1->2 alone: no path from 2 back to 1 in original graph
        # But together they form a cycle.
        errors = serializer.validate(wl, {0: [0, 1, 2]})

        assert len(errors) > 0

    def test_to_json_and_from_json(self):
        """Serializer can write and read JSON."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 100, node=1)
        wl = _make_workload([c0, c1])

        plan = serializer.serialize(wl)

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            plan.to_json(temp_path)

            # Read back
            loaded = ExecutionPlan.from_json(temp_path)

            assert loaded.version == "1.0"
            assert loaded.compute_order[0] == [0]
            assert loaded.compute_order[1] == [1]
        finally:
            os.unlink(temp_path)

    def test_multiple_nodes_independent(self):
        """Serializer handles multiple nodes independently."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()
        # Different nodes have different orderings
        c0 = _make_compute(0, 100, node=0, iteration=1, phase=Phase.FORWARD)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD)
        c2 = _make_compute(2, 100, node=1, iteration=0, phase=Phase.BACKWARD_INPUT)
        c3 = _make_compute(3, 100, node=1, iteration=0, phase=Phase.FORWARD)
        wl = _make_workload([c0, c1, c2, c3])

        plan = serializer.serialize(wl)

        # Node 0: iteration=0 then iteration=1
        assert plan.compute_order[0] == [1, 0]
        # Node 1: FORWARD then BACKWARD_INPUT
        assert plan.compute_order[1] == [3, 2]

    def test_optimizer_phase_last(self):
        """Optimizer phase comes last in ordering."""
        strategy = CppReferenceSerializer()
        c0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.OPTIMIZER)
        c1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD)
        wl = _make_workload([c0, c1])

        plan = strategy.serialize(wl)

        # FORWARD(0) should come before OPTIMIZER(3)
        assert plan.compute_order[0] == [1, 0]

    def test_backward_layers_reversed(self):
        """Backward phases have reversed layer order."""
        strategy = CppReferenceSerializer()
        # Forward: layer 0 -> 1 -> 2
        # Backward: layer 2 -> 1 -> 0
        f0 = _make_compute(0, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=0)
        f1 = _make_compute(1, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=1)
        f2 = _make_compute(2, 100, node=0, iteration=0, phase=Phase.FORWARD, layer_id=2)
        ig2 = _make_compute(3, 100, node=0, iteration=0, phase=Phase.BACKWARD_INPUT, layer_id=2)
        ig1 = _make_compute(4, 100, node=0, iteration=0, phase=Phase.BACKWARD_INPUT, layer_id=1)
        ig0 = _make_compute(5, 100, node=0, iteration=0, phase=Phase.BACKWARD_INPUT, layer_id=0)
        wl = _make_workload([f0, f1, f2, ig2, ig1, ig0])

        plan = strategy.serialize(wl)

        # Forward: 0,1,2 (layer 0,1,2)
        # Backward: 3,4,5 (layer 2,1,0 - reversed)
        assert plan.compute_order[0] == [0, 1, 2, 3, 4, 5]

    def test_complex_workload_full_ordering(self):
        """Test complex workload with multiple iterations, phases, layers."""
        topo = _make_star_topo()
        serializer = CppReferenceSerializer()

        # Create a realistic workload: 2 GAs, 3 layers each
        tasks = []
        task_id = 0
        for iteration in range(2):  # 2 GAs
            for layer in range(3):  # 3 layers
                # Forward pass
                tasks.append(_make_compute(
                    task_id, 100, node=0,
                    iteration=iteration, phase=Phase.FORWARD,
                    layer_id=layer, item_id=0
                ))
                task_id += 1

        wl = _make_workload(tasks)
        plan = serializer.serialize(wl)

        # Expected order: iteration=0,phase=FORWARD,layers=0,1,2 -> iteration=1,phase=FORWARD,layers=0,1,2
        expected = [0, 1, 2, 3, 4, 5]
        assert plan.compute_order[0] == expected


# ============================================================
# Tests for Custom TaskSerializer subclass
# ============================================================


class TestCustomTaskSerializer:
    """Tests for custom TaskSerializer subclass."""

    def test_custom_strategy(self):
        """Custom TaskSerializer subclass can provide different ordering."""
        class ReverseOrdering(TaskSerializer):
            def serialize(self, workload):
                # Reverse order by task_id
                compute_tasks = [t for t in workload.tasks if t.is_compute()]
                by_node: dict[int, list[Task]] = {}
                for t in compute_tasks:
                    if t.node not in by_node:
                        by_node[t.node] = []
                    by_node[t.node].append(t)
                result = {
                    node: [t.task_id for t in sorted(tasks, key=lambda x: -x.task_id)]
                    for node, tasks in by_node.items()
                }
                return ExecutionPlan(compute_order=result)

        serializer = ReverseOrdering()

        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 100, node=0)
        c2 = _make_compute(2, 100, node=0)
        wl = _make_workload([c0, c1, c2])

        plan = serializer.serialize(wl)

        # Should be in reverse order: 2, 1, 0
        assert plan.compute_order[0] == [2, 1, 0]
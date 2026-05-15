"""Tests for Puppeteer TTE analysis pass."""
import pytest

from src.static_analysis.passes.puppeteer_tte import (
    TTEInfo,
    FlowTiming,
    compute_optimistic_timing,
    compute_tte,
)
from src.static_analysis.passes.routing import BfsStrategy, BfsRouteTable
from src.static_analysis.passes.task_serializer import CppReferenceSerializer, ExecutionPlan
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


def _make_star_topo(bw=400.0, lat=0.5):
    topo = NetworkTopology()
    for leaf in [0, 1, 2]:
        topo.add_link(Link(src=leaf, dst=10, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
        topo.add_link(Link(src=10, dst=leaf, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
    return topo


def _make_compute(task_id, duration_us, node=0, deps=None):
    return Task(
        task_id=task_id, job_id=0, type=TaskType.COMPUTE,
        node=node, duration_us=duration_us, deps=deps or [],
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


def _make_empty_execution_plan() -> ExecutionPlan:
    return ExecutionPlan()


# ============================================================
# Tests for TTEInfo dataclass
# ============================================================


class TestTTEInfo:
    def test_basic_fields(self):
        info = TTEInfo(task_id=0, tte_us=100.0, priority_score=0.01, priority_class="elastic")
        assert info.task_id == 0
        assert info.tte_us == 100.0
        assert info.priority_class == "elastic"


class TestFlowTiming:
    def test_basic_fields(self):
        ft = FlowTiming(task_id=0, start_time_us=10, finish_time_us=100)
        assert ft.task_id == 0
        assert ft.start_time_us == 10
        assert ft.finish_time_us == 100


# ============================================================
# Tests for compute_optimistic_timing
# ============================================================


class TestOptimisticTiming:
    def test_empty_workload(self):
        topo = _make_star_topo()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        timing = compute_optimistic_timing(wl, route_table, topo, plan)
        assert timing == {}

    def test_single_compute(self):
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=0)
        wl = _make_workload([c0])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        timing = compute_optimistic_timing(wl, route_table, topo, plan)
        assert timing[0] == 1000

    def test_linear_chain(self):
        """A -> B -> C chain."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=0, deps=[0])
        c2 = _make_compute(2, 300, node=0, deps=[1])
        wl = _make_workload([c0, c1, c2])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        timing = compute_optimistic_timing(wl, route_table, topo, plan)
        assert timing[0] == 100
        assert timing[1] == 300  # 100 + 200
        assert timing[2] == 600  # 100 + 200 + 300

    def test_diamond_dag(self):
        """A split into B,C and merge at D."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=0, deps=[0])
        c2 = _make_compute(2, 300, node=0, deps=[0])
        c3 = _make_compute(3, 50, node=0, deps=[1, 2])
        wl = _make_workload([c0, c1, c2, c3])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        timing = compute_optimistic_timing(wl, route_table, topo, plan)
        # B fin: 100+200=300, C fin: 100+300=400, D start=max(300,400)=400, D fin: 450
        assert timing[0] == 100
        assert timing[1] == 300
        assert timing[2] == 400
        assert timing[3] == 450

    def test_compute_order_edges(self):
        """Compute-order edges affect timing."""
        topo = _make_star_topo()
        # Two compute tasks on same node, no DAG dependency
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=0)
        wl = _make_workload([c0, c1])
        route_table = BfsStrategy().compute_routes(wl, topo)

        # With serialized order 0->1
        plan = ExecutionPlan(compute_order={0: [0, 1]})
        timing = compute_optimistic_timing(wl, route_table, topo, plan)
        assert timing[0] == 100
        assert timing[1] == 300  # waits for 0

    def test_mixed_compute_flow(self):
        """Compute → flow → compute chain."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=1024 * 1024, deps=[0])  # ~20us at 400G
        c2 = _make_compute(2, 50, node=1, deps=[1])
        wl = _make_workload([c0, f1, c2])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        timing = compute_optimistic_timing(wl, route_table, topo, plan)
        assert timing[0] == 100
        assert timing[2] > 150  # flow duration > 0


# ============================================================
# Tests for compute_tte
# ============================================================


class TestComputeTTE:
    def test_no_children_background(self):
        """Flow with no children gets background priority."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        tte_info, _ = compute_tte(wl, route_table, topo, plan)
        assert 0 in tte_info
        assert tte_info[0].tte_us == float("inf")
        assert tte_info[0].priority_class == "background"

    def test_direct_child_has_tte(self):
        """Flow followed by a compute gives positive TTE."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        c1 = _make_compute(1, 500, node=1, deps=[0])
        wl = _make_workload([f0, c1])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        tte_info, flow_timing = compute_tte(wl, route_table, topo, plan)
        assert 0 in tte_info
        # TTE = child_start - flow_finish. Since flow finish pushes child start,
        # TTE should be 0 (critical)
        assert tte_info[0].tte_us <= 0
        assert tte_info[0].priority_class == "critical"

    def test_flow_with_slack(self):
        """Flow with elastic TTE (non-zero but small)."""
        topo = _make_star_topo()
        # Create a case where flow has some slack before its child must start
        c0 = _make_compute(0, 100, node=0)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])  # ~20us
        # The flow has a sibling on critical path that makes the child start later
        c2 = _make_compute(2, 1000, node=0, deps=[0])  # long compute
        c3 = _make_compute(3, 100, node=1, deps=[1, 2])  # child waiting for both
        wl = _make_workload([c0, f1, c2, c3])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        tte_info, _ = compute_tte(wl, route_table, topo, plan)
        assert 1 in tte_info
        # TTE should be > 0 because the child must wait for the long compute too
        assert tte_info[1].tte_us > 0

    def test_multiple_children_min_tte(self):
        """Flow with multiple children takes min TTE."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        # Two children of f1: one urgent, one not
        c2 = _make_compute(2, 50, node=1, deps=[1])  # urgent
        c3 = _make_compute(3, 10000, node=1, deps=[1, 0])  # also waits for c0, starts later
        wl = _make_workload([c0, f1, c2, c3])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        tte_info, _ = compute_tte(wl, route_table, topo, plan)
        assert 1 in tte_info
        # TTE should reflect the more urgent child
        assert tte_info[1].tte_us <= 0  # at least one child is critical

    def test_critical_threshold(self):
        """TTE near zero is classified as critical."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        c1 = _make_compute(1, 100, node=1, deps=[0])
        wl = _make_workload([f0, c1])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        tte_info, _ = compute_tte(wl, route_table, topo, plan)
        assert tte_info[0].priority_class == "critical"
        assert tte_info[0].priority_score == 0.0

    def test_flow_timing_output(self):
        """FlowTiming map has correct start/finish."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])
        route_table = BfsStrategy().compute_routes(wl, topo)
        plan = _make_empty_execution_plan()

        _, flow_timing = compute_tte(wl, route_table, topo, plan)
        assert 0 in flow_timing
        assert flow_timing[0].start_time_us >= 0
        assert flow_timing[0].finish_time_us > flow_timing[0].start_time_us

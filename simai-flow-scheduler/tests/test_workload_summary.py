"""
Tests for workload summary module.

Tests WorkloadSummary data structure and compute_workload_summary function
(basic statistics, comm/compute ratio, DAG width, critical path stats, hot links).
"""

import pytest

from src.static_analysis.passes.contention_analysis import find_contention_groups
from src.static_analysis.passes.critical_path import analyze_critical_path
from src.static_analysis.passes.routing import BfsStrategy
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.static_analysis.passes.workload_summary import WorkloadSummary, compute_workload_summary
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Task,
    TaskType,
)


# --- Helpers ---


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


def _make_star_topo(bw=400.0, lat=0.5):
    """Star: nodes 0,1,2 connect to switch 10 (bidirectional)."""
    topo = NetworkTopology()
    for leaf in [0, 1, 2]:
        topo.add_link(Link(src=leaf, dst=10, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
        topo.add_link(Link(src=10, dst=leaf, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
    return topo


def _analyze(wl, topo):
    route_table = BfsStrategy().compute_routes(wl, topo)
    cp = analyze_critical_path(wl, route_table, topo)
    cg = find_contention_groups(wl, route_table, topo, cp)
    return compute_workload_summary(wl, cp, cg)


# ============================================================
# Tests for compute_workload_summary
# ============================================================


class TestComputeWorkloadSummary:
    """Tests for the compute_workload_summary function."""

    def test_empty_workload(self):
        topo = _make_star_topo()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
        route_table = BfsStrategy().compute_routes(wl, topo)
        cp = analyze_critical_path(wl, route_table, topo)
        cg = find_contention_groups(wl, route_table, topo, cp)
        summary = compute_workload_summary(wl, cp, cg)

        assert summary.total_tasks == 0
        assert summary.total_compute_tasks == 0
        assert summary.total_flow_tasks == 0
        assert summary.total_communication_bytes == 0
        assert summary.total_compute_time_us == 0
        assert summary.comm_compute_ratio == 0.0
        assert summary.avg_dag_width == 0.0
        assert summary.critical_path_length_us == 0
        assert summary.hot_links == []

    def test_single_compute_task(self):
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=0)
        wl = _make_workload([c0])
        summary = _analyze(wl, topo)

        assert summary.total_tasks == 1
        assert summary.total_compute_tasks == 1
        assert summary.total_flow_tasks == 0
        assert summary.total_compute_time_us == 1000
        assert summary.comm_compute_ratio == 0.0  # no comm
        assert summary.critical_path_length_us == 1000

    def test_single_flow_task(self):
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024 * 1024)
        wl = _make_workload([f0])
        summary = _analyze(wl, topo)

        assert summary.total_tasks == 1
        assert summary.total_compute_tasks == 0
        assert summary.total_flow_tasks == 1
        assert summary.total_communication_bytes == 1024 * 1024
        assert summary.total_compute_time_us == 0
        assert summary.comm_compute_ratio == 0.0  # no compute to compare

    def test_mixed_compute_and_flow(self):
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024 * 1024)
        wl = _make_workload([c0, f0])
        summary = _analyze(wl, topo)

        assert summary.total_tasks == 2
        assert summary.total_compute_tasks == 1
        assert summary.total_flow_tasks == 1
        assert summary.total_communication_bytes == 1024 * 1024
        assert summary.total_compute_time_us == 1000
        # comm_compute_ratio = comm_time / compute_time
        assert summary.comm_compute_ratio > 0.0

    def test_comm_compute_ratio_time_based(self):
        """comm_compute_ratio uses time, not bytes."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([c0, f0])
        summary = _analyze(wl, topo)

        # Ratio should be dimensionless (time / time)
        assert 0.0 <= summary.comm_compute_ratio < 100.0

    def test_dag_width_single_task(self):
        """Single task has width 1."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        wl = _make_workload([c0])
        summary = _analyze(wl, topo)

        assert summary.avg_dag_width == 1.0

    def test_dag_width_linear_chain(self):
        """Linear chain has width 1."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 100, node=0, deps=[0])
        c2 = _make_compute(2, 100, node=0, deps=[1])
        wl = _make_workload([c0, c1, c2])
        summary = _analyze(wl, topo)

        assert summary.avg_dag_width == 1.0

    def test_dag_width_parallel_tasks(self):
        """Parallel tasks increase width."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 100, node=1)
        c2 = _make_compute(2, 100, node=2)
        wl = _make_workload([c0, c1, c2])
        summary = _analyze(wl, topo)

        # All at depth 0, so width = 3
        assert summary.avg_dag_width == 3.0

    def test_dag_width_diamond(self):
        """Diamond DAG: 1 root, 2 parallel, 1 sink."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 100, node=1, deps=[0])
        c2 = _make_compute(2, 100, node=2, deps=[0])
        c3 = _make_compute(3, 100, node=0, deps=[1, 2])
        wl = _make_workload([c0, c1, c2, c3])
        summary = _analyze(wl, topo)

        # Depth 0: 1 task, Depth 1: 2 tasks, Depth 2: 1 task
        # avg = (1 + 2 + 1) / 3 = 4/3 ≈ 1.33
        assert abs(summary.avg_dag_width - 4 / 3) < 0.01

    def test_critical_path_length(self):
        """Critical path length equals makespan."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=0, deps=[0])
        wl = _make_workload([c0, c1])
        summary = _analyze(wl, topo)

        assert summary.critical_path_length_us == 300

    def test_critical_path_comm_fraction_no_comm(self):
        """No communication on critical path."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        wl = _make_workload([c0])
        summary = _analyze(wl, topo)

        assert summary.critical_path_comm_fraction == 0.0

    def test_critical_path_comm_fraction_all_comm(self):
        """All critical path is communication."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])
        summary = _analyze(wl, topo)

        assert summary.critical_path_comm_fraction == 1.0

    def test_critical_path_comm_fraction_mixed(self):
        """Mixed compute and comm on critical path."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        wl = _make_workload([c0, f0])
        summary = _analyze(wl, topo)

        # Both tasks are on critical path
        assert 0.0 < summary.critical_path_comm_fraction < 1.0

    def test_hot_links_ordering(self):
        """Hot links are sorted by total bytes (descending)."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=2000)
        f2 = _make_flow(2, src=1, dst=2, size_bytes=500)
        wl = _make_workload([f0, f1, f2])
        summary = _analyze(wl, topo)

        # Link (0,10) has 3000 bytes, (10,1) has 3000 bytes, (1,10) has 500, (10,2) has 500
        assert len(summary.hot_links) > 0
        # First link should have most bytes
        assert summary.hot_links[0][1] >= summary.hot_links[-1][1]

    def test_hot_links_top_10(self):
        """Hot links limited to top 10."""
        topo = _make_star_topo()
        # Create 20 different flows
        tasks = []
        for i in range(20):
            tasks.append(_make_flow(i, src=0, dst=1, size_bytes=1000 * (i + 1)))
        wl = _make_workload(tasks)
        summary = _analyze(wl, topo)

        # Should have at most 10 hot links
        assert len(summary.hot_links) <= 10

    def test_hot_links_structure(self):
        """Hot links have correct structure (link_id, bytes, num_flows)."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        wl = _make_workload([f0])
        summary = _analyze(wl, topo)

        assert len(summary.hot_links) > 0
        link_id, bytes_count, num_flows = summary.hot_links[0]
        assert isinstance(link_id, tuple)
        assert len(link_id) == 2
        assert isinstance(bytes_count, int)
        assert isinstance(num_flows, int)
        assert bytes_count > 0
        assert num_flows > 0

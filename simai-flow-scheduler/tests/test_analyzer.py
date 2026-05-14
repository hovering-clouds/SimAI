"""
Tests for workload analyzer module.

Tests DefaultAnalyzer class and DefaultAnalysisResult dataclass.
Verifies that all analysis modules are called in the correct order and
that the unified analysis interface works end-to-end.
"""

import pytest

from src.static_analysis.strategies.default_strategy import DefaultAnalysisResult, DefaultAnalyzer
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
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


# ============================================================
# Tests for DefaultAnalyzer
# ============================================================


class TestDefaultAnalyzer:
    """Tests for the DefaultAnalyzer class."""

    def test_empty_workload(self):
        """Analyzer handles empty workload."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))

        result = analyzer.analyze(wl)

        assert isinstance(result, DefaultAnalysisResult)
        assert result.routing_hints is not None
        assert result.critical_path is not None
        assert result.contention_groups == {}
        assert result.node_views == {}
        assert result.traffic_matrix is not None
        assert result.summary is not None

    def test_single_compute_task(self):
        """Analyzer handles single compute task."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        c0 = _make_compute(0, 1000, node=0)
        wl = _make_workload([c0])

        result = analyzer.analyze(wl)

        assert result.critical_path.makespan_us == 1000
        assert result.summary.total_compute_tasks == 1
        assert result.summary.total_flow_tasks == 0
        assert 0 in result.node_views

    def test_single_flow_task(self):
        """Analyzer handles single flow task."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024 * 1024)
        wl = _make_workload([f0])

        result = analyzer.analyze(wl)

        assert result.critical_path.makespan_us > 0
        assert result.summary.total_flow_tasks == 1
        assert len(result.contention_groups) > 0
        assert result.traffic_matrix.get_traffic(0, 1) == 1024 * 1024

    def test_mixed_workload(self):
        """Analyzer handles mixed compute and flow tasks."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        c0 = _make_compute(0, 1000, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        c1 = _make_compute(2, 500, node=1, deps=[1])
        wl = _make_workload([c0, f0, c1])

        result = analyzer.analyze(wl)

        # All analysis modules should have results
        assert result.routing_hints is not None
        assert result.critical_path.makespan_us > 1500
        assert len(result.contention_groups) > 0
        assert len(result.node_views) >= 2
        assert result.traffic_matrix.get_total_traffic() == 1024
        assert result.summary.total_tasks == 3

    def test_result_has_all_fields(self):
        """DefaultAnalysisResult contains all expected fields."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        c0 = _make_compute(0, 100, node=0)
        wl = _make_workload([c0])

        result = analyzer.analyze(wl)

        # Check all fields exist
        assert hasattr(result, 'routing_hints')
        assert hasattr(result, 'critical_path')
        assert hasattr(result, 'contention_groups')
        assert hasattr(result, 'node_views')
        assert hasattr(result, 'traffic_matrix')
        assert hasattr(result, 'summary')

    def test_routing_hints_used_by_critical_path(self):
        """Critical path uses routing hints for multi-hop duration."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])

        result = analyzer.analyze(wl)

        # Flow should have non-zero duration (uses routing hints for path)
        timing = result.critical_path.task_timings[0]
        assert timing.earliest_finish_us > timing.earliest_start_us

    def test_contention_groups_use_routing_and_timing(self):
        """Contention groups use both routing hints and critical path timing."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=2048)
        wl = _make_workload([f0, f1])

        result = analyzer.analyze(wl)

        # Should have contention groups for shared links
        assert len(result.contention_groups) > 0
        # Groups should have timing information
        for group in result.contention_groups.values():
            assert group.num_flows > 0
            assert len(group.time_windows) > 0

    def test_node_views_use_critical_path_timing(self):
        """Node views use ASAP times from critical path."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        c0 = _make_compute(0, 500, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        wl = _make_workload([c0, f0])

        result = analyzer.analyze(wl)

        # Node 0 should have send time at 500us (after compute)
        node0 = result.node_views[0]
        assert len(node0.estimated_send_times) == 1
        assert node0.estimated_send_times[0][0] == 500

    def test_traffic_matrix_independent(self):
        """Traffic matrix works independently of other modules."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=1, dst=2, size_bytes=2000)
        wl = _make_workload([f0, f1])

        result = analyzer.analyze(wl)

        # Traffic matrix should aggregate correctly
        assert result.traffic_matrix.get_traffic(0, 1) == 1000
        assert result.traffic_matrix.get_traffic(1, 2) == 2000
        assert result.traffic_matrix.get_total_traffic() == 3000

    def test_summary_aggregates_results(self):
        """Summary aggregates results from critical path and contention groups."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)
        c0 = _make_compute(0, 1000, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        wl = _make_workload([c0, f0])

        result = analyzer.analyze(wl)

        # Summary should reflect workload characteristics
        assert result.summary.total_tasks == 2
        assert result.summary.total_compute_tasks == 1
        assert result.summary.total_flow_tasks == 1
        assert result.summary.critical_path_length_us == result.critical_path.makespan_us
        assert len(result.summary.hot_links) > 0

    def test_multiple_analyses_independent(self):
        """Multiple analyses on same analyzer are independent."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)

        wl1 = _make_workload([_make_compute(0, 100, node=0)])
        wl2 = _make_workload([_make_compute(0, 200, node=0)])

        result1 = analyzer.analyze(wl1)
        result2 = analyzer.analyze(wl2)

        # Results should be different
        assert result1.critical_path.makespan_us == 100
        assert result2.critical_path.makespan_us == 200

    def test_complex_workload_end_to_end(self):
        """End-to-end test with complex workload."""
        topo = _make_star_topo()
        analyzer = DefaultAnalyzer(topo)

        # Create a diamond DAG with flows
        c0 = _make_compute(0, 100, node=0)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        f2 = _make_flow(2, src=0, dst=2, size_bytes=2048, deps=[0])
        c3 = _make_compute(3, 200, node=0, deps=[1, 2])
        wl = _make_workload([c0, f1, f2, c3])

        result = analyzer.analyze(wl)

        # Verify all modules produced results
        assert result.routing_hints is not None
        assert result.critical_path.makespan_us > 300
        assert len(result.contention_groups) > 0
        assert len(result.node_views) >= 3
        assert result.traffic_matrix.get_total_traffic() == 3072
        assert result.summary.total_tasks == 4
        assert result.summary.avg_dag_width > 1.0  # Has parallel branches

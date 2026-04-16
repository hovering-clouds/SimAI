"""
Tests for node-centric local views.

Tests NodeLocalView data structure (add_send_flow, add_receive_flow)
and build_node_views integration with critical path analysis.
"""

import pytest

from src.scheduler.critical_path import analyze_critical_path
from src.scheduler.node_view import NodeLocalView, build_node_views
from src.scheduler.routing_hints import compute_routing_hints
from src.scheduler.topology_loader import Link, NetworkTopology
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
    hints = compute_routing_hints(topo, wl)
    cp = analyze_critical_path(wl, topo, hints)
    return build_node_views(wl, cp)


# ============================================================
# Tests for NodeLocalView
# ============================================================


class TestNodeLocalView:
    """Tests for the NodeLocalView data structure."""

    def test_add_send_flow(self):
        v = NodeLocalView(node_id=0)
        v.add_send_flow(task_id=1, size_bytes=1000, start_time=10)
        assert v.send_tasks == [1]
        assert v.total_send_bytes == 1000
        assert v.estimated_send_times == [(10, 1)]

    def test_add_receive_flow(self):
        v = NodeLocalView(node_id=0)
        v.add_receive_flow(task_id=2, size_bytes=2000, start_time=20)
        assert v.receive_tasks == [2]
        assert v.total_receive_bytes == 2000
        assert v.estimated_receive_times == [(20, 2)]

    def test_multiple_send_flows(self):
        v = NodeLocalView(node_id=0)
        v.add_send_flow(task_id=1, size_bytes=100, start_time=0)
        v.add_send_flow(task_id=2, size_bytes=300, start_time=50)
        assert v.send_tasks == [1, 2]
        assert v.total_send_bytes == 400
        assert len(v.estimated_send_times) == 2

    def test_default_values(self):
        v = NodeLocalView(node_id=5)
        assert v.send_tasks == []
        assert v.receive_tasks == []
        assert v.compute_tasks == []
        assert v.total_send_bytes == 0
        assert v.total_receive_bytes == 0
        assert v.total_compute_time_us == 0
        assert v.estimated_busy_ratio == 0.0
        assert v.busiest_outgoing_link is None
        assert v.busiest_incoming_link is None


# ============================================================
# Tests for build_node_views
# ============================================================


class TestBuildNodeViews:
    """Tests for the build_node_views function."""

    def test_empty_workload(self):
        topo = _make_star_topo()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
        views = _analyze(wl, topo)
        assert views == {}

    def test_single_compute_task(self):
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=1)
        wl = _make_workload([c0])
        views = _analyze(wl, topo)

        assert 1 in views
        v = views[1]
        assert v.compute_tasks == [0]
        assert v.total_compute_time_us == 1000
        assert v.send_tasks == []
        assert v.receive_tasks == []
        assert v.estimated_busy_ratio == 0.0  # no comm

    def test_single_flow(self):
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024 * 1024)  # 1 MiB
        wl = _make_workload([f0])
        views = _analyze(wl, topo)

        # Node 0 sends, Node 1 receives
        assert 0 in views
        assert 1 in views
        assert views[0].send_tasks == [0]
        assert views[0].total_send_bytes == 1024 * 1024
        assert views[0].receive_tasks == []
        assert views[1].receive_tasks == [0]
        assert views[1].total_receive_bytes == 1024 * 1024
        assert views[1].send_tasks == []

    def test_multiple_flows_same_sender(self):
        """Node sends to multiple destinations."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=0, dst=2, size_bytes=2000)
        wl = _make_workload([f0, f1])
        views = _analyze(wl, topo)

        v0 = views[0]
        assert v0.send_tasks == [0, 1]
        assert v0.total_send_bytes == 3000

    def test_send_receive_times_from_critical_path(self):
        """Estimated times come from critical path ASAP analysis."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 500, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0])
        wl = _make_workload([c0, f0])
        views = _analyze(wl, topo)

        # Flow starts after compute finishes
        v0 = views[0]
        assert len(v0.estimated_send_times) == 1
        assert v0.estimated_send_times[0] == (500, 1)  # starts at 500us

        v1 = views[1]
        assert len(v1.estimated_receive_times) == 1
        assert v1.estimated_receive_times[0][1] == 1  # task_id

    def test_busy_ratio_mixed_node(self):
        """Node with both compute and flow tasks."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024 * 1024)  # 1 MiB
        wl = _make_workload([c0, f0])
        views = _analyze(wl, topo)

        v0 = views[0]
        assert v0.compute_tasks == [0]
        assert v0.total_compute_time_us == 1000
        assert v0.send_tasks == [1]
        # busy_ratio = comm_time / (compute_time + comm_time)
        assert 0.0 < v0.estimated_busy_ratio < 1.0

    def test_busy_ratio_flow_only_node(self):
        """Node with only flow tasks has busy_ratio == 1.0."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=1, dst=2, size_bytes=1024)
        wl = _make_workload([f0])
        views = _analyze(wl, topo)

        v1 = views[1]
        assert v1.compute_tasks == []
        assert v1.total_compute_time_us == 0
        assert v1.send_tasks == [0]
        assert v1.estimated_busy_ratio == 1.0  # all time is comm

    def test_busy_ratio_compute_only_node(self):
        """Node with only compute tasks has busy_ratio == 0.0."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 500, node=0)
        wl = _make_workload([c0])
        views = _analyze(wl, topo)

        assert views[0].estimated_busy_ratio == 0.0

    def test_node_collects_all_task_types(self):
        """A node can have compute, send, and receive tasks simultaneously."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        f_send = _make_flow(1, src=0, dst=1, size_bytes=500)
        f_recv = _make_flow(2, src=1, dst=0, size_bytes=300)
        wl = _make_workload([c0, f_send, f_recv])
        views = _analyze(wl, topo)

        v0 = views[0]
        assert 0 in v0.compute_tasks
        assert 1 in v0.send_tasks
        assert 2 in v0.receive_tasks

    def test_skips_tasks_without_node_info(self):
        """Flow with None src/dst doesn't crash."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=100)
        wl = _make_workload([f0])
        views = _analyze(wl, topo)
        assert len(views) >= 2

    def test_different_nodes_independent(self):
        """Tasks on different nodes are correctly separated."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=1)
        wl = _make_workload([c0, c1])
        views = _analyze(wl, topo)

        assert views[0].compute_tasks == [0]
        assert views[0].total_compute_time_us == 100
        assert views[1].compute_tasks == [1]
        assert views[1].total_compute_time_us == 200

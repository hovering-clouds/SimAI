"""
Tests for traffic matrix module.

Tests TrafficMatrix data structure (get_traffic, get_total_traffic)
and compute_traffic_matrix function (traffic accumulation, top senders/receivers,
bidirectional pair detection).
"""

import pytest

from src.scheduler.traffic_matrix import TrafficMatrix, compute_traffic_matrix
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


# ============================================================
# Tests for TrafficMatrix
# ============================================================


class TestTrafficMatrix:
    """Tests for the TrafficMatrix data structure."""

    def test_get_traffic_existing(self):
        tm = TrafficMatrix()
        tm.traffic[(0, 1)] = 1000
        assert tm.get_traffic(0, 1) == 1000

    def test_get_traffic_nonexistent(self):
        tm = TrafficMatrix()
        assert tm.get_traffic(0, 1) == 0

    def test_get_total_traffic_empty(self):
        tm = TrafficMatrix()
        assert tm.get_total_traffic() == 0

    def test_get_total_traffic_multiple_pairs(self):
        tm = TrafficMatrix()
        tm.traffic[(0, 1)] = 1000
        tm.traffic[(1, 2)] = 2000
        tm.traffic[(0, 2)] = 500
        assert tm.get_total_traffic() == 3500

    def test_default_values(self):
        tm = TrafficMatrix()
        assert tm.traffic == {}
        assert tm.top_senders == []
        assert tm.top_receivers == []


# ============================================================
# Tests for compute_traffic_matrix
# ============================================================


class TestComputeTrafficMatrix:
    """Tests for the compute_traffic_matrix function."""

    def test_empty_workload(self):
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
        tm = compute_traffic_matrix(wl)
        assert tm.traffic == {}
        assert tm.top_senders == []
        assert tm.top_receivers == []
        assert tm.get_total_traffic() == 0

    def test_single_flow(self):
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic[(0, 1)] == 1024
        assert tm.get_total_traffic() == 1024
        assert tm.top_senders == [(0, 1024)]
        assert tm.top_receivers == [(1, 1024)]

    def test_traffic_accumulation_same_pair(self):
        """Multiple flows between same src-dst pair accumulate."""
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=500)
        f2 = _make_flow(2, src=0, dst=1, size_bytes=300)
        wl = _make_workload([f0, f1, f2])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic[(0, 1)] == 1800
        assert tm.get_total_traffic() == 1800

    def test_traffic_different_pairs(self):
        """Flows between different pairs are tracked separately."""
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=1, dst=2, size_bytes=2000)
        f2 = _make_flow(2, src=0, dst=2, size_bytes=500)
        wl = _make_workload([f0, f1, f2])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic[(0, 1)] == 1000
        assert tm.traffic[(1, 2)] == 2000
        assert tm.traffic[(0, 2)] == 500
        assert tm.get_total_traffic() == 3500

    def test_top_senders_ordering(self):
        """Top senders are sorted by total bytes sent (descending)."""
        f0 = _make_flow(0, src=0, dst=3, size_bytes=1000)
        f1 = _make_flow(1, src=1, dst=3, size_bytes=3000)
        f2 = _make_flow(2, src=2, dst=3, size_bytes=2000)
        wl = _make_workload([f0, f1, f2])
        tm = compute_traffic_matrix(wl)

        assert tm.top_senders == [(1, 3000), (2, 2000), (0, 1000)]

    def test_top_receivers_ordering(self):
        """Top receivers are sorted by total bytes received (descending)."""
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=0, dst=2, size_bytes=3000)
        f2 = _make_flow(2, src=0, dst=3, size_bytes=2000)
        wl = _make_workload([f0, f1, f2])
        tm = compute_traffic_matrix(wl)

        assert tm.top_receivers == [(2, 3000), (3, 2000), (1, 1000)]

    def test_top_senders_aggregates_multiple_destinations(self):
        """A sender's total includes all destinations."""
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=0, dst=2, size_bytes=500)
        f2 = _make_flow(2, src=1, dst=2, size_bytes=300)
        wl = _make_workload([f0, f1, f2])
        tm = compute_traffic_matrix(wl)

        assert tm.top_senders == [(0, 1500), (1, 300)]

    def test_top_receivers_aggregates_multiple_sources(self):
        """A receiver's total includes all sources."""
        f0 = _make_flow(0, src=0, dst=2, size_bytes=1000)
        f1 = _make_flow(1, src=1, dst=2, size_bytes=500)
        f2 = _make_flow(2, src=1, dst=3, size_bytes=300)
        wl = _make_workload([f0, f1, f2])
        tm = compute_traffic_matrix(wl)

        assert tm.top_receivers == [(2, 1500), (3, 300)]

    def test_skips_compute_tasks(self):
        """Compute tasks are ignored."""
        c0 = _make_compute(0, 1000, node=0)
        wl = _make_workload([c0])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic == {}
        assert tm.get_total_traffic() == 0

    def test_skips_flow_with_none_src(self):
        """Flow with None src is skipped."""
        f0 = Task(
            task_id=0, job_id=0, type=TaskType.FLOW,
            src=None, dst=1, size_bytes=1000,
            comm_type=CommType.TP_ALLREDUCE_RING, deps=[],
        )
        wl = _make_workload([f0])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic == {}

    def test_skips_flow_with_none_dst(self):
        """Flow with None dst is skipped."""
        f0 = Task(
            task_id=0, job_id=0, type=TaskType.FLOW,
            src=0, dst=None, size_bytes=1000,
            comm_type=CommType.TP_ALLREDUCE_RING, deps=[],
        )
        wl = _make_workload([f0])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic == {}

    def test_zero_size_flow_counted(self):
        """Flow with zero size_bytes is counted as 0."""
        f0 = _make_flow(0, src=0, dst=1, size_bytes=0)
        wl = _make_workload([f0])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic[(0, 1)] == 0
        assert tm.get_total_traffic() == 0

    def test_mixed_compute_and_flow(self):
        """Workload with both compute and flow tasks."""
        c0 = _make_compute(0, 100, node=0)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1000)
        c1 = _make_compute(2, 200, node=1)
        wl = _make_workload([c0, f0, c1])
        tm = compute_traffic_matrix(wl)

        assert tm.traffic[(0, 1)] == 1000
        assert tm.get_total_traffic() == 1000

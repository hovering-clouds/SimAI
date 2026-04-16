"""
Tests for link contention analysis module.

Tests LinkContentionGroup data structure (add_flow, concurrency queries,
sweep line peak detection) and find_contention_groups integration with
routing hints and critical path analysis.
"""

import pytest

from src.scheduler.critical_path import analyze_critical_path
from src.scheduler.contention_analysis import (
    LinkContentionGroup,
    find_contention_groups,
)
from src.scheduler.routing_hints import RoutingHints, compute_routing_hints
from src.scheduler.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Task,
    TaskType,
)


# --- Helpers ---


def _make_compute(task_id, duration_us, deps=None, node=0):
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
# Tests for LinkContentionGroup
# ============================================================


class TestLinkContentionGroup:
    """Tests for the LinkContentionGroup data structure."""

    def test_add_single_flow(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(task_id=0, size_bytes=1000, entry_time=0, exit_time=10)
        assert g.num_flows == 1
        assert g.all_flows == [0]
        assert g.time_windows[0] == (0, 10)
        assert g.total_data_bytes == 1000
        assert g.min_size_bytes == 1000
        assert g.max_size_bytes == 1000
        assert g.avg_size_bytes == 1000.0

    def test_add_multiple_flows_stats(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(task_id=0, size_bytes=100, entry_time=0, exit_time=5)
        g.add_flow(task_id=1, size_bytes=300, entry_time=0, exit_time=15)
        assert g.num_flows == 2
        assert g.total_data_bytes == 400
        assert g.min_size_bytes == 100
        assert g.max_size_bytes == 300
        assert g.avg_size_bytes == 200.0

    def test_worst_case_concurrency(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 10)
        g.add_flow(1, 200, 5, 15)
        g.add_flow(2, 300, 20, 30)
        assert g.worst_case_concurrency == 3

    def test_get_concurrency_at_time(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 10)
        g.add_flow(1, 200, 5, 15)
        g.add_flow(2, 300, 20, 30)

        assert g.get_concurrency_at_time(2) == 1   # Only flow 0
        assert g.get_concurrency_at_time(7) == 2   # Flows 0 and 1
        assert g.get_concurrency_at_time(12) == 1   # Only flow 1
        assert g.get_concurrency_at_time(25) == 1   # Only flow 2
        assert g.get_concurrency_at_time(50) == 0   # None active

    def test_get_concurrency_boundary(self):
        """Entry time is inclusive, exit time is exclusive."""
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 10)

        assert g.get_concurrency_at_time(0) == 1   # entry inclusive
        assert g.get_concurrency_at_time(9) == 1
        assert g.get_concurrency_at_time(10) == 0   # exit exclusive

    def test_peak_concurrency_window(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 20)
        g.add_flow(1, 200, 5, 15)
        g.add_flow(2, 300, 20, 30)

        g.analyze_temporal_contention()
        assert g.best_case_concurrency == 2

    def test_no_contention_when_separated(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 10)
        g.add_flow(1, 200, 20, 30)

        g.analyze_temporal_contention()
        assert g.best_case_concurrency == 1
        assert not g.has_temporal_contention

    def test_has_temporal_contention(self):
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 10)
        g.add_flow(1, 200, 5, 15)

        g.analyze_temporal_contention()
        assert g.has_temporal_contention

    def test_contention_ratio(self):
        g = LinkContentionGroup(link_id=(0, 1))
        # No flows
        assert g.contention_ratio == 0.0

        # Single flow
        g.add_flow(0, 100, 0, 10)
        g.analyze_temporal_contention()
        assert g.contention_ratio == 0.0

        # Two flows overlapping
        g.add_flow(1, 200, 5, 15)
        g.analyze_temporal_contention()
        assert g.contention_ratio == 1.0

    def test_contention_ratio_partial(self):
        """3 flows, only 2 overlap at peak → ratio = 2/3."""
        g = LinkContentionGroup(link_id=(0, 1))
        g.add_flow(0, 100, 0, 10)
        g.add_flow(1, 200, 5, 15)
        g.add_flow(2, 300, 100, 200)

        g.analyze_temporal_contention()
        assert g.best_case_concurrency == 2
        assert g.worst_case_concurrency == 3
        assert abs(g.contention_ratio - 2 / 3) < 1e-9

    def test_empty_group_peak(self):
        g = LinkContentionGroup(link_id=(0, 1))
        assert g.get_peak_concurrency_window() == (0, 0, 0)


# ============================================================
# Tests for find_contention_groups
# ============================================================


class TestFindContentionGroups:
    """Tests for the main find_contention_groups function."""

    def test_empty_workload(self):
        topo = _make_star_topo()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)
        assert groups == {}

    def test_single_flow(self):
        """Single flow creates one contention group per hop."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024 * 1024)  # 1 MiB
        wl = _make_workload([f0])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)

        # Path: 0 → 10 → 1, so groups for (0,10) and (10,1)
        assert (0, 10) in groups
        assert (10, 1) in groups
        assert groups[(0, 10)].num_flows == 1
        assert groups[(10, 1)].num_flows == 1

    def test_two_flows_sharing_link(self):
        """Two flows through same switch share a link."""
        topo = _make_star_topo()
        # Flow 0→1 and Flow 0→2 both use link (0, 10)
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024 * 1024)
        f1 = _make_flow(1, src=0, dst=2, size_bytes=1024 * 1024)
        wl = _make_workload([f0, f1])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)

        # Link (0, 10) has 2 flows
        assert groups[(0, 10)].num_flows == 2
        # Both start at time 0, so they overlap → temporal contention
        assert groups[(0, 10)].has_temporal_contention

    def test_temporally_separated_flows(self):
        """Sequential flows on same link don't contend."""
        topo = _make_star_topo()
        # Flow 0: compute(100us) → flow 0→1
        # Flow 1: compute(100us) → flow 0→1 (depends on flow 0)
        c0 = _make_compute(0, 100)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024 * 1024, deps=[0])
        f1 = _make_flow(2, src=0, dst=1, size_bytes=1024 * 1024, deps=[1])
        wl = _make_workload([c0, f0, f1])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)

        g = groups[(0, 10)]
        assert g.num_flows == 2
        # f0 starts at 100us, f1 starts after f0 finishes → no temporal overlap
        assert not g.has_temporal_contention

    def test_per_link_timing_correctness(self):
        """Per-link entry/exit times respect cumulative propagation latency."""
        topo = _make_star_topo(bw=400.0, lat=10.0)  # 10us latency per hop
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024 * 1024)
        wl = _make_workload([f0])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)

        # Path: 0 → 10 → 1
        # Link (0,10): entry=0, exit=0+tx
        # Link (10,1): entry=10us (latency of first hop), exit=10+tx
        entry_01, exit_01 = groups[(0, 10)].time_windows[0]
        entry_101, exit_101 = groups[(10, 1)].time_windows[0]

        assert entry_01 == 0
        assert entry_101 == pytest.approx(10, abs=1)  # ~10us propagation from first hop

    def test_skips_compute_tasks(self):
        """Compute tasks are ignored."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 100)
        wl = _make_workload([c0])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)
        assert groups == {}

    def test_skips_flow_zero_size(self):
        """Flow with zero size_bytes is skipped."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=0)
        wl = _make_workload([f0])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)
        assert groups == {}

    def test_contention_groups_with_critical_path_timing(self):
        """Flows with different start times (from deps) have correct timing."""
        topo = _make_star_topo()
        c0 = _make_compute(0, 1000)
        f0 = _make_flow(1, src=0, dst=1, size_bytes=1024 * 1024, deps=[0])
        f1 = _make_flow(2, src=0, dst=1, size_bytes=1024 * 1024)  # starts at 0
        wl = _make_workload([c0, f0, f1])
        hints = compute_routing_hints(topo, wl)
        cp = analyze_critical_path(wl, topo, hints)

        groups = find_contention_groups(wl, topo, hints, cp)

        g = groups[(0, 10)]
        # f1 starts at 0, f0 starts at 1000 → likely overlap depends on duration
        assert g.num_flows == 2
        # f1's entry should be 0, f0's entry should be 1000
        assert g.time_windows[2][0] == 0  # f1 starts at 0
        assert g.time_windows[1][0] == 1000  # f0 starts at 1000

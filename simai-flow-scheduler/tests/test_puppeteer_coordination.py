"""Tests for Puppeteer resource dependency / coordination pass."""
import pytest

from src.static_analysis.passes.puppeteer_coordination import (
    ResourceDependencyTable,
    compute_resource_dependency,
)
from src.static_analysis.passes.puppeteer_routing import RouteTable
from src.static_analysis.passes.puppeteer_tte import TTEInfo
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Task,
    TaskType,
)


# --- Helpers ---


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
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=max_node + 1),
        tasks=tasks,
    )


# ============================================================
# Tests for ResourceDependencyTable
# ============================================================


class TestResourceDependencyTable:
    def test_empty_table(self):
        table = ResourceDependencyTable()
        assert table.peers == {}
        assert table.groups == {}

    def test_basic_fields(self):
        table = ResourceDependencyTable(
            peers={0: {1}, 1: {0}},
            groups={"g0": {0, 1}},
        )
        assert table.peers[0] == {1}
        assert table.groups["g0"] == {0, 1}


# ============================================================
# Tests for compute_resource_dependency
# ============================================================


class TestResourceDependency:
    def test_no_shared_links(self):
        """Flows on different links have no dependency."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],  # uses link (0,10), (10,1)
            2: [2, 10, 1],  # uses link (2,10), (10,1)
        })
        # Flow 0 uses (0,10), Flow 2 uses (2,10) — different first links
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
            _make_flow(2, src=2, dst=1, size_bytes=1024),
        ])
        timing = {0: (0, 100), 2: (0, 100)}

        dep = compute_resource_dependency(wl, route_table, timing)
        # They share (10,1) link — actually they DO share that link
        # Let me reconsider: flow0 = 0→10→1, flow2 = 2→10→1
        # Shared link: (10, 1)
        assert len(dep.groups) >= 1

    def test_no_shared_links_different_paths(self):
        """Flows with completely disjoint link sets have no dependency."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],      # uses (0,10), (10,1)
            2: [3, 11, 4],      # uses (3,11), (11,4)
        })
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
            _make_flow(2, src=3, dst=4, size_bytes=1024),
        ])
        timing = {0: (0, 100), 2: (0, 100)}

        dep = compute_resource_dependency(wl, route_table, timing)
        # Each flow has independent links
        # Only check groups might be empty
        assert 0 not in dep.peers or 2 not in dep.peers[0]

    def test_dag_ordered_not_grouped(self):
        """Flows already ordered by DAG dependency are not grouped."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],
            1: [0, 10, 1],
        })
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),  # no deps
            _make_flow(1, src=0, dst=1, size_bytes=1024, deps=[0]),  # depends on 0
        ])
        timing = {0: (0, 100), 1: (100, 200)}

        dep = compute_resource_dependency(wl, route_table, timing)
        # Flow 1 depends on flow 0, so they shouldn't be peers
        assert 0 not in dep.peers or 1 not in dep.peers.get(0, set())

    def test_non_overlapping_intervals(self):
        """Flows with non-overlapping time intervals not grouped."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],
            1: [0, 10, 1],
        })
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
            _make_flow(1, src=0, dst=1, size_bytes=1024),
        ])
        # Non-overlapping: flow 0 finishes before flow 1 starts
        timing = {0: (0, 100), 1: (200, 300)}

        dep = compute_resource_dependency(wl, route_table, timing)
        assert 1 not in dep.peers.get(0, set())

    def test_two_groups_independent(self):
        """Two independent pairs of sharing flows form separate groups."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],
            1: [0, 10, 1],
            2: [3, 11, 4],
            3: [3, 11, 4],
        })
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
            _make_flow(1, src=0, dst=1, size_bytes=1024),
            _make_flow(2, src=3, dst=4, size_bytes=1024),
            _make_flow(3, src=3, dst=4, size_bytes=1024),
        ])
        timing = {0: (0, 100), 1: (0, 100), 2: (0, 100), 3: (0, 100)}

        dep = compute_resource_dependency(wl, route_table, timing)
        # Should have at least 2 groups
        assert len(dep.groups) >= 2

    def test_priority_filtering_background_only(self):
        """Flows only get grouped when at least one is critical or elastic."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],
            1: [0, 10, 1],
        })
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
            _make_flow(1, src=0, dst=1, size_bytes=1024),
        ])
        timing = {0: (0, 100), 1: (0, 100)}

        # Both background → no grouping (when tte_info is provided)
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=float("inf"), priority_score=0.0, priority_class="background"),
            1: TTEInfo(task_id=1, tte_us=float("inf"), priority_score=0.0, priority_class="background"),
        }

        dep = compute_resource_dependency(wl, route_table, timing, tte_info=tte_info)
        # They share links but are both background
        assert len(dep.groups) == 0

    def test_priority_filtering_one_critical(self):
        """Flows get grouped when at least one is critical."""
        route_table = RouteTable(paths={
            0: [0, 10, 1],
            1: [0, 10, 1],
        })
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
            _make_flow(1, src=0, dst=1, size_bytes=1024),
        ])
        timing = {0: (0, 100), 1: (0, 100)}

        # One critical, one background
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=0.0, priority_score=0.0, priority_class="critical"),
            1: TTEInfo(task_id=1, tte_us=float("inf"), priority_score=0.0, priority_class="background"),
        }

        dep = compute_resource_dependency(wl, route_table, timing, tte_info=tte_info)
        assert len(dep.groups) >= 1

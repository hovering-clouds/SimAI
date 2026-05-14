"""Tests for Puppeteer resource dependency / coordination pass.

⚠️ compute_resource_dependency is a stub until Phase 3.
"""
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
# Tests for ResourceDependencyTable dataclass
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
# Tests for compute_resource_dependency (Phase 2 stub)
# ============================================================


class TestComputeResourceDependency:
    """Phase 2 stub — always returns empty result."""

    def test_returns_empty_by_default(self):
        dep = compute_resource_dependency(
            _make_workload([]), RouteTable(), {},
        )
        assert dep.peers == {}
        assert dep.groups == {}

    def test_empty_with_data(self):
        """Even with real inputs, stub returns empty."""
        route_table = RouteTable(paths={0: [0, 10, 1]})
        wl = _make_workload([
            _make_flow(0, src=0, dst=1, size_bytes=1024),
        ])
        dep = compute_resource_dependency(wl, route_table, {0: (0, 100)})
        assert dep.peers == {}
        assert dep.groups == {}

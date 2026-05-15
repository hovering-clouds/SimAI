"""Tests for Puppeteer offline greedy routing pass."""
import pytest

from src.static_analysis.passes.routing import (
    GreedyRouteTable,
    GreedyStrategy,
    k_shortest_paths,
)
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Task,
    TaskType,
)


# --- Helpers ---


def _make_star_topo(bw=400.0, lat=0.5):
    """Star: nodes 0,1,2 connect to switch 10 (bidirectional)."""
    topo = NetworkTopology()
    for leaf in [0, 1, 2]:
        topo.add_link(Link(src=leaf, dst=10, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
        topo.add_link(Link(src=10, dst=leaf, bandwidth_gbps=bw, latency_us=lat, error_rate=0))
    return topo


def _make_mesh_topo():
    """Mesh: 0-1-2 with direct links forming a line."""
    topo = NetworkTopology()
    topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=1, dst=0, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=1, dst=2, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=2, dst=1, bandwidth_gbps=100, latency_us=1, error_rate=0))
    return topo


def _make_dual_path_topo():
    """Two parallel paths between leaf switches via two spines."""
    topo = NetworkTopology()
    # Leaf 0 → Spine 10, Spine 11
    topo.add_link(Link(src=0, dst=10, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=10, dst=0, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=0, dst=11, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=11, dst=0, bandwidth_gbps=100, latency_us=1, error_rate=0))
    # Leaf 1 → Spine 10, Spine 11
    topo.add_link(Link(src=1, dst=10, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=10, dst=1, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=1, dst=11, bandwidth_gbps=100, latency_us=1, error_rate=0))
    topo.add_link(Link(src=11, dst=1, bandwidth_gbps=100, latency_us=1, error_rate=0))
    return topo


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
# Tests for RouteTable
# ============================================================


class TestRouteTable:
    def test_store_and_retrieve(self):
        table = GreedyRouteTable()
        table.paths[0] = [0, 10, 1]
        assert table.paths[0] == [0, 10, 1]

    def test_get_path_returns_copy(self):
        table = GreedyRouteTable()
        table.paths[0] = [0, 10, 1]
        got = list(table.paths[0])
        got.append(99)
        assert table.paths[0] == [0, 10, 1]

    def test_missing_path_raises(self):
        table = GreedyRouteTable()
        with pytest.raises(KeyError):
            table.paths[42]


# ============================================================
# Tests for k_shortest_paths
# ============================================================


class TestKShortestPaths:
    def test_src_eq_dst(self):
        topo = _make_star_topo()
        paths = k_shortest_paths(topo, 0, 0, k=4)
        assert paths == [[0]]

    def test_single_path_star(self):
        topo = _make_star_topo()
        paths = k_shortest_paths(topo, 0, 1, k=4)
        # Only one route: 0 -> 10 -> 1
        assert [0, 10, 1] in paths
        assert len(paths) == 1

    def test_dual_path(self):
        topo = _make_dual_path_topo()
        paths = k_shortest_paths(topo, 0, 1, k=4)
        # Two equal-length paths: 0->10->1 and 0->11->1
        assert len(paths) == 2
        assert all(len(p) == 3 for p in paths)
        # Both paths must be valid
        for p in paths:
            assert p[0] == 0
            assert p[-1] == 1

    def test_k_limit(self):
        topo = _make_dual_path_topo()
        paths = k_shortest_paths(topo, 0, 1, k=1)
        assert len(paths) == 1

    def test_deterministic_order(self):
        topo = _make_dual_path_topo()
        paths1 = k_shortest_paths(topo, 0, 1, k=4)
        paths2 = k_shortest_paths(topo, 0, 1, k=4)
        assert paths1 == paths2

    def test_no_path(self):
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100, latency_us=1, error_rate=0))
        topo.add_link(Link(src=1, dst=0, bandwidth_gbps=100, latency_us=1, error_rate=0))
        # Node 2 is isolated
        paths = k_shortest_paths(topo, 0, 2, k=4)
        assert paths == []


# ============================================================
# Tests for compute_greedy_routes
# ============================================================


class TestGreedyRoutes:
    def test_single_flow(self):
        """Single flow gets the only available path."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])
        timing = {0: (0, 100)}

        routes = GreedyStrategy(timing, k=4).compute_routes(wl, topo)
        assert 0 in routes.paths
        assert routes.paths[0] == [0, 10, 1]

    def test_two_flows_same_path_star(self):
        """Two flows sharing a link in a star are both routed."""
        topo = _make_star_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        f1 = _make_flow(1, src=0, dst=2, size_bytes=1024)
        wl = _make_workload([f0, f1])
        timing = {0: (0, 100), 1: (0, 100)}

        routes = GreedyStrategy(timing, k=4).compute_routes(wl, topo)
        assert 0 in routes.paths
        assert 1 in routes.paths

    def test_deterministic_output(self):
        """Same inputs produce same route table."""
        topo = _make_dual_path_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0])
        timing = {0: (0, 100)}

        routes1 = GreedyStrategy(timing, k=4).compute_routes(wl, topo)
        routes2 = GreedyStrategy(timing, k=4).compute_routes(wl, topo)
        assert routes1.paths[0] == routes2.paths[0]

    def test_dual_path_spreads_flows(self):
        """Two concurrent flows between same nodes spread across spines."""
        topo = _make_dual_path_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0, f1])
        timing = {0: (0, 100), 1: (0, 100)}

        routes = GreedyStrategy(timing, k=4).compute_routes(wl, topo)
        # Two flows between same nodes should use different spines
        assert routes.paths[0] != routes.paths[1]

    def test_sequential_flows_same_path(self):
        """Non-overlapping flows may use same path."""
        topo = _make_dual_path_topo()
        f0 = _make_flow(0, src=0, dst=1, size_bytes=1024)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=1024)
        wl = _make_workload([f0, f1])
        # Non-overlapping timing
        timing = {0: (0, 100), 1: (200, 300)}

        routes = GreedyStrategy(timing, k=4).compute_routes(wl, topo)
        # Both can use shortest path since they don't overlap
        assert len(routes.paths[0]) == len(routes.paths[1]) == 3

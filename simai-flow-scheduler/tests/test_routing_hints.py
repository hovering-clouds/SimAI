"""
Tests for routing hints module.

Tests BFS shortest path computation, route table lookups,
and integration with P2PWorkload + NetworkTopology.
"""

import os

import pytest

from src.static_analysis.passes.routing import BfsRouteTable, BfsStrategy, bfs_shortest_path
from src.static_analysis.passes.topology_loader import Link, NetworkTopology, TopologyLoader
from src.workload_format.schema import (
    CommType,
    Job,
    Meta,
    P2PWorkload,
    Task,
    TaskType,
    Phase,
)


# Path to real topology file for integration tests
SIMAI_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPECTRUM_X_TOPO = os.path.join(
    SIMAI_ROOT,
    "astra-sim-alibabacloud",
    "inputs",
    "topo",
    "Spectrum-X_8g_8gps_400Gbps_H100",
)


# --- Helper: build simple topologies ---


def _make_linear_topo(n: int) -> NetworkTopology:
    """
    Linear topology: 0 -> 1 -> 2 -> ... -> n-1.
    Each link: 100Gbps, 1.0us latency.
    """
    topo = NetworkTopology()
    topo.gpu_nodes = list(range(n))
    topo.total_nodes = n
    for i in range(n - 1):
        topo.add_link(Link(src=i, dst=i + 1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
    return topo


def _make_star_topo(center: int, leaves: list[int]) -> NetworkTopology:
    """
    Star topology: all leaves connect to center bidirectionally.
    Each link: 400Gbps, 0.5us latency.
    """
    topo = NetworkTopology()
    topo.gpu_nodes = leaves
    topo.switch_nodes = [center]
    topo.total_nodes = max(leaves + [center]) + 1
    for leaf in leaves:
        topo.add_link(Link(src=leaf, dst=center, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))
        topo.add_link(Link(src=center, dst=leaf, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))
    return topo


def _make_workload_with_flows(
    flow_specs: list[tuple[int, int, int]],
) -> P2PWorkload:
    """
    Create a minimal P2PWorkload with flow tasks.

    Args:
        flow_specs: list of (src, dst, size_bytes) tuples
    """
    tasks = []
    for i, (src, dst, size_bytes) in enumerate(flow_specs):
        tasks.append(
            Task(
                task_id=i,
                job_id=0,
                type=TaskType.FLOW,
                src=src,
                dst=dst,
                size_bytes=size_bytes,
                comm_type=CommType.TP_ALLREDUCE_RING,
            )
        )
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=max(max(s, d) for s, d, _ in flow_specs) + 1),
        tasks=tasks,
    )


# ============================================================
# Tests for bfs_shortest_path
# ============================================================


class TestBfsShortestPath:
    """Tests for the BFS shortest path algorithm."""

    def test_same_node(self):
        """src == dst returns [src]."""
        topo = _make_linear_topo(3)
        result = bfs_shortest_path(topo, 1, 1)
        assert result == [1]

    def test_direct_neighbor(self):
        """Adjacent nodes: path = [src, dst]."""
        topo = _make_linear_topo(3)
        result = bfs_shortest_path(topo, 0, 1)
        assert result == [0, 1]

    def test_two_hop_path(self):
        """Two hops: 0 -> 1 -> 2."""
        topo = _make_linear_topo(3)
        result = bfs_shortest_path(topo, 0, 2)
        assert result == [0, 1, 2]

    def test_multi_hop_linear(self):
        """Linear 0 -> 1 -> 2 -> 3 -> 4."""
        topo = _make_linear_topo(5)
        result = bfs_shortest_path(topo, 0, 4)
        assert result == [0, 1, 2, 3, 4]

    def test_no_path_raises_value_error(self):
        """Disconnected nodes raise ValueError."""
        topo = NetworkTopology()
        # Only one link: 0->1, no path from 0 to 2
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        with pytest.raises(ValueError, match="No path found from node 0 to node 2"):
            bfs_shortest_path(topo, 0, 2)

    def test_star_two_hop(self):
        """Star topology: leaf -> center -> other leaf."""
        topo = _make_star_topo(center=10, leaves=[0, 1, 2])
        result = bfs_shortest_path(topo, 0, 1)
        assert result == [0, 10, 1]

    def test_chooses_shortest_not_first(self):
        """BFS guarantees shortest path by hop count."""
        topo = NetworkTopology()
        # Direct link 0->2 (1 hop)
        topo.add_link(Link(src=0, dst=2, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        # Also path 0->1->2 (2 hops)
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        topo.add_link(Link(src=1, dst=2, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))

        result = bfs_shortest_path(topo, 0, 2)
        assert result == [0, 2]  # Direct link is shorter


# ============================================================
# Tests for BfsRouteTable.get_path_by_endpoints
# ============================================================


class TestBfsRouteTableGetPath:
    """Tests for BfsRouteTable path lookups."""

    def test_returns_path(self):
        topo = _make_linear_topo(3)
        table = BfsRouteTable(topo)
        table.ensure_path(0, 2, bfs_shortest_path(topo, 0, 2))
        path = table.get_path_by_endpoints(0, 2)
        assert path == [0, 1, 2]

    def test_stores_path(self):
        """Path is stored in _paths after ensure_path."""
        topo = _make_linear_topo(3)
        table = BfsRouteTable(topo)
        table.ensure_path(0, 2, bfs_shortest_path(topo, 0, 2))
        assert (0, 2) in table._paths
        assert table._paths[(0, 2)] == [0, 1, 2]

    def test_no_path_raises_key_error(self):
        """Missing endpoint pair raises KeyError."""
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        table = BfsRouteTable(topo)
        with pytest.raises(KeyError):
            table.get_path_by_endpoints(0, 5)

    def test_same_node_path(self):
        """src == dst returns [src]."""
        topo = _make_linear_topo(3)
        table = BfsRouteTable(topo)
        table.ensure_path(1, 1, bfs_shortest_path(topo, 1, 1))
        path = table.get_path_by_endpoints(1, 1)
        assert path == [1]


# ============================================================
# Tests for BfsStrategy.compute_routes
# ============================================================


class TestBfsStrategy:
    """Tests for BfsStrategy.compute_routes."""

    def test_single_flow_direct_link(self):
        """Single flow on direct link: path stored in route table."""
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        workflow = _make_workload_with_flows([(0, 1, 1000)])

        table = BfsStrategy().compute_routes(workflow, topo)

        assert (0, 1) in table._paths
        assert table._paths[(0, 1)] == [0, 1]

    def test_single_flow_two_hops(self):
        """Single flow through switch: path stored in route table."""
        topo = _make_star_topo(center=10, leaves=[0, 1])
        workflow = _make_workload_with_flows([(0, 1, 1000)])

        table = BfsStrategy().compute_routes(workflow, topo)

        assert (0, 1) in table._paths
        assert table._paths[(0, 1)] == [0, 10, 1]

    def test_multiple_flows_different_src(self):
        """Multiple flows with different sources each get their own path entry."""
        topo = _make_star_topo(center=10, leaves=[0, 1, 2])
        workflow = _make_workload_with_flows([(0, 1, 1000), (2, 1, 2000)])

        table = BfsStrategy().compute_routes(workflow, topo)

        assert (0, 1) in table._paths
        assert table._paths[(0, 1)] == [0, 10, 1]
        assert (2, 1) in table._paths
        assert table._paths[(2, 1)] == [2, 10, 1]

    def test_shared_paths_deduped(self):
        """Flows with same (src, dst) share one path entry."""
        topo = _make_star_topo(center=10, leaves=[0, 1])
        workflow = _make_workload_with_flows([(0, 1, 1000), (0, 1, 2000)])

        table = BfsStrategy().compute_routes(workflow, topo)

        # Only one entry for (0, 1)
        assert (0, 1) in table._paths
        assert table._paths[(0, 1)] == [0, 10, 1]
        assert len(table._paths) == 1

    def test_skips_compute_tasks(self):
        """Compute tasks are ignored."""
        topo = _make_linear_topo(3)
        workflow = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            tasks=[
                Task(task_id=0, job_id=0, type=TaskType.COMPUTE, node=0, duration_us=100),
                Task(task_id=1, job_id=0, type=TaskType.FLOW, src=0, dst=2, size_bytes=1000),
            ],
        )

        table = BfsStrategy().compute_routes(workflow, topo)

        # Only the flow task contributes a path
        assert len(table._paths) == 1
        assert (0, 2) in table._paths

    def test_skips_flow_without_src_dst(self):
        """Flow tasks with None src/dst are skipped."""
        topo = _make_linear_topo(3)
        workflow = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            tasks=[
                Task(task_id=0, job_id=0, type=TaskType.FLOW, dst=2, size_bytes=1000),
            ],
        )

        table = BfsStrategy().compute_routes(workflow, topo)
        assert table._paths == {}

    def test_empty_workload(self):
        """Empty workload produces empty route table."""
        topo = _make_linear_topo(3)
        workflow = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=3))

        table = BfsStrategy().compute_routes(workflow, topo)
        assert table._paths == {}


# ============================================================
# Integration test with real topology
# ============================================================


class TestIntegrationSpectrumX:
    """Integration tests using real Spectrum-X topology.

    Note: Spectrum-X topology file only defines unidirectional links
    (GPU -> NVSwitch, GPU -> ASW, ASW -> PSW). There are no reverse links.
    So BFS routing only works in the forward direction.
    """

    @pytest.fixture
    def topo(self):
        loader = TopologyLoader()
        return loader.load(SPECTRUM_X_TOPO)

    def test_gpu_to_asw_path(self, topo):
        """GPU 0 -> ASW 9: direct link (1 hop)."""
        table = BfsRouteTable(topo)
        table.ensure_path(0, 9, bfs_shortest_path(topo, 0, 9))
        path = table.get_path_by_endpoints(0, 9)
        assert path == [0, 9]

    def test_gpu_to_psw_path(self, topo):
        """GPU 0 -> PSW 17: through ASW (2 hops)."""
        table = BfsRouteTable(topo)
        table.ensure_path(0, 17, bfs_shortest_path(topo, 0, 17))
        path = table.get_path_by_endpoints(0, 17)
        assert path == [0, 9, 17]

    def test_gpu_to_nvswitch(self, topo):
        """GPU 0 -> NV Switch 8: direct link."""
        table = BfsRouteTable(topo)
        table.ensure_path(0, 8, bfs_shortest_path(topo, 0, 8))
        path = table.get_path_by_endpoints(0, 8)
        assert path == [0, 8]

    def test_reverse_path_exists(self, topo):
        """Reverse path from PSW 17 to ASW 9 exists (bidirectional links)."""
        table = BfsRouteTable(topo)
        table.ensure_path(17, 9, bfs_shortest_path(topo, 17, 9))
        path = table.get_path_by_endpoints(17, 9)
        assert path == [17, 9]

    def test_strategy_computes_routes(self, topo):
        """BfsStrategy computes routes for a Spectrum-X workload."""
        workflow = _make_workload_with_flows([
            (0, 9, 1000),   # Direct link exists
            (0, 17, 1000),  # Two-hop path exists
            (4, 13, 1000),  # Direct link exists
        ])

        table = BfsStrategy().compute_routes(workflow, topo)

        assert (0, 9) in table._paths
        assert table._paths[(0, 9)] == [0, 9]
        assert (0, 17) in table._paths
        assert table._paths[(0, 17)] == [0, 9, 17]
        assert (4, 13) in table._paths
        assert table._paths[(4, 13)] == [4, 13]

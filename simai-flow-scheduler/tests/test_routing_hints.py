"""
Tests for routing hints module.

Tests BFS shortest path computation, path caching, link load aggregation,
and integration with P2PWorkload + NetworkTopology.
"""

import os

import pytest

from src.scheduler.routing_hints import RoutingHints, compute_routing_hints, bfs_shortest_path
from src.scheduler.topology_loader import Link, NetworkTopology, TopologyLoader
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
    Linear topology: 0 → 1 → 2 → ... → n-1.
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
        """Two hops: 0 → 1 → 2."""
        topo = _make_linear_topo(3)
        result = bfs_shortest_path(topo, 0, 2)
        assert result == [0, 1, 2]

    def test_multi_hop_linear(self):
        """Linear 0 → 1 → 2 → 3 → 4."""
        topo = _make_linear_topo(5)
        result = bfs_shortest_path(topo, 0, 4)
        assert result == [0, 1, 2, 3, 4]

    def test_no_path_returns_none(self):
        """Disconnected nodes return None."""
        topo = NetworkTopology()
        # Only one link: 0→1, no path from 0 to 2
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        result = bfs_shortest_path(topo, 0, 2)
        assert result is None

    def test_star_two_hop(self):
        """Star topology: leaf → center → other leaf."""
        topo = _make_star_topo(center=10, leaves=[0, 1, 2])
        result = bfs_shortest_path(topo, 0, 1)
        assert result == [0, 10, 1]

    def test_chooses_shortest_not_first(self):
        """BFS guarantees shortest path by hop count."""
        topo = NetworkTopology()
        # Direct link 0→2 (1 hop)
        topo.add_link(Link(src=0, dst=2, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        # Also path 0→1→2 (2 hops)
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        topo.add_link(Link(src=1, dst=2, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))

        result = bfs_shortest_path(topo, 0, 2)
        assert result == [0, 2]  # Direct link is shorter


# ============================================================
# Tests for RoutingHints.get_path
# ============================================================


class TestRoutingHintsGetPath:
    """Tests for RoutingHints.get_path with caching."""

    def test_returns_path(self):
        topo = _make_linear_topo(3)
        hints = RoutingHints()
        path = hints.get_path(topo, 0, 2)
        assert path == [0, 1, 2]

    def test_caches_result(self):
        """Second call returns cached result (no re-computation)."""
        topo = _make_linear_topo(3)
        hints = RoutingHints()

        path1 = hints.get_path(topo, 0, 2)
        path2 = hints.get_path(topo, 0, 2)
        assert path1 is path2  # Same list object (cached)
        assert (0, 2) in hints._cached_paths

    def test_no_path_raises_exception(self):
        """No path → raises ValueError."""
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        hints = RoutingHints()
        with pytest.raises(ValueError, match="No path found from node 0 to node 5"):
            hints.get_path(topo, 0, 5)

    def test_same_node_path(self):
        """src == dst returns [src]."""
        topo = _make_linear_topo(3)
        hints = RoutingHints()
        path = hints.get_path(topo, 1, 1)
        assert path == [1]


# ============================================================
# Tests for RoutingHints.get_flow_links
# ============================================================


class TestRoutingHintsGetFlowLinks:
    """Tests for converting flow tasks to physical link tuples."""

    def test_direct_link_flow(self):
        """Flow on direct link → one physical link tuple."""
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.FLOW, src=0, dst=1, size_bytes=1000)

        links = hints.get_flow_links(task, topo)
        assert links == [(0, 1)]

    def test_two_hop_flow(self):
        """Flow through switch → two physical link tuples."""
        topo = _make_star_topo(center=10, leaves=[0, 1])
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.FLOW, src=0, dst=1, size_bytes=1000)

        links = hints.get_flow_links(task, topo)
        assert links == [(0, 10), (10, 1)]

    def test_multi_hop_flow(self):
        """Flow through linear 0→1→2→3."""
        topo = _make_linear_topo(4)
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.FLOW, src=0, dst=3, size_bytes=1000)

        links = hints.get_flow_links(task, topo)
        assert links == [(0, 1), (1, 2), (2, 3)]

    def test_no_path_raises_exception(self):
        """Flow with no path raises ValueError."""
        topo = NetworkTopology()
        # No link between 0 and 1
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.FLOW, src=0, dst=1, size_bytes=1000)

        with pytest.raises(ValueError, match="No path found from node 0 to node 1"):
            hints.get_flow_links(task, topo)

    def test_compute_task_returns_empty(self):
        """Compute task (no src/dst) returns empty list."""
        topo = _make_linear_topo(3)
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.COMPUTE, node=0, duration_us=100)

        links = hints.get_flow_links(task, topo)
        assert links == []

    def test_flow_with_none_src(self):
        """Flow task with src=None returns empty list."""
        topo = _make_linear_topo(3)
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.FLOW, dst=1, size_bytes=1000)

        links = hints.get_flow_links(task, topo)
        assert links == []


# ============================================================
# Tests for RoutingHints.get_most_used_links
# ============================================================


class TestGetMostUsedLinks:
    """Tests for link load ranking."""

    def test_most_used_ordering(self):
        """Links used by more flows appear first."""
        hints = RoutingHints()
        hints.link_loads = {(0, 1): 5, (1, 2): 10, (2, 3): 3}

        result = hints.get_most_used_links(top_k=3)
        assert result == [((1, 2), 10), ((0, 1), 5), ((2, 3), 3)]

    def test_top_k_limits_results(self):
        """top_k parameter limits output length."""
        hints = RoutingHints()
        hints.link_loads = {(i, i + 1): i + 1 for i in range(10)}

        result = hints.get_most_used_links(top_k=3)
        assert len(result) == 3
        assert result[0] == ((9, 10), 10)

    def test_empty_link_loads(self):
        """No loads → empty list."""
        hints = RoutingHints()
        assert hints.get_most_used_links() == []

    def test_default_top_k(self):
        """Default top_k=20 returns all if fewer than 20."""
        hints = RoutingHints()
        hints.link_loads = {(0, 1): 1, (1, 2): 2}

        result = hints.get_most_used_links()
        assert len(result) == 2


# ============================================================
# Tests for compute_routing_hints
# ============================================================


class TestComputeRoutingHints:
    """Tests for the main compute_routing_hints function."""

    def test_single_flow_direct_link(self):
        """Single flow on direct link: one link gets load=1."""
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0))
        workload = _make_workload_with_flows([(0, 1, 1000)])

        hints = compute_routing_hints(topo, workload)

        assert hints.link_loads == {(0, 1): 1}

    def test_single_flow_two_hops(self):
        """Single flow through switch: two links each get load=1."""
        topo = _make_star_topo(center=10, leaves=[0, 1])
        workload = _make_workload_with_flows([(0, 1, 1000)])

        hints = compute_routing_hints(topo, workload)

        assert hints.link_loads == {(0, 10): 1, (10, 1): 1}

    def test_multiple_flows_same_link(self):
        """Multiple flows sharing a link accumulate load count."""
        topo = _make_star_topo(center=10, leaves=[0, 1, 2])
        # Both flows 0→1 and 2→1 go through link (10, 1)
        workload = _make_workload_with_flows([(0, 1, 1000), (2, 1, 2000)])

        hints = compute_routing_hints(topo, workload)

        assert hints.link_loads[(10, 1)] == 2  # Both flows use (10, 1)
        assert hints.link_loads[(0, 10)] == 1
        assert hints.link_loads[(2, 10)] == 1

    def test_computes_shared_paths(self):
        """Flows with same (src, dst) reuse cached path."""
        topo = _make_star_topo(center=10, leaves=[0, 1])
        # Two flows with same src/dst
        workload = _make_workload_with_flows([(0, 1, 1000), (0, 1, 2000)])

        hints = compute_routing_hints(topo, workload)

        # Both flows traverse same path, each link gets load=2
        assert hints.link_loads[(0, 10)] == 2
        assert hints.link_loads[(10, 1)] == 2
        # Path cached once
        assert (0, 1) in hints._cached_paths

    def test_skips_compute_tasks(self):
        """Compute tasks are ignored."""
        topo = _make_linear_topo(3)
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            tasks=[
                Task(task_id=0, job_id=0, type=TaskType.COMPUTE, node=0, duration_us=100),
                Task(task_id=1, job_id=0, type=TaskType.FLOW, src=0, dst=2, size_bytes=1000),
            ],
        )

        hints = compute_routing_hints(topo, workload)

        # Only the flow task contributes to link loads
        assert hints.link_loads == {(0, 1): 1, (1, 2): 1}

    def test_skips_flow_without_src_dst(self):
        """Flow tasks with None src/dst are skipped."""
        topo = _make_linear_topo(3)
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            tasks=[
                Task(task_id=0, job_id=0, type=TaskType.FLOW, dst=2, size_bytes=1000),
            ],
        )

        hints = compute_routing_hints(topo, workload)
        assert hints.link_loads == {}

    def test_empty_workload(self):
        """Empty workload produces empty routing hints."""
        topo = _make_linear_topo(3)
        workload = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=3))

        hints = compute_routing_hints(topo, workload)
        assert hints.link_loads == {}
        assert hints._cached_paths == {}

    def test_most_used_links_after_compute(self):
        """get_most_used_links works after compute_routing_hints."""
        topo = _make_star_topo(center=10, leaves=[0, 1, 2])
        # 3 flows all through center switch
        workload = _make_workload_with_flows([
            (0, 1, 1000),
            (0, 2, 1000),
            (2, 1, 1000),
        ])

        hints = compute_routing_hints(topo, workload)
        top_links = hints.get_most_used_links(top_k=2)

        # (10, 1) used by flows 0→1 and 2→1, load=2
        # (0, 10) used by flows 0→1 and 0→2, load=2
        assert len(top_links) == 2
        assert top_links[0][1] == 2
        assert top_links[1][1] == 2


# ============================================================
# Integration test with real topology
# ============================================================


class TestIntegrationSpectrumX:
    """Integration tests using real Spectrum-X topology.

    Note: Spectrum-X topology file only defines unidirectional links
    (GPU→NVSwitch, GPU→ASW, ASW→PSW). There are no reverse links.
    So BFS routing only works in the forward direction.
    """

    @pytest.fixture
    def topo(self):
        loader = TopologyLoader()
        return loader.load(SPECTRUM_X_TOPO)

    def test_gpu_to_asw_path(self, topo):
        """GPU 0 → ASW 9: direct link (1 hop)."""
        hints = RoutingHints()
        path = hints.get_path(topo, 0, 9)
        assert path == [0, 9]

    def test_gpu_to_psw_path(self, topo):
        """GPU 0 → PSW 17: through ASW (2 hops)."""
        hints = RoutingHints()
        path = hints.get_path(topo, 0, 17)
        assert path == [0, 9, 17]

    def test_gpu_to_nvswitch(self, topo):
        """GPU 0 → NV Switch 8: direct link."""
        hints = RoutingHints()
        path = hints.get_path(topo, 0, 8)
        assert path == [0, 8]

    def test_reverse_path_exists(self, topo):
        """Reverse path from PSW 17 to ASW 9 exists (bidirectional links)."""
        hints = RoutingHints()
        path = hints.get_path(topo, 17, 9)
        assert path == [17, 9]  # Direct reverse link exists

    def test_compute_routing_hints_with_spectrum(self, topo):
        """Routing with unidirectional topology: flows use fallback direct links."""
        # GPU 0→9 and GPU 0→17 have actual paths, GPU 0→1 has no path
        workload = _make_workload_with_flows([
            (0, 9, 1000),   # Direct link exists
            (0, 17, 1000),  # Two-hop path exists
            (4, 13, 1000),  # Direct link exists
        ])

        hints = compute_routing_hints(topo, workload)

        # Flow 0→9 uses path [0, 9], link (0, 9) gets load +1
        # Flow 0→17 uses path [0, 9, 17], links (0, 9) + (9, 17) each get load +1
        # Flow 4→13 uses path [4, 13], link (4, 13) gets load +1
        assert hints.link_loads[(0, 9)] == 2    # Used by flows 0→9 and 0→17
        assert hints.link_loads[(9, 17)] == 1   # Used by flow 0→17
        assert hints.link_loads[(4, 13)] == 1   # Used by flow 4→13


# ============================================================
# Tests for custom routing strategies
# ============================================================


class TestCustomRoutingStrategy:
    """Tests for custom routing strategy support."""

    def test_custom_strategy_called(self):
        """Custom routing strategy is invoked instead of default BFS."""
        topo = _make_linear_topo(3)

        # Custom strategy that always returns a fixed path
        def custom_strategy(topology, src, dst):
            return [src, dst]  # Direct path regardless of topology

        hints = RoutingHints(routing_strategy=custom_strategy)
        path = hints.get_path(topo, 0, 2)

        # Should use custom strategy, not BFS (which would return [0, 1, 2])
        assert path == [0, 2]

    def test_custom_strategy_with_compute_routing_hints(self):
        """compute_routing_hints accepts custom routing strategy."""
        topo = _make_linear_topo(3)
        workload = _make_workload_with_flows([(0, 2, 1000)])

        # Custom strategy that returns direct paths
        def direct_routing(topology, src, dst):
            if src == dst:
                return [src]
            return [src, dst]

        hints = compute_routing_hints(topo, workload, routing_strategy=direct_routing)

        # Path should be direct, not multi-hop
        assert hints.get_path(topo, 0, 2) == [0, 2]
        # Link loads should reflect direct path
        assert hints.link_loads == {(0, 2): 1}

    def test_default_strategy_when_none_provided(self):
        """Default BFS strategy is used when no custom strategy provided."""
        topo = _make_linear_topo(3)

        # No custom strategy provided
        hints = RoutingHints()
        path = hints.get_path(topo, 0, 2)

        # Should use default BFS
        assert path == [0, 1, 2]

    def test_custom_strategy_no_path_raises_exception(self):
        """Custom strategy returning None raises ValueError."""
        topo = _make_linear_topo(3)

        # Custom strategy that always returns None
        def broken_strategy(topology, src, dst):
            return None

        hints = RoutingHints(routing_strategy=broken_strategy)

        with pytest.raises(ValueError, match="No path found from node 0 to node 2"):
            hints.get_path(topo, 0, 2)

    def test_custom_strategy_caching(self):
        """Custom strategy results are cached."""
        topo = _make_linear_topo(3)

        call_count = 0

        def counting_strategy(topology, src, dst):
            nonlocal call_count
            call_count += 1
            if src == dst:
                return [src]
            return [src, dst]

        hints = RoutingHints(routing_strategy=counting_strategy)

        # First call
        path1 = hints.get_path(topo, 0, 2)
        assert call_count == 1

        # Second call should use cache
        path2 = hints.get_path(topo, 0, 2)
        assert call_count == 1  # Not incremented
        assert path1 is path2  # Same cached object


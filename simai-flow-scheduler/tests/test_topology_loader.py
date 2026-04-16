"""
Tests for topology loader.

Uses real topology files from astra-sim-alibabacloud and also
tests with programmatically created temp files for edge cases.
"""

import os
import tempfile

import pytest

from src.scheduler.topology_loader import Link, NetworkTopology, NodeType, TopologyLoader


# Path to real topology files (relative from simai-flow-scheduler/)
SIMAI_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SPECTRUM_X_TOPO = os.path.join(
    SIMAI_ROOT,
    "astra-sim-alibabacloud",
    "inputs",
    "topo",
    "Spectrum-X_8g_8gps_400Gbps_H100",
)
ALIBABA_HPN_TOPO = os.path.join(
    SIMAI_ROOT,
    "astra-sim-alibabacloud",
    "inputs",
    "topo",
    "AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_H100",
)


# --- Helper: create temp topology file ---


def _write_temp_topo(content: str) -> str:
    """Write content to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".topo")
    with os.fdopen(fd, "w") as f:
        f.write(content)
    return path


# Minimal 4-GPU Spectrum-X-like topology for unit tests
MINI_TOPO_CONTENT = """\
10 4 1 5 12 H100
4 5 6 7 8 9
0 4 2880Gbps 0.000025ms 0
0 5 400Gbps 0.0005ms 0
1 4 2880Gbps 0.000025ms 0
1 6 400Gbps 0.0005ms 0
2 4 2880Gbps 0.000025ms 0
2 7 400Gbps 0.0005ms 0
3 4 2880Gbps 0.000025ms 0
3 8 400Gbps 0.0005ms 0
5 9 400Gbps 0.0005ms 0
6 9 400Gbps 0.0005ms 0
7 9 400Gbps 0.0005ms 0
8 9 400Gbps 0.0005ms 0
"""


# ============================================================
# Tests with real topology files
# ============================================================


class TestSpectrumXTopology:
    """Tests using the real Spectrum-X topology file."""

    @pytest.fixture
    def topo(self):
        loader = TopologyLoader()
        return loader.load(SPECTRUM_X_TOPO)

    def test_header_metadata(self, topo):
        """Verify header parsing: total_nodes, gpu_count, switch_count."""
        assert topo.gpu_count == 8
        assert topo.total_nodes == 18
        assert topo.switch_count == 10  # 1 NV + 9 other
        assert topo.gpu_type == "H100"

    def test_gpu_nodes(self, topo):
        """Verify GPU nodes are 0..7."""
        assert topo.gpu_nodes == [0, 1, 2, 3, 4, 5, 6, 7]
        assert len(topo.gpu_nodes) == 8

    def test_switch_nodes(self, topo):
        """Verify switch nodes are 8..17."""
        assert topo.switch_nodes == [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        assert len(topo.switch_nodes) == 10

    def test_node_classification(self, topo):
        """Verify GPU vs switch node classification."""
        assert topo.is_gpu_node(0)
        assert topo.is_gpu_node(7)
        assert not topo.is_gpu_node(8)

        assert topo.is_switch_node(8)
        assert topo.is_switch_node(17)
        assert not topo.is_switch_node(0)

    def test_node_type_assignment(self, topo):
        """Verify specific node type assignments."""
        # Node 8 is the NV switch (first in switch list, nv_switch_count=1)
        assert topo.node_types[8] == NodeType.NV_SWITCH
        # Nodes 9-17 are network switches
        assert topo.node_types[9] == NodeType.ASW_SWITCH
        assert topo.node_types[17] == NodeType.ASW_SWITCH

    def test_nvlink_links(self, topo):
        """Verify NVLink links: GPU→NV_switch at 2880Gbps."""
        # GPU 0 → NV switch 8
        link = topo.get_link(0, 8)
        assert link is not None
        assert link.bandwidth_gbps == 2880.0
        assert link.latency_us == pytest.approx(0.025, abs=0.001)

    def test_gpu_to_switch_links(self, topo):
        """Verify GPU→ASW links at 400Gbps."""
        # GPU 0 → ASW 9
        link = topo.get_link(0, 9)
        assert link is not None
        assert link.bandwidth_gbps == 400.0
        assert link.latency_us == pytest.approx(0.5, abs=0.001)

    def test_switch_to_psw_links(self, topo):
        """Verify ASW→PSW links at 400Gbps."""
        # ASW 9 → PSW 17
        link = topo.get_link(9, 17)
        assert link is not None
        assert link.bandwidth_gbps == 400.0

    def test_total_link_count(self, topo):
        """Verify total number of parsed links (bidirectional: each line = 2 links)."""
        # 8 NVLinks + 8 GPU→ASW + 8 ASW→PSW = 24 lines × 2 directions = 48
        assert len(topo.links) == 48

    def test_adjacency_gpu_outgoing(self, topo):
        """Each GPU has 2 outgoing links (NV + ASW)."""
        for gpu_id in range(8):
            neighbors = topo.get_neighbors(gpu_id)
            assert len(neighbors) == 2, f"GPU {gpu_id} should have 2 outgoing links"

    def test_adjacency_switch_outgoing(self, topo):
        """ASW switches (9-16) each have 2 outgoing: to GPU (reverse) + to PSW."""
        for sw_id in range(9, 17):
            neighbors = topo.get_neighbors(sw_id)
            assert len(neighbors) == 2, f"ASW {sw_id} should have 2 outgoing links"
            neighbor_ids = [n[0] for n in neighbors]
            assert sw_id - 9 in neighbor_ids  # Reverse link to GPU
            assert 17 in neighbor_ids  # Link to PSW

    def test_bidirectional_links(self, topo):
        """Links are bidirectional: both directions exist."""
        # NV switch → GPU (reverse of GPU→NV)
        assert topo.get_link(8, 0) is not None
        assert topo.get_link(8, 0).bandwidth_gbps == 2880.0
        # PSW → ASW (reverse of ASW→PSW)
        assert topo.get_link(17, 9) is not None
        assert topo.get_link(17, 9).bandwidth_gbps == 400.0
        # ASW → GPU (reverse of GPU→ASW)
        assert topo.get_link(9, 0) is not None
        assert topo.get_link(9, 0).bandwidth_gbps == 400.0

    def test_nonexistent_link(self, topo):
        """Lookup for non-existent link returns None."""
        assert topo.get_link(0, 17) is None  # GPU→PSW: no direct link
        assert topo.get_link(100, 200) is None  # Completely invalid


@pytest.mark.skipif(
    not os.path.exists(ALIBABA_HPN_TOPO),
    reason="AlibabaHPN topology file not available",
)
class TestAlibabaHPNTopology:
    """Tests using the AlibabaHPN topology file."""

    @pytest.fixture
    def topo(self):
        loader = TopologyLoader()
        return loader.load(ALIBABA_HPN_TOPO)

    def test_header_metadata(self, topo):
        """Verify AlibabaHPN header: 16 GPUs (inferred from total - switches), dual plane."""
        assert topo.total_nodes == 38
        assert topo.gpu_count == 16  # 38 - 22 switches
        assert topo.switch_count == 22  # 2 NV + 20 other

    def test_gpu_nodes_count(self, topo):
        """Verify 16 GPU nodes (0-15)."""
        assert len(topo.gpu_nodes) == 16
        assert topo.gpu_nodes == list(range(16))

    def test_link_count_nonzero(self, topo):
        """Verify links were parsed."""
        assert len(topo.links) > 0


# ============================================================
# Tests with mini topology (programmatic)
# ============================================================


class TestMiniTopology:
    """Tests using a small 4-GPU topology for detailed validation."""

    @pytest.fixture
    def topo_path(self):
        path = _write_temp_topo(MINI_TOPO_CONTENT)
        yield path
        os.unlink(path)

    @pytest.fixture
    def topo(self, topo_path):
        loader = TopologyLoader()
        return loader.load(topo_path)

    def test_header(self, topo):
        """Verify header: 10 total, 4 GPUs, 5 switches (1 NV + 4 other), 12 links."""
        assert topo.total_nodes == 10
        assert topo.gpu_count == 4
        assert topo.switch_count == 6  # 1 NV + 5 other
        assert topo.gpu_type == "H100"

    def test_gpu_and_switch_nodes(self, topo):
        """Verify node classification."""
        assert topo.gpu_nodes == [0, 1, 2, 3]
        assert topo.switch_nodes == [4, 5, 6, 7, 8, 9]

    def test_link_count(self, topo):
        """Verify exactly 12 links parsed (bidirectional: 12 lines × 2 = 24)."""
        assert len(topo.links) == 24

    def test_nvlink_bandwidth(self, topo):
        """NVLink links at 2880Gbps."""
        link = topo.get_link(0, 4)
        assert link is not None
        assert link.bandwidth_gbps == 2880.0

    def test_gpu_to_asw_bandwidth(self, topo):
        """GPU→ASW links at 400Gbps."""
        link = topo.get_link(0, 5)
        assert link is not None
        assert link.bandwidth_gbps == 400.0

    def test_asw_to_psw_bandwidth(self, topo):
        """ASW→PSW links at 400Gbps."""
        link = topo.get_link(5, 9)
        assert link is not None
        assert link.bandwidth_gbps == 400.0

    def test_adjacency_complete(self, topo):
        """All nodes appear in adjacency (bidirectional: PSW has reverse links)."""
        # PSW node 9 now has outgoing reverse links back to ASWs
        assert 9 in topo.adjacency
        neighbors = topo.get_neighbors(9)
        assert len(neighbors) == 4  # Reverse links to ASW 5, 6, 7, 8
        neighbor_ids = sorted([n[0] for n in neighbors])
        assert neighbor_ids == [5, 6, 7, 8]

    def test_neighbors_gpu(self, topo):
        """Each GPU has 2 neighbors (NV switch + ASW)."""
        neighbors = topo.get_neighbors(0)
        neighbor_ids = [n[0] for n in neighbors]
        assert sorted(neighbor_ids) == [4, 5]


# ============================================================
# Unit tests for data structures
# ============================================================


class TestLinkDataclass:
    """Tests for Link dataclass."""

    def test_link_id_property(self):
        link = Link(src=0, dst=8, bandwidth_gbps=2880.0, latency_us=0.025, error_rate=0)
        assert link.link_id == (0, 8)

    def test_link_hash_and_equality(self):
        link1 = Link(src=0, dst=8, bandwidth_gbps=2880.0, latency_us=0.025, error_rate=0)
        link2 = Link(src=0, dst=8, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0)
        # Equality based on (src, dst) only
        assert link1 == link2
        assert hash(link1) == hash(link2)

    def test_link_inequality(self):
        link1 = Link(src=0, dst=8, bandwidth_gbps=2880.0, latency_us=0.025, error_rate=0)
        link2 = Link(src=8, dst=0, bandwidth_gbps=2880.0, latency_us=0.025, error_rate=0)
        assert link1 != link2

    def test_link_not_equal_to_non_link(self):
        link = Link(src=0, dst=8, bandwidth_gbps=2880.0, latency_us=0.025, error_rate=0)
        assert link != "not a link"


class TestNetworkTopology:
    """Tests for NetworkTopology methods."""

    def test_add_link(self):
        topo = NetworkTopology()
        link = Link(src=0, dst=1, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0)
        topo.add_link(link)

        assert topo.get_link(0, 1) is link
        assert len(topo.links) == 1

    def test_add_link_updates_adjacency(self):
        topo = NetworkTopology()
        link = Link(src=0, dst=1, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0)
        topo.add_link(link)

        neighbors = topo.get_neighbors(0)
        assert len(neighbors) == 1
        assert neighbors[0][0] == 1
        assert neighbors[0][1] is link

    def test_add_link_ensures_dst_in_adjacency(self):
        topo = NetworkTopology()
        link = Link(src=0, dst=1, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0)
        topo.add_link(link)

        # Node 1 should be in adjacency even though it has no outgoing links
        assert 1 in topo.adjacency
        assert topo.adjacency[1] == []

    def test_get_link_not_found(self):
        topo = NetworkTopology()
        assert topo.get_link(0, 1) is None

    def test_get_neighbors_empty(self):
        topo = NetworkTopology()
        assert topo.get_neighbors(0) == []

    def test_is_gpu_node(self):
        topo = NetworkTopology()
        topo.gpu_nodes = [0, 1, 2]
        assert topo.is_gpu_node(0)
        assert topo.is_gpu_node(2)
        assert not topo.is_gpu_node(3)

    def test_is_switch_node(self):
        topo = NetworkTopology()
        topo.switch_nodes = [8, 9, 10]
        assert topo.is_switch_node(8)
        assert topo.is_switch_node(10)
        assert not topo.is_switch_node(0)

    def test_multiple_links_same_src(self):
        topo = NetworkTopology()
        topo.add_link(Link(src=0, dst=8, bandwidth_gbps=2880.0, latency_us=0.025, error_rate=0))
        topo.add_link(Link(src=0, dst=9, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))

        neighbors = topo.get_neighbors(0)
        assert len(neighbors) == 2
        neighbor_ids = sorted([n[0] for n in neighbors])
        assert neighbor_ids == [8, 9]


# ============================================================
# Tests for parsing edge cases
# ============================================================


class TestParsingEdgeCases:
    """Tests for bandwidth/latency parsing and error handling."""

    @pytest.fixture
    def loader(self):
        return TopologyLoader()

    def test_parse_bandwidth_gbps(self, loader):
        assert loader._parse_bandwidth("400Gbps") == 400.0
        assert loader._parse_bandwidth("2880Gbps") == 2880.0

    def test_parse_bandwidth_mbps(self, loader):
        assert loader._parse_bandwidth("10000Mbps") == 10.0

    def test_parse_bandwidth_tbps(self, loader):
        assert loader._parse_bandwidth("1Tbps") == 1000.0

    def test_parse_bandwidth_case_insensitive(self, loader):
        assert loader._parse_bandwidth("400gbps") == 400.0
        assert loader._parse_bandwidth("400GBPS") == 400.0

    def test_parse_bandwidth_invalid(self, loader):
        with pytest.raises(ValueError, match="Invalid bandwidth format"):
            loader._parse_bandwidth("invalid")

    def test_parse_latency_ms(self, loader):
        assert loader._parse_latency("0.0005ms") == pytest.approx(0.5, abs=0.001)
        assert loader._parse_latency("0.000025ms") == pytest.approx(0.025, abs=0.001)

    def test_parse_latency_us(self, loader):
        assert loader._parse_latency("25us") == 25.0

    def test_parse_latency_ns(self, loader):
        assert loader._parse_latency("500ns") == pytest.approx(0.5, abs=0.001)

    def test_parse_latency_s(self, loader):
        assert loader._parse_latency("0.001s") == pytest.approx(1000.0, abs=0.01)

    def test_parse_latency_invalid(self, loader):
        with pytest.raises(ValueError, match="Invalid latency format"):
            loader._parse_latency("invalid")

    def test_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load("/nonexistent/path/topology")

    def test_invalid_header_too_few_fields(self):
        path = _write_temp_topo("18 8 1\n8 9\n")
        try:
            loader = TopologyLoader()
            with pytest.raises(ValueError, match="Invalid header line"):
                loader.load(path)
        finally:
            os.unlink(path)

    def test_empty_file(self):
        path = _write_temp_topo("")
        try:
            loader = TopologyLoader()
            with pytest.raises(ValueError, match="missing header lines"):
                loader.load(path)
        finally:
            os.unlink(path)

    def test_header_only_file(self):
        path = _write_temp_topo("18 8 1 9 24 H100\n")
        try:
            loader = TopologyLoader()
            with pytest.raises(ValueError, match="missing header lines"):
                loader.load(path)
        finally:
            os.unlink(path)

    def test_links_only_header_and_switches(self):
        """File with header + switch IDs but no links -> empty topology."""
        path = _write_temp_topo("10 4 1 5 0 H100\n4 5 6 7 8 9\n")
        try:
            loader = TopologyLoader()
            topo = loader.load(path)
            assert topo.gpu_count == 4
            assert len(topo.links) == 0
        finally:
            os.unlink(path)

    def test_invalid_link_line_skipped(self):
        """Lines with fewer than 5 fields are silently skipped."""
        content = "10 4 1 4 1 H100\n4 5 6 7 8 9\n0 8 400Gbps\n"
        path = _write_temp_topo(content)
        try:
            loader = TopologyLoader()
            topo = loader.load(path)
            assert len(topo.links) == 0  # Invalid line skipped
        finally:
            os.unlink(path)

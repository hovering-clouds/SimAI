"""
Tests for AICB Workload Parser.

Covers:
1. Header parsing (HYBRID_TRANSFORMER_FWD_IN_BCKWD format)
2. Work item parsing (12-field tab/whitespace-separated lines)
3. parse_comm_type static method
4. Full file parsing (HYBRID and MICRO formats)
5. Real example file integration tests
"""

import pytest
from pathlib import Path

from src.workload_generator.aicb_parser import AicbParser, AicbHeader, AicbWorkItem


# ---------------------------------------------------------------------------
# Helper: create temp AICB files
# ---------------------------------------------------------------------------

HYBRID_HEADER_FULL = (
    "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
    "model_parallel_NPU_group: 2 ep: 16 pp: 12 vpp: 8 ga: 24 "
    "all_gpus: 9216 checkpoints: 0 checkpoint_initiates: 0 pp_comm 50331648"
)

HYBRID_HEADER_NO_PP_COMM = (
    "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
    "model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 8 ga: 1 "
    "all_gpus: 8 checkpoints: 0 checkpoint_initiates: 0"
)

# A few representative items from workload_analytical.txt
SAMPLE_ITEMS = """\
grad_gather\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tALLGATHER\t2807758848\t100
grad_param_comm\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tREDUCESCATTER\t5615517696\t100
embedding_layer\t-1\t622731\tALLREDUCE\t50331648\t1\tNONE\t0\t15091072\tNONE\t0\t100
attention_column\t-1\t1750840\tALLGATHER\t50331648\t875420\tREDUCESCATTER\t0\t875420\tNONE\t0\t100
"""


def _write_aicb_file(tmp_path: Path, header: str, items: str) -> Path:
    """Write a minimal AICB workload file and return its path."""
    count = len([l for l in items.strip().split("\n") if l.strip()])
    content = f"{header}\n{count}\n{items}"
    p = tmp_path / "workload.txt"
    p.write_text(content)
    return p


def _write_micro_file(tmp_path: Path, items: str) -> Path:
    """Write a minimal MICRO benchmark file and return its path."""
    count = len([l for l in items.strip().split("\n") if l.strip()])
    content = f"MICRO\n{count}\n{items}"
    p = tmp_path / "micro.txt"
    p.write_text(content)
    return p


# ===========================================================================
# TestParseCommType
# ===========================================================================

class TestParseCommType:
    """Tests for AicbParser.parse_comm_type static method."""

    def test_none(self):
        assert AicbParser.parse_comm_type("NONE") == ("NONE", "")

    def test_empty_string(self):
        assert AicbParser.parse_comm_type("") == ("NONE", "")

    def test_allreduce_no_suffix_is_tp(self):
        assert AicbParser.parse_comm_type("ALLREDUCE") == ("ALLREDUCE", "tp")

    def test_allgather_no_suffix_is_tp(self):
        assert AicbParser.parse_comm_type("ALLGATHER") == ("ALLGATHER", "tp")

    def test_reducescatter_no_suffix_is_tp(self):
        assert AicbParser.parse_comm_type("REDUCESCATTER") == ("REDUCESCATTER", "tp")

    def test_alltoall_no_suffix_is_tp(self):
        """ALLTOALL without suffix defaults to TP per astra-sim Workload.cc."""
        assert AicbParser.parse_comm_type("ALLTOALL") == ("ALLTOALL", "tp")

    def test_allgather_dp_ep(self):
        assert AicbParser.parse_comm_type("ALLGATHER_DP_EP") == ("ALLGATHER", "dp_ep")

    def test_reducescatter_dp_ep(self):
        assert AicbParser.parse_comm_type("REDUCESCATTER_DP_EP") == ("REDUCESCATTER", "dp_ep")

    def test_alltoall_ep(self):
        assert AicbParser.parse_comm_type("ALLTOALL_EP") == ("ALLTOALL", "ep")

    def test_allgather_dp(self):
        assert AicbParser.parse_comm_type("ALLGATHER_DP") == ("ALLGATHER", "dp")

    def test_reducescatter_dp(self):
        assert AicbParser.parse_comm_type("REDUCESCATTER_DP") == ("REDUCESCATTER", "dp")


# ===========================================================================
# TestParseHeader
# ===========================================================================

class TestParseHeader:
    """Tests for header line parsing."""

    def test_full_header_with_pp_comm(self, tmp_path):
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, "embedding_layer\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tNONE\t0\t100\n")
        parser = AicbParser()
        header, _ = parser.parse(str(p))
        assert header.tp == 2
        assert header.ep == 16
        assert header.pp == 12
        assert header.vpp == 8
        assert header.ga == 24
        assert header.all_gpus == 9216
        assert header.pp_comm_size == 50331648

    def test_header_without_pp_comm(self, tmp_path):
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_NO_PP_COMM, "embedding_layer\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tNONE\t0\t100\n")
        parser = AicbParser()
        header, _ = parser.parse(str(p))
        assert header.tp == 8
        assert header.ep == 1
        assert header.pp == 1
        assert header.vpp == 8
        assert header.ga == 1
        assert header.all_gpus == 8
        assert header.pp_comm_size == 0  # absent → 0

    def test_header_field_types(self, tmp_path):
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, "embedding_layer\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tNONE\t0\t100\n")
        parser = AicbParser()
        header, _ = parser.parse(str(p))
        assert isinstance(header, AicbHeader)
        for field_name in ("tp", "ep", "pp", "vpp", "ga", "all_gpus", "pp_comm_size"):
            assert isinstance(getattr(header, field_name), int), f"{field_name} should be int"


# ===========================================================================
# TestParseItems
# ===========================================================================

class TestParseItems:
    """Tests for individual work item parsing."""

    def test_parse_items_count(self, tmp_path):
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, SAMPLE_ITEMS)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        assert len(items) == 4

    def test_grad_gather_item(self, tmp_path):
        """grad_gather has DP ALLGATHER."""
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, SAMPLE_ITEMS)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        item = items[0]
        assert item.name == "grad_gather"
        assert item.forward_compute_time == 1
        assert item.forward_comm == "NONE"
        assert item.forward_comm_size == 0
        assert item.backward_compute_time == 1
        assert item.backward_comm == "NONE"
        assert item.backward_comm_size == 0
        assert item.dp_compute_time == 1
        assert item.dp_comm == "ALLGATHER"
        assert item.dp_comm_size == 2807758848

    def test_embedding_layer_item(self, tmp_path):
        """embedding_layer has forward ALLREDUCE."""
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, SAMPLE_ITEMS)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        item = items[2]
        assert item.name == "embedding_layer"
        assert item.forward_compute_time == 622731
        assert item.forward_comm == "ALLREDUCE"
        assert item.forward_comm_size == 50331648
        assert item.backward_compute_time == 1
        assert item.backward_comm == "NONE"
        assert item.dp_compute_time == 15091072
        assert item.dp_comm == "NONE"

    def test_attention_column_item(self, tmp_path):
        """attention_column has forward ALLGATHER and backward REDUCESCATTER."""
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, SAMPLE_ITEMS)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        item = items[3]
        assert item.name == "attention_column"
        assert item.forward_comm == "ALLGATHER"
        assert item.forward_comm_size == 50331648
        assert item.backward_comm == "REDUCESCATTER"
        assert item.backward_comm_size == 0

    def test_item_field_types(self, tmp_path):
        p = _write_aicb_file(tmp_path, HYBRID_HEADER_FULL, SAMPLE_ITEMS)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        item = items[0]
        assert isinstance(item, AicbWorkItem)
        assert isinstance(item.forward_compute_time, int)
        assert isinstance(item.forward_comm, str)
        assert isinstance(item.forward_comm_size, int)

    def test_whitespace_separated_items(self, tmp_path):
        """Items can be separated by spaces (not just tabs), matching C++ >> behavior."""
        content = (
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            "model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 8 ga: 1 "
            "all_gpus: 8 checkpoints: 0 checkpoint_initiates: 0\n"
            "1\n"
            "embedding_layer     -1 556000  ALLREDUCE   16777216      1       NONE 0        1      NONE   0      1\n"
        )
        p = tmp_path / "ws_workload.txt"
        p.write_text(content)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        assert len(items) == 1
        assert items[0].name == "embedding_layer"
        assert items[0].forward_compute_time == 556000
        assert items[0].forward_comm == "ALLREDUCE"
        assert items[0].forward_comm_size == 16777216

    def test_item_with_insufficient_fields_skipped(self, tmp_path):
        """Lines with fewer than 12 fields should be skipped."""
        content = (
            "HYBRID_TRANSFORMER_FWD_IN_BCKWD "
            "model_parallel_NPU_group: 8 ep: 1 pp: 1 vpp: 8 ga: 1 "
            "all_gpus: 8 checkpoints: 0 checkpoint_initiates: 0\n"
            "2\n"
            "embedding_layer\t-1\t556000\tALLREDUCE\t16777216\t1\tNONE\t0\t1\tNONE\t0\t1\n"
            "bad_line\t-1\t100\tNONE\n"  # only 4 fields
        )
        p = tmp_path / "short_line.txt"
        p.write_text(content)
        parser = AicbParser()
        _, items = parser.parse(str(p))
        assert len(items) == 1  # only the valid line


# ===========================================================================
# TestParseMicro
# ===========================================================================

class TestParseMicro:
    """Tests for MICRO format parsing."""

    MICRO_ITEMS = """\
micro_test\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tALLREDUCE\t4096\t1
micro_test\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tALLREDUCE\t8192\t1
micro_test\t-1\t1\tNONE\t0\t1\tNONE\t0\t1\tALLREDUCE\t16777216\t1
"""

    def test_micro_returns_items_only(self, tmp_path):
        p = _write_micro_file(tmp_path, self.MICRO_ITEMS)
        parser = AicbParser()
        items = parser.parse_micro(str(p))
        assert isinstance(items, list)
        assert len(items) == 3

    def test_micro_item_fields(self, tmp_path):
        p = _write_micro_file(tmp_path, self.MICRO_ITEMS)
        parser = AicbParser()
        items = parser.parse_micro(str(p))
        item = items[0]
        assert item.name == "micro_test"
        assert item.dp_comm == "ALLREDUCE"
        assert item.dp_comm_size == 4096

    def test_micro_last_item(self, tmp_path):
        p = _write_micro_file(tmp_path, self.MICRO_ITEMS)
        parser = AicbParser()
        items = parser.parse_micro(str(p))
        assert items[2].dp_comm_size == 16777216


# ===========================================================================
# TestRealFiles — integration tests using actual AICB example files
# ===========================================================================

class TestRealFiles:
    """Integration tests with real AICB workload files from the SimAI repo."""

    SIMAI_ROOT = Path(__file__).resolve().parents[2]  # simai-flow-scheduler/tests → SimAI/

    @pytest.mark.skipif(
        not (SIMAI_ROOT / "example/microAllReduce.txt").exists(),
        reason="Real example files not available",
    )
    def test_parse_micro_all_reduce(self):
        """Parse example/microAllReduce.txt — 2 embedding_layer items."""
        f = self.SIMAI_ROOT / "example/microAllReduce.txt"
        parser = AicbParser()
        header, items = parser.parse(str(f))
        assert header.tp == 8
        assert header.ep == 1
        assert header.pp == 1
        assert header.ga == 1
        assert header.all_gpus == 8
        assert len(items) == 2
        assert items[0].name == "embedding_layer"
        assert items[0].forward_comm == "ALLREDUCE"
        assert items[0].forward_comm_size == 16777216
        assert items[1].forward_comm_size == 67108864

    @pytest.mark.skipif(
        not (SIMAI_ROOT / "example/workload_analytical.txt").exists(),
        reason="Real example files not available",
    )
    def test_parse_workload_analytical_header(self):
        """Parse example/workload_analytical.txt header."""
        f = self.SIMAI_ROOT / "example/workload_analytical.txt"
        parser = AicbParser()
        header, items = parser.parse(str(f))
        assert header.tp == 2
        assert header.ep == 16
        assert header.pp == 12
        assert header.vpp == 8
        assert header.ga == 24
        assert header.all_gpus == 9216
        assert header.pp_comm_size == 50331648
        assert len(items) == 1789

    @pytest.mark.skipif(
        not (SIMAI_ROOT / "example/workload_analytical.txt").exists(),
        reason="Real example files not available",
    )
    def test_parse_workload_analytical_first_items(self):
        """Verify first items of workload_analytical.txt.

        Item order from actual file:
          0: grad_gather, 1: grad_param_comm, 2: grad_param_compute,
          3: embedding_grads, 4: moe_grad_norm1, 5: moe_grad_norm2,
          6: embedding_layer, 7: attention_column
        """
        f = self.SIMAI_ROOT / "example/workload_analytical.txt"
        parser = AicbParser()
        _, items = parser.parse(str(f))

        # Item 0: grad_gather — DP ALLGATHER
        assert items[0].name == "grad_gather"
        assert items[0].dp_comm == "ALLGATHER"
        assert items[0].dp_comm_size == 2807758848

        # Item 1: grad_param_comm — DP REDUCESCATTER
        assert items[1].name == "grad_param_comm"
        assert items[1].dp_comm == "REDUCESCATTER"

        # Item 3: embedding_grads — backward ALLREDUCE
        assert items[3].name == "embedding_grads"
        assert items[3].backward_comm == "ALLREDUCE"
        assert items[3].backward_comm_size == 50331648

        # Item 4: moe_grad_norm1 — DP_EP ALLGATHER
        assert items[4].name == "moe_grad_norm1"
        assert items[4].dp_comm == "ALLGATHER_DP_EP"

        # Item 5: moe_grad_norm2 — DP_EP REDUCESCATTER
        assert items[5].name == "moe_grad_norm2"
        assert items[5].dp_comm == "REDUCESCATTER_DP_EP"

        # Item 6: embedding_layer — forward ALLREDUCE
        assert items[6].name == "embedding_layer"
        assert items[6].forward_comm == "ALLREDUCE"
        assert items[6].forward_comm_size == 50331648

        # Item 7: attention_column — forward ALLGATHER, backward REDUCESCATTER
        assert items[7].name == "attention_column"
        assert items[7].forward_comm == "ALLGATHER"
        assert items[7].backward_comm == "REDUCESCATTER"

    @pytest.mark.skipif(
        not (SIMAI_ROOT / "aicb/workload/simAI/micro_test/all_reduce.txt").exists(),
        reason="Real micro test files not available",
    )
    def test_parse_micro_all_reduce_file(self):
        """Parse aicb micro_test/all_reduce.txt — MICRO format."""
        f = self.SIMAI_ROOT / "aicb/workload/simAI/micro_test/all_reduce.txt"
        parser = AicbParser()
        items = parser.parse_micro(str(f))
        assert len(items) == 19
        assert items[0].dp_comm == "ALLREDUCE"
        assert items[0].dp_comm_size == 4096
        # Last item should have the largest size
        assert items[-1].dp_comm_size == 1073741824

    @pytest.mark.skipif(
        not (SIMAI_ROOT / "aicb/workload/simAI/micro_test/all_gather.txt").exists(),
        reason="Real micro test files not available",
    )
    def test_parse_micro_all_gather_file(self):
        """Parse aicb micro_test/all_gather.txt — MICRO format."""
        f = self.SIMAI_ROOT / "aicb/workload/simAI/micro_test/all_gather.txt"
        parser = AicbParser()
        items = parser.parse_micro(str(f))
        assert len(items) == 19
        assert items[0].dp_comm == "ALLGATHER"

    @pytest.mark.skipif(
        not (SIMAI_ROOT / "aicb/workload/simAI/micro_test/all_to_all.txt").exists(),
        reason="Real micro test files not available",
    )
    def test_parse_micro_all_to_all_file(self):
        """Parse aicb micro_test/all_to_all.txt — MICRO format."""
        f = self.SIMAI_ROOT / "aicb/workload/simAI/micro_test/all_to_all.txt"
        parser = AicbParser()
        items = parser.parse_micro(str(f))
        assert len(items) == 19
        assert items[0].dp_comm == "ALLTOALL"

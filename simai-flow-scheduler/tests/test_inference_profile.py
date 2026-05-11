"""
Tests for InferenceProfileStore.
"""

import pytest
from src.workload_generator.inference_profile import InferenceProfileStore, LayerProfile


@pytest.fixture
def tsv_content():
    """Sample TSV profiling data."""
    return "layer_id\tlayer_name\tcomp_time\tcomm_size\n" \
           "0\tattention\t1051867\t41943040\n" \
           "0\tmlp\t1037890\t41943040\n" \
           "1\tattention\t1051867\t41943040\n" \
           "1\tmlp\t1037890\t41943040\n" \
           "2\tattention\t1051867\t41943040\n" \
           "2\tmoe\t2000000\t83886080\n"


@pytest.fixture
def tsv_path(tmp_path, tsv_content):
    """Write TSV content to a temp file."""
    f = tmp_path / "profile.tsv"
    f.write_text(tsv_content)
    return str(f)


class TestInferenceProfileStore:

    def test_load_tsv(self, tsv_path):
        """Test loading a TSV profiling file."""
        store = InferenceProfileStore()
        store.load(tsv_path, "prefill_bs4_seq4096")

        profiles = store.get_profile("prefill_bs4_seq4096")
        assert len(profiles) == 6

    def test_comp_time_conversion_ns_to_us(self, tsv_path):
        """Test that comp_time is converted from ns to us."""
        store = InferenceProfileStore()
        store.load(tsv_path, "key")

        profiles = store.get_profile("key")
        # 1051867 ns / 1000 = 1051 us (int truncation)
        assert profiles[0].comp_time_us == 1051
        assert profiles[1].comp_time_us == 1037

    def test_get_layers_by_name(self, tsv_path):
        """Test filtering by layer name."""
        store = InferenceProfileStore()
        store.load(tsv_path, "key")

        att_layers = store.get_layers_by_name("key", "attention")
        assert len(att_layers) == 3
        assert all(p.layer_name == "attention" for p in att_layers)

        moe_layers = store.get_layers_by_name("key", "moe")
        assert len(moe_layers) == 1

    def test_load_multiple_profiles(self, tsv_path):
        """Test loading multiple profiles with different keys."""
        store = InferenceProfileStore()
        store.load(tsv_path, "prefill_bs4_seq4096")
        store.load(tsv_path, "decode_bs4_seq1")

        assert len(store.list_profiles()) == 2
        assert "prefill_bs4_seq4096" in store.list_profiles()
        assert "decode_bs4_seq1" in store.list_profiles()

    def test_duplicate_key_raises(self, tsv_path):
        """Test that loading with an existing key raises KeyError."""
        store = InferenceProfileStore()
        store.load(tsv_path, "key")

        with pytest.raises(KeyError):
            store.load(tsv_path, "key")

    def test_unknown_key_raises(self, tsv_path):
        """Test that querying unknown key raises KeyError with helpful message."""
        store = InferenceProfileStore()
        store.load(tsv_path, "key")

        with pytest.raises(KeyError, match="Available keys"):
            store.get_profile("unknown_key")

    def test_empty_file_raises(self, tmp_path):
        """Test that empty file raises ValueError."""
        f = tmp_path / "empty.tsv"
        f.write_text("")

        store = InferenceProfileStore()
        with pytest.raises(ValueError):
            store.load(str(f), "key")

    def test_missing_columns_raises(self, tmp_path):
        """Test that missing required columns raises ValueError."""
        f = tmp_path / "bad.tsv"
        f.write_text("layer_id\tlayer_name\n" "0\tattention\n")

        store = InferenceProfileStore()
        with pytest.raises(ValueError):
            store.load(str(f), "key")

    def test_csv_comma_separated(self, tmp_path):
        """Test loading a comma-separated CSV (fallback)."""
        f = tmp_path / "profile.csv"
        f.write_text(
            "layer_id,layer_name,comp_time,comm_size\n"
            "0,attention,1051867,41943040\n"
            "1,mlp,1037890,41943040\n"
        )

        store = InferenceProfileStore()
        store.load(str(f), "key")

        profiles = store.get_profile("key")
        assert len(profiles) == 2
        assert profiles[0].layer_name == "attention"

    def test_list_profiles(self, tsv_path):
        """Test listing available profile keys."""
        store = InferenceProfileStore()
        assert store.list_profiles() == []

        store.load(tsv_path, "prefill_bs4_seq4096")
        assert store.list_profiles() == ["prefill_bs4_seq4096"]

    def test_load_directory_new_format(self, tmp_path, tsv_content):
        """Test load_directory with new filename format."""
        # Create files with new format: {phase}_bs{bs}_seq{seq}_tp{tp}_ep{ep}_pp{pp}.csv
        (tmp_path / "prefill_bs4_seq4096_tp2_ep1_pp1.csv").write_text(tsv_content)
        (tmp_path / "decode_bs4_seq1_tp2_ep1_pp1.csv").write_text(tsv_content)
        (tmp_path / "prefill_bs8_seq2048_tp4_ep1_pp1.csv").write_text(tsv_content)

        store = InferenceProfileStore(tp=2, ep=1, pp=1)
        loaded = store.load_directory(str(tmp_path))

        assert loaded == 2  # Only tp=2 files loaded
        assert "prefill_bs4_seq4096" in store.list_profiles()
        assert "decode_bs4_seq1" in store.list_profiles()
        assert "prefill_bs8_seq2048" not in store.list_profiles()

    def test_load_directory_vidur_format(self, tmp_path, tsv_content):
        """Test load_directory with Vidur filename format."""
        # Create files with Vidur format: vidur-{model}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs{bs}-seq{seq}-{phase}.csv
        (tmp_path / "vidur-DeepSeek-671B-world_size1-tp1-pp1-ep1-bs1-seq512-prefill.csv").write_text(tsv_content)
        (tmp_path / "vidur-DeepSeek-671B-world_size1-tp1-pp1-ep1-bs1-seq1-decode.csv").write_text(tsv_content)
        (tmp_path / "vidur-DeepSeek-671B-world_size2-tp2-pp1-ep1-bs4-seq1024-prefill.csv").write_text(tsv_content)

        store = InferenceProfileStore(tp=1, ep=1, pp=1)
        loaded = store.load_directory(str(tmp_path))

        assert loaded == 2  # Only tp=1 files loaded
        assert "prefill_bs1_seq512" in store.list_profiles()
        assert "decode_bs1_seq1" in store.list_profiles()
        assert "prefill_bs4_seq1024" not in store.list_profiles()

    def test_load_directory_mixed_formats(self, tmp_path, tsv_content):
        """Test load_directory with mixed filename formats."""
        (tmp_path / "prefill_bs4_seq4096_tp1_ep1_pp1.csv").write_text(tsv_content)
        (tmp_path / "vidur-Model-world_size1-tp1-pp1-ep1-bs8-seq2048-decode.csv").write_text(tsv_content)
        (tmp_path / "invalid_filename.csv").write_text(tsv_content)

        store = InferenceProfileStore(tp=1, ep=1, pp=1)
        loaded = store.load_directory(str(tmp_path))

        assert loaded == 2  # Only valid files loaded
        assert "prefill_bs4_seq4096" in store.list_profiles()
        assert "decode_bs8_seq2048" in store.list_profiles()

    def test_load_directory_no_filter(self, tmp_path, tsv_content):
        """Test load_directory without parallelism filters."""
        (tmp_path / "prefill_bs4_seq4096_tp2_ep1_pp1.csv").write_text(tsv_content)
        (tmp_path / "decode_bs4_seq1_tp4_ep2_pp1.csv").write_text(tsv_content)

        store = InferenceProfileStore()  # No filters
        loaded = store.load_directory(str(tmp_path))

        assert loaded == 2  # All files loaded
        assert len(store.list_profiles()) == 2

    def test_load_directory_empty(self, tmp_path):
        """Test load_directory with no matching files."""
        store = InferenceProfileStore(tp=1, ep=1, pp=1)
        loaded = store.load_directory(str(tmp_path))

        assert loaded == 0
        assert store.list_profiles() == []

    def test_load_directory_not_a_directory(self, tmp_path):
        """Test load_directory with invalid path."""
        store = InferenceProfileStore()
        with pytest.raises(ValueError, match="Not a directory"):
            store.load_directory(str(tmp_path / "nonexistent"))

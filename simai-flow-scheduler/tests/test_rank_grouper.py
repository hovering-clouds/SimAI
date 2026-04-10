"""
Tests for Rank Grouper.

Covers:
1. Basic TP/DP/PP grouping (no EP)
2. TP/DP/EP/PP four-dimensional grouping
3. Custom rank order verification
4. Edge cases (single GPU per group)
5. Integration with AicbParser.parse_comm_type()
"""

import pytest
from src.workload_generator.rank_grouper import RankGrouper
from src.workload_format.schema import ParallelismConfig


# ===========================================================================
# Helper functions
# ===========================================================================

def make_grouper(ranks: list[int], tp: int = 1, dp: int = 1, pp: int = 1, ep: int = 1) -> RankGrouper:
    """Create a RankGrouper with the given ranks and parallelism config."""
    config = ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep)
    return RankGrouper(assigned_nodes=ranks, parallelism=config)


# ===========================================================================
# TestBasicTPDPPP - Basic 3D grouping without EP
# ===========================================================================

class TestBasicTPDPPP:
    """Test basic TP/DP/PP grouping (EP=1)."""

    def test_single_tp_group(self):
        """8 GPUs, TP=8, DP=1, PP=1: all GPUs in one TP group."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=8, dp=1, pp=1)
        # Only one TP group containing all ranks
        tp_group = grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0)
        assert tp_group == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_two_tp_groups_dp2(self):
        """8 GPUs, TP=4, DP=2, PP=1: two TP groups, each with 4 ranks."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=4, dp=2, pp=1)
        # TP group 0 in DP group 0
        tp_group_0 = grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0)
        assert tp_group_0 == [0, 1, 2, 3]
        # TP group 1 in DP group 1
        tp_group_1 = grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0)
        assert tp_group_1 == [4, 5, 6, 7]

    def test_four_tp_groups_dp4(self):
        """8 GPUs, TP=2, DP=4, PP=1: four TP groups, each with 2 ranks."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=4, pp=1)
        # TP groups: [0,1], [2,3], [4,5], [6,7]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0) == [2, 3]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=2, ep_idx=0) == [4, 5]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=3, ep_idx=0) == [6, 7]

    def test_dp_groups(self):
        """8 GPUs, TP=2, DP=4, PP=1: verify DP groups."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=4, pp=1)
        # DP groups: [0,2,4,6], [1,3,5,7] for each tp_idx
        # With tp_idx=0: ranks at positions 0, 2, 4, 6
        assert grouper.get_dp_group(pp_idx=0, ep_idx=0, tp_idx=0) == [0, 2, 4, 6]
        # With tp_idx=1: ranks at positions 1, 3, 5, 7
        assert grouper.get_dp_group(pp_idx=0, ep_idx=0, tp_idx=1) == [1, 3, 5, 7]

    def test_pp_groups(self):
        """8 GPUs, TP=2, DP=2, PP=2: verify PP groups."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)
        # PP stage 0: ranks [0,1,2,3], PP stage 1: ranks [4,5,6,7]
        # For dp_idx=0, ep_idx=0, tp_idx=0: PP group is [0, 4]
        assert grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=0) == [0, 4]
        # For dp_idx=0, ep_idx=0, tp_idx=1: PP group is [1, 5]
        assert grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=1) == [1, 5]
        # For dp_idx=1, ep_idx=0, tp_idx=0: PP group is [2, 6]
        assert grouper.get_pp_group(dp_idx=1, ep_idx=0, tp_idx=0) == [2, 6]
        # For dp_idx=1, ep_idx=0, tp_idx=1: PP group is [3, 7]
        assert grouper.get_pp_group(dp_idx=1, ep_idx=0, tp_idx=1) == [3, 7]


# ===========================================================================
# TestWithEP - Four-dimensional grouping with EP
# ===========================================================================

class TestWithEP:
    """Test TP/DP/EP/PP four-dimensional grouping."""

    def test_tp_with_ep(self):
        """16 GPUs, TP=2, DP=2, EP=2, PP=2: verify TP groups with EP."""
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=2, pp=2, ep=2)
        # Total = 2*2*2*2 = 16
        # For pp_idx=0, dp_idx=0, ep_idx=0: TP group = [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0, 1]
        # For pp_idx=0, dp_idx=0, ep_idx=1: TP group = [2, 3]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=1) == [2, 3]
        # For pp_idx=0, dp_idx=1, ep_idx=0: TP group = [4, 5]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0) == [4, 5]

    def test_ep_groups(self):
        """16 GPUs, TP=2, DP=2, EP=2, PP=2: verify EP groups."""
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=2, pp=2, ep=2)
        # For pp_idx=0, dp_idx=0, tp_idx=0: EP group = [0, 2]
        assert grouper.get_ep_group(pp_idx=0, dp_idx=0, tp_idx=0) == [0, 2]
        # For pp_idx=0, dp_idx=0, tp_idx=1: EP group = [1, 3]
        assert grouper.get_ep_group(pp_idx=0, dp_idx=0, tp_idx=1) == [1, 3]

    def test_dp_ep_group(self):
        """16 GPUs, TP=2, DP=2, EP=2, PP=2: verify DP×EP combined groups."""
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=2, pp=2, ep=2)
        # For pp_idx=0, tp_idx=0: DP×EP group = [0, 2, 4, 6]
        # (dp_idx=0,ep_idx=0), (dp_idx=0,ep_idx=1), (dp_idx=1,ep_idx=0), (dp_idx=1,ep_idx=1)
        assert grouper.get_dp_ep_group(pp_idx=0, tp_idx=0) == [0, 2, 4, 6]
        # For pp_idx=0, tp_idx=1: DP×EP group = [1, 3, 5, 7]
        assert grouper.get_dp_ep_group(pp_idx=0, tp_idx=1) == [1, 3, 5, 7]


# ===========================================================================
# TestCustomRankOrder - Verify custom rank ordering changes grouping
# ===========================================================================

class TestCustomRankOrder:
    """Test that changing rank order changes grouping results."""

    def test_default_order_tp_groups(self):
        """Default order: [0,1,2,3,4,5,6,7] with TP=2."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)
        # TP groups should be consecutive pairs within each PP/DP combo
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0) == [2, 3]

    def test_reordered_tp_groups(self):
        """Reordered: user wants [0,4] and [1,5] as TP groups.

        To achieve this with [PP][DP][TP] layout:
        PP stage 0, DP 0: [0, 4, ...]
        PP stage 0, DP 1: [1, 5, ...]
        etc.
        """
        # This tests that users can control grouping by reordering ranks
        # The exact reordering depends on the desired mapping
        ranks = [0, 4, 1, 5, 2, 6, 3, 7]
        grouper = make_grouper(ranks, tp=2, dp=2, pp=2)
        # With this ordering, TP group for pp=0, dp=0 should include 0 and 4
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0, 4]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0) == [1, 5]


# ===========================================================================
# TestEdgeCases - Single GPU per group, boundary conditions
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_gpu(self):
        """Single GPU: TP=1, DP=1, PP=1, EP=1."""
        grouper = make_grouper([0])
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0]
        assert grouper.get_dp_group(pp_idx=0, ep_idx=0, tp_idx=0) == [0]
        assert grouper.get_ep_group(pp_idx=0, dp_idx=0, tp_idx=0) == [0]
        assert grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=0) == [0]

    def test_tp1_multiple_dp(self):
        """TP=1 means each GPU is its own TP group."""
        grouper = make_grouper([0, 1, 2, 3], tp=1, dp=4, pp=1)
        # Each rank is its own TP group
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0) == [1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=2, ep_idx=0) == [2]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=3, ep_idx=0) == [3]

    def test_assertion_failure_on_mismatch(self):
        """AssertionError when len(ranks) != tp * dp * ep * pp."""
        with pytest.raises(AssertionError):
            make_grouper([0, 1, 2], tp=2, dp=2, pp=1)  # 3 != 2*2*1*1=4

    def test_pp1_only_one_pp_group(self):
        """PP=1 means only one PP stage."""
        grouper = make_grouper([0, 1, 2, 3], tp=2, dp=2, pp=1)
        # Only pp_idx=0 is valid
        # For dp_idx=0, ep_idx=0, tp_idx=0: PP group has only one element [0]
        assert grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=0) == [0]
        # For dp_idx=0, ep_idx=0, tp_idx=1: PP group is [1]
        assert grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=1) == [1]
        # For dp_idx=1, ep_idx=0, tp_idx=0: PP group is [2]
        assert grouper.get_pp_group(dp_idx=1, ep_idx=0, tp_idx=0) == [2]


# ===========================================================================
# TestIntegration - Integration with parse_comm_type patterns
# ===========================================================================

class TestIntegration:
    """Test integration patterns matching how WorkloadBuilder will use RankGrouper."""

    def test_tp_comm_type_mapping(self):
        """No suffix → TP group mapping."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)
        # ALLREDUCE without suffix → TP group
        tp_ranks = grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0)
        assert tp_ranks == [0, 1]  # First TP group

    def test_dp_ep_comm_type_mapping(self):
        """_DP_EP suffix → DP×EP combined group mapping."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=1, ep=2)
        # ALLGATHER_DP_EP → DP×EP group
        dp_ep_ranks = grouper.get_dp_ep_group(pp_idx=0, tp_idx=0)
        # With tp=2, dp=2, ep=2, pp=1: indices are [pp][dp][ep][tp]
        # pp_idx=0, tp_idx=0: iterate over dp_idx (0,1) and ep_idx (0,1)
        # global_idx = 0*(2*2*2) + dp_idx*(2*2) + ep_idx*2 + 0
        # dp_idx=0, ep_idx=0: 0; dp_idx=0, ep_idx=1: 2; dp_idx=1, ep_idx=0: 4; dp_idx=1, ep_idx=1: 6
        assert dp_ep_ranks == [0, 2, 4, 6]

    def test_full_example_8gpu(self):
        """Full example from design doc: 8 GPUs, TP=2, DP=2, PP=2.

        Layout: [PP=2][DP=2][TP=2]
        - PP stage 0: ranks[0:4] = [0,1,2,3]
        - PP stage 1: ranks[4:8] = [4,5,6,7]

        Within PP stage 0:
        - DP group 0: [0,1], DP group 1: [2,3]
        - TP group 0: [0,2], TP group 1: [1,3]
        """
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)

        # PP stage 0, DP group 0: TP group should be [0, 1]
        # Wait, let me recalculate based on the formula:
        # global_idx = pp_idx*(dp*tp) + dp_idx*tp + tp_idx
        # For pp_idx=0, dp_idx=0: indices are 0*2+0=0, 0*2+1=1 → [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0) == [0, 1]

        # For pp_idx=0, dp_idx=1: indices are 0*4+1*2+0=2, 0*4+1*2+1=3 → [2, 3]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1, ep_idx=0) == [2, 3]

        # For pp_idx=1, dp_idx=0: indices are 1*4+0*2+0=4, 1*4+0*2+1=5 → [4, 5]
        assert grouper.get_tp_group(pp_idx=1, dp_idx=0, ep_idx=0) == [4, 5]

        # DP groups:
        # For pp_idx=0, ep_idx=0, tp_idx=0: dp_idx varies → 0*4+0*2+0=0, 0*4+1*2+0=2 → [0, 2]
        assert grouper.get_dp_group(pp_idx=0, ep_idx=0, tp_idx=0) == [0, 2]
        # For pp_idx=0, ep_idx=0, tp_idx=1: → 0*4+0*2+1=1, 0*4+1*2+1=3 → [1, 3]
        assert grouper.get_dp_group(pp_idx=0, ep_idx=0, tp_idx=1) == [1, 3]

        # PP groups:
        # For dp_idx=0, ep_idx=0, tp_idx=0: pp_idx varies → 0*4+0+0=0, 1*4+0+0=4 → [0, 4]
        assert grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=0) == [0, 4]

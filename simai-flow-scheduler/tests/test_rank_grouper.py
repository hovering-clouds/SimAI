"""
Tests for Rank Grouper (3D [PP][DP][TP] layout).

Covers:
1. Base RankGrouper — shared 3D grouping methods (TP/DP/PP)
2. MegatronRankGrouper — EP as sub-division of DP
3. VllmRankGrouper — EP spanning all non-PP GPUs
4. Custom rank order verification
5. Edge cases (single GPU per group)
6. Integration with AicbParser.parse_comm_type()
"""

import pytest
from src.workload_generator.rank_grouper import (
    RankGrouper,
    MegatronRankGrouper,
    VllmRankGrouper,
)
from src.workload_format.schema import ParallelismConfig


# ===========================================================================
# Helper
# ===========================================================================

def make_grouper(
    ranks: list[int],
    tp: int = 1,
    dp: int = 1,
    pp: int = 1,
    ep: int = 1,
) -> MegatronRankGrouper:
    """Create a MegatronRankGrouper (default EP strategy)."""
    return MegatronRankGrouper(ranks, ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep))


def make_base(ranks, tp=1, dp=1, pp=1):
    """Create a base RankGrouper (no EP methods)."""
    return RankGrouper(ranks, ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=1))


# ===========================================================================
# TestBasicTPDPPP - Basic 3D grouping without EP
# ===========================================================================

class TestBasicTPDPPP:
    """Test basic TP/DP/PP grouping (EP=1, base RankGrouper)."""

    def test_single_tp_group(self):
        """8 GPUs, TP=8, DP=1, PP=1: all GPUs in one TP group."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=8, dp=1, pp=1)
        tp_group = grouper.get_tp_group(pp_idx=0, dp_idx=0)
        assert tp_group == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_two_tp_groups_dp2(self):
        """8 GPUs, TP=4, DP=2, PP=1: two TP groups, each with 4 ranks."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=4, dp=2, pp=1)
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 1, 2, 3]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [4, 5, 6, 7]

    def test_four_tp_groups_dp4(self):
        """8 GPUs, TP=2, DP=4, PP=1: four TP groups, each with 2 ranks."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=4, pp=1)
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [2, 3]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=2) == [4, 5]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=3) == [6, 7]

    def test_dp_groups(self):
        """8 GPUs, TP=2, DP=4, PP=1: verify DP groups (all replicas)."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=4, pp=1)
        assert grouper.get_dp_group(pp_idx=0, tp_idx=0) == [0, 2, 4, 6]
        assert grouper.get_dp_group(pp_idx=0, tp_idx=1) == [1, 3, 5, 7]

    def test_pp_groups(self):
        """8 GPUs, TP=2, DP=2, PP=2: verify PP groups."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)
        assert grouper.get_pp_group(dp_idx=0, tp_idx=0) == [0, 4]
        assert grouper.get_pp_group(dp_idx=0, tp_idx=1) == [1, 5]
        assert grouper.get_pp_group(dp_idx=1, tp_idx=0) == [2, 6]
        assert grouper.get_pp_group(dp_idx=1, tp_idx=1) == [3, 7]


# ===========================================================================
# TestMegatronEP - Megatron-style EP (ep sub-divides dp)
# ===========================================================================

class TestMegatronEP:
    """Test MegatronRankGrouper: EP as sub-division of DP dimension."""

    def test_tp_groups_with_ep(self):
        """16 GPUs, TP=2, DP=4, EP=2, PP=2: verify TP groups with EP."""
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=4, pp=2, ep=2)
        # In 3D layout [PP=2][DP=4][TP=2], dp=4 includes EP factor
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [2, 3]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=2) == [4, 5]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=3) == [6, 7]

    def test_ep_groups(self):
        """16 GPUs, TP=2, DP=4, EP=2, PP=2: verify EP groups.

        EP spans consecutive ep=2 DP replicas.
        dp_ep = dp / ep = 4 / 2 = 2 groups per expert shard.
        """
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=4, pp=2, ep=2)
        # PP=0, dp_ep=0 (DP replicas 0-1), tp=0: ranks at dp=0{0}, dp=1{2} → [0, 2]
        assert grouper.get_ep_group(pp_idx=0, dp_ep_idx=0, tp_idx=0) == [0, 2]
        # PP=0, dp_ep=0 (DP replicas 0-1), tp=1: dp=0{1}, dp=1{3} → [1, 3]
        assert grouper.get_ep_group(pp_idx=0, dp_ep_idx=0, tp_idx=1) == [1, 3]
        # PP=0, dp_ep=1 (DP replicas 2-3), tp=0: dp=2{4}, dp=3{6} → [4, 6]
        assert grouper.get_ep_group(pp_idx=0, dp_ep_idx=1, tp_idx=0) == [4, 6]

    def test_dp_ep_groups(self):
        """DP_EP groups: stride by ep through the DP dimension.

        DP_EP groups ranks that share the same expert shard across
        different DP replica groups (stride by ep).
        """
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=4, pp=2, ep=2)
        # PP=0, ep=0 (same expert shard), tp=0: dp=0{0}, dp=2{4} → [0, 4]
        assert grouper.get_dp_ep_group(pp_idx=0, ep_idx=0, tp_idx=0) == [0, 4]
        # PP=0, ep=0, tp=1: dp=0{1}, dp=2{5} → [1, 5]
        assert grouper.get_dp_ep_group(pp_idx=0, ep_idx=0, tp_idx=1) == [1, 5]
        # PP=0, ep=1 (different expert shard), tp=0: dp=1{2}, dp=3{6} → [2, 6]
        assert grouper.get_dp_ep_group(pp_idx=0, ep_idx=1, tp_idx=0) == [2, 6]
        # PP=1, ep=0, tp=0: dp=0{8}, dp=2{12} → [8, 12]
        assert grouper.get_dp_ep_group(pp_idx=1, ep_idx=0, tp_idx=0) == [8, 12]

    def test_dp_groups_with_ep(self):
        """DP (all replicas) when EP > 1 — returns all dp_total ranks."""
        ranks = list(range(16))
        grouper = make_grouper(ranks, tp=2, dp=4, pp=2, ep=2)
        # PP=0, tp=0: all dp=0,1,2,3 → [0, 2, 4, 6]
        assert grouper.get_dp_group(pp_idx=0, tp_idx=0) == [0, 2, 4, 6]
        # PP=0, tp=1: [1, 3, 5, 7]
        assert grouper.get_dp_group(pp_idx=0, tp_idx=1) == [1, 3, 5, 7]

    def test_metatron_dp_ep_invariant(self):
        """Megatron constraint: dp % ep == 0."""
        with pytest.raises(AssertionError):
            make_grouper(list(range(6)), tp=2, dp=3, pp=1, ep=2)


# ===========================================================================
# TestVllmEP - vLLM-style EP (EP spans all GPUs in a PP stage)
# ===========================================================================

class TestVllmEP:
    """Test VllmRankGrouper: EP spans ALL non-PP GPUs."""

    def test_ep_all_ranks(self):
        """8 GPUs, TP=2, DP=4, PP=1: EP group is all 8 ranks."""
        grouper = VllmRankGrouper(
            list(range(8)),
            ParallelismConfig(tp=2, dp=4, pp=1, ep=4),
        )
        # Single PP stage, EP = all 8 GPUs
        assert grouper.get_ep_group(pp_idx=0) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_ep_per_stage(self):
        """16 GPUs, TP=2, DP=4, PP=2: each stage has its own EP group."""
        grouper = VllmRankGrouper(
            list(range(16)),
            ParallelismConfig(tp=2, dp=4, pp=2, ep=4),
        )
        # PP=0 → nodes[0:8], PP=1 → nodes[8:16]
        assert grouper.get_ep_group(pp_idx=0) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert grouper.get_ep_group(pp_idx=1) == [8, 9, 10, 11, 12, 13, 14, 15]

    def test_tp_and_dp_still_work(self):
        """vLLM EP shares TP/DP/PP grouping from base class."""
        grouper = VllmRankGrouper(
            list(range(8)),
            ParallelismConfig(tp=2, dp=4, pp=1, ep=4),
        )
        # TP groups still work: [0,1], [2,3], [4,5], [6,7]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [2, 3]
        # DP groups still work
        assert grouper.get_dp_group(pp_idx=0, tp_idx=0) == [0, 2, 4, 6]


# ===========================================================================
# TestCustomRankOrder - Verify custom rank ordering changes grouping
# ===========================================================================

class TestCustomRankOrder:
    """Test that changing rank order changes grouping results."""

    def test_default_order_tp_groups(self):
        """Default order: [0,1,2,3,4,5,6,7] with TP=2."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [2, 3]

    def test_reordered_tp_groups(self):
        """Reordered: user wants [0,4] and [1,5] as TP groups."""
        ranks = [0, 4, 1, 5, 2, 6, 3, 7]
        grouper = make_base(ranks, tp=2, dp=2, pp=2)
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 4]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [1, 5]


# ===========================================================================
# TestEdgeCases - Single GPU per group, boundary conditions
# ===========================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_gpu(self):
        """Single GPU: TP=1, DP=1, PP=1, EP=1."""
        grouper = make_grouper([0])
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0]
        assert grouper.get_dp_group(pp_idx=0, tp_idx=0) == [0]
        assert grouper.get_ep_group(pp_idx=0, dp_ep_idx=0, tp_idx=0) == [0]
        assert grouper.get_pp_group(dp_idx=0, tp_idx=0) == [0]

    def test_tp1_multiple_dp(self):
        """TP=1 means each GPU is its own TP group."""
        grouper = make_grouper([0, 1, 2, 3], tp=1, dp=4, pp=1)
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=2) == [2]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=3) == [3]

    def test_assertion_failure_on_mismatch(self):
        """AssertionError when len(ranks) != tp * dp * pp."""
        with pytest.raises(AssertionError):
            make_grouper([0, 1, 2], tp=2, dp=2, pp=1)  # 3 != 2*2*1=4

    def test_pp1_only_one_pp_group(self):
        """PP=1 means only one PP stage."""
        grouper = make_grouper([0, 1, 2, 3], tp=2, dp=2, pp=1)
        assert grouper.get_pp_group(dp_idx=0, tp_idx=0) == [0]
        assert grouper.get_pp_group(dp_idx=0, tp_idx=1) == [1]
        assert grouper.get_pp_group(dp_idx=1, tp_idx=0) == [2]

    def test_base_class_ep_not_implemented(self):
        """Base RankGrouper raises NotImplementedError for get_ep_group."""
        grouper = make_base([0, 1, 2, 3], tp=2, dp=2, pp=1)
        with pytest.raises(NotImplementedError):
            grouper.get_ep_group(pp_idx=0, dp_idx=0, tp_idx=0)

    def test_get_all_stage_ranks(self):
        """get_all_stage_ranks returns all dp*tp ranks for a PP stage."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=4, pp=1)
        assert grouper.get_all_stage_ranks(pp_idx=0) == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_get_all_stage_ranks_multi_pp(self):
        """get_all_stage_ranks works with PP>1."""
        grouper = make_grouper(list(range(16)), tp=2, dp=4, pp=2, ep=2)
        assert grouper.get_all_stage_ranks(pp_idx=0) == [0, 1, 2, 3, 4, 5, 6, 7]
        assert grouper.get_all_stage_ranks(pp_idx=1) == [8, 9, 10, 11, 12, 13, 14, 15]


# ===========================================================================
# TestIntegration - Integration with parse_comm_type patterns
# ===========================================================================

class TestIntegration:
    """Test integration patterns matching how WorkloadBuilder will use RankGrouper."""

    def test_tp_comm_type_mapping(self):
        """No suffix → TP group mapping."""
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)
        tp_ranks = grouper.get_tp_group(pp_idx=0, dp_idx=0)
        assert tp_ranks == [0, 1]

    def test_dp_ep_comm_type_mapping(self):
        """_DP_EP suffix → DP_EP combined group mapping."""
        grouper = make_grouper([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=4, pp=1, ep=2)
        # ALLGATHER_DP_EP → DP_EP group (stride by ep)
        # tp=2, dp=4, ep=2 → dp_ep = 2
        # pp=0, ep_idx=0, tp=0: stride dp=(0,2) → [0, 4]
        dp_ep_ranks = grouper.get_dp_ep_group(pp_idx=0, ep_idx=0, tp_idx=0)
        assert dp_ep_ranks == [0, 4]

    def test_full_example_8gpu(self):
        """Full example: 8 GPUs, TP=2, DP=2, PP=2.

        Layout: [PP=2][DP=2][TP=2]
        - PP stage 0: ranks[0:4] = [0,1,2,3]
        - PP stage 1: ranks[4:8] = [4,5,6,7]

        Within PP stage 0:
        - DP group 0: [0,1], DP group 1: [2,3]
        - TP group 0: [0,2], TP group 1: [1,3]
        """
        grouper = make_base([0, 1, 2, 3, 4, 5, 6, 7], tp=2, dp=2, pp=2)

        # TP groups
        assert grouper.get_tp_group(pp_idx=0, dp_idx=0) == [0, 1]
        assert grouper.get_tp_group(pp_idx=0, dp_idx=1) == [2, 3]
        assert grouper.get_tp_group(pp_idx=1, dp_idx=0) == [4, 5]

        # DP groups (all replicas)
        assert grouper.get_dp_group(pp_idx=0, tp_idx=0) == [0, 2]
        assert grouper.get_dp_group(pp_idx=0, tp_idx=1) == [1, 3]

        # PP groups
        assert grouper.get_pp_group(dp_idx=0, tp_idx=0) == [0, 4]

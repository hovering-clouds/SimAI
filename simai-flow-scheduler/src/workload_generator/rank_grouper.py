"""
Rank Grouper for AI Training Parallelism.

Derives TP/DP/EP/PP rank groups from a flat rank list and parallelism config.
EP groups are derived from the 3D layout rather than occupying their own
dimension in the grid, allowing different EP partitioning strategies.

Layout convention: [PP][DP][TP] (3D) — TP innermost, PP outermost.
Index formula: global_idx = pp_idx * (dp * tp) + dp_idx * tp + tp_idx

EP groups are a DERIVED concept, not a grid dimension. Different strategies
(Megatron-style, vLLM-style, custom) implement get_ep_group() differently
while sharing the same base layout.

Example: 8 GPUs, tp=2, dp=4, pp=1
    ranks = [0, 1, 2, 3, 4, 5, 6, 7]

    By [DP=4][TP=2]:
      TP groups: [0,1], [2,3], [4,5], [6,7]    — one per DP replica
      DP groups: [0,2,4,6], [1,3,5,7]          — all replicas, per tp_idx
"""

from src.workload_format.schema import ParallelismConfig


class RankGrouper:
    """
    Base rank grouper with 3D [PP][DP][TP] layout.

    Subclasses must implement get_ep_group() with strategy-specific signatures.
    All other group accessors are shared across strategies.

    The rank list ordering convention is [PP][DP][TP].
    The index formula is:
        global_idx = pp_idx * (dp * tp) + dp_idx * tp + tp_idx
    """

    def __init__(self, assigned_nodes: list[int], parallelism: ParallelismConfig):
        self.nodes = assigned_nodes
        self.tp = parallelism.tp
        self.dp = parallelism.dp
        self.pp = parallelism.pp
        self.ep = parallelism.ep
        assert len(assigned_nodes) == self.tp * self.dp * self.pp, (
            f"Expected {self.tp * self.dp * self.pp} nodes, "
            f"got {len(assigned_nodes)}"
        )

    # ── Private index helper ──────────────────────────────────────────────

    def _idx(self, pp_idx: int, dp_idx: int, tp_idx: int) -> int:
        """3D [PP][DP][TP] linear index."""
        return pp_idx * (self.dp * self.tp) + dp_idx * self.tp + tp_idx

    # ── Shared group accessors ────────────────────────────────────────────

    def get_tp_group(self, pp_idx: int, dp_idx: int) -> list[int]:
        """TP group — all ranks sharing the same (PP, DP) position."""
        return [
            self.nodes[self._idx(pp_idx, dp_idx, tp_idx)]
            for tp_idx in range(self.tp)
        ]

    def get_dp_group(self, pp_idx: int, tp_idx: int) -> list[int]:
        """DP group — all model replicas at the same (PP, TP) position."""
        return [
            self.nodes[self._idx(pp_idx, dp_idx, tp_idx)]
            for dp_idx in range(self.dp)
        ]

    def get_pp_group(self, dp_idx: int, tp_idx: int) -> list[int]:
        """PP group — ranks across pipeline stages at the same (DP, TP) position."""
        return [
            self.nodes[self._idx(pp_idx, dp_idx, tp_idx)]
            for pp_idx in range(self.pp)
        ]

    def get_pp_rank(self, pp_idx: int, dp_idx: int, tp_idx: int) -> int:
        """Single rank at a specific (PP, DP, TP) position."""
        return self.nodes[self._idx(pp_idx, dp_idx, tp_idx)]

    def get_all_stage_ranks(self, pp_idx: int) -> list[int]:
        """All ranks belonging to a single PP stage (dp * tp ranks)."""
        start = pp_idx * (self.dp * self.tp)
        return self.nodes[start:start + self.dp * self.tp]

    # ── Subclass responsibility ───────────────────────────────────────────

    def get_ep_group(self, *args, **kwargs):
        """Retrieve EP group ranks.

        Signature and semantics depend on the specific EP partitioning
        strategy.  Override in subclasses.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement get_ep_group()"
        )


# ── Strategy: Megatron-style EP ──────────────────────────────────────────────

class MegatronRankGrouper(RankGrouper):
    """Megatron-style EP: experts distributed across consecutive DP replicas.

    EP groups are formed by taking ``ep`` consecutive DP replicas and
    collecting the rank at a given ``tp_idx`` from each.  This is the
    standard MoE sharding pattern where each TP group holds a full model
    copy and experts are spread across ``ep`` such copies.

    Constraint: ``dp % ep == 0``, with ``dp_ep = dp // ep`` data-parallel
    replicas per expert group.
    """

    def __init__(self, assigned_nodes: list[int], parallelism: ParallelismConfig):
        super().__init__(assigned_nodes, parallelism)
        assert self.dp % self.ep == 0, (
            f"MegatronRankGrouper requires dp ({self.dp}) "
            f"to be divisible by ep ({self.ep})"
        )
        self.dp_ep: int = self.dp // self.ep

    def get_ep_group(self, pp_idx: int, dp_ep_idx: int, tp_idx: int) -> list[int]:
        """EP group starting at ``dp_ep_idx * ep`` in the DP dimension.

        Args:
            pp_idx: Pipeline stage index.
            dp_ep_idx: Expert's data-parallel group index (0 .. dp_ep - 1).
            tp_idx: Tensor-parallel rank within each replica.

        Returns:
            List of ``ep`` ranks, one from each consecutive DP replica.
        """
        start = dp_ep_idx * self.ep
        return [
            self.nodes[self._idx(pp_idx, start + local_ep, tp_idx)]
            for local_ep in range(self.ep)
        ]

    def get_dp_ep_group(self, pp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """DP_EP group — ranks sharing the same expert shard.

        Stride by ``ep`` through the DP dimension: take the k-th data-parallel
        replica of expert shard ``ep_idx``, for k = 0, 1, ..., dp_ep - 1.

        Used for data-parallel weight gradient sync *within* an expert shard.
        Megatron-specific: VllmRankGrouper has no equivalent concept.
        """
        return [
            self.nodes[self._idx(pp_idx, dp_idx, tp_idx)]
            for dp_idx in range(ep_idx, self.dp, self.ep)
        ]


# ── Strategy: vLLM-style EP ──────────────────────────────────────────────────

class VllmRankGrouper(RankGrouper):
    """vLLM-style EP: experts span ALL GPUs within a PP stage.

    The EP group is the entire set of ``dp * tp`` non-PP GPUs.
    There is no sub-division of DP — all ranks participate in every
    expert all-to-all.

    ``ep_size = dp * tp`` per PP stage.
    """

    def get_ep_group(self, pp_idx: int) -> list[int]:
        """EP group — all non-PP ranks in the given PP stage.

        Args:
            pp_idx: Pipeline stage index.

        Returns:
            All ``dp * tp`` ranks from that PP stage.
        """
        return self.get_all_stage_ranks(pp_idx)

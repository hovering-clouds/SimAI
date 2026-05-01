"""
Rank Grouper for AI Training Parallelism.

Derives TP/DP/EP/PP rank groups from a flat rank list and parallelism config.
The rank list order implicitly encodes the multi-dimensional parallel grid mapping.

Convention: ranks are laid out in [PP][DP][EP][TP] order (TP innermost, PP outermost).
Index formula: global_idx = pp_idx*(dp*ep*tp) + dp_idx*(ep*tp) + ep_idx*tp + tp_idx

Example: 8 GPUs, tp=2, dp=2, pp=2
    ranks = [0, 1, 2, 3, 4, 5, 6, 7]

    By [PP=2][DP=2][TP=2]:
      PP stage 0: ranks[0:4] = [0,1,2,3]
      PP stage 1: ranks[4:8] = [4,5,6,7]

    Within PP stage 0, dp=2, tp=2:
      DP group 0: [0,1]    DP group 1: [2,3]
      TP group 0: [0,2]    TP group 1: [1,3]

Users control grouping by reordering the assigned_nodes list.
"""

from src.workload_format.schema import ParallelismConfig


class RankGrouper:
    """
    Derive parallel groups from assigned_nodes list order + parallelism config.

    The list ordering convention is [PP=pp][DP=dp][EP=ep][TP=tp].
    The index formula is:
        global_idx = pp_idx * (dp * ep * tp) + dp_idx * (ep * tp) + ep_idx * tp + tp_idx
    """

    def __init__(self, assigned_nodes: list[int], parallelism: ParallelismConfig):
        self.nodes = assigned_nodes
        self.tp = parallelism.tp
        self.dp = parallelism.dp
        self.ep = parallelism.ep
        self.pp = parallelism.pp
        assert len(assigned_nodes) == self.tp * self.dp * self.ep * self.pp, \
            f"Expected {self.tp * self.dp * self.ep * self.pp} nodes, got {len(assigned_nodes)}"

    def get_tp_group(self, pp_idx: int, dp_idx: int, ep_idx: int) -> list[int]:
        """Get TP group ranks for the given (PP, DP, EP) indices."""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for tp_idx in range(self.tp)
        ]

    def get_dp_group(self, pp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """Get DP group ranks for the given (PP, EP, TP) indices."""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for dp_idx in range(self.dp)
        ]

    def get_ep_group(self, pp_idx: int, dp_idx: int, tp_idx: int) -> list[int]:
        """Get EP group ranks for the given (PP, DP, TP) indices."""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for ep_idx in range(self.ep)
        ]

    def get_dp_ep_group(self, pp_idx: int, tp_idx: int) -> list[int]:
        """Get DP×EP combined group ranks for the given (PP, TP) indices."""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for dp_idx in range(self.dp)
            for ep_idx in range(self.ep)
        ]

    def get_pp_group(self, dp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """Get PP group ranks for the given (DP, EP, TP) indices."""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for pp_idx in range(self.pp)
        ]

    def get_pp_rank(self, pp_idx: int, dp_idx: int, ep_idx: int, tp_idx: int) -> int:
        """Get the single rank at a specific (PP, DP, EP, TP) position."""
        return self.nodes[
            pp_idx * (self.dp * self.ep * self.tp) +
            dp_idx * (self.ep * self.tp) +
            ep_idx * self.tp +
            tp_idx
        ]

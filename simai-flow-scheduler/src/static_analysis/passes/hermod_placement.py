"""Hermod-only logical-rank to physical-GPU placement helpers."""
from ...workload_format.schema import ParallelismConfig


PLACEMENTS = {"contiguous", "cyclic_pp_dp"}


def assigned_nodes_for(
    parallelism: ParallelismConfig,
    placement: str,
    gpus_per_server: int,
) -> list[int]:
    """Map logical [PP][DP][TP] ranks onto physical GPU IDs.

    ``cyclic_pp_dp`` is an experimental contention placement for homogeneous
    servers: every TP group remains local, while both adjacent PP stages and
    DP replicas rotate across servers.  It is not a paper placement.
    """
    if parallelism.ep < 1 or parallelism.dp % parallelism.ep:
        raise ValueError(
            f"Hermod Megatron placement requires dp ({parallelism.dp}) "
            f"to be divisible by ep ({parallelism.ep})"
        )
    # EP is a subdivision of the DP dimension, not another world-size
    # multiplier.  Keep this identical to ParallelismConfig.world_size and
    # MegatronRankGrouper's [PP][DP][TP] layout.
    total = parallelism.world_size
    if placement == "contiguous":
        return list(range(total))
    if total % gpus_per_server or gpus_per_server % parallelism.tp:
        raise ValueError("cyclic_pp_dp requires whole TP groups on equal-size servers")
    server_count = total // gpus_per_server
    slots_per_server = gpus_per_server // parallelism.tp
    slots_used = [0] * server_count
    nodes: list[int] = []
    for pp_idx in range(parallelism.pp):
        for dp_idx in range(parallelism.dp):
            server = (pp_idx + dp_idx) % server_count
            slot = slots_used[server]
            if slot >= slots_per_server:
                raise ValueError("cyclic_pp_dp cannot balance this PP/DP/server configuration")
            slots_used[server] += 1
            start = server * gpus_per_server + slot * parallelism.tp
            nodes.extend(range(start, start + parallelism.tp))
    return nodes

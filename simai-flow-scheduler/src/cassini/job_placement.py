"""GPU placement strategies for Cassini multi-job experiments.

Provides deterministic GPU assignment policies that create or avoid
cross-server contention for experiment reproducibility.
"""


def resolve_parallelism(header, dp_override):
    """Resolve TP/DP/PP/EP from AICB header with optional DP override."""
    tp, pp, ep = header.tp, header.pp, header.ep
    if dp_override and dp_override >= 1:
        dp = dp_override
        total_gpus = tp * dp * pp * ep
    else:
        total_gpus = header.all_gpus
        dp = total_gpus // (tp * pp * ep)
    return tp, dp, pp, ep, total_gpus


def _replica_size(cfg):
    """Number of GPUs for one DP replica across PP/EP/TP dimensions."""
    return cfg["tp"] * cfg["pp"] * cfg["ep"]


def _job_size(cfg):
    return cfg["tp"] * cfg["dp"] * cfg["pp"] * cfg["ep"]


def _server_count(gpu_count, gpus_per_server):
    if gpus_per_server <= 0:
        raise ValueError("gpus_per_server must be positive")
    if gpu_count % gpus_per_server != 0:
        raise ValueError(
            f"gpu_count ({gpu_count}) must be divisible by "
            f"gpus_per_server ({gpus_per_server})"
        )
    return gpu_count // gpus_per_server


def _partition_even(items, num_groups):
    if num_groups <= 0:
        raise ValueError("placement_clusters must be positive")
    groups = []
    n = len(items)
    start = 0
    for i in range(num_groups):
        size = n // num_groups + (1 if i < n % num_groups else 0)
        groups.append(items[start:start + size])
        start += size
    return groups


def _can_fit(server_id, size, gpus_per_server, server_free):
    return server_free[server_id] + size <= (server_id + 1) * gpus_per_server


def _take_from_server(server_id, size, gpus_per_server, server_free):
    if size > gpus_per_server:
        raise RuntimeError(
            f"Replica needs {size} GPUs but one server only has {gpus_per_server}. "
            "Use smaller tp/pp/ep or a topology with more GPUs per server."
        )
    if not _can_fit(server_id, size, gpus_per_server, server_free):
        raise RuntimeError(
            f"Server {server_id} does not have {size} contiguous GPUs left "
            f"(next={server_free[server_id]}, limit={(server_id + 1) * gpus_per_server})."
        )
    base = server_free[server_id]
    server_free[server_id] += size
    return list(range(base, base + size))


def _assignment_from_dp_servers(cfg, dp_servers, gpus_per_server, server_free):
    """Build assigned_nodes in RankGrouper's [PP][DP][EP][TP] order."""
    if len(dp_servers) != cfg["dp"]:
        raise RuntimeError(
            f"Expected {cfg['dp']} DP server assignments, got {len(dp_servers)}"
        )

    blocks = []
    size = _replica_size(cfg)
    for server_id in dp_servers:
        blocks.append(_take_from_server(server_id, size, gpus_per_server, server_free))

    nodes = []
    for pp_idx in range(cfg["pp"]):
        for dp_idx in range(cfg["dp"]):
            block = blocks[dp_idx]
            for ep_idx in range(cfg["ep"]):
                for tp_idx in range(cfg["tp"]):
                    offset = pp_idx * (cfg["ep"] * cfg["tp"]) + ep_idx * cfg["tp"] + tp_idx
                    nodes.append(block[offset])

    expected = _job_size(cfg)
    if len(nodes) != expected:
        raise RuntimeError(f"Expected {expected} assigned nodes, got {len(nodes)}")
    return nodes


# ---------------------------------------------------------------------------
# Placement strategies
# ---------------------------------------------------------------------------


def contiguous_gpus(job_configs):
    """Assign contiguous GPU ranges — minimal cross-server contention."""
    assignments = []
    offset = 0
    for cfg in job_configs:
        n = cfg["tp"] * cfg["dp"] * cfg["pp"] * cfg["ep"]
        assignments.append(list(range(offset, offset + n)))
        offset += n
    return assignments


def contention_spread_gpus(
    job_configs,
    gpu_count,
    gpus_per_server,
    placement_clusters=2,
    **_,
):
    """Deterministically place jobs to create cross-cluster DP contention."""
    n_servers = _server_count(gpu_count, gpus_per_server)
    server_groups = _partition_even(list(range(n_servers)), placement_clusters)
    if any(not group for group in server_groups):
        raise RuntimeError(
            f"placement_clusters={placement_clusters} is too high for "
            f"{n_servers} servers"
        )

    server_free = [s * gpus_per_server for s in range(n_servers)]
    reserved_free = list(server_free)
    cluster_cursor = [0 for _ in server_groups]
    assignments = []

    for j, cfg in enumerate(job_configs):
        replica_size = _replica_size(cfg)
        dp_servers = []
        for dp_idx in range(cfg["dp"]):
            cluster_idx = (j + dp_idx) % placement_clusters
            candidates = server_groups[cluster_idx]
            picked = None
            for attempt in range(len(candidates)):
                pos = (cluster_cursor[cluster_idx] + attempt) % len(candidates)
                server_id = candidates[pos]
                if _can_fit(server_id, replica_size, gpus_per_server, reserved_free):
                    picked = server_id
                    reserved_free[server_id] += replica_size
                    cluster_cursor[cluster_idx] = (pos + 1) % len(candidates)
                    break
            if picked is None:
                raise RuntimeError(
                    f"Cannot allocate job {j} DP replica {dp_idx}: no server in "
                    f"cluster {cluster_idx} has {replica_size} GPUs left. "
                    "Reduce num_jobs/dp/tp/pp/ep or increase topology size."
                )
            dp_servers.append(picked)

        assignments.append(
            _assignment_from_dp_servers(cfg, dp_servers, gpus_per_server, server_free)
        )

    return assignments


PLACEMENT_MAP = {
    "contiguous": contiguous_gpus,
    "contention-spread": contention_spread_gpus,
}


# ---------------------------------------------------------------------------
# Placement report
# ---------------------------------------------------------------------------


def print_placement_report(workload, topology, gpus_per_server, placement_clusters=2):
    """Print GPU placement summary for each job."""
    print("\nGPU Placement Report")
    print("-" * 40)

    job_servers = {}
    n_servers = max(1, topology.gpu_count // gpus_per_server)
    for job in workload.jobs:
        nodes = sorted(job.assigned_nodes)
        servers = sorted(set(g // gpus_per_server for g in nodes))
        clusters = sorted(set(
            min(s * placement_clusters // n_servers, placement_clusters - 1)
            for s in servers
        ))
        job_servers[job.job_id] = servers

        ranges = []
        start = nodes[0]
        end = nodes[0]
        for g in nodes[1:]:
            if g == end + 1:
                end = g
            else:
                ranges.append(f"{start}-{end}" if start != end else str(start))
                start = end = g
        ranges.append(f"{start}-{end}" if start != end else str(start))
        gpu_str = ",".join(ranges)

        tag = "cross-server" if len(servers) > 1 else "single-server"
        server_str = ",".join(str(s) for s in servers)
        cluster_str = ",".join(str(c) for c in clusters)
        print(f"  Job {job.job_id}: GPUs [{gpu_str}] -> "
              f"server(s) [{server_str}], cluster(s) [{cluster_str}] "
              f"({tag}, {len(servers)} server(s))")

    n_cross = sum(1 for s in job_servers.values() if len(s) > 1)

    if n_cross >= 2:
        print(f"  -> {n_cross} jobs span multiple servers: "
              f"DP traffic will cross shared spine (natural contention)")
    elif n_cross == 0:
        print(f"  -> All jobs single-server: no spine traffic, no contention")
        print(f"     Hint: use larger model, higher DP, or contention-spread placement")
    else:
        print(f"  -> Only {n_cross} job(s) cross servers: limited contention")
        print(f"     Hint: ensure all jobs span 2+ servers for natural contention")
    print()

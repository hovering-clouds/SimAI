"""Tests for Cassini experiment GPU placement helpers."""

import importlib.util
from pathlib import Path

from src.workload_format.schema import ParallelismConfig
from src.workload_generator.rank_grouper import RankGrouper


def _load_runner():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_cassini_experiments.py"
    spec = importlib.util.spec_from_file_location("run_cassini_experiments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _servers(nodes, gpus_per_server=4):
    return sorted(set(n // gpus_per_server for n in nodes))


def _clusters(nodes, gpus_per_server=4, clusters=2, servers=6):
    return sorted(set((n // gpus_per_server) * clusters // servers for n in nodes))


def test_contention_spread_places_dp_replicas_across_clusters():
    runner = _load_runner()
    cfg = {"tp": 2, "dp": 3, "pp": 2, "ep": 1}

    assignments = runner._contention_spread_gpus(
        [cfg, cfg],
        gpu_count=24,
        gpus_per_server=4,
        placement_clusters=2,
    )

    assert len(assignments) == 2
    assert len(assignments[0]) == 12
    assert len(assignments[1]) == 12
    assert set(assignments[0]).isdisjoint(assignments[1])
    assert set(assignments[0]) | set(assignments[1]) == set(range(24))

    assert _servers(assignments[0]) == [0, 1, 3]
    assert _servers(assignments[1]) == [2, 4, 5]
    assert _clusters(assignments[0]) == [0, 1]
    assert _clusters(assignments[1]) == [0, 1]


def test_contention_spread_preserves_rank_grouper_order_for_dp_groups():
    runner = _load_runner()
    cfg = {"tp": 2, "dp": 3, "pp": 2, "ep": 1}
    nodes = runner._contention_spread_gpus(
        [cfg],
        gpu_count=24,
        gpus_per_server=4,
        placement_clusters=2,
    )[0]

    grouper = RankGrouper(
        nodes,
        ParallelismConfig(tp=2, dp=3, pp=2, ep=1),
    )

    for pp_idx in range(2):
        for tp_idx in range(2):
            dp_group = grouper.get_dp_group(pp_idx=pp_idx, ep_idx=0, tp_idx=tp_idx)
            assert _clusters(dp_group) == [0, 1]

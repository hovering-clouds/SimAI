"""Tests for explicit GPU placement used by dynamic Hermod experiments."""
import runpy
from pathlib import Path

from src.workload_format.schema import ParallelismConfig


RUNNER = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "scripts" / "run_hermod_dynamic_e2e.py"),
    run_name="hermod_dynamic_placement_test",
)
assigned_nodes_for = RUNNER["assigned_nodes_for"]


def _server(group: list[int]) -> int:
    return group[0] // 8


def test_cyclic_pp_dp_keeps_tp_local_and_spreads_pp_dp_across_servers():
    parallelism = ParallelismConfig(tp=4, dp=4, pp=2, ep=1)
    nodes = assigned_nodes_for(parallelism, "cyclic_pp_dp", gpus_per_server=8)
    groups = [nodes[index:index + 4] for index in range(0, len(nodes), 4)]

    assert all(all(node // 8 == _server(group) for node in group) for group in groups)
    # Logical order is [PP][DP][TP].  PP peers at a fixed DP index and DP
    # peers at a fixed PP index must both leave the local server.
    assert all(_server(groups[dp]) != _server(groups[4 + dp]) for dp in range(4))
    assert len({_server(groups[dp]) for dp in range(4)}) == 4
    assert len({_server(groups[4 + dp]) for dp in range(4)}) == 4


def test_contiguous_placement_remains_the_legacy_mapping():
    parallelism = ParallelismConfig(tp=4, dp=2, pp=2, ep=1)
    assert assigned_nodes_for(parallelism, "contiguous", gpus_per_server=8) == list(range(16))


def test_contiguous_ep_is_a_dp_subdivision_not_world_size_multiplier():
    parallelism = ParallelismConfig(tp=2, dp=2, pp=4, ep=2)
    assert assigned_nodes_for(parallelism, "contiguous", gpus_per_server=8) == list(range(16))


def test_cyclic_ep_keeps_tp_local_and_spreads_expert_group_across_servers():
    parallelism = ParallelismConfig(tp=2, dp=2, pp=4, ep=2)
    nodes = assigned_nodes_for(parallelism, "cyclic_pp_dp", gpus_per_server=8)
    tp_groups = [nodes[index:index + 2] for index in range(0, len(nodes), 2)]

    assert all(group[0] // 8 == group[1] // 8 for group in tp_groups)
    for pp_idx in range(parallelism.pp):
        dp0 = tp_groups[pp_idx * parallelism.dp]
        dp1 = tp_groups[pp_idx * parallelism.dp + 1]
        assert dp0[0] // 8 != dp1[0] // 8

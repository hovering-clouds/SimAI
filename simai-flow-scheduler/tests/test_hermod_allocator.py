import pytest

from src.executor.bandwidth_allocators.hermod_allocator import HermodAllocator
from src.executor.runtime import ActiveFlow
from src.static_analysis.passes.hermod_priority import HermodPriorityAnalysis
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import CommType, Meta, P2PWorkload, Task, TaskType


def make_analysis():
    tasks = [
        Task(1, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100, comm_type=CommType.PP_SEND,
             coflow_id="pp", microbatch_id=0, logical_layer_id=0),
        Task(2, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100, comm_type=CommType.DP_ALLREDUCE,
             coflow_id="dp", microbatch_id=0, logical_layer_id=0),
        Task(3, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100, comm_type=CommType.PP_SEND,
             coflow_id="pp", microbatch_id=0, logical_layer_id=0),
    ]
    return HermodPriorityAnalysis.from_workload(P2PWorkload("1", Meta(0, 2), tasks=tasks))


def active(tid):
    return ActiveFlow(tid, 0, 1, 100, 100, [0, 1], 0, 0)


def test_strict_priority_then_fair_share_within_coflow():
    topo = NetworkTopology(); topo.add_link(Link(0, 1, 100, 1, 0))
    result = HermodAllocator(make_analysis()).allocate([active(1), active(2), active(3)], topo)
    assert result[1] == pytest.approx(50)
    assert result[3] == pytest.approx(50)
    assert result[2] == 0


def test_disjoint_links_do_not_block_lower_priority_coflow():
    topo = NetworkTopology()
    topo.add_link(Link(0, 1, 100, 1, 0)); topo.add_link(Link(2, 3, 100, 1, 0))
    low = ActiveFlow(2, 2, 3, 100, 100, [2, 3], 0, 0)
    result = HermodAllocator(make_analysis()).allocate([active(1), low], topo)
    assert result == {1: 100, 2: 100}

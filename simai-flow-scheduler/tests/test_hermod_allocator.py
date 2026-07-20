import pytest

from src.executor.bandwidth_allocators.hermod_allocator import HermodAllocator
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.hermod_policy import HermodSchedulingPolicy
from src.executor.runtime import ActiveFlow
from src.static_analysis.passes.hermod_metadata import HermodMetadataRecord
from src.static_analysis.passes.hermod_priority import HermodPriorityAnalysis
from src.static_analysis.passes.routing import BfsStrategy
from src.static_analysis.passes.task_serializer import ExecutionPlan
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.static_analysis.strategies.default_strategy import DefaultAnalysisResult
from src.static_analysis.strategies.hermod_strategy import HermodAnalysisResult
from src.workload_format.schema import CommType, Meta, P2PWorkload, Task, TaskType


def _record(task_id, coflow_id, mid, lid, ctype="pp"):
    return HermodMetadataRecord(task_id, coflow_id, mid, lid, ctype, "test")


def make_workload_and_records():
    tasks = [
        Task(1, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100, comm_type=CommType.PP_SEND),
        Task(2, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100, comm_type=CommType.DP_ALLREDUCE),
        Task(3, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100, comm_type=CommType.PP_SEND),
    ]
    records = {
        1: _record(1, "pp", 0, 0, "pp"),
        2: _record(2, "dp", 0, 0, "dp"),
        3: _record(3, "pp", 0, 0, "pp"),
    }
    return P2PWorkload("1", Meta(0, 2), tasks=tasks), records


def active(tid):
    return ActiveFlow(tid, 0, 1, 100, 100, [0, 1], 0, 0)


def test_strict_priority_then_fair_share_within_coflow():
    wl, records = make_workload_and_records()
    analysis = HermodPriorityAnalysis.from_workload(wl, records)
    topo = NetworkTopology(); topo.add_link(Link(0, 1, 100, 1, 0))
    result = HermodAllocator(analysis).allocate([active(1), active(2), active(3)], topo)
    assert result[1] == pytest.approx(50)
    assert result[3] == pytest.approx(50)
    assert result[2] == 0


def test_disjoint_links_do_not_block_lower_priority_coflow():
    wl, records = make_workload_and_records()
    analysis = HermodPriorityAnalysis.from_workload(wl, records)
    topo = NetworkTopology()
    topo.add_link(Link(0, 1, 100, 1, 0)); topo.add_link(Link(2, 3, 100, 1, 0))
    low = ActiveFlow(2, 2, 3, 100, 100, [2, 3], 0, 0)
    result = HermodAllocator(analysis).allocate([active(1), low], topo)
    assert result == {1: 100, 2: 100}


def test_partial_path_overlap_uses_progressive_filling():
    tasks = [
        Task(task_id, 0, TaskType.FLOW, src=0, dst=1, size_bytes=100,
             comm_type=CommType.PP_SEND)
        for task_id in (1, 2, 3)
    ]
    records = {tid: _record(tid, "pp", 0, 0, "pp") for tid in (1, 2, 3)}
    wl = P2PWorkload("1", Meta(0, 4), tasks=tasks)
    analysis = HermodPriorityAnalysis.from_workload(wl, records)
    topo = NetworkTopology()
    topo.add_link(Link(0, 1, 10, 1, 0))
    topo.add_link(Link(1, 2, 10, 1, 0))
    topo.add_link(Link(1, 3, 5, 1, 0))
    flows = [
        ActiveFlow(1, 0, 2, 100, 100, [0, 1, 2], 0, 0),
        ActiveFlow(2, 0, 3, 100, 100, [0, 1, 3], 0, 0),
        ActiveFlow(3, 0, 3, 100, 100, [0, 1, 3], 0, 0),
    ]
    result = HermodAllocator(analysis).allocate(flows, topo)
    assert result == pytest.approx({1: 5, 2: 2.5, 3: 2.5})
    reversed_result = HermodAllocator(analysis).allocate(list(reversed(flows)), topo)
    assert reversed_result == pytest.approx(result)


def test_priority_improves_a_critical_pp_dp_contention_chain():
    """A minimal E2E case where §4.1 has a measurable critical-path benefit."""
    topo = NetworkTopology()
    topo.total_nodes = topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.node_types = {0: "gpu", 1: "gpu"}
    topo.add_link(Link(0, 1, 100, 0, 0))
    topo.add_link(Link(1, 0, 100, 0, 0))
    workload = P2PWorkload("1", Meta(1, 2), tasks=[
        Task(1, 0, TaskType.FLOW, src=0, dst=1, size_bytes=125_000,
             comm_type=CommType.PP_SEND),
        Task(2, 0, TaskType.FLOW, src=0, dst=1, size_bytes=125_000,
             comm_type=CommType.DP_ALLREDUCE),
        Task(3, 0, TaskType.COMPUTE, node=1, duration_us=10, deps=[1]),
    ])
    records = {1: _record(1, "pp", 0, 0, "pp"), 2: _record(2, "dp", 0, 0, "dp")}
    routes = BfsStrategy().compute_routes(workload, topo)
    plan = ExecutionPlan(compute_order={1: [3]})
    default = AnalyticalExecutor(topo, DefaultSchedulingPolicy(
        DefaultAnalysisResult(routes, plan),
    )).execute(workload)
    hermod = AnalyticalExecutor(topo, HermodSchedulingPolicy(
        HermodAnalysisResult(routes, plan,
                             HermodPriorityAnalysis.from_workload(workload, records)),
    )).execute(workload)

    assert default.makespan_us == 30
    assert hermod.makespan_us == 20

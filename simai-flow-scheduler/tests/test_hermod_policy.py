from src.executor.policies.hermod_policy import HermodSchedulingPolicy
from src.static_analysis.passes.hermod_metadata import HermodMetadataRecord
from src.static_analysis.passes.hermod_priority import HermodPriorityAnalysis
from src.static_analysis.passes.routing import BfsRouteTable
from src.static_analysis.passes.task_serializer import ExecutionPlan
from src.static_analysis.passes.topology_loader import NetworkTopology
from src.static_analysis.strategies.hermod_strategy import HermodAnalysisResult
from src.workload_format.schema import CommType, Meta, P2PWorkload, Task, TaskType


def _analysis(workload, records) -> HermodAnalysisResult:
    return HermodAnalysisResult(
        BfsRouteTable(NetworkTopology()),
        ExecutionPlan(),
        HermodPriorityAnalysis.from_workload(workload, records),
    )


def test_dynamic_analysis_update_merges_hermod_coflows():
    first = P2PWorkload("1", Meta(1, 2), tasks=[
        Task(1, 0, TaskType.FLOW, src=0, dst=1, size_bytes=1,
             comm_type=CommType.PP_SEND),
    ])
    r1 = {1: HermodMetadataRecord(1, "job0:pp", 0, 0, "pp", "test")}
    second = P2PWorkload("1", Meta(1, 2), tasks=[
        Task(2, 1, TaskType.FLOW, src=0, dst=1, size_bytes=1,
             comm_type=CommType.DP_ALLREDUCE),
    ])
    r2 = {2: HermodMetadataRecord(2, "job1:dp", 1, 1, "dp", "test")}
    policy = HermodSchedulingPolicy(_analysis(first, r1))
    policy.update_analysis(second, _analysis(second, r2))

    merged = policy.allocator.priority_analysis
    assert set(merged.coflows) == {"job0:pp", "job1:dp"}
    assert merged.task_to_coflow[1] == "job0:pp"
    assert merged.task_to_coflow[2] == "job1:dp"

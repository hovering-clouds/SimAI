"""MFS analysis strategy — default routing + compute ordering + MFS metadata.

Mirrors DefaultAnalyzer and adds MFS context metadata on top.
"""
from dataclasses import dataclass, field

from ..passes.routing import RouteTable, BfsStrategy
from ..passes.topology_loader import NetworkTopology
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ..passes.mfs_context import MfsContext, build_mfs_context
from ..passes.mfs_feasibility import FeasibilityInfo, build_feasibility_info
from ...workload_format.schema import P2PWorkload, BatchTaskInfo


@dataclass
class MfsAnalysisResult:
    route_table: RouteTable
    execution_plan: ExecutionPlan
    mfs_context: MfsContext
    feasibility_info: FeasibilityInfo | None = None


class MfsAnalyzer:
    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(
        self,
        workload: P2PWorkload,
        batch_task_info: list[BatchTaskInfo],
        trace: dict,
    ) -> MfsAnalysisResult:
        route_table = BfsStrategy().compute_routes(workload, self.topology)
        execution_plan = CppReferenceSerializer().serialize(workload)
        context = build_mfs_context(workload, batch_task_info, trace=trace)
        feasibility_info = build_feasibility_info(
            workload, context, route_table, self.topology,
        )
        return MfsAnalysisResult(
            route_table=route_table,
            execution_plan=execution_plan,
            mfs_context=context,
            feasibility_info=feasibility_info,
        )

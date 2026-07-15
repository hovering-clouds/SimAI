"""Routing, compute serialization, and Hermod §4.1 priority analysis."""
from dataclasses import dataclass
from ..passes.routing import BfsStrategy, RouteTable
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ..passes.topology_loader import NetworkTopology
from ..passes.hermod_priority import HermodPriorityAnalysis, HermodScheduleVariant
from ...workload_format.schema import P2PWorkload


@dataclass
class HermodAnalysisResult:
    route_table: RouteTable
    execution_plan: ExecutionPlan
    priority_analysis: HermodPriorityAnalysis


class HermodAnalyzer:
    def __init__(self, topology: NetworkTopology,
                 variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B):
        self.topology = topology
        self.variant = variant

    def analyze(self, workload: P2PWorkload) -> HermodAnalysisResult:
        return HermodAnalysisResult(
            route_table=BfsStrategy().compute_routes(workload, self.topology),
            execution_plan=CppReferenceSerializer().serialize(workload),
            priority_analysis=HermodPriorityAnalysis.from_workload(workload, self.variant),
        )

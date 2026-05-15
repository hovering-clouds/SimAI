"""Default analysis strategy — minimal: routing hints + compute ordering.

Used by DefaultSchedulingPolicy. For full-featured reference (all passes)
see ExampleAnalyzer in example_strategy.py.
"""
from dataclasses import dataclass, field

from ..passes.routing import RouteTable, BfsStrategy
from ..passes.topology_loader import NetworkTopology
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ...workload_format.schema import P2PWorkload


@dataclass
class DefaultAnalysisResult:
    """Minimal analysis result: routing hints + compute ordering.

    Used by DefaultSchedulingPolicy. For full analysis see
    ExampleAnalysisResult (example_strategy.py).
    """

    route_table: RouteTable
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)


class DefaultAnalyzer:
    """Minimal analyzer: routing hints + compute ordering.

    Used by DefaultSchedulingPolicy. For full analysis see
    ExampleAnalyzer (example_strategy.py).
    """

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        """Run minimized passes and return the combined result."""
        route_table = BfsStrategy().compute_routes(workload, self.topology)
        serializer = CppReferenceSerializer()
        execution_plan = serializer.serialize(workload)

        return DefaultAnalysisResult(
            route_table=route_table,
            execution_plan=execution_plan,
        )

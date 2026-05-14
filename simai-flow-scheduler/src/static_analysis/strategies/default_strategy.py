"""Default analysis strategy — minimal: routing hints + compute ordering.

Used by DefaultSchedulingPolicy. For full-featured reference (all passes)
see ExampleAnalyzer in example_strategy.py.
"""
from dataclasses import dataclass, field

from ..passes.routing_hints import RoutingHints, compute_routing_hints
from ..passes.topology_loader import NetworkTopology
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ...workload_format.schema import P2PWorkload


@dataclass
class DefaultAnalysisResult:
    """Minimal analysis result: routing hints + compute ordering.

    Used by DefaultSchedulingPolicy. For full analysis see
    ExampleAnalysisResult (example_strategy.py).
    """

    routing_hints: RoutingHints
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
        routing_hints = compute_routing_hints(self.topology, workload)
        serializer = CppReferenceSerializer()
        execution_plan = serializer.serialize(workload)

        return DefaultAnalysisResult(
            routing_hints=routing_hints,
            execution_plan=execution_plan,
        )

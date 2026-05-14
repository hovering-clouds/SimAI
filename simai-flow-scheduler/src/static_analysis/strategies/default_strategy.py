"""Default analysis strategy — runs all passes in the standard order.

Merged from the original WorkloadAnalyzer + WorkloadAnalysisResult to serve
as the single default analysis entry point. Custom phase-2 strategies
(e.g. Puppeteer-specific analysis) can reuse the pipeline or reorder passes.
"""
from dataclasses import dataclass, field

from ..passes.contention_analysis import (
    LinkContentionGroup,
    find_contention_groups,
)
from ..passes.critical_path import CriticalPathInfo, analyze_critical_path
from ..passes.node_view import NodeLocalView, build_node_views
from ..passes.routing_hints import RoutingHints, compute_routing_hints
from ..passes.topology_loader import NetworkTopology
from ..passes.traffic_matrix import TrafficMatrix, compute_traffic_matrix
from ..passes.workload_summary import WorkloadSummary, compute_workload_summary
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ...workload_format.schema import P2PWorkload


@dataclass
class DefaultAnalysisResult:
    """Complete default workload analysis result."""

    routing_hints: RoutingHints
    critical_path: CriticalPathInfo
    contention_groups: dict[tuple[int, int], LinkContentionGroup]
    node_views: dict[int, NodeLocalView]
    traffic_matrix: TrafficMatrix
    summary: WorkloadSummary
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)


class DefaultAnalyzer:
    """Runs all analysis passes in the standard dependency order.

    Order:
        1. routing_hints  — BFS shortest paths
        2. critical_path  — CPM with multi-hop duration
        3. contention_groups — spatial + temporal contention
        4. node_views     — per-node scheduling perspective
        5. traffic_matrix — node-to-node volume
        6. summary        — global workload statistics
    """

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        """Run all analysis passes and return the combined result."""
        routing_hints = compute_routing_hints(self.topology, workload)
        critical_path = analyze_critical_path(workload, routing_hints)
        contention_groups = find_contention_groups(
            workload, routing_hints, critical_path,
        )
        node_views = build_node_views(workload, critical_path)
        traffic_matrix = compute_traffic_matrix(workload)
        summary = compute_workload_summary(workload, critical_path, contention_groups)

        serializer = CppReferenceSerializer()
        execution_plan = serializer.serialize(workload)

        return DefaultAnalysisResult(
            routing_hints=routing_hints,
            critical_path=critical_path,
            contention_groups=contention_groups,
            node_views=node_views,
            traffic_matrix=traffic_matrix,
            summary=summary,
            execution_plan=execution_plan,
        )

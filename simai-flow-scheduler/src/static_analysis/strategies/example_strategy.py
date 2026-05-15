"""Full-featured reference analyzer — runs all passes.

Custom strategies (e.g. Puppeteer) can use this as a template
to reorder or extend the analysis pipeline.
"""
from dataclasses import dataclass, field

from ..passes.contention_analysis import (
    LinkContentionGroup,
    find_contention_groups,
)
from ..passes.critical_path import CriticalPathInfo, analyze_critical_path
from ..passes.node_view import NodeLocalView, build_node_views
from ..passes.routing import RouteTable, BfsStrategy
from ..passes.topology_loader import NetworkTopology
from ..passes.traffic_matrix import TrafficMatrix, compute_traffic_matrix
from ..passes.workload_summary import WorkloadSummary, compute_workload_summary
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ...workload_format.schema import P2PWorkload


@dataclass
class ExampleAnalysisResult:
    """Complete workload analysis result — reference example.

    Runs all analysis passes. Custom strategies (e.g. Puppeteer) can use
    this as a template to reorder or extend the pipeline.
    """

    route_table: RouteTable
    critical_path: CriticalPathInfo
    contention_groups: dict[tuple[int, int], LinkContentionGroup]
    node_views: dict[int, NodeLocalView]
    traffic_matrix: TrafficMatrix
    summary: WorkloadSummary
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)


class ExampleAnalyzer:
    """Runs all analysis passes in the standard dependency order.

    Custom strategies (e.g. Puppeteer) can use this as a template
    to reorder or extend the analysis pipeline.

    Order:
        1. route_table  — BFS shortest paths
        2. critical_path  — CPM with multi-hop duration
        3. contention_groups — spatial + temporal contention
        4. node_views     — per-node scheduling perspective
        5. traffic_matrix — node-to-node volume
        6. summary        — global workload statistics
    """

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> ExampleAnalysisResult:
        """Run all analysis passes and return the combined result."""
        route_table = BfsStrategy().compute_routes(workload, self.topology)
        critical_path = analyze_critical_path(workload, route_table, self.topology)
        contention_groups = find_contention_groups(
            workload, route_table, self.topology, critical_path,
        )
        node_views = build_node_views(workload, critical_path)
        traffic_matrix = compute_traffic_matrix(workload)
        summary = compute_workload_summary(workload, critical_path, contention_groups)

        serializer = CppReferenceSerializer()
        execution_plan = serializer.serialize(workload)

        return ExampleAnalysisResult(
            route_table=route_table,
            critical_path=critical_path,
            contention_groups=contention_groups,
            node_views=node_views,
            traffic_matrix=traffic_matrix,
            summary=summary,
            execution_plan=execution_plan,
        )

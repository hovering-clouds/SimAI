"""
Workload analyzer - unified entry point for all analysis modules.

Orchestrates the execution of all analysis modules in the correct dependency order:
1. routing_hints (depends on topology only)
2. critical_path (depends on routing_hints)
3. contention_groups (depends on routing_hints + critical_path)
4. node_views (depends on critical_path)
5. traffic_matrix (independent)
6. summary (depends on critical_path + contention_groups)

Useful for:
- One-stop analysis of a workload
- Ensuring correct execution order
- Consistent analysis across different use cases
"""

from dataclasses import dataclass

from ..workload_format.schema import P2PWorkload
from .passes.contention_analysis import LinkContentionGroup, find_contention_groups
from .passes.critical_path import CriticalPathInfo, analyze_critical_path
from .passes.node_view import NodeLocalView, build_node_views
from .passes.routing_hints import RoutingHints, compute_routing_hints
from .passes.topology_loader import NetworkTopology
from .passes.traffic_matrix import TrafficMatrix, compute_traffic_matrix
from .passes.workload_summary import WorkloadSummary, compute_workload_summary


@dataclass
class WorkloadAnalysisResult:
    """Complete workload analysis result."""

    routing_hints: RoutingHints
    critical_path: CriticalPathInfo
    contention_groups: dict[tuple[int, int], LinkContentionGroup]
    node_views: dict[int, NodeLocalView]
    traffic_matrix: TrafficMatrix
    summary: WorkloadSummary


class WorkloadAnalyzer:
    """Unified entry point for workload analysis."""

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> WorkloadAnalysisResult:
        """Run all analysis modules on the workload."""
        routing_hints = compute_routing_hints(self.topology, workload)
        critical_path = analyze_critical_path(workload, routing_hints)
        contention_groups = find_contention_groups(
            workload, routing_hints, critical_path
        )
        node_views = build_node_views(workload, critical_path)
        traffic_matrix = compute_traffic_matrix(workload)
        summary = compute_workload_summary(workload, critical_path, contention_groups)

        return WorkloadAnalysisResult(
            routing_hints=routing_hints,
            critical_path=critical_path,
            contention_groups=contention_groups,
            node_views=node_views,
            traffic_matrix=traffic_matrix,
            summary=summary,
        )

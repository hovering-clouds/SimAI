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
from .contention_analysis import LinkContentionGroup, find_contention_groups
from .critical_path import CriticalPathInfo, analyze_critical_path
from .node_view import NodeLocalView, build_node_views
from .routing_hints import RoutingHints, compute_routing_hints
from .topology_loader import NetworkTopology
from .traffic_matrix import TrafficMatrix, compute_traffic_matrix
from .workload_summary import WorkloadSummary, compute_workload_summary


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
        """
        Initialize analyzer with network topology.

        Args:
            topology: Network topology graph
        """
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> WorkloadAnalysisResult:
        """
        Run all analysis modules on the workload.

        Execution order matters (routing must come before critical path):
        1. routing_hints  — depends on topology only (BFS shortest paths)
        2. critical_path  — depends on routing_hints (multi-hop duration estimation)
        3. contention_groups — depends on routing_hints + critical_path (paths + timing)
        4. node_views     — depends on critical_path (uses ASAP times)
        5. traffic_matrix — no dependencies
        6. summary        — depends on critical_path + contention_groups

        Args:
            workload: P2P workload to analyze

        Returns:
            WorkloadAnalysisResult with all analysis results
        """
        # 1. Routing hints (BFS shortest paths - must come first!)
        routing_hints = compute_routing_hints(self.topology, workload)

        # 2. Critical path (uses routing_hints for accurate multi-hop duration)
        critical_path = analyze_critical_path(workload, self.topology, routing_hints)

        # 3. Link contention (uses paths from routing_hints + timing from critical_path)
        contention_groups = find_contention_groups(
            workload, self.topology, routing_hints, critical_path
        )

        # 4. Node views (uses ASAP times from critical_path)
        node_views = build_node_views(workload, critical_path)

        # 5. Traffic matrix (independent)
        traffic_matrix = compute_traffic_matrix(workload)

        # 6. Summary (aggregates results from above)
        summary = compute_workload_summary(workload, critical_path, contention_groups)

        return WorkloadAnalysisResult(
            routing_hints=routing_hints,
            critical_path=critical_path,
            contention_groups=contention_groups,
            node_views=node_views,
            traffic_matrix=traffic_matrix,
            summary=summary,
        )

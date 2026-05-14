"""Default analysis strategy — runs all passes in the standard order.

Matches the execution order of the original WorkloadAnalyzer.analyze().
Kept as a separate strategy class so that custom phase-2 strategies
(e.g. Puppeteer-specific analysis) can reuse the pipeline or reorder passes.
"""
from ..passes.contention_analysis import find_contention_groups
from ..passes.critical_path import analyze_critical_path
from ..passes.node_view import build_node_views
from ..passes.routing_hints import compute_routing_hints
from ..passes.topology_loader import NetworkTopology
from ..passes.traffic_matrix import compute_traffic_matrix
from ..passes.workload_summary import compute_workload_summary
from ..analyzer import WorkloadAnalysisResult
from ...workload_format.schema import P2PWorkload


class DefaultAnalysisStrategy:
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

    def analyze(self, workload: P2PWorkload) -> WorkloadAnalysisResult:
        """Run all analysis passes and return the combined result."""
        routing_hints = compute_routing_hints(self.topology, workload)
        critical_path = analyze_critical_path(workload, routing_hints)
        contention_groups = find_contention_groups(
            workload, routing_hints, critical_path,
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

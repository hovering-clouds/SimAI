"""
Scheduler module - workload analysis and scheduling infrastructure.

Provides topology loading, routing hints, critical path analysis,
link contention analysis, and unified workload analysis.
"""

from .topology_loader import TopologyLoader, NetworkTopology, Link, NodeType
from .routing_hints import (
    RoutingHints,
    compute_routing_hints,
    bfs_shortest_path,
    RoutingStrategy,
)
from .critical_path import (
    TaskTimingInfo,
    CriticalPathInfo,
    CriticalPathStrategy,
    analyze_critical_path,
    analyze_cpm,
)
from .contention_analysis import (
    LinkContentionGroup,
    find_contention_groups,
)
from .node_view import (
    NodeLocalView,
    build_node_views,
)
from .traffic_matrix import (
    TrafficMatrix,
    compute_traffic_matrix,
)
from .workload_summary import (
    WorkloadSummary,
    compute_workload_summary,
)
from .analyzer import (
    WorkloadAnalysisResult,
    WorkloadAnalyzer,
)

__all__ = [
    "TopologyLoader",
    "NetworkTopology",
    "Link",
    "NodeType",
    "RoutingHints",
    "compute_routing_hints",
    "bfs_shortest_path",
    "RoutingStrategy",
    "TaskTimingInfo",
    "CriticalPathInfo",
    "CriticalPathStrategy",
    "analyze_critical_path",
    "analyze_cpm",
    "LinkContentionGroup",
    "find_contention_groups",
    "NodeLocalView",
    "build_node_views",
    "TrafficMatrix",
    "compute_traffic_matrix",
    "WorkloadSummary",
    "compute_workload_summary",
    "WorkloadAnalysisResult",
    "WorkloadAnalyzer",
]

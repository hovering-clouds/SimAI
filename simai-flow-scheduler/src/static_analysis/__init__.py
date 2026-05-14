"""
Static analysis module - pre-scheduling workload analysis infrastructure.

Provides topology loading, routing hints, critical path analysis,
link contention analysis, and unified workload analysis.
"""

from .passes.topology_loader import TopologyLoader, NetworkTopology, Link, NodeType
from .passes.routing_hints import (
    RoutingHints,
    compute_routing_hints,
    bfs_shortest_path,
    RoutingStrategy,
)
from .passes.critical_path import (
    TaskTimingInfo,
    CriticalPathInfo,
    CriticalPathStrategy,
    analyze_critical_path,
    analyze_cpm,
    _estimate_duration,
    _topological_sort,
)
from .passes.contention_analysis import (
    LinkContentionGroup,
    find_contention_groups,
)
from .passes.node_view import (
    NodeLocalView,
    build_node_views,
)
from .passes.traffic_matrix import (
    TrafficMatrix,
    compute_traffic_matrix,
)
from .passes.workload_summary import (
    WorkloadSummary,
    compute_workload_summary,
)
from .strategies import DefaultAnalyzer, DefaultAnalysisResult
from .passes.task_serializer import (
    ExecutionPlan,
    CppReferenceSerializer,
    TaskSerializer,
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
    "DefaultAnalyzer",
    "DefaultAnalysisResult",
    "ExecutionPlan",
    "CppReferenceSerializer",
    "TaskSerializer",
]

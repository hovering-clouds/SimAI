"""Default analysis strategy — minimal: routing hints + compute ordering.

Used by DefaultSchedulingPolicy. For full-featured reference (all passes)
see ExampleAnalyzer in example_strategy.py.
"""
from dataclasses import dataclass, field

from ..passes.routing import BfsRouteTable, BfsStrategy, RouteTable
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


class LightweightAnalyzer:
    """轻量分析器 — 只构建 compute_order（C++ 参考顺序），不做 CPM/BFS。

    用于 Cassini 动态模式：路由和时间片已在 Phase 1 one-shot 分析中预计算，
    每个 iteration 展开只需注册新 task_ids 到 compute_order，供
    SchedulingPolicy.update_analysis() 增量合并。

    compute_order 使用 CppReferenceSerializer 复现 C++ 参考执行顺序，
    与静态路径的 DefaultAnalyzer/CassiniAnalyzer 一致。
    """

    def __init__(self):
        self._serializer = CppReferenceSerializer()

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        """构建 C++ 参考顺序的 compute_order，路由表留空。

        Args:
            workload: 单个 iteration 展开后的 P2PWorkload。

        Returns:
            DefaultAnalysisResult:
                route_table: 空 BfsRouteTable（update_routes 幂等，不产生新条目）。
                execution_plan: 含 C++ 参考顺序的 compute_order。
        """
        plan = self._serializer.serialize(workload)

        return DefaultAnalysisResult(
            route_table=BfsRouteTable(None),
            execution_plan=plan,
        )

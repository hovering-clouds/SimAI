"""Default analysis strategy — minimal: routing hints + compute ordering.

Used by DefaultSchedulingPolicy. For full-featured reference (all passes)
see ExampleAnalyzer in example_strategy.py.
"""
from dataclasses import dataclass, field

from ..passes.routing import BfsRouteTable, BfsStrategy, RouteTable
from ..passes.topology_loader import NetworkTopology
from ..passes.task_serializer import CppReferenceSerializer, OneFOneBSerializer, ExecutionPlan
from ...workload_format.schema import P2PWorkload
from ...workload_generator.rank_grouper import RankGrouper


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


class OneFOneBAnalyzer:
    """1F1B 分析器 — 路由 + 1F1B compute_order。

    与 DefaultAnalyzer 类似：从 workload 的 Job 中推导 PP 映射，
    同时做 BFS 路由和 1F1B compute_order。
    """

    def __init__(self, topology: NetworkTopology):
        """
        Args:
            topology: 网络拓扑（用于 BFS 路由）。
        """
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        """构建路由表 + 1F1B compute_order。

        Args:
            workload: P2PWorkload（需包含 jobs 信息用于推导 PP 映射）。

        Returns:
            DefaultAnalysisResult: 含路由表和 1F1B compute_order。
        """
        # 从 workload 的 job 推导 node_to_stage 映射
        node_to_stage: dict[int, int] = {}
        pp = 1
        for job in workload.jobs:
            grouper = RankGrouper(job.assigned_nodes, job.parallelism)
            pp = grouper.pp
            stage_size = grouper.dp * grouper.ep * grouper.tp
            for stage_id in range(grouper.pp):
                for i in range(stage_size):
                    node = grouper.nodes[stage_id * stage_size + i]
                    node_to_stage[node] = stage_id

        # BFS 路由
        route_table = BfsStrategy().compute_routes(workload, self.topology)

        # 1F1B compute_order
        serializer = OneFOneBSerializer(pp=pp, node_to_stage=node_to_stage)
        execution_plan = serializer.serialize(workload)

        return DefaultAnalysisResult(
            route_table=route_table,
            execution_plan=execution_plan,
        )

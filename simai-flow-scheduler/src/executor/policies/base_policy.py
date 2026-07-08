"""Scheduling policy interface."""
from abc import ABC, abstractmethod

from ..runtime import ActiveFlow
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...workload_format.schema import P2PWorkload, Task


class SchedulingPolicy(ABC):
    """调度策略接口。

    Executor 拥有事件队列和 DAG 进度管理；Policy 决定准入、路径和带宽分配。
    """

    @abstractmethod
    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> None:
        """初始化策略，存储 workload / topology 引用供后续决策使用。"""
        ...

    @abstractmethod
    def emit_ready_tasks(
        self,
        current_time: int,
        ready_tasks: list[Task],
    ) -> list[int]:
        """准入控制：从当前可调度 task 中选择允许立即启动的 task ID 列表。"""
        ...

    @abstractmethod
    def get_flow_path(self, task: Task) -> list[int]:
        """返回 flow task 的传输路径 [src, hop1, ..., dst]。"""
        ...

    @abstractmethod
    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        """带宽分配：返回 task_id → allocated_bw_gbps 的映射。"""
        ...

    @abstractmethod
    def on_task_emitted(self, current_time: int, task: Task) -> None:
        """task 通过准入时的通知回调。"""
        ...

    @abstractmethod
    def on_task_completed(self, current_time: int, task: Task) -> None:
        """task 完成时的通知回调。"""
        ...

    def update_analysis(self, workload: P2PWorkload, analysis_result) -> None:
        """增量合并新 Job 的 workload + 分析结果（用于动态展开模式）。

        Args:
            workload: 新 Job 展开后的完整 P2PWorkload（含 tasks）。
            analysis_result: 对应 Analyzer.analyze() 的返回值
                （如 DefaultAnalysisResult / CassiniAnalysisResult）。

        基类提供空实现，每个策略按需覆盖。
        """



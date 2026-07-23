"""Isolated analyzers for advanced pipeline experiments.

These analyzers are intentionally not exported from ``strategies.__init__``
and are not registered in Default, Hermod, or Puppeteer. They own the
strategy-specific expansion sidecar and compose one concrete serializer with
the existing BFS route pass.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..passes.pipeline_task_serializers import (
    AdvancedPipelineSerializer,
    BidirectionalPipelineSerializer,
    InterleavedOneFOneBSerializer,
    PipelineTaskInfo,
    ZeroBubbleSerializer,
)
from ..passes.routing import BfsStrategy
from ..passes.topology_loader import NetworkTopology
from .default_strategy import DefaultAnalysisResult
from ...workload_format.schema import P2PWorkload


class _AdvancedPipelineAnalyzer(ABC):
    """Strategy-local sidecar owner and analysis composition."""

    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.expansion_task_info: dict[int, object] = {}
        self.schedule_task_info: dict[int, PipelineTaskInfo] = {}

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        serializer = self._build_serializer()
        execution_plan = serializer.serialize(workload)
        self.schedule_task_info = dict(serializer.task_info)
        return DefaultAnalysisResult(
            route_table=BfsStrategy().compute_routes(workload, self.topology),
            execution_plan=execution_plan,
        )

    @abstractmethod
    def _build_serializer(self) -> AdvancedPipelineSerializer:
        ...


class InterleavedPipelineAnalyzer(_AdvancedPipelineAnalyzer):
    """Isolated analyzer for direct Interleaved 1F1B workloads."""

    def __init__(
        self,
        topology: NetworkTopology,
        virtual_pipeline_size: int = 2,
        interleave_group_size: int | None = None,
    ):
        super().__init__(topology)
        self.virtual_pipeline_size = virtual_pipeline_size
        self.interleave_group_size = interleave_group_size

    def _build_serializer(self) -> InterleavedOneFOneBSerializer:
        return InterleavedOneFOneBSerializer(
            virtual_pipeline_size=self.virtual_pipeline_size,
            interleave_group_size=self.interleave_group_size,
            expansion_task_info=self.expansion_task_info,
        )


class ZeroBubblePipelineAnalyzer(_AdvancedPipelineAnalyzer):
    """Isolated analyzer for direct ZB1P-style workloads."""

    def __init__(
        self,
        topology: NetworkTopology,
        max_inflight_microbatches: int | None = None,
    ):
        super().__init__(topology)
        self.max_inflight_microbatches = max_inflight_microbatches

    def _build_serializer(self) -> ZeroBubbleSerializer:
        return ZeroBubbleSerializer(
            expansion_task_info=self.expansion_task_info,
            max_inflight_microbatches=self.max_inflight_microbatches,
        )


class BidirectionalPipelineAnalyzer(_AdvancedPipelineAnalyzer):
    """Isolated analyzer for the basic two-replica Chimera workload."""

    def _build_serializer(self) -> BidirectionalPipelineSerializer:
        return BidirectionalPipelineSerializer(
            expansion_task_info=self.expansion_task_info,
        )

"""Hermod §4.1 policy: normal admission/routing, priority bandwidth service."""
from .default_policy import DefaultSchedulingPolicy
from ..bandwidth_allocators.hermod_allocator import HermodAllocator
from ...static_analysis.strategies.hermod_strategy import HermodAnalysisResult


class HermodSchedulingPolicy(DefaultSchedulingPolicy):
    def __init__(self, analysis: HermodAnalysisResult):
        super().__init__(analysis)
        self.allocator = HermodAllocator(analysis.priority_analysis)

    def update_analysis(self, workload, analysis_result) -> None:
        """Reject dynamic injection until Hermod metadata can be merged safely."""
        raise NotImplementedError(
            "Dynamic Hermod scheduling is not implemented: newly injected tasks "
            "need incremental MID/LID/coflow priority analysis"
        )

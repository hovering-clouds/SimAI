"""Hermod §4.1 policy: normal admission/routing, priority bandwidth service."""
from .default_policy import DefaultSchedulingPolicy
from ..bandwidth_allocators.hermod_allocator import HermodAllocator
from ...static_analysis.strategies.hermod_strategy import HermodAnalysisResult
from ...static_analysis.passes.hermod_priority import HermodPriorityAnalysis


class HermodSchedulingPolicy(DefaultSchedulingPolicy):
    def __init__(self, analysis: HermodAnalysisResult):
        super().__init__(analysis)
        self.allocator = HermodAllocator(analysis.priority_analysis)

    def update_analysis(self, workload, analysis_result: HermodAnalysisResult) -> None:
        """Merge a dynamically injected batch's routes, compute order and coflows."""
        super().update_analysis(workload, analysis_result)
        current = self.allocator.priority_analysis
        incoming = analysis_result.priority_analysis
        duplicate = set(current.coflows) & set(incoming.coflows)
        if duplicate:
            raise ValueError(f"Duplicate Hermod coflow IDs during dynamic update: {sorted(duplicate)}")
        if current.variant != incoming.variant or current.ep_mode != incoming.ep_mode:
            raise ValueError("Dynamic Hermod analysis uses incompatible priority configuration")
        merged = {**current.coflows, **incoming.coflows}
        self.allocator.priority_analysis = HermodPriorityAnalysis(
            merged, current.variant, current.ep_mode,
        )

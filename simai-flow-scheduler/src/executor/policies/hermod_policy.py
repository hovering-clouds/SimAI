"""Hermod §4.1 policy: normal admission/routing, priority bandwidth service."""
from .default_policy import DefaultSchedulingPolicy
from ..bandwidth_allocators.hermod_allocator import HermodAllocator
from ...static_analysis.strategies.hermod_strategy import HermodAnalysisResult


class HermodSchedulingPolicy(DefaultSchedulingPolicy):
    def __init__(self, analysis: HermodAnalysisResult):
        super().__init__(analysis)
        self.allocator = HermodAllocator(analysis.priority_analysis)

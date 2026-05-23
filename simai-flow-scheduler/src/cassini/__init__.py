"""
Cassini: Network-Aware Job Scheduling - Core Algorithm Module.

Implements the geometric circle abstraction for multi-job time-shift optimization
based on periodic communication patterns.

Phase 1 modules:
    - communication_pattern: Extract periodic Up/Down phases from P2PWorkload
    - circle_abstraction: Geometric circle + rotation + LCM unified circle
    - pair_compatibility: Link compatibility time-shift optimization via grid search

Phase 2 modules:
    - cassini_strategy: Full analysis pipeline (routing → patterns → time-shifts)
    - cassini_policy: Scheduling policy that applies per-job time-shifts at runtime

Phase 3 modules:
    - affinity_graph: Bipartite graph traversal for cluster-level time-shift solving
"""

from .circle_abstraction import CircleAbstraction
from .communication_pattern import CommunicationPattern, extract_communication_patterns
from .pair_compatibility import CompatibilityResult, optimize_link_compatibility, compute_score

__all__ = [
    "CommunicationPattern",
    "extract_communication_patterns",
    "CircleAbstraction",
    "CompatibilityResult",
    "optimize_link_compatibility",
    "compute_score",
    "AffinityGraph",
    "build_affinity_graph",
    "compute_cluster_time_shifts",
]


def __getattr__(name: str):
    """Lazy-load affinity_graph to avoid circular imports.

    The affinity_graph module imports circle_abstraction, which imports
    communication_pattern, which (transitively) depends on static_analysis
    — which in turn imports cassini_strategy.  Deferring the import breaks
    that cycle.
    """
    _AFFINITY_EXPORTS = {"AffinityGraph", "build_affinity_graph", "compute_cluster_time_shifts"}
    if name in _AFFINITY_EXPORTS:
        from . import affinity_graph as _ag
        return getattr(_ag, name)
    raise AttributeError(f"module 'src.cassini' has no attribute {name!r}")

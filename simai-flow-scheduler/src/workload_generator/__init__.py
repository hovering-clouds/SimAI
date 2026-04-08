"""
Workload generator module - Collective communication to P2P flow expanders.
"""

from .collective_expander import CollectiveExpander, FlowTask

__all__ = [
    "CollectiveExpander",
    "FlowTask",
]

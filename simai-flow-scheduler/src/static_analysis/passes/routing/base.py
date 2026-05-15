"""Abstract base classes for route tables and routing strategies."""
from abc import ABC, abstractmethod

from ....workload_format.schema import P2PWorkload, Task
from ..topology_loader import NetworkTopology


class RouteTable(ABC):
    """Abstract route table — maps flow tasks to paths.

    Subclasses decide the internal storage layout (e.g., by (src,dst) or by task_id)
    and implement get_path accordingly.
    """

    @abstractmethod
    def get_path(self, task: Task) -> list[int]:
        """Return the path for a flow task."""
        ...


class RouteStrategy(ABC):
    """Abstract routing strategy — computes a RouteTable from workload + topology."""

    @abstractmethod
    def compute_routes(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> RouteTable:
        """Compute and return a route table for the given workload."""
        ...

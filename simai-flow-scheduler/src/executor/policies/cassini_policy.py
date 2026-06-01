"""Cassini scheduling policy — time-shift driven multi-job iteration alignment.

Applies per-job time-shifts computed by CassiniAnalyzer to stagger iteration
starts, reducing link contention without changing intra-job DAG structure.

Reuses FairShareAllocator for bandwidth and BfsRouteTable for path lookup.
"""

from collections import defaultdict
from typing import Optional

from .base_policy import SchedulingPolicy
from ..bandwidth_allocators.base_allocator import BandwidthAllocator
from ..bandwidth_allocators.fair_share_allocator import FairShareAllocator
from ..runtime import ActiveFlow
from ...static_analysis.passes.routing import RouteTable
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...static_analysis.strategies.cassini_strategy import CassiniAnalysisResult
from ...workload_format.schema import P2PWorkload, Task


class CassiniSchedulingPolicy(SchedulingPolicy):
    """Time-shift based scheduling for multi-job workloads.

    Each job receives a global time-shift (microseconds).  Tasks belonging to
    a job are withheld until current_time >= time_shift[job_id].  After that
    point the job runs unmodified — the DAG's cross-iteration dependencies
    naturally maintain the periodic alignment.

    Compute tasks still respect the C++ reference serial ordering within each
    node.  Flow tasks are fully admitted once the job's time-shift has elapsed.
    """

    def __init__(
        self,
        analysis: CassiniAnalysisResult,
        allocator: BandwidthAllocator | None = None,
    ):
        self.route_table: RouteTable = analysis.route_table
        self.time_shifts: dict[int, int] = dict(analysis.time_shifts)
        self.compute_order = analysis.execution_plan.compute_order
        self.allocator = allocator if allocator is not None else FairShareAllocator()

        self._topology: Optional[NetworkTopology] = None
        self._job_started: dict[int, bool] = {}

        # Per-node compute cursor for serial execution
        self.compute_position: dict[int, int] = {
            task_id: idx
            for node, ids in self.compute_order.items()
            for idx, task_id in enumerate(ids)
        }
        self.compute_cursor: dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # SchedulingPolicy interface
    # ------------------------------------------------------------------

    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> None:
        self._topology = topology
        self._job_started = {job.job_id: False for job in workload.jobs}
        self.compute_cursor = defaultdict(int)

    def emit_ready_tasks(
        self,
        current_time: int,
        ready_tasks: list[Task],
    ) -> list[int]:
        emitted: list[int] = []
        for task in sorted(ready_tasks, key=lambda t: t.task_id):
            if not self._job_cleared(task.job_id, current_time):
                continue
            if task.is_flow():
                emitted.append(task.task_id)
            elif task.is_compute() and self._is_next_compute(task):
                emitted.append(task.task_id)
        return emitted

    def get_flow_path(self, task: Task) -> list[int]:
        return self.route_table.get_path(task)

    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        return self.allocator.allocate(active_flows, self._topology, current_time)

    def on_task_emitted(self, current_time: int, task: Task) -> None:
        pass

    def on_task_completed(self, current_time: int, task: Task) -> None:
        if task.is_compute():
            self.compute_cursor[task.node] += 1

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _job_cleared(self, job_id: int, current_time: int) -> bool:
        """Return True if the job's time-shift has elapsed."""
        if self._job_started.get(job_id, True):
            return True
        shift = self.time_shifts.get(job_id, 0)
        if current_time >= shift:
            self._job_started[job_id] = True
            return True
        return False

    def _is_next_compute(self, task: Task) -> bool:
        node_id = task.node
        cursor = self.compute_cursor.get(node_id, 0)
        order = self.compute_order.get(node_id, [])
        return cursor < len(order) and order[cursor] == task.task_id

"""
Node-centric local views - per-node scheduling perspective.

Builds a local view for each node showing its compute tasks, send/receive flows,
estimated schedule times (from critical path analysis), and idle ratio.

Useful for:
- Understanding per-node load balance
- Identifying communication-heavy nodes
- Quick scheduling decisions based on local information
"""

from dataclasses import dataclass, field

from ..workload_format.schema import P2PWorkload
from .critical_path import CriticalPathInfo


@dataclass
class NodeLocalView:
    """Local scheduling view for a single node."""

    node_id: int

    # Flows this node sends
    send_tasks: list[int] = field(default_factory=list)  # task_ids
    total_send_bytes: int = 0

    # Flows this node receives
    receive_tasks: list[int] = field(default_factory=list)  # task_ids
    total_receive_bytes: int = 0

    # Compute tasks on this node
    compute_tasks: list[int] = field(default_factory=list)  # task_ids
    total_compute_time_us: int = 0

    # Estimated schedule (based on ASAP from critical path)
    estimated_send_times: list[tuple[int, int]] = field(default_factory=list)
    # (start_time_us, task_id) — when the node starts transmitting
    estimated_receive_times: list[tuple[int, int]] = field(default_factory=list)
    # (arrival_time_us, task_id) — when the node finishes receiving

    # Estimated idle ratio (communication time / total active time)
    # High value = node spends more time waiting on communication
    estimated_idle_ratio: float = 0.0

    def add_send_flow(self, task_id: int, size_bytes: int, start_time: int):
        self.send_tasks.append(task_id)
        self.total_send_bytes += size_bytes
        self.estimated_send_times.append((start_time, task_id))

    def add_receive_flow(self, task_id: int, size_bytes: int, arrival_time: int):
        self.receive_tasks.append(task_id)
        self.total_receive_bytes += size_bytes
        self.estimated_receive_times.append((arrival_time, task_id))


def build_node_views(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
) -> dict[int, NodeLocalView]:
    """
    Build per-node local views.

    For each node:
    1. Collect send flows (where node is src)
    2. Collect receive flows (where node is dst)
    3. Collect compute tasks (where node is node)
    4. Estimate schedule using ASAP times from critical path
    5. Compute idle ratio (comm time / total active time)
    """
    views: dict[int, NodeLocalView] = {}

    # Collect all node IDs
    all_nodes: set[int] = set()
    for task in workload.tasks:
        if task.is_compute() and task.node is not None:
            all_nodes.add(task.node)
        if task.is_flow():
            if task.src is not None:
                all_nodes.add(task.src)
            if task.dst is not None:
                all_nodes.add(task.dst)

    for node_id in all_nodes:
        views[node_id] = NodeLocalView(node_id=node_id)

    # Build lookup sets for flow task IDs per node
    send_task_ids: dict[int, set[int]] = {nid: set() for nid in all_nodes}
    recv_task_ids: dict[int, set[int]] = {nid: set() for nid in all_nodes}

    # Populate views
    for task in workload.tasks:
        if task.is_compute() and task.node is not None:
            node_id = task.node
            views[node_id].compute_tasks.append(task.task_id)
            views[node_id].total_compute_time_us += task.duration_us or 0

        elif task.is_flow():
            if task.task_id not in critical_path.task_timings:
                raise ValueError(
                    f"Task {task.task_id} not found in critical_path.task_timings. "
                    f"Critical path analysis should cover all tasks in the workload."
                )
            timing = critical_path.task_timings[task.task_id]
            size_bytes = task.size_bytes or 0

            # Sender: starts transmitting at earliest_start_us
            if task.src is not None and task.src in views:
                views[task.src].add_send_flow(
                    task.task_id, size_bytes, timing.earliest_start_us
                )
                send_task_ids[task.src].add(task.task_id)

            # Receiver: data arrives at earliest_finish_us
            if task.dst is not None and task.dst in views:
                views[task.dst].add_receive_flow(
                    task.task_id, size_bytes, timing.earliest_finish_us
                )
                recv_task_ids[task.dst].add(task.task_id)

    # Compute idle ratios using flow durations from critical path
    for node_id, view in views.items():
        # All flow task IDs this node participates in (union to avoid double-count)
        node_flow_ids = send_task_ids[node_id] | recv_task_ids[node_id]

        # Sum flow durations from critical path timing
        total_comm_time = 0
        for tid in node_flow_ids:
            timing = critical_path.task_timings[tid]
            total_comm_time += timing.earliest_finish_us - timing.earliest_start_us

        total_time = view.total_compute_time_us + total_comm_time
        if total_time > 0:
            view.estimated_idle_ratio = total_comm_time / total_time

    return views

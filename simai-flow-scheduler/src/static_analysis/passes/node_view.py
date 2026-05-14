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

from ...workload_format.schema import P2PWorkload
from .critical_path import CriticalPathInfo


@dataclass
class NodeLocalView:
    node_id: int
    send_tasks: list[int] = field(default_factory=list)
    total_send_bytes: int = 0
    receive_tasks: list[int] = field(default_factory=list)
    total_receive_bytes: int = 0
    compute_tasks: list[int] = field(default_factory=list)
    total_compute_time_us: int = 0
    estimated_send_times: list[tuple[int, int]] = field(default_factory=list)
    estimated_receive_times: list[tuple[int, int]] = field(default_factory=list)
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
    views: dict[int, NodeLocalView] = {}

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

            if task.src is not None and task.src in views:
                views[task.src].add_send_flow(
                    task.task_id, size_bytes, timing.earliest_start_us
                )
            if task.dst is not None and task.dst in views:
                views[task.dst].add_receive_flow(
                    task.task_id, size_bytes, timing.earliest_finish_us
                )

    for node_id, view in views.items():
        node_flow_ids = set(
            tid for tid in view.send_tasks
        ) | set(tid for tid in view.receive_tasks)

        total_comm_time = 0
        for tid in node_flow_ids:
            timing = critical_path.task_timings[tid]
            total_comm_time += timing.earliest_finish_us - timing.earliest_start_us

        total_time = view.total_compute_time_us + total_comm_time
        if total_time > 0:
            view.estimated_idle_ratio = total_comm_time / total_time

    return views

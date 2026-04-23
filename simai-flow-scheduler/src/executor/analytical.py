"""Analytical Executor - discrete event simulation for P2P workloads."""
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..static_analysis.routing_hints import RoutingHints
from ..static_analysis.topology_loader import NetworkTopology
from ..workload_format.schema import P2PWorkload, Task, TaskType
from .bandwidth import BandwidthAllocator, FairShareAllocator
from .result import ExecutionResult, TaskTiming


@dataclass(order=True)
class Event:
    """事件：按 time 排序，seq 打破平局。"""
    time: int
    seq: int = field(compare=True)
    kind: str = field(compare=False)
    task_id: int = field(compare=False)
    version: int = field(compare=False, default=0)


@dataclass
class ActiveFlow:
    """当前正在传输的流。"""
    task_id: int
    src: int
    dst: int
    size_bytes: int
    remaining_bytes: int
    path: list[int]
    start_time: int
    last_update_time: int
    current_bw_gbps: float = 0.0
    estimated_end_time: int = 0
    version: int = 0


class AnalyticalExecutor:
    """离散事件模拟器：处理 P2PWorkload + ExecutionPlan。"""

    def __init__(
        self,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        allocator: Optional[BandwidthAllocator] = None,
    ):
        self.topology = topology
        self.routing_hints = routing_hints
        self.allocator = allocator or FairShareAllocator()

    def execute(self, workload: P2PWorkload, plan) -> ExecutionResult:
        """
        运行离散事件模拟。

        Args:
            workload: 原始 P2P Workload
            plan: ExecutionPlan（包含每个节点的 compute 排序）

        Returns:
            ExecutionResult
        """
        # ── 构建索引 ──
        task_map: dict[int, Task] = {t.task_id: t for t in workload.tasks}

        # dep_count: 每个 task 还需要多少 dep 完成
        dep_count: dict[int, int] = {}
        # dependents: task A 完成后，哪些 task 的 dep 计数减 1
        dependents: dict[int, list[int]] = defaultdict(list)

        for task in workload.tasks:
            dep_count[task.task_id] = len(task.deps)
            for dep in task.deps:
                dependents[dep].append(task.task_id)

        # compute_cursor: 每个节点当前执行到的 compute_order 位置
        compute_cursor: dict[int, int] = defaultdict(int)

        # 时间记录
        start_times: dict[int, int] = {}
        end_times: dict[int, int] = {}

        # 事件队列和 seq 计数器（用 list 包装以实现可变引用）
        event_queue: list[Event] = []
        seq = [0]

        def next_seq() -> int:
            s = seq[0]
            seq[0] += 1
            return s

        def push_event(time: int, kind: str, task_id: int, version: int = 0):
            heapq.heappush(event_queue, Event(
                time=time, seq=next_seq(), kind=kind,
                task_id=task_id, version=version,
            ))

        # active flows
        active_flows: dict[int, ActiveFlow] = {}

        # ── 初始化事件 ──
        # 每个节点的第一个 compute task（仅当 deps 已满足时）
        for node_id, compute_ids in plan.compute_order.items():
            if compute_ids:
                first_id = compute_ids[0]
                if dep_count[first_id] == 0:
                    push_event(time=0, kind="compute_ready", task_id=first_id)

        # 没有依赖的 flow task
        for task in workload.tasks:
            if task.is_flow() and dep_count[task.task_id] == 0:
                push_event(time=0, kind="flow_ready", task_id=task.task_id)

        # ── 事件循环 ──
        while event_queue:
            event = heapq.heappop(event_queue)
            current_time = event.time

            if event.kind == "compute_ready":
                self._handle_compute_ready(
                    event, task_map, start_times, push_event,
                )

            elif event.kind == "compute_done":
                self._handle_compute_done(
                    event, task_map, end_times, dep_count, dependents,
                    plan, compute_cursor, push_event,
                )

            elif event.kind == "flow_ready":
                self._handle_flow_ready(
                    event, task_map, start_times, active_flows,
                    push_event, current_time,
                )

            elif event.kind == "flow_completion":
                self._handle_flow_completion(
                    event, task_map, active_flows, end_times,
                    dep_count, dependents, push_event, current_time,
                    compute_cursor, plan,
                )

        # ── 构建结果 ──
        return self._build_result(task_map, start_times, end_times)

    # ── 事件处理器 ──

    def _handle_compute_ready(self, event, task_map, start_times, push_event):
        """处理 compute_ready 事件：记录开始时间，安排 compute_done。"""
        task_id = event.task_id
        start_times[task_id] = event.time
        task = task_map[task_id]
        push_event(
            time=event.time + task.duration_us,
            kind="compute_done",
            task_id=task_id,
        )

    def _handle_compute_done(
        self, event, task_map, end_times, dep_count, dependents,
        plan, compute_cursor, push_event,
    ):
        """处理 compute_done 事件：释放下游依赖，推进 compute_cursor。"""
        task_id = event.task_id
        end_times[task_id] = event.time
        task = task_map[task_id]
        node_id = task.node

        # 释放下游依赖
        self._release_dependents(
            task_id, task_map, dep_count, dependents, event.time, push_event,
            compute_cursor=compute_cursor, plan=plan,
        )

        # 推进该节点的 compute_cursor，检查下一个 compute
        compute_cursor[node_id] += 1
        cursor = compute_cursor[node_id]
        order = plan.compute_order.get(node_id, [])
        if cursor < len(order):
            next_task_id = order[cursor]
            if dep_count[next_task_id] == 0:
                push_event(time=event.time, kind="compute_ready", task_id=next_task_id)

    def _handle_flow_ready(
        self, event, task_map, start_times, active_flows,
        push_event, current_time,
    ):
        """处理 flow_ready 事件：创建 ActiveFlow，分配带宽。"""
        task_id = event.task_id
        task = task_map[task_id]
        start_times[task_id] = current_time

        # 查询路径
        path = self.routing_hints.get_path(task.src, task.dst)

        # 创建 ActiveFlow
        flow = ActiveFlow(
            task_id=task_id,
            src=task.src,
            dst=task.dst,
            size_bytes=task.size_bytes,
            remaining_bytes=task.size_bytes,
            path=path,
            start_time=current_time,
            last_update_time=current_time,
        )
        active_flows[task_id] = flow

        # 重新分配带宽（会设置 flow 的 bw 和 estimated_end_time）
        self._reallocate_bandwidth(current_time, active_flows, push_event)

    def _handle_flow_completion(
        self, event, task_map, active_flows, end_times,
        dep_count, dependents, push_event, current_time,
        compute_cursor, plan,
    ):
        """处理 flow_completion 事件：懒删除检查，释放流，触发下游。"""
        task_id = event.task_id
        version = event.version

        # 懒删除检查
        if task_id not in active_flows:
            return
        if active_flows[task_id].version != version:
            return

        end_times[task_id] = current_time
        del active_flows[task_id]

        # 释放下游依赖
        self._release_dependents(
            task_id, task_map, dep_count, dependents, current_time, push_event,
            compute_cursor=compute_cursor, plan=plan,
        )

        # 重新分配剩余 flow 的带宽
        if active_flows:
            self._reallocate_bandwidth(current_time, active_flows, push_event)

    # ── 辅助方法 ──

    def _release_dependents(
        self, task_id, task_map, dep_count, dependents,
        current_time, push_event,
        compute_cursor=None, plan=None,
    ):
        """释放 task_id 的下游依赖：减少 dep_count，触发满足条件的下游 task。"""
        for downstream_id in dependents.get(task_id, []):
            dep_count[downstream_id] -= 1
            if dep_count[downstream_id] == 0:
                downstream_task = task_map[downstream_id]
                if downstream_task.is_flow():
                    push_event(
                        time=current_time, kind="flow_ready",
                        task_id=downstream_id,
                    )
                elif downstream_task.is_compute() and compute_cursor is not None and plan is not None:
                    # 检查该 compute 是否是其节点 compute_order 中的当前待执行任务
                    node_id = downstream_task.node
                    order = plan.compute_order.get(node_id, [])
                    cursor = compute_cursor.get(node_id, 0)
                    if cursor < len(order) and order[cursor] == downstream_id:
                        push_event(
                            time=current_time, kind="compute_ready",
                            task_id=downstream_id,
                        )

    def _reallocate_bandwidth(
        self, current_time, active_flows, push_event,
    ):
        """重新分配带宽：更新 remaining_bytes → 调用 allocator → 安排新的 completion 事件。"""
        flows_list = list(active_flows.values())
        if not flows_list:
            return

        # Step 1: 更新所有 active flow 的 remaining_bytes
        for flow in flows_list:
            elapsed = current_time - flow.last_update_time
            if elapsed > 0 and flow.current_bw_gbps > 0:
                # elapsed 是微秒，bw 是 Gbps
                # transmitted = elapsed_us * 1e-6 * bw_gbps * 1e9 / 8
                transmitted_bytes = int(elapsed * flow.current_bw_gbps * 1e3 / 8)
                flow.remaining_bytes = max(0, flow.remaining_bytes - transmitted_bytes)
            flow.last_update_time = current_time

        # Step 2: 调用 allocator 得到新的带宽分配
        new_bw = self.allocator.allocate(
            flows_list, self.topology, self.routing_hints, current_time,
        )

        # Step 3: 更新每条 flow 的带宽和预计完成时间
        for flow in flows_list:
            flow.current_bw_gbps = new_bw.get(flow.task_id, 0.0)
            flow.version += 1

            if flow.remaining_bytes == 0:
                # 流已真正完成：安排立即完成事件
                flow.estimated_end_time = current_time
                push_event(
                    time=flow.estimated_end_time,
                    kind="flow_completion",
                    task_id=flow.task_id,
                    version=flow.version,
                )
            elif flow.current_bw_gbps > 0:
                # 正常传输：计算传播延迟 + 传输时间
                propagation_delay = self._compute_propagation_delay(flow.path)
                # remaining_bytes * 8 = bits, bw_gbps * 1e3 = bits per us
                transmission_us = int(
                    flow.remaining_bytes * 8 / (flow.current_bw_gbps * 1e3)
                )
                flow.estimated_end_time = current_time + propagation_delay + transmission_us
                push_event(
                    time=flow.estimated_end_time,
                    kind="flow_completion",
                    task_id=flow.task_id,
                    version=flow.version,
                )
            else:
                # 带宽为零：流被暂停，不安排完成事件
                # 等待下次 active_flows 变化时重新分配带宽
                # estimated_end_time 保持不变（不会被使用）
                pass

    def _compute_propagation_delay(self, path: list[int]) -> int:
        """计算路径的总传播延迟（微秒）。"""
        delay = 0
        for i in range(len(path) - 1):
            link = self.topology.get_link(path[i], path[i + 1])
            if link:
                delay += int(link.latency_us)
        return delay

    def _build_result(
        self, task_map, start_times, end_times,
    ) -> ExecutionResult:
        """从时间记录构建 ExecutionResult。"""
        per_task: dict[int, TaskTiming] = {}
        for task_id, start in start_times.items():
            end = end_times.get(task_id, start)
            task = task_map[task_id]
            per_task[task_id] = TaskTiming(
                task_id=task_id,
                node=task.node if task.is_compute() else (task.src or 0),
                task_type="compute" if task.is_compute() else "flow",
                start_time_us=start,
                end_time_us=end,
            )

        # 计算 job_iteration_times
        job_min_max: dict[int, tuple[int, int]] = {}
        for task_id, timing in per_task.items():
            job_id = task_map[task_id].job_id
            if job_id not in job_min_max:
                job_min_max[job_id] = (timing.start_time_us, timing.end_time_us)
            else:
                s, e = job_min_max[job_id]
                job_min_max[job_id] = (
                    min(s, timing.start_time_us),
                    max(e, timing.end_time_us),
                )

        job_iteration_times = {
            jid: end - start for jid, (start, end) in job_min_max.items()
        }

        all_starts = [t.start_time_us for t in per_task.values()]
        all_ends = [t.end_time_us for t in per_task.values()]
        total_time = max(all_ends) if all_ends else 0
        makespan = (max(all_ends) - min(all_starts)) if all_starts and all_ends else 0

        return ExecutionResult(
            per_task=per_task,
            job_iteration_times=job_iteration_times,
            total_time_us=total_time,
            makespan_us=makespan,
        )

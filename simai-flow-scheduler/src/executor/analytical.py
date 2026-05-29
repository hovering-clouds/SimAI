"""Analytical Executor - discrete event simulation for P2P workloads."""
import heapq
from collections import defaultdict
from dataclasses import dataclass, field

from ..static_analysis.passes.topology_loader import NetworkTopology
from ..workload_format.schema import P2PWorkload, Task, TaskType
from .policies.base_policy import SchedulingPolicy
from .result import ExecutionResult, TaskTiming
from .runtime import ActiveFlow

# 带宽变化幅度阈值（百分比）：小于该值时跳过事件推送。设为 0.0 关闭过滤。
BW_CHANGE_THRESHOLD_PCT = 1.0


@dataclass(order=True)
class Event:
    """事件：按 time 排序，seq 打破平局。"""
    time: int
    seq: int = field(compare=True)
    kind: str = field(compare=False)
    task_id: int = field(compare=False)
    version: int = field(compare=False, default=0)


class AnalyticalExecutor:
    """离散事件模拟器：处理 P2PWorkload。

    Executor 拥有事件队列和 DAG 状态；准入、路径查询和带宽分配通过 SchedulingPolicy 委托。
    """

    def __init__(self, topology: NetworkTopology, policy: SchedulingPolicy):
        self.topology = topology
        self.policy = policy

    def execute(self, workload: P2PWorkload) -> ExecutionResult:
        """运行离散事件模拟。"""
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

        active_flows: dict[int, ActiveFlow] = {}
        ready_pool: set[int] = set()

        # ── 初始化策略 ──
        self.policy.initialize(workload, self.topology)

        # ── 初始化 ready_pool：所有无依赖的 task ──
        for task in workload.tasks:
            if dep_count[task.task_id] == 0:
                ready_pool.add(task.task_id)

        # ── 初次 drain ──
        self._drain_ready_pool(
            current_time=0, ready_pool=ready_pool, task_map=task_map,
            start_times=start_times, active_flows=active_flows,
            push_event=push_event,
        )

        # ── 事件循环 ──
        last_time = 0
        total_tasks = len(workload.tasks)
        event_count = 0
        _PROGRESS_INTERVAL = 10000  # 每处理 10K 事件输出一次进度
        while event_queue:
            event = heapq.heappop(event_queue)
            current_time = event.time

            # 时间单调性检查：防止逆向因果（未来事件触发过去时刻的事件）
            assert current_time >= last_time, (
                f"Time went backwards: {last_time} → {current_time} "
                f"(event: {event.kind} task_id={event.task_id})"
            )
            last_time = current_time

            if event.kind == "compute_done":
                self._handle_compute_done(
                    event, task_map, end_times, dep_count, dependents,
                    push_event, start_times, active_flows, ready_pool,
                )

            elif event.kind == "flow_completion":
                self._handle_flow_completion(
                    event, task_map, active_flows, end_times,
                    dep_count, dependents, push_event,
                    start_times, ready_pool,
                )

            event_count += 1
            if event_count % _PROGRESS_INTERVAL == 0:
                pct = len(end_times) / total_tasks * 100
                print(
                    f"  [progress] processed {event_count:,} events, "
                    f"{len(end_times):,}/{total_tasks:,} tasks done "
                    f"({pct:.1f}%)  time={current_time}  "
                    f"queue={len(event_queue):,}",
                    flush=True,
                )

        # ── 死锁检查 ──
        if ready_pool:
            raise RuntimeError(
                f"Deadlock detected: {len(ready_pool)} ready tasks pending "
                f"with empty event queue. Policy: {type(self.policy).__name__}. "
                f"Pending task IDs: {sorted(ready_pool)}"
            )

        # ── 构建结果 ──
        return self._build_result(task_map, start_times, end_times)

    # ── Ready pool ──

    def _drain_ready_pool(
        self, current_time, ready_pool, task_map,
        start_times, active_flows, push_event,
    ):
        """从 ready_pool 获取 task，通过 policy 准入，启动被允许的 task。"""
        while ready_pool:
            ready_tasks = [task_map[tid] for tid in ready_pool]
            emitted_ids = self.policy.emit_ready_tasks(current_time, ready_tasks)
            if not emitted_ids:
                break

            flows_started = False
            for tid in emitted_ids:
                ready_pool.discard(tid)
                task = task_map[tid]
                self.policy.on_task_emitted(current_time, task)

                if task.is_compute():
                    self._start_compute(task, current_time, start_times, push_event)
                elif task.is_flow():
                    self._start_flow(
                        task, current_time, start_times, active_flows, push_event,
                    )
                    flows_started = True

            if flows_started and active_flows:
                self._reallocate_bandwidth(current_time, active_flows, push_event)

    def _mark_task_ready(
        self, task_id, current_time, ready_pool, task_map,
        start_times, active_flows, push_event,
    ):
        """将 task 加入 ready_pool 并立即 drain。"""
        ready_pool.add(task_id)
        self._drain_ready_pool(
            current_time, ready_pool, task_map,
            start_times, active_flows, push_event,
        )

    def _start_compute(self, task, current_time, start_times, push_event):
        """启动 compute task：记录开始时间，安排 compute_done 事件。"""
        start_times[task.task_id] = current_time
        push_event(
            time=current_time + task.duration_us,
            kind="compute_done",
            task_id=task.task_id,
        )

    def _start_flow(
        self, task, current_time, start_times, active_flows, push_event,
    ):
        """启动 flow task：创建 ActiveFlow，0 字节流直接安排完成事件。"""
        start_times[task.task_id] = current_time

        path = self.policy.get_flow_path(task)

        flow = ActiveFlow(
            task_id=task.task_id,
            src=task.src,
            dst=task.dst,
            size_bytes=task.size_bytes,
            remaining_bytes=task.size_bytes,
            path=path,
            start_time=current_time,
            last_update_time=current_time,
        )
        active_flows[task.task_id] = flow

        if not task.size_bytes:
            propagation_delay = self._compute_propagation_delay(path)
            push_event(
                time=current_time + propagation_delay,
                kind="flow_completion",
                task_id=task.task_id,
                version=0,
            )

    # ── 事件处理器 ──

    def _handle_compute_done(
        self, event, task_map, end_times, dep_count, dependents,
        push_event, start_times, active_flows, ready_pool,
    ):
        """处理 compute_done 事件：释放下游依赖，通知策略。"""
        task_id = event.task_id
        end_times[task_id] = event.time
        task = task_map[task_id]

        # 1. 通知策略（推进 cursor + current_layer）
        #    必须在释放下游依赖之前调用，这样带宽重分配时 current_layer 已更新
        self.policy.on_task_completed(event.time, task)

        # 2. 释放下游依赖（触发带宽重分配）
        self._release_dependents(
            task_id, task_map, dep_count, dependents, event.time, ready_pool,
            start_times, active_flows, push_event,
        )

        # 3. drain ready pool（cursor 已推进，下一个 compute 可能可准入）
        self._drain_ready_pool(
            event.time, ready_pool, task_map,
            start_times, active_flows, push_event,
        )

        # 4. 重新分配带宽（current_layer 可能已推进，需重新评估 active flow 的 RLI）
        if active_flows:
            self._reallocate_bandwidth(event.time, active_flows, push_event)

    def _handle_flow_completion(
        self, event, task_map, active_flows, end_times,
        dep_count, dependents, push_event,
        start_times, ready_pool,
    ):
        """处理 flow_completion 事件：懒删除检查，释放流，触发下游。"""
        task_id = event.task_id
        version = event.version

        # 懒删除检查
        if task_id not in active_flows:
            return
        if active_flows[task_id].version != version:
            return

        end_times[task_id] = event.time
        del active_flows[task_id]

        # 1. 释放下游依赖
        self._release_dependents(
            task_id, task_map, dep_count, dependents, event.time, ready_pool,
            start_times, active_flows, push_event,
        )

        # 2. 通知策略
        self.policy.on_task_completed(event.time, task_map[task_id])

        # 3. 重新分配剩余 flow 的带宽
        if active_flows:
            self._reallocate_bandwidth(event.time, active_flows, push_event)

        self._drain_ready_pool(
            event.time, ready_pool, task_map,
            start_times, active_flows, push_event,
        )

    # ── 辅助方法 ──

    def _release_dependents(
        self, task_id, task_map, dep_count, dependents,
        current_time, ready_pool,
        start_times, active_flows, push_event,
    ):
        """释放 task_id 的下游依赖：减少 dep_count，触发满足条件的下游 task。"""
        for downstream_id in dependents.get(task_id, []):
            dep_count[downstream_id] -= 1
            if dep_count[downstream_id] == 0:
                self._mark_task_ready(
                    downstream_id, current_time, ready_pool, task_map,
                    start_times, active_flows, push_event,
                )

    def _reallocate_bandwidth(
        self, current_time, active_flows, push_event,
    ):
        """重新分配带宽：更新 remaining_bytes → 调用 policy → 安排新的 completion 事件。"""
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

        # Step 2: 调用 policy 得到新的带宽分配
        new_bw = self.policy.allocate_bandwidth(current_time, flows_list)

        # Step 3: 更新每条 flow 的带宽和预计完成时间
        for flow in flows_list:
            old_bw = flow.current_bw_gbps
            flow.current_bw_gbps = new_bw.get(flow.task_id, 0.0)

            if flow.remaining_bytes == 0:
                # 传输已完成，正在传播阶段。
                # 旧事件已包含正确的完成时间（transmission_done + propagation_delay），
                # 不递增 version，不推送新事件，让旧事件自然触发。
                pass
            elif flow.current_bw_gbps == old_bw:
                # 带宽未变：已有事件的时间仍然是正确的（时间推导见下面注释），
                # 不递增 version，不推送新事件，让旧事件自然触发。
                # 推导：旧事件时间 = T₁ + prop_delay + R₁ * 8 / (bw * 1e3)
                #       Step 1 衰减后 R₂ = R₁ - elapsed * bw * 1e3 / 8
                #       新事件时间 = T₂ + prop_delay + R₂ * 8 / (bw * 1e3)
                #                  = T₂ + prop_delay + R₁*8/(bw*1e3) - (T₂ - T₁)
                #                  = T₁ + prop_delay + R₁*8/(bw*1e3)
                #                  = 旧事件时间 ✅
                pass
            elif (old_bw > 0 and flow.current_bw_gbps > 0
                  and abs(flow.current_bw_gbps - old_bw) / max(old_bw, flow.current_bw_gbps) * 100
                      <= BW_CHANGE_THRESHOLD_PCT):
                # 变化幅度小于阈值：旧事件的完成时间误差在阈值范围内，不推送新事件。
                pass
            elif flow.current_bw_gbps > 0:
                # 带宽发生变化（0→正值 或 正数→不同正数）：用新事件替换旧事件
                flow.version += 1
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
                # 带宽降到零：流被暂停，递增 version 使旧事件失效，不安排新事件
                flow.version += 1

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

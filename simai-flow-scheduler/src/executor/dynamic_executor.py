"""
DynamicExecutor — extends AnalyticalExecutor with dynamic job expansion.

Inherits all event-processing logic from the parent class.
Adds JobManager + delayed-task queue for on-demand Job expansion.
"""

import heapq
import time
from collections import defaultdict

from ..workload_format.schema import Task
from ..workload_format.compact_workload import (
    CompactWorkload, JobDAG, JobExpansionInfo,
    ExpandedJob, SimulationState, TaskIdAllocator,
)

from .analytical import AnalyticalExecutor, Event, EVENT_BATCH_GAP_US, PROGRESS_INTERVAL
from .job_manager import JobManager
from .job_expander import JobExpander
from .job_policy import JobPolicy
from .runtime import ActiveFlow
from .result import ExecutionResult, TaskTiming
from ..workload_format.compact_workload import expanded_jobs_to_workload


class DynamicExecutor(AnalyticalExecutor):
    """动态 Job 展开执行器。

    继承 AnalyticalExecutor, 复用全部事件处理辅助方法。
    父类不做任何修改。
    """

    def __init__(self, topology, policy, analyzer):
        super().__init__(topology, policy)
        self._analyzer = analyzer
        self._job_manager: JobManager | None = None
        self._task_meta: dict[int, dict] = {}  # task_id → {phase, layer_id, comm_type, src, dst}
        # 渐进构造 ExecutionResult
        self._per_task: dict[int, TaskTiming] = {}
        self._job_iteration_times: dict[int, int] = {}
        self._min_start_us: int = 2 ** 63
        self._max_end_us: int = 0

    # ── Main entry ──────────────────────────────────────────────────────────

    def execute_dynamic(
        self,
        job_dag: JobDAG,
        job_expansion_info: dict[int, JobExpansionInfo],
        job_policy: JobPolicy,
        job_expander: JobExpander,
    ) -> ExecutionResult:
        """动态模式入口。"""
        # 1. Build JobManager
        self._job_manager = JobManager(
            job_dag=job_dag,
            job_expansion_info=job_expansion_info,
            job_policy=job_policy,
            job_expander=job_expander,
        )

        # ------------------------------------------------------------------
        # 2. Local state — same structure as parent's execute()
        # ------------------------------------------------------------------
        task_map: dict[int, Task] = {}
        dep_count: dict[int, int] = {}
        dependents: dict[int, list[int]] = defaultdict(list)
        start_times: dict[int, int] = {}
        end_times: dict[int, int] = {}
        event_queue: list[Event] = []
        seq = [0]
        active_flows: dict[int, ActiveFlow] = {}
        ready_pool: set[int] = set()
        delayed_queue: list[tuple[int, int]] = []  # (release_time, task_id) min-heap

        def next_seq() -> int:
            s = seq[0]; seq[0] += 1; return s

        def push_event(time: int, kind: str, task_id: int, version: int = 0):
            heapq.heappush(event_queue, Event(
                time=time, seq=next_seq(), kind=kind,
                task_id=task_id, version=version,
            ))

        # 3. Expand root jobs → build initial workload → init policy → analyze & inject
        root_jobs = self._job_manager.initialize()
        if not root_jobs:
            return self._build_result(task_map, start_times, end_times)

        init_wl = expanded_jobs_to_workload(root_jobs)
        self.policy.initialize(init_wl, self.topology)

        for ej in root_jobs:
            self._analyze_and_inject(
                ej,
                delayed_queue, task_map, dep_count, dependents, ready_pool, 0,
            )

        # 4. Initial drain
        self._release_delayed(delayed_queue, 0, ready_pool)
        self._drain_ready_pool(0, ready_pool, task_map,
                               start_times, active_flows, push_event)

        # 5. Event loop
        last_time = 0
        total_injected = len(task_map)
        event_count = 0
        _start_time = time.monotonic()

        while event_queue or ready_pool or delayed_queue or not self._job_manager.is_all_jobs_done():

            if not event_queue:
                # runtime deadlock check
                if ready_pool:
                    raise RuntimeError(
                        f"Deadlock detected: {len(ready_pool)} ready tasks pending "
                        f"with empty event queue. Policy: {type(self.policy).__name__}. "
                        f"Pending task IDs: {sorted(ready_pool)}"
                    )
                # ── No events or ready tasks → try expand eligible or jump delayed ──
                if delayed_queue:
                    next_time = delayed_queue[0][0]
                    self._release_delayed(delayed_queue, next_time, ready_pool)
                    self._drain_ready_pool(next_time, ready_pool, task_map,
                                        start_times, active_flows, push_event)
                    continue
                # Expand next eligible job if any remain (currently not necessary, but may be useful for future dynamic injection policies)
                sim_state = SimulationState(current_time_us=last_time)
                new_jobs = self._job_manager.try_expand_eligible(sim_state)
                if new_jobs:
                    for ej in new_jobs:
                        self._analyze_and_inject(
                            ej, delayed_queue, task_map, dep_count,
                            dependents, ready_pool, last_time,
                        )
                    self._release_delayed(delayed_queue, last_time, ready_pool)
                    self._drain_ready_pool(last_time, ready_pool, task_map,
                                           start_times, active_flows, push_event)
                    continue
                break

            # ── Process events ──
            current_time = event_queue[0].time
            assert current_time >= last_time
            last_time = current_time

            self._skip_reallocate = True
            batch_completed: set[int] = set()
            while True:
                batch_events = []
                while event_queue and event_queue[0].time - current_time <= EVENT_BATCH_GAP_US:
                    batch_events.append(heapq.heappop(event_queue))
                if not batch_events:
                    break
                for event in batch_events:
                    if event.kind == "compute_done":
                        self._handle_compute_done(
                            event, task_map, end_times, dep_count, dependents,
                            push_event, start_times, active_flows, ready_pool,
                        )
                        batch_completed.add(event.task_id)
                    elif event.kind == "flow_completion":
                        self._handle_flow_completion(
                            event, task_map, active_flows, end_times,
                            dep_count, dependents, push_event,
                            start_times, ready_pool,
                        )
                        batch_completed.add(event.task_id)

                self._release_delayed(delayed_queue, current_time, ready_pool)
                self._drain_ready_pool(current_time, ready_pool, task_map,
                                       start_times, active_flows, push_event)

                event_count += len(batch_events)

            self._skip_reallocate = False
            if active_flows:
                self._reallocate_bandwidth(current_time, active_flows, push_event)

            # ★ Dynamic injection trigger ★
            sim_state = SimulationState(current_time_us=current_time)
            new_batches = self._job_manager.on_tasks_completed(
                batch_completed, sim_state,
            )
            # 展开后继 jobs
            for ej in new_batches:
                n = self._analyze_and_inject(
                    ej,
                    delayed_queue, task_map, dep_count, dependents,
                    ready_pool, current_time,
                )
                total_injected += n
            # 回收已完成 Job 的 task details（无论是否有后继 jobs）
            self._reclaim_completed(
                task_map, dep_count, dependents, start_times, end_times,
            )
            self._release_delayed(delayed_queue, current_time, ready_pool)
            self._drain_ready_pool(current_time if current_time > 0 else 0,
                                   ready_pool, task_map,
                                   start_times, active_flows, push_event)

            # Progress
            completed_count = len(self._per_task)
            if event_count % PROGRESS_INTERVAL == 0:
                pct = completed_count / total_injected * 100 if total_injected else 0
                elapsed = time.monotonic() - _start_time
                eta = elapsed / max(pct, 0.1) * (100 - pct) if pct > 0 else 0
                print(
                    f"\r  [progress elps/eta={elapsed:.0f}s/{eta:.0f}s] "
                    f"jobs={len(self._job_manager.completed_jobs)}/"
                    f"{len(self._job_manager._dag.jobs)} "
                    f"{completed_count:,}/{total_injected:,} tasks "
                    f"({pct:.1f}%)  t={current_time:,} us",
                    end="\r", flush=True,
                )

        print()

        return ExecutionResult(
            per_task=self._per_task,
            job_iteration_times=self._job_iteration_times,
            total_time_us=self._max_end_us,
            makespan_us=self._max_end_us - self._min_start_us,
        )

    # ── Dynamic helpers ─────────────────────────────────────────────────────

    def _analyze_and_inject(
        self,
        ej: ExpandedJob,
        delayed_queue, task_map, dep_count, dependents, ready_pool, current_time,
    ) -> int:
        """分析一个 ExpandedJob 的 tasks，合并到 policy，注入 executor。"""
        # 1. Static analysis on the job's tasks
        mini_wl = expanded_jobs_to_workload([ej])
        result = self._analyzer.analyze(mini_wl)
        self.policy.update_analysis(mini_wl, result)

        # 2. Record task metadata for visualization
        for t in ej.tasks:
            self._task_meta[t.task_id] = {
                k: getattr(t, k, None)
                for k in ("phase", "layer_id", "comm_type", "src", "dst", "node")
            }

        # 3. Inject tasks (with delay if applicable)
        entry_ids = set(ej.entry_task_ids)
        for task in ej.tasks:
            tid = task.task_id
            task_map[tid] = task
            dep_count[tid] = len(task.deps)
            for dep in task.deps:
                dependents[dep].append(tid)

            if dep_count[tid] == 0:
                if ej.delay_us > 0 and tid in entry_ids:
                    heapq.heappush(delayed_queue, (current_time + ej.delay_us, tid))
                else:
                    ready_pool.add(tid)
        return len(ej.tasks)

    def _reclaim_completed(
        self,
        task_map, dep_count, dependents, start_times, end_times,
    ):
        """回收已完成 Job 的 task details，转为 TaskTiming 存入 _per_task。"""
        for job_id, task_ids in self._job_manager.drain_completed():
            job_start = 2 ** 63
            job_end = 0

            for tid in task_ids:
                task = task_map.get(tid)
                if task is None:
                    continue
                st = start_times.pop(tid, 0)
                et = end_times.pop(tid, 0)
                tt = TaskTiming(
                    task_id=tid,
                    node=task.node if task.is_compute() else (task.src or 0),
                    task_type="compute" if task.is_compute() else "flow",
                    start_time_us=st,
                    end_time_us=et,
                )
                self._per_task[tid] = tt
                job_start = min(job_start, st)
                job_end = max(job_end, et)

                # Clean up
                task_map.pop(tid, None)
                dep_count.pop(tid, None)
                dependents.pop(tid, None)
                self._task_meta.pop(tid, None)

            self._job_iteration_times[job_id] = job_end - job_start
            self._min_start_us = min(self._min_start_us, job_start)
            self._max_end_us = max(self._max_end_us, job_end)

    def _release_delayed(
        self,
        delayed_queue: list[tuple[int, int]],
        current_time: int,
        ready_pool: set[int],
    ):
        """释放到期的延迟 tasks 到 ready_pool。"""
        while delayed_queue and delayed_queue[0][0] <= current_time:
            _, tid = heapq.heappop(delayed_queue)
            ready_pool.add(tid)



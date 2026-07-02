"""
JobManager — manages Job lifecycle in dynamic mode.

Tracks active/completed Jobs, checks JobDAG for newly eligible Jobs,
and coordinates expansion via JobExpander + JobPolicy.
"""

from ..workload_format.schema import Job, Task
from ..workload_format.compact_workload import (
    JobDAG, JobExpansionInfo, ExpandedJob, SimulationState,
)
from .job_expander import JobExpander
from .job_policy import JobPolicy


class JobManager:
    """管理 JobDAG 中 Jobs 的生命周期。

    展开 → 注入 → 监视完成 → 回收 → 展开后继。

    Job 完成检测使用反向索引 O(1) 查表，而非 O(N) 扫全量 terminal task。
    """

    def __init__(
        self,
        job_dag: JobDAG,
        job_expansion_info: dict[int, JobExpansionInfo],
        job_policy: JobPolicy,
        job_expander: JobExpander,
    ):
        self._dag: JobDAG = job_dag
        self._expansion_info: dict[int, JobExpansionInfo] = job_expansion_info
        self._policy: JobPolicy = job_policy
        self._expander: JobExpander = job_expander

        # Runtime state
        self.completed_jobs: set[int] = set()
        self._active_jobs: set[int] = set()                     # active job IDs
        # 已完成但尚未回收的job信息 
        self._job_task_ids: dict[int, set[int]] = {}            # job_id → all task_ids

        # Job 完成检测（反向索引，O(1) per task）
        self._terminal_to_job: dict[int, int] = {}              # terminal_task_id → job_id
        self._remaining_terminal: dict[int, int] = {}            # job_id → 剩余 terminal 数

    # ── Public API ──────────────────────────────────────────────────────────

    def initialize(self) -> list[ExpandedJob]:
        """展开所有 root jobs, 返回初始 ExpandedJob 列表。"""
        eligible = self._dag.get_eligible_jobs()
        return self._expand_selected(eligible, SimulationState(current_time_us=0))

    def on_tasks_completed(
        self,
        completed_task_ids: set[int],
        sim_state: SimulationState,
    ) -> list[ExpandedJob]:
        """报告一批 tasks 完成：检查完成 → 回收 → 展开新 eligible Jobs。

        completed_task_ids 可以是增量批次或累积全集——反向索引保证每个
        terminal task 只被消费一次。
        """
        newly_done: list[int] = []
        for tid in completed_task_ids:
            job_id = self._terminal_to_job.pop(tid, None)
            if job_id is not None:
                remaining = self._remaining_terminal[job_id] - 1
                if remaining <= 0:
                    newly_done.append(job_id)
                    del self._remaining_terminal[job_id]
                else:
                    self._remaining_terminal[job_id] = remaining

        if not newly_done:
            return []

        for job_id in newly_done:
            self.completed_jobs.add(job_id)
            self._active_jobs.discard(job_id)
            self._dag.mark_completed(job_id)

        eligible = self._filter_eligible(self._dag.get_eligible_jobs())
        if not eligible:
            return []

        return self._expand_selected(eligible, sim_state)

    def is_all_jobs_done(self) -> bool:
        return len(self.completed_jobs) == len(self._dag.jobs)

    def try_expand_eligible(self, sim_state: SimulationState) -> list[ExpandedJob]:
        """展开所有符合条件的 eligible jobs（事件队列空时调用）。"""
        eligible = self._filter_eligible(self._dag.get_eligible_jobs())
        if not eligible:
            return []
        return self._expand_selected(eligible, sim_state)

    def drain_completed(self) -> list[tuple[int, set[int]]]:
        """返回本轮新完成的 (job_id, task_ids) 列表，供 DynamicExecutor 回收。"""
        done = [(jid, self._job_task_ids.pop(jid))
                for jid in list(self.completed_jobs)
                if jid in self._job_task_ids]
        return done

    # ── Internal ────────────────────────────────────────────────────────────

    def _filter_eligible(self, candidate_ids: list[int]) -> list[int]:
        return [
            jid for jid in candidate_ids
            if jid not in self.completed_jobs and jid not in self._active_jobs
        ]

    def _expand_selected(
        self, job_ids: list[int], sim_state: SimulationState,
    ) -> list[ExpandedJob]:
        """按 JobPolicy 决策展开一组 Jobs。"""
        if not job_ids:
            return []

        specs = [self._dag.jobs[jid] for jid in job_ids]
        ordered = self._policy.order_expansion(specs, sim_state)

        results: list[ExpandedJob] = []
        for job in ordered:
            if not self._policy.can_emit(job, sim_state):
                continue

            job.assigned_nodes = self._policy.get_placement(job, sim_state)
            info = self._expansion_info[job.job_id]
            expanded = self._expander.expand_job(job, info)

            # Register runtime state
            self._active_jobs.add(job.job_id)
            self._job_task_ids[job.job_id] = {t.task_id for t in expanded.tasks}

            # Build terminal task index for O(1) completion detection
            for tid in expanded.terminal_task_ids:
                self._terminal_to_job[tid] = job.job_id
            self._remaining_terminal[job.job_id] = len(expanded.terminal_task_ids)

            # Delay
            expanded.delay_us = self._policy.get_delay_us(job, sim_state)
            results.append(expanded)

        return results

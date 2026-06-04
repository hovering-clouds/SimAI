"""
JobExpander — expands a single Job (Job + JobExpansionInfo) into an ExpandedJob.

Reuses existing expanders (InferenceTraceExpander, WorkloadBuilder) to generate
tasks on demand during dynamic simulation.
"""

import json

from ..workload_format.schema import Job, Task
from ..workload_format.compact_workload import (
    JobExpansionInfo, ExpandedJob, TaskIdAllocator,
)
from ..workload_generator.inference_trace_expander import InferenceTraceExpander
from ..workload_generator.inference_profile import InferenceProfileStore
from ..workload_generator.workload_builder import WorkloadBuilder
from ..workload_generator.aicb_parser import AicbParser
from ..static_analysis.passes.topology_loader import NetworkTopology


class JobExpander:
    """Expands a single Job (Job + JobExpansionInfo) into ExpandedJob on demand.

    Holds a trace cache to avoid re-parsing the same trace file.
    """

    def __init__(
        self,
        task_id_allocator: TaskIdAllocator,
        profile_store: InferenceProfileStore,
        topology: NetworkTopology,
    ):
        self._allocator = task_id_allocator
        self._profile_store = profile_store
        self._topology = topology
        self._trace_cache: dict[str, dict] = {}

    def _load_trace(self, trace_src: str) -> dict:
        """Load a trace JSON file, caching by path."""
        if trace_src not in self._trace_cache:
            with open(trace_src) as f:
                self._trace_cache[trace_src] = json.load(f)
        return self._trace_cache[trace_src]

    # ── Dispatch ──────────────────────────────────────────────────────────────

    def expand_job(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        """Expand a single Job into tasks."""
        if info.job_type == "training":
            return self._expand_training(job, info)
        return self._expand_inference(job, info)

    # ── Inference ─────────────────────────────────────────────────────────────

    def _expand_inference(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        """Expand one inference batch into an ExpandedJob."""
        trace = self._load_trace(info.trace_src)
        batch = trace["batches"][info.trace_job_index]

        expander = InferenceTraceExpander(
            profile_store=self._profile_store,
            tp=job.parallelism.tp,
            ep=job.parallelism.ep,
            pp=job.parallelism.pp,
            assigned_nodes=job.assigned_nodes or None,
        )

        batch_lookup = {b["batch_id"]: b for b in trace["batches"]}
        total_layers = expander._get_total_layers(trace)
        resolved, reuse_info = expander._setup_kv_reuse(trace, None)

        tasks, exits, infos, _ = expander.expand_single_batch(
            batch=batch,
            batch_lookup=batch_lookup,
            batch_exits={},       # dynamic mode: no predecessor batch tasks
            job_id=job.job_id,
            task_id_start=self._allocator.next_id,
            total_layers=total_layers,
            trace=trace,
            request_reuse_info=reuse_info,
            resolved_storage_node_ids=resolved,
        )

        self._allocator.allocate(len(tasks))
        terminal_ids = sorted({
            tid for tids in exits.values() for tid in tids
        })
        entry_ids = sorted({
            t.task_id for t in tasks
            if not t.deps
        })

        return ExpandedJob(
            job_id=job.job_id,
            tasks=[t.to_task() for t in tasks],
            entry_task_ids=entry_ids,
            terminal_task_ids=terminal_ids,
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def _expand_training(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        """Expand one training iteration (full AICB trace) into an ExpandedJob."""
        # Re-parse the AICB trace to get header + items
        header, items = AicbParser().parse(info.trace_src)

        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(
            aicb_header=header,
            aicb_items=items,
            job=job,
        )

        # Reassign task IDs with global allocator
        offset = self._allocator.next_id
        self._allocator.allocate(len(workload.tasks))
        for i, task in enumerate(workload.tasks):
            old_id = task.task_id
            new_id = offset + i
            task.task_id = new_id
            task.deps = [d + offset for d in task.deps]

        # Find entry tasks (no deps) and terminal tasks (no dependents)
        all_ids = {t.task_id for t in workload.tasks}
        has_dependents: set[int] = set()
        for t in workload.tasks:
            has_dependents.update(t.deps)
        entry_ids = sorted(all_ids - has_dependents)
        terminal_ids = sorted(task_id for t in workload.tasks
                              for task_id in t.deps if task_id not in all_ids)

        return ExpandedJob(
            job_id=job.job_id,
            tasks=workload.tasks,
            entry_task_ids=entry_ids,
            terminal_task_ids=terminal_ids,
        )

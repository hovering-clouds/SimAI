"""Dynamic training expansion with strategy-specific pipeline DAG overlays."""

from __future__ import annotations

from ..workload_format.compact_workload import ExpandedJob, JobExpansionInfo
from ..workload_format.schema import Job, Meta, P2PWorkload
from ..workload_generator.pipeline_workload_overlay import (
    BidirectionalPipelineTaskInfo,
    InterleavedPipelineTaskInfo,
    apply_pipeline_workload_overlay,
)
from .job_expander import JobExpander


class PipelineJobExpander(JobExpander):
    """Wrap ``JobExpander`` and transform training jobs before injection.

    The wrapper owns its overlay sidecar and reserves extra IDs through the
    same global allocator used by the base expander.  Inference expansion and
    pipeline modes without a workload overlay are unchanged.
    """

    def __init__(
        self,
        *args,
        pipeline_mode: str,
        pipeline_vpp: int = 2,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipeline_mode = pipeline_mode
        self.pipeline_vpp = pipeline_vpp
        self.pipeline_task_info: dict[
            int, InterleavedPipelineTaskInfo | BidirectionalPipelineTaskInfo
        ] = {}

    def expand_job(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        expanded = super().expand_job(job, info)
        if info.job_type != "training" or self.pipeline_mode not in {
            "interleaved_1f1b", "bidirectional",
        }:
            return expanded

        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(job.assigned_nodes)),
            jobs=[job],
            tasks=expanded.tasks,
        )
        overlay = apply_pipeline_workload_overlay(
            self.pipeline_mode,
            workload,
            virtual_pipeline_size=self.pipeline_vpp,
            allocate_task_ids=self._allocate_ids,
        )
        self.pipeline_task_info.update(overlay.task_info)
        tasks = overlay.workload.tasks
        entry_ids = sorted(task.task_id for task in tasks if not task.deps)
        has_dependents = {
            dependency for task in tasks for dependency in task.deps
        }
        terminal_ids = sorted(
            task.task_id for task in tasks if task.task_id not in has_dependents
        )
        return ExpandedJob(
            job_id=expanded.job_id,
            tasks=tasks,
            entry_task_ids=entry_ids,
            terminal_task_ids=terminal_ids,
            delay_us=expanded.delay_us,
        )

    def _allocate_ids(self, count: int) -> list[int]:
        start, end = self._allocator.allocate(count)
        return list(range(start, end + 1))

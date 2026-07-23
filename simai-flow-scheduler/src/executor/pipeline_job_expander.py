"""Dynamic training expansion with strategy-specific direct pipeline DAGs."""

from __future__ import annotations

from dataclasses import replace

from ..workload_format.compact_workload import ExpandedJob, JobExpansionInfo
from ..workload_format.schema import Job
from ..workload_generator.aicb_parser import AicbParser
from ..workload_generator.interleaved_pipeline_builder import (
    InterleavedPipelineWorkloadBuilder,
)
from ..workload_generator.zero_bubble_pipeline_builder import (
    ZeroBubblePipelineWorkloadBuilder,
)
from ..workload_generator.bidirectional_pipeline_builder import (
    BidirectionalPipelineWorkloadBuilder,
)
from .job_expander import JobExpander


class PipelineJobExpander(JobExpander):
    """Expand advanced training pipelines before dynamic task injection.

    The wrapper owns the shared expansion sidecar and reserves all task IDs
    through the same global allocator used by the base expander. Inference
    expansion and pipeline modes without a strategy builder are unchanged.
    """

    def __init__(
        self,
        *args,
        pipeline_mode: str,
        pipeline_vpp: int = 2,
        pipeline_gradient_sync_bytes: int | None = None,
        pipeline_task_info: dict[int, object] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.pipeline_mode = pipeline_mode
        self.pipeline_vpp = pipeline_vpp
        self.pipeline_gradient_sync_bytes = pipeline_gradient_sync_bytes
        self.pipeline_task_info = (
            pipeline_task_info if pipeline_task_info is not None else {}
        )

    def expand_job(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        if info.job_type == "training" and self.pipeline_mode in {
            "interleaved_1f1b", "zero_bubble", "bidirectional",
        }:
            return self._expand_direct_training(job, info)
        return super().expand_job(job, info)

    def _expand_direct_training(
        self,
        job: Job,
        info: JobExpansionInfo,
    ) -> ExpandedJob:
        header, items = AicbParser().parse(info.trace_src)
        local_info: dict[int, object] = {}
        if self.pipeline_mode == "interleaved_1f1b":
            builder = InterleavedPipelineWorkloadBuilder(
                self.pipeline_vpp,
                local_info,
            )
        elif self.pipeline_mode == "zero_bubble":
            builder = ZeroBubblePipelineWorkloadBuilder(local_info)
        else:
            builder = BidirectionalPipelineWorkloadBuilder(
                local_info,
                gradient_sync_bytes=self.pipeline_gradient_sync_bytes,
            )
        workload = builder.build_from_aicb(header, items, job)

        offset = self._allocator.next_id
        self._allocator.allocate(len(workload.tasks))
        old_to_new = {
            task.task_id: offset + index
            for index, task in enumerate(workload.tasks)
        }
        for task in workload.tasks:
            old_id = task.task_id
            task.task_id = old_to_new[old_id]
            task.deps = [old_to_new[dependency] for dependency in task.deps]

        for old_id, metadata in local_info.items():
            new_id = old_to_new[old_id]
            self.pipeline_task_info[new_id] = replace(
                metadata,
                task_id=new_id,
            )

        entry_ids = sorted(task.task_id for task in workload.tasks if not task.deps)
        has_dependents = {
            dependency for task in workload.tasks for dependency in task.deps
        }
        terminal_ids = sorted(
            task.task_id for task in workload.tasks
            if task.task_id not in has_dependents
        )
        return ExpandedJob(
            job_id=job.job_id,
            tasks=workload.tasks,
            entry_task_ids=entry_ids,
            terminal_task_ids=terminal_ids,
        )

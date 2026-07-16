"""On-demand AICB training expansion with Hermod metadata enrichment."""
from pathlib import Path

from ..workload_format.compact_workload import ExpandedJob, JobExpansionInfo, TaskIdAllocator
from ..workload_format.schema import Job
from ..workload_generator.aicb_parser import AicbParser
from ..workload_generator.hermod_aicb_metadata import HermodAicbMetadataAdapter
from ..workload_generator.workload_builder import WorkloadBuilder


class HermodTrainingJobExpander:
    """Expand one complete AICB training iteration on demand.

    ``JobExpansionInfo.trace_src`` identifies the AICB workload.  Each dynamic
    Job receives globally unique task IDs, while WorkloadBuilder's coflow IDs
    stay unique because they embed the dynamic job ID.
    """

    def __init__(self, task_id_allocator: TaskIdAllocator):
        self._allocator = task_id_allocator
        self._aicb_cache: dict[str, tuple] = {}

    def _load_aicb(self, path: str):
        if path not in self._aicb_cache:
            self._aicb_cache[path] = AicbParser().parse(Path(path))
        return self._aicb_cache[path]

    def expand_job(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        if info.job_type != "training":
            raise ValueError(f"Hermod dynamic expander does not support {info.job_type!r}")
        header, items = self._load_aicb(info.trace_src)
        workload = WorkloadBuilder().build_from_aicb(
            aicb_header=header,
            aicb_items=items,
            job=job,
            comm_algo="ring",
        )
        HermodAicbMetadataAdapter(header, items).apply(workload)

        offset = self._allocator.next_id
        self._allocator.allocate(len(workload.tasks))
        for index, task in enumerate(workload.tasks):
            task.task_id = offset + index
            task.deps = [dependency + offset for dependency in task.deps]

        entry_ids = sorted(task.task_id for task in workload.tasks if not task.deps)
        all_ids = {task.task_id for task in workload.tasks}
        non_terminal = {dependency for task in workload.tasks for dependency in task.deps}
        return ExpandedJob(
            job_id=job.job_id,
            tasks=workload.tasks,
            entry_task_ids=entry_ids,
            terminal_task_ids=sorted(all_ids - non_terminal),
        )

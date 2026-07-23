"""Routing, compute serialization, and Hermod §4.1 priority analysis."""
from dataclasses import dataclass
from pathlib import Path

from ..passes.hermod_metadata import HermodAicbMetadataAdapter, HermodMetadataRecord
from ..passes.hermod_priority import (
    HermodPriorityAnalysis, HermodScheduleVariant, HermodEpMode, classify_coflow_type,
)
from ..passes.routing import BfsStrategy, RouteTable
from ..passes.pipeline_task_serializers import (
    PipelineTaskInfo,
    build_pipeline_serializer,
)
from ..passes.task_serializer import ExecutionPlan
from ..passes.topology_loader import NetworkTopology
from ...workload_format.schema import P2PWorkload
from ...workload_generator.aicb_parser import AicbHeader, AicbParser, AicbWorkItem


@dataclass
class HermodAnalysisResult:
    route_table: RouteTable
    execution_plan: ExecutionPlan
    priority_analysis: HermodPriorityAnalysis


def build_hermod_records(
    workload: P2PWorkload,
    header: AicbHeader,
    aicb_items: list[AicbWorkItem] | None = None,
    pipeline_mode: str | None = None,
) -> dict[int, HermodMetadataRecord]:
    """Build Hermod metadata sidecar from a workload and its AICB source.

    Usage:
        records = build_hermod_records(workload, header, items)
        analysis = HermodAnalyzer(...).analyze(workload, hermod_records=records)
    """
    return HermodAicbMetadataAdapter(header, aicb_items).apply(
        workload,
        split_pp_by_layer=pipeline_mode == "interleaved_1f1b",
        split_pp_by_direction=pipeline_mode == "bidirectional",
    )


class HermodAnalyzer:
    def __init__(self, topology: NetworkTopology,
                 variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
                 ep_mode: HermodEpMode = HermodEpMode.REJECT,
                 pipeline_mode: str = "1f1b",
                 pipeline_vpp: int = 2,
                 interleave_group_size: int | None = None):
        self.topology = topology
        self.variant = variant
        self.ep_mode = ep_mode
        self.pipeline_mode = pipeline_mode
        self.pipeline_vpp = pipeline_vpp
        self.interleave_group_size = interleave_group_size
        self.pipeline_task_info: dict[int, PipelineTaskInfo] = {}

    def analyze(self, workload: P2PWorkload,
                hermod_records: dict[int, HermodMetadataRecord]) -> HermodAnalysisResult:
        serializer = build_pipeline_serializer(
            self.pipeline_mode,
            workload,
            virtual_pipeline_size=self.pipeline_vpp,
            interleave_group_size=self.interleave_group_size,
        )
        execution_plan = serializer.serialize(workload)
        pipeline_task_info = dict(getattr(serializer, "task_info", {}))
        self.pipeline_task_info.update(pipeline_task_info)
        return HermodAnalysisResult(
            route_table=BfsStrategy().compute_routes(workload, self.topology),
            execution_plan=execution_plan,
            priority_analysis=HermodPriorityAnalysis.from_workload(
                workload, hermod_records, self.variant, self.ep_mode),
        )


class HermodDynamicAnalyzer(HermodAnalyzer):
    """Hermod analyzer for DynamicExecutor mini-workloads.

    DynamicExecutor supplies tasks without their Job objects.  Retaining a
    registry lets the 1F1B serializer recover each injected job's PP stages.
    The trace_src_by_job mapping lets the analyzer reconstruct Hermod metadata
    from the original AICB source files.
    """

    def __init__(self, topology: NetworkTopology, jobs_by_id: dict[int, object],
                 trace_src_by_job: dict[int, str],
                 variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
                 ep_mode: HermodEpMode = HermodEpMode.REJECT,
                 pipeline_mode: str = "1f1b",
                 pipeline_vpp: int = 2,
                 interleave_group_size: int | None = None):
        super().__init__(
            topology, variant, ep_mode, pipeline_mode,
            pipeline_vpp, interleave_group_size,
        )
        self.jobs_by_id = jobs_by_id
        self._trace_src_by_job = trace_src_by_job
        self._aicb_cache: dict[str, tuple[AicbHeader, list[AicbWorkItem]]] = {}

    def analyze(self, workload: P2PWorkload) -> HermodAnalysisResult:
        job_ids = {task.job_id for task in workload.tasks}
        missing = sorted(job_ids - self.jobs_by_id.keys())
        if missing:
            raise ValueError(f"Hermod dynamic analysis lacks jobs for task job IDs: {missing}")
        workload.jobs = [self.jobs_by_id[job_id] for job_id in sorted(job_ids)]
        records = self._build_hermod_records(workload)
        return super().analyze(workload, hermod_records=records)

    def _build_hermod_records(self, workload) -> dict[int, HermodMetadataRecord]:
        """Build Hermod sidecar from AICB source files (parsed on demand)."""
        for task in workload.get_flow_tasks():
            if classify_coflow_type(task.comm_type) is None:
                continue
            src = self._trace_src_by_job.get(task.job_id)
            if src is None:
                return {}
            if src not in self._aicb_cache:
                self._aicb_cache[src] = AicbParser().parse(Path(src))
            header, items = self._aicb_cache[src]
            return build_hermod_records(
                workload, header, items, pipeline_mode=self.pipeline_mode,
            )
        return {}

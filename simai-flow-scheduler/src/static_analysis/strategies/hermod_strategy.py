"""Routing, compute serialization, and Hermod §4.1 priority analysis."""
from dataclasses import dataclass
from ..passes.routing import BfsStrategy, RouteTable
from ..passes.task_serializer import CppReferenceSerializer, OneFOneBSerializer, ExecutionPlan
from ..passes.topology_loader import NetworkTopology
from ..passes.hermod_priority import HermodPriorityAnalysis, HermodScheduleVariant, HermodEpMode
from ...workload_format.schema import P2PWorkload
from ...workload_generator.rank_grouper import RankGrouper


@dataclass
class HermodAnalysisResult:
    route_table: RouteTable
    execution_plan: ExecutionPlan
    priority_analysis: HermodPriorityAnalysis


class HermodAnalyzer:
    def __init__(self, topology: NetworkTopology,
                 variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
                 ep_mode: HermodEpMode = HermodEpMode.REJECT,
                 pipeline_mode: str = "1f1b"):
        self.topology = topology
        self.variant = variant
        self.ep_mode = ep_mode
        self.pipeline_mode = pipeline_mode

    def analyze(self, workload: P2PWorkload) -> HermodAnalysisResult:
        node_to_stage = {}
        pp = 1
        for job in workload.jobs:
            grouper = RankGrouper(job.assigned_nodes, job.parallelism)
            pp = max(pp, grouper.pp)
            stage_size = grouper.dp * grouper.ep * grouper.tp
            for stage_id in range(grouper.pp):
                for node in grouper.nodes[stage_id * stage_size:(stage_id + 1) * stage_size]:
                    node_to_stage[node] = stage_id
        if self.pipeline_mode == "gpipe":
            execution_plan = CppReferenceSerializer().serialize(workload)
        elif self.pipeline_mode == "1f1b":
            execution_plan = OneFOneBSerializer(pp, node_to_stage).serialize(workload)
        else:
            raise ValueError(
                f"Unsupported Hermod pipeline mode {self.pipeline_mode!r}. "
                "Register a serializer before exposing a new mode."
            )
        return HermodAnalysisResult(
            route_table=BfsStrategy().compute_routes(workload, self.topology),
            execution_plan=execution_plan,
            priority_analysis=HermodPriorityAnalysis.from_workload(
                workload, self.variant, self.ep_mode),
        )


class HermodDynamicAnalyzer(HermodAnalyzer):
    """Hermod analyzer for DynamicExecutor mini-workloads.

    DynamicExecutor supplies tasks without their Job objects.  Retaining a
    registry lets the 1F1B serializer recover each injected job's PP stages.
    """

    def __init__(self, topology: NetworkTopology, jobs_by_id: dict[int, object],
                 variant: HermodScheduleVariant = HermodScheduleVariant.CONVENTIONAL_1F1B,
                 ep_mode: HermodEpMode = HermodEpMode.REJECT,
                 pipeline_mode: str = "1f1b"):
        super().__init__(topology, variant, ep_mode, pipeline_mode)
        self.jobs_by_id = jobs_by_id

    def analyze(self, workload: P2PWorkload) -> HermodAnalysisResult:
        job_ids = {task.job_id for task in workload.tasks}
        missing = sorted(job_ids - self.jobs_by_id.keys())
        if missing:
            raise ValueError(f"Hermod dynamic analysis lacks jobs for task job IDs: {missing}")
        workload.jobs = [self.jobs_by_id[job_id] for job_id in sorted(job_ids)]
        return super().analyze(workload)

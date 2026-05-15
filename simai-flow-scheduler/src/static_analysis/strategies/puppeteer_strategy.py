"""Puppeteer analysis strategy — orchestrates passes to build a scheduling policy.

Bootstrap pipeline:
1. Compute routing hints (shortest paths) for initial flow duration estimates
2. Compute TTE with shortest-path durations
3. Compute offline greedy route table
4. Recompute TTE with greedy routes for refined priorities
5. Compute resource dependency / coordination groups
6. Package into PuppeteerAnalysisResult
"""
from dataclasses import dataclass, field

from ..passes.puppeteer_coordination import (
    ResourceDependencyTable,
    compute_resource_dependency,
)
from ..passes.routing import GreedyRouteTable, GreedyStrategy, BfsStrategy
from ..passes.puppeteer_tte import TTEInfo, FlowTiming, compute_tte
from ..passes.task_serializer import ExecutionPlan
from ..passes.topology_loader import NetworkTopology
from ...workload_format.schema import P2PWorkload


@dataclass
class PuppeteerAnalysisResult:
    """Complete Puppeteer analysis result.

    Contains all precomputed tables needed by PuppeteerSchedulingPolicy.
    """
    route_table: GreedyRouteTable
    tte_info: dict[int, TTEInfo]
    flow_timing: dict[int, FlowTiming]
    resource_dependency: ResourceDependencyTable
    execution_plan: ExecutionPlan


class PuppeteerAnalyzer:
    """Puppeteer analysis orchestrator.

    Runs a multi-pass bootstrap pipeline to produce all tables
    needed by PuppeteerSchedulingPolicy.

    Args:
        topology: Network topology
        k_paths: Number of candidate paths for greedy routing (default: 4)
        tte_threshold_us: Threshold for elastic vs background classification
        recompute_tte: Whether to recompute TTE after greedy routing
    """

    def __init__(
        self,
        topology: NetworkTopology,
        k_paths: int = 4,
        tte_threshold_us: float = 1000.0,
        recompute_tte: bool = True,
    ):
        self.topology = topology
        self.k_paths = k_paths
        self.tte_threshold_us = tte_threshold_us
        self.recompute_tte = recompute_tte

    def analyze(self, workload: P2PWorkload) -> PuppeteerAnalysisResult:
        """Run the full Puppeteer analysis pipeline.

        Args:
            workload: P2P workload to analyze

        Returns:
            PuppeteerAnalysisResult with all precomputed tables
        """
        # Step 1: Serialize compute order
        from ..passes.task_serializer import CppReferenceSerializer
        serializer = CppReferenceSerializer()
        execution_plan = serializer.serialize(workload)

        # Step 2: Initial routing hints (shortest paths)
        route_table = BfsStrategy().compute_routes(workload, self.topology)

        # Step 3: TTE with shortest paths
        tte_info, flow_timing = compute_tte(
            workload, route_table, self.topology, execution_plan,
            route_paths=None,
            small_threshold_us=self.tte_threshold_us,
        )

        # Step 4: Greedy routing
        timing_dict: dict[int, tuple[int, int]] = {
            ft.task_id: (ft.start_time_us, ft.finish_time_us)
            for ft in flow_timing.values()
        }
        route_table = GreedyStrategy(timing_dict, k=self.k_paths).compute_routes(
            workload, self.topology,
        )

        # Step 5: Recompute TTE with greedy routes (if enabled)
        if self.recompute_tte:
            tte_info, flow_timing = compute_tte(
                workload, route_table, self.topology, execution_plan,
                route_paths=route_table.paths,
                small_threshold_us=self.tte_threshold_us,
            )
            timing_dict = {
                ft.task_id: (ft.start_time_us, ft.finish_time_us)
                for ft in flow_timing.values()
            }

        # Step 6: Resource dependency
        resource_dependency = compute_resource_dependency(
            workload, route_table, timing_dict, tte_info,
        )

        return PuppeteerAnalysisResult(
            route_table=route_table,
            tte_info=tte_info,
            flow_timing=flow_timing,
            resource_dependency=resource_dependency,
            execution_plan=execution_plan,
        )

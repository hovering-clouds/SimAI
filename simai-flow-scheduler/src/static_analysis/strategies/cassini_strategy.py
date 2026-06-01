"""Cassini analysis pipeline — network-aware multi-job time-shift scheduling.

Assembles Phase 1 modules (communication pattern extraction, circle abstraction,
pairwise compatibility) into a single analysable pipeline:

    1. BFS routing
    2. Critical path analysis (CPM)
    3. Communication pattern extraction
    4. Affinity graph traversal → per-job global time-shifts
    5. Compute execution plan
"""

from collections import defaultdict
from dataclasses import dataclass, field

from ..passes.critical_path import CriticalPathInfo, analyze_critical_path
from ..passes.routing import RouteTable, BfsStrategy
from ..passes.task_serializer import CppReferenceSerializer, ExecutionPlan
from ..passes.topology_loader import NetworkTopology
from ...cassini.affinity_graph import build_affinity_graph, compute_cluster_time_shifts
from ...cassini.communication_pattern import (
    CommunicationPattern,
    extract_communication_patterns,
)
from ...workload_format.schema import P2PWorkload


@dataclass
class CassiniAnalysisResult:
    """Complete Cassini analysis result.

    Attributes:
        route_table: BFS shortest-path routes for every flow task.
        critical_path: CPM forward/backward pass timing.
        communication_patterns: Per-job periodic communication patterns.
        time_shifts: Per-job global time-shift in microseconds.
        execution_plan: C++ reference compute ordering.
    """

    route_table: RouteTable
    critical_path: CriticalPathInfo
    communication_patterns: dict[int, CommunicationPattern]
    time_shifts: dict[int, int] = field(default_factory=dict)
    execution_plan: ExecutionPlan = field(default_factory=ExecutionPlan)


class CassiniAnalyzer:
    """Cassini analysis pipeline.

    Runs the full pre-execution analysis:
    routing → critical path → pattern extraction → affinity graph → time-shifts.

    The affinity graph traversal (Section 4.3 of the paper) ranks links by
    contention weight, optimises from most to least contended, and propagates
    each job's single global time-shift across all links it traverses.
    """

    def __init__(
        self,
        topology: NetworkTopology,
        step_deg: int = 5,
    ):
        self.topology = topology
        self.step_deg = step_deg

    def analyze(
        self,
        workload: P2PWorkload,
        route_table: RouteTable | None = None,
    ) -> CassiniAnalysisResult:
        if route_table is None:
            route_table = BfsStrategy().compute_routes(workload, self.topology)
        critical_path = analyze_critical_path(workload, route_table, self.topology)
        patterns = extract_communication_patterns(workload, critical_path, route_table, self.topology)

        time_shifts = self._compute_time_shifts(patterns, route_table, workload)

        serializer = CppReferenceSerializer()
        execution_plan = serializer.serialize(workload)

        return CassiniAnalysisResult(
            route_table=route_table,
            critical_path=critical_path,
            communication_patterns=patterns,
            time_shifts=time_shifts,
            execution_plan=execution_plan,
        )

    # ------------------------------------------------------------------
    # Internal: time-shift computation
    # ------------------------------------------------------------------

    def _compute_time_shifts(
        self,
        patterns: dict[int, CommunicationPattern],
        route_table: RouteTable,
        workload: P2PWorkload,
    ) -> dict[int, int]:
        """Compute per-job global time-shifts via affinity graph traversal.

        Builds the bipartite job-link graph, ranks links by contention
        weight, and traverses from most to least contended — fixing each
        job's time-shift on its first encounter and propagating it to
        subsequent links.
        """
        if len(patterns) < 2:
            return {jid: 0 for jid in patterns}

        job_links = self._build_job_links(workload, route_table)
        link_capacities = self._build_link_capacities(job_links)
        graph = build_affinity_graph(patterns, job_links, link_capacities)
        return compute_cluster_time_shifts(graph, self.step_deg)

    def _build_job_links(
        self,
        workload: P2PWorkload,
        route_table: RouteTable,
    ) -> dict[int, set[tuple[int, int]]]:
        """Map each job to the set of links its flows traverse."""
        job_links: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for task in workload.tasks:
            if not task.is_flow():
                continue
            try:
                path = route_table.get_path(task)
            except (KeyError, ValueError):
                continue
            for i in range(len(path) - 1):
                job_links[task.job_id].add((path[i], path[i + 1]))
        return dict(job_links)

    def _build_link_capacities(
        self,
        job_links: dict[int, set[tuple[int, int]]],
    ) -> dict[tuple[int, int], float]:
        """Collect capacities for every link referenced by any job."""
        caps: dict[tuple[int, int], float] = {}
        for links in job_links.values():
            for lid in links:
                if lid not in caps:
                    link = self.topology.get_link(lid[0], lid[1])
                    caps[lid] = link.bandwidth_gbps if link else 100.0
        return caps

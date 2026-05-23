"""Bipartite affinity graph and BFS-based time-shift solver.

Cassini models cluster-wide scheduling as a bipartite graph traversal
(Section 4.3 of the paper):

    - Left vertices: Jobs
    - Right vertices: Links
    - Edge: Job traverses Link (from RouteTable)

Algorithm (per connected component):
    1. Pick a root job (highest total communication demand).
    2. BFS from root: job → its contended links → new jobs on each link.
    3. On each visited link, optimise new (unfixed) jobs against
       already-fixed jobs already seen on that link.
    4. Mark new jobs as fixed (one global shift per job) and enqueue.
    5. When a job has multiple candidate links to unvisited jobs,
       prioritise the link with the highest contention weight.
       Lower-weight edges that lead to already-visited jobs are discarded
       by the visited set — this is the "acyclic BFS tree" simplification
       of the paper's multi-candidate acyclic-graph selection.

The core constraint is that each job has exactly ONE global time-shift,
shared across all links it traverses.
"""

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .circle_abstraction import CircleAbstraction
from .communication_pattern import CommunicationPattern
from .pair_compatibility import (
    CompatibilityResult,
    optimize_link_compatibility,
)


@dataclass
class AffinityGraph:
    """Bipartite graph: jobs <-> links.

    Attributes:
        job_links: Mapping from job_id to the set of links its flows traverse.
        link_jobs: Mapping from link_id to the set of jobs that traverse it.
        link_capacities: Mapping from link_id to capacity in Gbps.
        patterns: Per-job communication patterns.
    """

    job_links: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    link_jobs: dict[tuple[int, int], set[int]] = field(default_factory=dict)
    link_capacities: dict[tuple[int, int], float] = field(default_factory=dict)
    patterns: dict[int, CommunicationPattern] = field(default_factory=dict)


def build_affinity_graph(
    patterns: dict[int, CommunicationPattern],
    job_links: dict[int, set[tuple[int, int]]],
    link_capacities: dict[tuple[int, int], float],
) -> AffinityGraph:
    """Build a bipartite affinity graph from existing analysis data.

    Args:
        patterns: Per-job communication patterns.
        job_links: Pre-computed mapping job_id → set of traversed links.
        link_capacities: Link bandwidth capacities in Gbps.

    Returns:
        A fully populated AffinityGraph.
    """
    link_jobs: dict[tuple[int, int], set[int]] = defaultdict(set)
    for jid, links in job_links.items():
        for lid in links:
            link_jobs[lid].add(jid)

    # Only keep links shared by 2+ jobs (contended links)
    contended = {
        lid: jids for lid, jids in link_jobs.items() if len(jids) >= 2
    }

    return AffinityGraph(
        job_links=dict(job_links),
        link_jobs=contended,
        link_capacities=dict(link_capacities),
        patterns=dict(patterns),
    )


def compute_cluster_time_shifts(
    graph: AffinityGraph,
    step_deg: int = 5,
) -> dict[int, int]:
    """Compute per-job global time-shifts via BFS on each connected component.

    1. Decompose the bipartite graph into connected components.
    2. For each component, pick a root job and BFS outward.
    3. At each visited link, fix newly-discovered jobs against
       already-fixed jobs on that link via _optimize_with_fixed.
    4. Jobs never appearing on any contended link get shift 0.

    Args:
        graph: The populated affinity graph.
        step_deg: Angular step for grid search (default 5 degrees).

    Returns:
        Dict mapping job_id → time_shift in microseconds.
    """
    if not graph.link_jobs:
        return {jid: 0 for jid in graph.patterns}

    fixed_shifts_us: dict[int, int] = {}

    components = _find_connected_components(graph)
    for comp_jobs in components:
        _bfs_traverse_component(graph, comp_jobs, fixed_shifts_us, step_deg)

    for jid in graph.patterns:
        if jid not in fixed_shifts_us:
            fixed_shifts_us[jid] = 0

    return fixed_shifts_us


# ---------------------------------------------------------------------------
# Connected component decomposition
# ---------------------------------------------------------------------------


def _find_connected_components(graph: AffinityGraph) -> list[set[int]]:
    """Decompose the bipartite graph into connected components.

    Two jobs belong to the same component if there exists a path
    through shared contended links connecting them.
    """
    # Only jobs that appear on at least one contended link are relevant
    active_jobs: set[int] = set()
    for jids in graph.link_jobs.values():
        active_jobs.update(jids)

    unvisited = set(active_jobs)
    components: list[set[int]] = []

    while unvisited:
        start = unvisited.pop()
        comp: set[int] = {start}
        queue = [start]
        while queue:
            jid = queue.pop(0)
            for lid in graph.job_links.get(jid, set()):
                if lid not in graph.link_jobs:
                    continue
                for neighbour in graph.link_jobs[lid]:
                    if neighbour not in comp:
                        comp.add(neighbour)
                        queue.append(neighbour)
                        unvisited.discard(neighbour)
        components.append(comp)

    return components


def _job_total_demand(graph: AffinityGraph, job_id: int) -> float:
    """Total average bandwidth demand of a job across all its contended links."""
    pattern = graph.patterns.get(job_id)
    if pattern is None:
        return 0.0
    total = 0.0
    for lid in graph.job_links.get(job_id, set()):
        if lid not in graph.link_jobs:
            continue
        demands = pattern.link_demands.get(lid, {})
        if demands:
            total += sum(demands.values()) / len(demands)
    return total


# ---------------------------------------------------------------------------
# BFS traversal
# ---------------------------------------------------------------------------


def _bfs_traverse_component(
    graph: AffinityGraph,
    comp_jobs: set[int],
    fixed_shifts_us: dict[int, int],
    step_deg: int,
) -> None:
    """BFS from the highest-demand root job through contended links.

    Each iteration:
      1. Dequeue a job (already fixed).
      2. Collect its unvisited contended links that have unfixed jobs.
      3. Sort by contention weight (highest first).
      4. For each such link, optimise new jobs against already-fixed
         jobs on that link, mark the new jobs as fixed, and enqueue them.
    """
    if not comp_jobs:
        return

    root = max(comp_jobs, key=lambda jid: _job_total_demand(graph, jid))
    fixed_shifts_us[root] = 0

    visited_jobs: set[int] = {root}
    visited_links: set[tuple[int, int]] = set()
    queue: list[int] = [root]

    while queue:
        current = queue.pop(0)
        candidates = _collect_candidate_links(
            graph, current, visited_jobs, visited_links,
        )

        for lid, unfixed_on_link in candidates:
            if lid in visited_links:
                continue

            all_jobs_on_link = graph.link_jobs.get(lid, set())
            fixed_on_link = {
                jid: fixed_shifts_us[jid]
                for jid in all_jobs_on_link if jid in fixed_shifts_us
            }

            if not unfixed_on_link:
                visited_links.add(lid)
                continue

            _process_link(
                graph, lid, all_jobs_on_link,
                fixed_on_link, unfixed_on_link,
                fixed_shifts_us, step_deg,
            )
            visited_links.add(lid)

            for jid in unfixed_on_link:
                visited_jobs.add(jid)
                queue.append(jid)


def _collect_candidate_links(
    graph: AffinityGraph,
    job_id: int,
    visited_jobs: set[int],
    visited_links: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], list[int]]]:
    """Return unvisited contended links of *job_id* that lead to unfixed jobs.

    Sorted by contention weight descending so the most-congested edge
    is traversed first.
    """
    result: list[tuple[float, tuple[int, int], list[int]]] = []
    for lid in graph.job_links.get(job_id, set()):
        if lid in visited_links or lid not in graph.link_jobs:
            continue
        unfixed = [jid for jid in graph.link_jobs[lid]
                   if jid not in visited_jobs]
        if not unfixed:
            continue
        w = _link_contention_weight(graph, lid, graph.link_jobs[lid])
        result.append((w, lid, unfixed))
    result.sort(key=lambda x: x[0], reverse=True)
    return [(lid, unfixed) for _, lid, unfixed in result]


def _process_link(
    graph: AffinityGraph,
    link_id: tuple[int, int],
    all_jobs_on_link: set[int],
    fixed_on_link: dict[int, int],
    unfixed_on_link: list[int],
    fixed_shifts_us: dict[int, int],
    step_deg: int,
) -> None:
    """Optimise time-shifts on one link and record newly-fixed jobs."""
    circles, circle_to_job = _build_link_circles(graph, link_id, all_jobs_on_link)
    job_to_circle = {jid: idx for idx, jid in circle_to_job.items()}

    if len(circles) < 2:
        for jid in unfixed_on_link:
            fixed_shifts_us[jid] = 0
        return

    capacity = graph.link_capacities.get(link_id, 100.0)

    fixed_deg: dict[int, int] = {}
    for jid, shift_us in fixed_on_link.items():
        circ_idx = job_to_circle.get(jid)
        if circ_idx is not None and circ_idx in circles:
            perimeter = circles[circ_idx].perimeter
            if perimeter > 0:
                fixed_deg[circ_idx] = round(shift_us * 360 / perimeter) % 360

    result = optimize_link_compatibility(
        circles, capacity, step_deg, fixed_shifts_deg=fixed_deg or None,
    )

    for circ_idx, shift_us in result.time_shifts_us.items():
        jid = circle_to_job.get(circ_idx)
        if jid is not None and jid not in fixed_shifts_us:
            fixed_shifts_us[jid] = shift_us


# ---------------------------------------------------------------------------
# Internal: link contention weight
# ---------------------------------------------------------------------------


def _link_contention_weight(
    graph: AffinityGraph,
    link_id: tuple[int, int],
    job_ids: set[int],
) -> float:
    """Contention weight: num_jobs * avg_demand on this link."""
    if not job_ids:
        return 0.0
    total_demand = 0.0
    count = 0
    for jid in job_ids:
        pattern = graph.patterns.get(jid)
        if pattern is None:
            continue
        demands = pattern.link_demands.get(link_id, {})
        if demands:
            total_demand += sum(demands.values()) / len(demands)
            count += 1
    if count == 0:
        return 0.0
    return len(job_ids) * (total_demand / count)


# ---------------------------------------------------------------------------
# Internal: circle construction for a single link
# ---------------------------------------------------------------------------


def _build_link_circles(
    graph: AffinityGraph,
    link_id: tuple[int, int],
    job_ids: set[int],
) -> tuple[dict[int, CircleAbstraction], dict[int, int]]:
    """Build circles for all jobs on a link.

    Uses LCM unified circles when jobs have different iteration times.

    Returns:
        (circles, circle_to_job) where circles maps circle_index → Circle
        and circle_to_job maps circle_index → job_id.
    """
    patterns = []
    jid_order = []
    for jid in job_ids:
        p = graph.patterns.get(jid)
        if p is not None and link_id in p.link_demands and p.iteration_time_us > 0:
            patterns.append(p)
            jid_order.append(jid)

    if len(patterns) < 2:
        circles: dict[int, CircleAbstraction] = {}
        circle_to_job: dict[int, int] = {}
        for i, jid in enumerate(jid_order):
            p = graph.patterns.get(jid)
            if p is not None and link_id in p.link_demands:
                circles[i] = CircleAbstraction.from_pattern(p, link_id)
                circle_to_job[i] = jid
        return circles, circle_to_job

    lcm_perimeter, unified = CircleAbstraction.build_unified(patterns, link_id)

    if unified:
        circles = {i: c for i, c in enumerate(unified)}
        circle_to_job = {i: jid_order[i] for i in range(len(jid_order))}
        return circles, circle_to_job

    # Fallback: use raw per-job circles
    circles = {}
    circle_to_job = {}
    for i, jid in enumerate(jid_order):
        p = graph.patterns.get(jid)
        if p is not None and link_id in p.link_demands:
            circles[i] = CircleAbstraction.from_pattern(p, link_id)
            circle_to_job[i] = jid
    return circles, circle_to_job



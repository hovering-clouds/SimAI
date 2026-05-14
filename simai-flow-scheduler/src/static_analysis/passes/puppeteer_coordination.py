"""Resource dependency analysis for Puppeteer-like coordination.

Builds co-start groups for flows sharing planned resources.
This approximates Puppeteer's runtime coordination mechanism by
identifying flows that should start together to avoid network imbalance.
"""
from dataclasses import dataclass, field

from ...workload_format.schema import P2PWorkload
from .puppeteer_routing import RouteTable
from .puppeteer_tte import TTEInfo


@dataclass
class ResourceDependencyTable:
    """Resource dependency and coordination groups.

    peers: flow task_id -> set of peer flow ids that share resources
    groups: group_id -> set of flow task_ids that should co-start
    """
    peers: dict[int, set[int]] = field(default_factory=dict)
    groups: dict[str, set[int]] = field(default_factory=dict)


def compute_resource_dependency(
    workload: P2PWorkload,
    route_table: RouteTable,
    flow_timing: dict[int, tuple[int, int]],
    tte_info: dict[int, TTEInfo] | None = None,
) -> ResourceDependencyTable:
    """Build coordination groups for flows sharing planned resources.

    Conditions for group membership:
    1. Flows share at least one physical link in their planned paths
    2. Their optimistic active intervals overlap
    3. They are not already ordered by workload DAG dependencies
    4. At least one flow is critical or near-critical (when tte_info provided)

    Args:
        workload: P2P workload
        route_table: Precomputed route table
        flow_timing: Optimistic flow timing: task_id -> (start_us, finish_us)
        tte_info: Optional TTE info for priority-based filtering

    Returns:
        ResourceDependencyTable with group membership
    """
    # Build DAG dependency set for quick lookup
    dag_ordered: set[tuple[int, int]] = set()
    for t in workload.tasks:
        for dep in t.deps:
            dag_ordered.add((dep, t.task_id))

    # Build flow link sets from route table
    flow_links: dict[int, set[tuple[int, int]]] = {}
    for t in workload.tasks:
        if not t.is_flow():
            continue
        tid = t.task_id
        if tid not in route_table.paths:
            continue
        path = route_table.paths[tid]
        links = {(path[i], path[i + 1]) for i in range(len(path) - 1)}
        flow_links[tid] = links

    # Find overlapping flow pairs
    flow_ids = list(flow_links.keys())
    peers: dict[int, set[int]] = {fid: set() for fid in flow_ids}

    for i in range(len(flow_ids)):
        for j in range(i + 1, len(flow_ids)):
            a, b = flow_ids[i], flow_ids[j]

            # Skip if already DAG-ordered
            if (a, b) in dag_ordered or (b, a) in dag_ordered:
                continue

            # Check shared links
            if not (flow_links[a] & flow_links[b]):
                continue

            # Check overlapping intervals
            a_start, a_finish = flow_timing.get(a, (0, 0))
            b_start, b_finish = flow_timing.get(b, (0, 0))

            if not (a_start < b_finish and b_start < a_finish):
                continue

            # Priority filtering: require at least one critical/elastic flow
            if tte_info is not None:
                a_prio = tte_info.get(a)
                b_prio = tte_info.get(b)
                a_is_urgent = a_prio is not None and a_prio.priority_class in ("critical", "elastic")
                b_is_urgent = b_prio is not None and b_prio.priority_class in ("critical", "elastic")
                if not (a_is_urgent or b_is_urgent):
                    continue

            peers[a].add(b)
            peers[b].add(a)

    # Build groups from peer graph (connected components)
    groups: dict[str, set[int]] = {}
    visited: set[int] = set()
    group_counter = 0

    for fid in flow_ids:
        if fid in visited or not peers[fid]:
            continue

        # BFS for connected component
        component: set[int] = set()
        queue = [fid]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for peer in peers[current]:
                if peer not in visited:
                    queue.append(peer)

        if component:
            groups[f"coord_group_{group_counter}"] = component
            group_counter += 1

    return ResourceDependencyTable(peers=peers, groups=groups)

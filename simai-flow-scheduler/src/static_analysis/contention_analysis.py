"""
Link contention analysis - identify flows competing for the same physical links.

Combines spatial (which links) and temporal (when) analysis to identify
actual contention. Uses per-link timing (not global flow timing) for accuracy.

Key design:
- entry_time = flow_start + cumulative_propagation_latency (from previous hops)
- exit_time  = entry_time + transmission_delay (size / this_link_bandwidth)
- Sweep line algorithm computes peak concurrency per link

Dependencies:
- Task 1 (RoutingHints): provides paths through topology
- Task 2 (CriticalPathInfo): provides ASAP timing for each flow
"""

from dataclasses import dataclass, field

from ..workload_format.schema import P2PWorkload
from .critical_path import CriticalPathInfo
from .routing_hints import RoutingHints


@dataclass
class LinkContentionGroup:
    """
    All flow tasks that traverse a specific physical link, with temporal analysis.

    Uses ASAP timing from critical path analysis (v1 static analysis).
    Actual timing depends on dynamic scheduler decisions in Phase 4.
    """

    link_id: tuple[int, int]  # (node_a, node_b)

    # All flows using this link (spatial contention)
    all_flows: list[int] = field(default_factory=list)  # task_ids

    # Statistics
    total_data_bytes: int = 0
    min_size_bytes: int = 0
    max_size_bytes: int = 0
    avg_size_bytes: float = 0.0
    num_flows: int = 0

    # Concurrency estimates (based on ASAP timing)
    worst_case_concurrency: int = 0  # = num_flows
    best_case_concurrency: int = 0  # peak concurrent at any single time point

    # Per-link time windows: task_id -> (entry_time_us, exit_time_us)
    time_windows: dict[int, tuple[int, int]] = field(default_factory=dict)

    def add_flow(
        self, task_id: int, size_bytes: int, entry_time: int, exit_time: int
    ):
        """
        Add a flow with per-link timing.

        Args:
            task_id: The flow task ID
            size_bytes: Data size to transfer
            entry_time: flow_start + cumulative propagation latency (us)
            exit_time: entry_time + transmission_delay on this link (us)
        """
        self.all_flows.append(task_id)
        self.total_data_bytes += size_bytes
        self.num_flows += 1
        self.time_windows[task_id] = (entry_time, exit_time)

        if self.min_size_bytes == 0 or size_bytes < self.min_size_bytes:
            self.min_size_bytes = size_bytes
        if size_bytes > self.max_size_bytes:
            self.max_size_bytes = size_bytes
        self.avg_size_bytes = self.total_data_bytes / self.num_flows
        self.worst_case_concurrency = self.num_flows

    def get_concurrency_at_time(self, timestamp: int) -> int:
        """How many flows are active on this link at a given timestamp?"""
        count = 0
        for fid in self.all_flows:
            entry, exit_ = self.time_windows[fid]
            if entry <= timestamp < exit_:
                count += 1
        return count

    def get_peak_concurrency_window(self) -> tuple[int, int, int]:
        """Find the time window with maximum concurrency (sweep line)."""
        if not self.time_windows:
            return (0, 0, 0)

        events: list[tuple[int, int]] = []
        for entry, exit_ in self.time_windows.values():
            events.append((entry, 1))
            events.append((exit_, -1))

        # Sort by time; exits before entries at same timestamp
        events.sort(key=lambda x: (x[0], x[1]))

        max_concurrent = 0
        current = 0
        peak_start = 0
        peak_end = 0
        in_peak = False

        for time_, delta in events:
            prev_current = current
            current += delta

            if delta == 1 and current > max_concurrent:
                max_concurrent = current
                peak_start = time_
                in_peak = True
            elif delta == -1 and in_peak and prev_current == max_concurrent:
                peak_end = time_
                in_peak = False

        return (peak_start, peak_end, max_concurrent)

    def analyze_temporal_contention(self):
        """Post-process: compute best_case_concurrency via sweep line."""
        _, _, peak = self.get_peak_concurrency_window()
        self.best_case_concurrency = peak

    @property
    def has_temporal_contention(self) -> bool:
        """Check if this link has any potential temporal contention."""
        return self.best_case_concurrency > 1

    @property
    def contention_ratio(self) -> float:
        """
        Ratio of actual concurrency to worst case.
        High (~1.0) = most flows overlap. Low (~0.0) = well-separated.
        """
        if self.worst_case_concurrency < 2:
            return 0.0
        return self.best_case_concurrency / self.worst_case_concurrency


def find_contention_groups(
    workload: P2PWorkload,
    routing_hints: RoutingHints,
    critical_path_info: CriticalPathInfo,
) -> dict[tuple[int, int], LinkContentionGroup]:
    """
    Group flows by physical link with per-link temporal analysis.

    For each flow task:
    1. Get global start time from critical path (ASAP)
    2. Get full path through topology (via routing_hints)
    3. Calculate per-link timing:
       - entry_time = flow_start + cumulative_propagation_latency
       - exit_time  = entry_time + transmission_delay (size / link_bandwidth)
    4. Add to contention group
    5. Post-process to compute peak concurrency via sweep line

    Uses topology bound in routing_hints for path and link lookups.

    Complexity: O(F * H) where F = num flows, H = avg hop count
    """
    topology = routing_hints.topology
    groups: dict[tuple[int, int], LinkContentionGroup] = {}

    for task in workload.tasks:
        if not task.is_flow():
            continue
        if task.src is None or task.dst is None:
            continue

        size_bytes = task.size_bytes or 0
        if size_bytes == 0:
            continue

        size_bits = size_bytes * 8

        # Global start time from ASAP analysis
        timing = critical_path_info.task_timings.get(task.task_id)
        flow_start_time = timing.earliest_start_us if timing else 0

        # Full path through topology
        path = routing_hints.get_path(task.src, task.dst)
        if len(path) < 2:
            continue

        cumulative_latency_us = 0.0

        for i in range(len(path) - 1):
            link_id = (path[i], path[i + 1])
            link = topology.get_link(link_id[0], link_id[1])

            if link is None:
                break

            # Transmission delay on THIS specific link (microseconds)
            tx_delay_us = (size_bits / (link.bandwidth_gbps * 1e9)) * 1e6

            # Per-link entry/exit times
            entry_time = flow_start_time + int(cumulative_latency_us)
            exit_time = entry_time + int(tx_delay_us)

            if link_id not in groups:
                groups[link_id] = LinkContentionGroup(link_id=link_id)

            groups[link_id].add_flow(task.task_id, size_bytes, entry_time, exit_time)

            # Accumulate propagation latency for next hop
            cumulative_latency_us += link.latency_us

    # Post-process: compute temporal contention per group
    for group in groups.values():
        group.analyze_temporal_contention()

    return groups

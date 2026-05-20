"""RMLQ-style bandwidth allocator for MFS.

Implements strict-priority queues with fair sharing within each queue.
RLI is computed on-demand using dynamic current_layer state.
P2D flows use MLU-based promotion driven by request-level ttft_slo_us.
"""
from dataclasses import dataclass

from .base_allocator import BandwidthAllocator
from ...static_analysis.passes.topology_loader import NetworkTopology
from ...static_analysis.passes.mfs_context import MfsContext, MfsStage

_P2D_SENTINEL_RLI = 10_000
_BG_SENTINEL_RLI = 10_000


@dataclass
class MfsAllocatorConfig:
    num_queues: int = 4
    p2d_initial_queue: int = 0
    early_rli0_queue: int = 2
    early_default_queue: int = 1
    urgent_p2d_queue: int = 3
    p2d_mlu_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9)


class MfsAllocator(BandwidthAllocator):
    """RMLQ-style allocator: strict priority across queues, fair share within."""

    def __init__(
        self,
        context: MfsContext,
        config: MfsAllocatorConfig | None = None,
    ):
        self.context = context
        self.config = config or MfsAllocatorConfig()
        # Dynamic state updated by MfsSchedulingPolicy
        self.current_layer_by_stage: dict[tuple[int, int], int] = {}
        self.request_start_time: dict[int, int] = {}

    def _compute_rli(self, task_id: int) -> int:
        """Compute RLI on-demand using current dynamic layer state."""
        info = self.context.task_info.get(task_id)
        if info is None:
            return _BG_SENTINEL_RLI

        if info.mfs_stage == MfsStage.P2D:
            return _P2D_SENTINEL_RLI
        if info.mfs_stage == MfsStage.BACKGROUND:
            return _BG_SENTINEL_RLI

        # EARLY: RLI = max(target_layer - current_layer, 0)
        stage_key = (info.job_id, info.stage_id)
        current = self.current_layer_by_stage.get(stage_key, 0)
        return max(info.target_layer - current, 0)

    def _queue_for(self, flow, current_time: int, topology: NetworkTopology | None = None) -> int:
        info = self.context.task_info.get(flow.task_id)
        if info is None:
            return 0

        if info.mfs_stage == MfsStage.P2D:
            deadline = self._earliest_deadline(info.request_ids)
            if deadline is not None:
                return self._queue_for_mlu(flow, current_time, deadline, topology)
            return self.config.p2d_initial_queue

        if info.mfs_stage == MfsStage.EARLY:
            rli = self._compute_rli(flow.task_id)
            if rli == 0:
                return self.config.early_rli0_queue
            return self.config.early_default_queue

        return 0

    def _earliest_deadline(self, request_ids: tuple[int, ...]) -> int | None:
        """Return the earliest deadline (start_time + ttft_slo_us) among requests."""
        deadlines = []
        for rid in request_ids:
            ri = self.context.request_info.get(rid)
            if ri is None or ri.ttft_slo_us is None:
                continue
            start = self.request_start_time.get(rid)
            if start is not None:
                deadlines.append(start + ri.ttft_slo_us)
        return min(deadlines) if deadlines else None

    def _queue_for_mlu(
        self,
        flow,
        current_time: int,
        deadline_us: int,
        topology: NetworkTopology | None,
    ) -> int:
        """Determine P2D queue using MLU based on deadline."""
        remaining_time = deadline_us - current_time

        # Past deadline -> urgent
        if remaining_time <= 0:
            return self.config.urgent_p2d_queue

        # Compute required bandwidth
        remaining_bits = flow.remaining_bytes * 8
        required_bw_gbps = remaining_bits / (remaining_time * 1e3)

        # Estimate bottleneck bandwidth on flow path
        bottleneck_bw = float("inf")
        if topology is not None:
            path = flow.path
            for i in range(len(path) - 1):
                link = topology.get_link(path[i], path[i + 1])
                if link is not None:
                    bottleneck_bw = min(bottleneck_bw, link.bandwidth_gbps)

        if bottleneck_bw <= 0 or bottleneck_bw == float("inf"):
            # Cannot estimate, keep at initial queue
            return self.config.p2d_initial_queue

        mlu = required_bw_gbps / bottleneck_bw

        # Map MLU to queue based on thresholds
        thresholds = self.config.p2d_mlu_thresholds
        if mlu >= thresholds[-1]:
            return self.config.urgent_p2d_queue
        if len(thresholds) >= 2 and mlu >= thresholds[-2]:
            return self.config.early_default_queue
        if mlu >= thresholds[0]:
            return self.config.early_rli0_queue
        return self.config.p2d_initial_queue

    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        current_time: int = 0,
    ) -> dict[int, float]:
        if not active_flows:
            return {}

        cfg = self.config

        # Assign each flow to a queue
        queue_flows: dict[int, list] = {q: [] for q in range(cfg.num_queues)}
        for flow in active_flows:
            q = self._queue_for(flow, current_time, topology)
            q = max(0, min(q, cfg.num_queues - 1))
            queue_flows[q].append(flow)

        # Build per-link remaining capacity
        link_rem: dict[tuple[int, int], float] = {}
        for flow in active_flows:
            path = flow.path
            for i in range(len(path) - 1):
                link = (path[i], path[i + 1])
                if link not in link_rem:
                    link_obj = topology.get_link(link[0], link[1])
                    link_rem[link] = link_obj.bandwidth_gbps if link_obj else 0.0

        # Build per-queue per-link flow count for fair sharing
        queue_link_count: dict[int, dict[tuple[int, int], int]] = {}
        for q, flows in queue_flows.items():
            counts: dict[tuple[int, int], int] = {}
            for flow in flows:
                path = flow.path
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    counts[link] = counts.get(link, 0) + 1
            queue_link_count[q] = counts

        result: dict[int, float] = {}

        # Allocate from highest queue to lowest
        for q in range(cfg.num_queues - 1, -1, -1):
            flows = queue_flows[q]
            if not flows:
                continue

            link_counts = queue_link_count[q]

            # Compute per-flow allocation (bottleneck across path links)
            queue_allocs: list[tuple[object, float]] = []
            for flow in flows:
                min_alloc = float("inf")
                path = flow.path
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    rem = link_rem.get(link, 0.0)
                    count = link_counts.get(link, 0)
                    if count > 0 and rem > 0:
                        alloc = rem / count
                        min_alloc = min(min_alloc, alloc)

                if min_alloc != float("inf") and min_alloc > 0:
                    queue_allocs.append((flow, min_alloc))
                else:
                    result[flow.task_id] = 0.0

            # Deduct consumed capacity after computing all fair-share values
            for flow, bw in queue_allocs:
                result[flow.task_id] = bw
                path = flow.path
                for i in range(len(path) - 1):
                    link = (path[i], path[i + 1])
                    if link in link_rem:
                        link_rem[link] = max(0.0, link_rem[link] - bw)

        # Ensure all flows have an entry
        for flow in active_flows:
            if flow.task_id not in result:
                result[flow.task_id] = 0.0

        return result

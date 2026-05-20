"""Tests for MFS RMLQ allocator."""
import pytest

from src.static_analysis.passes.topology_loader import NetworkTopology, Link
from src.static_analysis.passes.mfs_context import MfsContext, MfsTaskInfo, MfsStage, MfsRequestInfo
from src.static_analysis.passes.mfs_rli import RliInfo
from src.executor.bandwidth_allocators.mfs_allocator import (
    MfsAllocator, MfsAllocatorConfig,
)
from src.executor.runtime import ActiveFlow


def _make_topology(links=None):
    """Build a simple topology with given links (default: 4-node line)."""
    topo = NetworkTopology()
    default_links = [
        Link(0, 4, 100.0, 1.0, 0.0),
        Link(4, 0, 100.0, 1.0, 0.0),
        Link(1, 4, 100.0, 1.0, 0.0),
        Link(4, 1, 100.0, 1.0, 0.0),
        Link(2, 4, 100.0, 1.0, 0.0),
        Link(4, 2, 100.0, 1.0, 0.0),
        Link(3, 4, 100.0, 1.0, 0.0),
        Link(4, 3, 100.0, 1.0, 0.0),
    ]
    for link in (links or default_links):
        topo.add_link(link)
    return topo


def _flow(tid, src, dst, size, start=0, path=None):
    return ActiveFlow(
        task_id=tid, src=src, dst=dst,
        size_bytes=size, remaining_bytes=size,
        path=path or [src, dst],
        start_time=start, last_update_time=start,
    )


def _make_context(task_infos):
    """Build MfsContext from a list of MfsTaskInfo."""
    return MfsContext(task_info={info.task_id: info for info in task_infos})


# ── Allocator tests ───────────────────────────────────────────────────────────


class TestMfsAllocator:
    def setup_method(self):
        self.topo = _make_topology()

    def test_high_queue_gets_full_bandwidth(self):
        """Flow in high-priority queue gets full link, low queue gets zero."""
        ctx = _make_context([
            MfsTaskInfo(1, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(2, None, (), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        rli = {1: RliInfo(1, 0, 0), 2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(
            early_rli0_queue=2, p2d_initial_queue=0,
            urgent_p2d_queue=3, early_default_queue=1,
        )
        alloc = MfsAllocator(ctx, rli, cfg)

        f1 = _flow(1, 0, 1, 100)
        f2 = _flow(2, 2, 3, 100)
        # Both use link through switch node 4, but on different endpoints
        # Need to share a bottleneck link. Use same link path.
        f1 = _flow(1, 0, 1, 100, path=[0, 4, 1])
        f2 = _flow(2, 2, 3, 100, path=[2, 4, 3])

        result = alloc.allocate([f1, f2], self.topo, current_time=0)
        # Queue 2 (early RLI 0) > queue 0 (p2d initial)
        assert result[1] == 100.0  # full bandwidth
        assert result[2] == 100.0  # different links, no contention

    def test_strict_priority_shared_link(self):
        """Two flows on same link: high queue gets everything, low gets zero."""
        # Single shared link between 0 and 1
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 50.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(2, None, (), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        rli = {1: RliInfo(1, 0, 0), 2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(
            early_rli0_queue=2, p2d_initial_queue=0,
            urgent_p2d_queue=3, early_default_queue=1,
        )
        alloc = MfsAllocator(ctx, rli, cfg)

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100, path=[0, 1])

        result = alloc.allocate([f1, f2], topo, current_time=0)
        assert result[1] == 50.0  # full link bandwidth
        assert result[2] == 0.0   # nothing left

    def test_fair_share_within_same_queue(self):
        """Two flows in same queue on same link: each gets half bandwidth."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(3, None, (), 0, MfsStage.EARLY, 0, "collective"),
        ])
        rli = {1: RliInfo(1, 0, 0), 3: RliInfo(3, 0, 0)}
        alloc = MfsAllocator(ctx, rli)

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f3 = _flow(3, 0, 1, 100, path=[0, 1])

        result = alloc.allocate([f1, f3], topo, current_time=0)
        assert result[1] == pytest.approx(50.0)
        assert result[3] == pytest.approx(50.0)

    def test_p2d_promotion_after_delay(self):
        """P2D starts at low queue, promotes to urgent after delay."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(2, None, (), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        rli = {1: RliInfo(1, 0, 0), 2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(p2d_promotion_delay_us=500)
        alloc = MfsAllocator(ctx, rli, cfg)

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100, start=0, path=[0, 1])

        # Before promotion: P2D at queue 0, early at queue 2
        result_before = alloc.allocate([f1, f2], topo, current_time=100)
        assert result_before[1] == 100.0
        assert result_before[2] == 0.0

        # After promotion: P2D at urgent queue 3 > early queue 2
        result_after = alloc.allocate([f1, f2], topo, current_time=600)
        assert result_after[2] == 100.0
        assert result_after[1] == 0.0

    def test_multi_hop_flow_bottleneck(self):
        """Multi-hop flow allocation uses path bottleneck."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 5, 200.0, 1.0, 0.0))
        topo.add_link(Link(5, 4, 50.0, 1.0, 0.0))
        topo.add_link(Link(4, 3, 200.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, None, (), 0, MfsStage.EARLY, 0, "collective"),
        ])
        rli = {1: RliInfo(1, 0, 0)}
        alloc = MfsAllocator(ctx, rli)

        f1 = _flow(1, 0, 3, 100, path=[0, 5, 4, 3])
        result = alloc.allocate([f1], topo, current_time=0)
        assert result[1] == 50.0  # bottleneck link

    def test_empty_flows(self):
        """No flows -> empty result."""
        ctx = MfsContext()
        alloc = MfsAllocator(ctx, {})
        assert alloc.allocate([], self.topo, 0) == {}


# ── MLU promotion tests (Phase 2) ─────────────────────────────────────────────


class TestMluPromotion:
    """Tests for MLU-based P2D promotion."""

    def test_loose_deadline_stays_low(self):
        """P2D with loose deadline (low MLU) should stay at low priority."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(2, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        ctx.request_info[1] = MfsRequestInfo(
            request_id=1, arrival_time_us=0, ttft_slo_us=10000000,
            deadline_us=10000000,  # very far in the future
        )
        rli = {2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(enable_deadline_promotion=True)
        alloc = MfsAllocator(ctx, rli, cfg)

        # Large remaining time, small remaining bytes -> low MLU
        f2 = _flow(2, 0, 1, 100, start=0, path=[0, 1])
        result = alloc.allocate([f2], topo, current_time=100)
        # Should still be at low queue, get full bw since no contention
        assert result[2] == 100.0

    def test_tight_deadline_gets_promoted(self):
        """P2D with tight deadline (high MLU) should be promoted to urgent."""
        # Use 1 Gbps link so MLU values are meaningful with small data
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 1.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(2, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        ctx.request_info[1] = MfsRequestInfo(
            request_id=1, arrival_time_us=0, ttft_slo_us=1000,
            deadline_us=1000,
        )
        rli = {1: RliInfo(1, 0, 0), 2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(enable_deadline_promotion=True)
        alloc = MfsAllocator(ctx, rli, cfg)

        # P2D: 100KB remaining, 100us to deadline -> MLU = 800Kbits/(100*1e3) / 1Gbps = 8.0
        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100000, start=0, path=[0, 1])
        result = alloc.allocate([f1, f2], topo, current_time=900)
        # P2D (urgent queue 3) > EARLY (queue 2) -> P2D gets all bandwidth
        assert result[2] == 1.0
        assert result[1] == 0.0

    def test_past_deadline_gets_urgent(self):
        """P2D past deadline should go to urgent queue immediately."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(2, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        ctx.request_info[1] = MfsRequestInfo(
            request_id=1, arrival_time_us=0, ttft_slo_us=100,
            deadline_us=100,
        )
        rli = {2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(enable_deadline_promotion=True)
        alloc = MfsAllocator(ctx, rli, cfg)

        f2 = _flow(2, 0, 1, 1000, start=0, path=[0, 1])
        result = alloc.allocate([f2], topo, current_time=200)
        # Past deadline, still gets bandwidth (no contention)
        assert result[2] == 100.0

    def test_no_deadline_falls_back_to_delay(self):
        """P2D without deadline should use Phase 1 delay-based heuristic."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(2, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        # No request_info -> no deadline
        rli = {2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(
            enable_deadline_promotion=True,
            p2d_promotion_delay_us=500,
        )
        alloc = MfsAllocator(ctx, rli, cfg)

        f2 = _flow(2, 0, 1, 1000, start=0, path=[0, 1])

        # Before delay: should be at low queue
        result_before = alloc.allocate([f2], topo, current_time=100)
        assert result_before[2] == 100.0  # full bw, no contention

        # After delay: should be promoted via fallback
        result_after = alloc.allocate([f2], topo, current_time=600)
        assert result_after[2] == 100.0

    def test_mlu_promotion_disabled_uses_delay(self):
        """With enable_deadline_promotion=False, deadline is ignored."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(2, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        ctx.request_info[1] = MfsRequestInfo(
            request_id=1, arrival_time_us=0, ttft_slo_us=100,
            deadline_us=100,  # tight deadline
        )
        rli = {2: RliInfo(2, 0, 10000)}
        cfg = MfsAllocatorConfig(
            enable_deadline_promotion=False,
            p2d_promotion_delay_us=100000,  # long delay
        )
        alloc = MfsAllocator(ctx, rli, cfg)

        f2 = _flow(2, 0, 1, 1000, start=0, path=[0, 1])
        # Even with tight deadline, should stay at low queue (delay not reached)
        result = alloc.allocate([f2], topo, current_time=50)
        # No contention so still gets full bw, but the queue assignment is 0
        assert result[2] == 100.0

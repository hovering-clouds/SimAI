"""Tests for MFS RMLQ allocator."""
import pytest

from src.static_analysis.passes.topology_loader import NetworkTopology, Link
from src.static_analysis.passes.mfs_context import MfsContext, MfsTaskInfo, MfsStage, MfsRequestInfo
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


def _make_context(task_infos, request_infos=None):
    """Build MfsContext from lists of MfsTaskInfo and optional MfsRequestInfo."""
    ctx = MfsContext(task_info={info.task_id: info for info in task_infos})
    if request_infos:
        for ri in request_infos:
            ctx.request_info[ri.request_id] = ri
    return ctx


# ── Allocator tests ───────────────────────────────────────────────────────────


class TestMfsAllocator:
    def setup_method(self):
        self.topo = _make_topology()

    def test_high_queue_gets_full_bandwidth(self):
        """Flow in high-priority queue gets full link, low queue gets zero."""
        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(2, 0, None, (), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        cfg = MfsAllocatorConfig()
        alloc = MfsAllocator(ctx, cfg)

        f1 = _flow(1, 0, 1, 100, path=[0, 4, 1])
        f2 = _flow(2, 2, 3, 100, path=[2, 4, 3])

        result = alloc.allocate([f1, f2], self.topo, current_time=0)
        assert result[1] == 100.0
        assert result[2] == 100.0  # different links, no contention

    def test_strict_priority_shared_link(self):
        """Two flows on same link: high queue gets everything, low gets zero."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 50.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 0, "collective"),
            MfsTaskInfo(2, 0, None, (), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        cfg = MfsAllocatorConfig()
        alloc = MfsAllocator(ctx, cfg)

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100, path=[0, 1])

        result = alloc.allocate([f1, f2], topo, current_time=0)
        assert result[1] == 50.0
        assert result[2] == 0.0

    def test_fair_share_within_same_queue(self):
        """Two EARLY default flows in same queue on same link: fair share."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        # RLI > 0 → early_default_queue, no RED quantile splitting
        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 2, "collective"),
            MfsTaskInfo(3, 0, None, (), 0, MfsStage.EARLY, 2, "collective"),
        ])
        alloc = MfsAllocator(ctx)

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f3 = _flow(3, 0, 1, 100, path=[0, 1])

        result = alloc.allocate([f1, f3], topo, current_time=0)
        assert result[1] == pytest.approx(50.0)
        assert result[3] == pytest.approx(50.0)

    def test_multi_hop_flow_bottleneck(self):
        """Multi-hop flow allocation uses path bottleneck."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 5, 200.0, 1.0, 0.0))
        topo.add_link(Link(5, 4, 50.0, 1.0, 0.0))
        topo.add_link(Link(4, 3, 200.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 0, "collective"),
        ])
        alloc = MfsAllocator(ctx)

        f1 = _flow(1, 0, 3, 100, path=[0, 5, 4, 3])
        result = alloc.allocate([f1], topo, current_time=0)
        assert result[1] == 50.0  # bottleneck link

    def test_empty_flows(self):
        """No flows -> empty result."""
        ctx = MfsContext()
        alloc = MfsAllocator(ctx)
        assert alloc.allocate([], self.topo, 0) == {}


# ── MLU promotion tests ───────────────────────────────────────────────────────


class TestMluPromotion:
    """Tests for MLU-based P2D promotion with relative ttft_slo_us."""

    def test_loose_deadline_stays_low(self):
        """P2D with loose deadline (low MLU) should stay at low priority."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context(
            [MfsTaskInfo(2, 0, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer")],
            [MfsRequestInfo(request_id=1, ttft_slo_us=10000000)],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[1] = 0

        f2 = _flow(2, 0, 1, 100, start=0, path=[0, 1])
        result = alloc.allocate([f2], topo, current_time=100)
        assert result[2] == 100.0  # full bw, no contention

    def test_tight_deadline_gets_promoted(self):
        """P2D with tight deadline (high MLU) should be promoted to urgent."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 1.0, 1.0, 0.0))

        ctx = _make_context(
            [
                MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 0, "collective"),
                MfsTaskInfo(2, 0, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
            ],
            [MfsRequestInfo(request_id=1, ttft_slo_us=1000)],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[1] = 0

        # P2D: 100KB remaining, 100us to deadline -> MLU = 8.0
        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100000, start=0, path=[0, 1])
        result = alloc.allocate([f1, f2], topo, current_time=900)
        # P2D (urgent queue 4) > EARLY (queue 2) -> P2D gets all bandwidth
        assert result[2] == 1.0
        assert result[1] == 0.0

    def test_past_deadline_gets_urgent(self):
        """P2D past deadline should go to urgent queue immediately."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context(
            [MfsTaskInfo(2, 0, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer")],
            [MfsRequestInfo(request_id=1, ttft_slo_us=100)],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[1] = 0

        f2 = _flow(2, 0, 1, 1000, start=0, path=[0, 1])
        result = alloc.allocate([f2], topo, current_time=200)
        assert result[2] == 100.0

    def test_no_slo_stays_at_initial(self):
        """P2D without ttft_slo_us should stay at initial queue."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context(
            [MfsTaskInfo(2, 0, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer")],
            # No SLO set
            [MfsRequestInfo(request_id=1, ttft_slo_us=None)],
        )
        alloc = MfsAllocator(ctx)

        f2 = _flow(2, 0, 1, 1000, start=0, path=[0, 1])
        result = alloc.allocate([f2], topo, current_time=100)
        assert result[2] == 100.0  # full bw, no contention

    def test_no_request_info_stays_at_initial(self):
        """P2D with no request_info at all should stay at initial queue."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context(
            [MfsTaskInfo(2, 0, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer")],
        )
        alloc = MfsAllocator(ctx)

        f2 = _flow(2, 0, 1, 1000, start=0, path=[0, 1])
        result = alloc.allocate([f2], topo, current_time=100)
        assert result[2] == 100.0


# ── Dynamic RLI tests ─────────────────────────────────────────────────────────


class TestDynamicRli:
    """Tests for on-demand RLI computation with dynamic current_layer."""

    def test_rli_zero_gets_high_queue(self):
        """EARLY flow with RLI 0 should go to early_rli0_queue."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 0, "collective"),
        ])
        alloc = MfsAllocator(ctx)
        # current_layer defaults to 0, target_layer is 0 -> RLI = 0

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        result = alloc.allocate([f1], topo, current_time=0)
        assert result[1] == 100.0  # gets full bw

    def test_rli_advances_with_current_layer(self):
        """RLI should decrease as current_layer advances."""
        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 3, "collective"),
        ])
        alloc = MfsAllocator(ctx)

        # At current_layer=0: RLI = 3-0 = 3 -> early_default_queue (1)
        assert alloc._compute_rli(1) == 3

        # Advance current_layer to 2: RLI = 3-2 = 1 -> still early_default_queue
        alloc.current_layer_by_stage[(0, 0)] = 2
        assert alloc._compute_rli(1) == 1

        # Advance to 3: RLI = 3-3 = 0 -> early_rli0_queue (2)
        alloc.current_layer_by_stage[(0, 0)] = 3
        assert alloc._compute_rli(1) == 0

    def test_rli_non_negative(self):
        """RLI should never go below 0."""
        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (), 0, MfsStage.EARLY, 2, "collective"),
        ])
        alloc = MfsAllocator(ctx)
        alloc.current_layer_by_stage[(0, 0)] = 5
        assert alloc._compute_rli(1) == 0

    def test_p2d_gets_sentinel_rli(self):
        """P2D flows should get large sentinel RLI."""
        ctx = _make_context([
            MfsTaskInfo(2, 0, None, (1,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        alloc = MfsAllocator(ctx)
        assert alloc._compute_rli(2) > 100


# ── RED tests ─────────────────────────────────────────────────────────────────


class TestRed:
    """Tests for RED (Robust Effective Deadline) computation."""

    def test_red_single_request(self):
        """Single request returns its own deadline."""
        ctx = _make_context(
            [MfsTaskInfo(1, 0, None, (10,), 0, MfsStage.EARLY, 0, "collective")],
            [MfsRequestInfo(request_id=10, ttft_slo_us=1000)],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[10] = 100
        # deadline = 100 + 1000 = 1100
        assert alloc._compute_red((10,)) == 1100

    def test_red_uniformly_tight(self):
        """Uniformly tight deadlines: RED close to minimum."""
        ctx = _make_context(
            [MfsTaskInfo(1, 0, None, (10, 11), 0, MfsStage.EARLY, 0, "collective")],
            [
                MfsRequestInfo(request_id=10, ttft_slo_us=1000),
                MfsRequestInfo(request_id=11, ttft_slo_us=1010),
            ],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[10] = 0
        alloc.request_start_time[11] = 0
        red = alloc._compute_red((10, 11))
        # Both start at 0, deadlines 1000 and 1010, gap=10 is small
        # Should be close to 1000
        assert 900 <= red <= 1100

    def test_red_tight_outlier_reduced(self):
        """One tight outlier + many loose: RED pulled toward loose group."""
        ctx = _make_context(
            [MfsTaskInfo(1, 0, None, (10, 11, 12), 0, MfsStage.EARLY, 0, "collective")],
            [
                MfsRequestInfo(request_id=10, ttft_slo_us=100),    # tight outlier
                MfsRequestInfo(request_id=11, ttft_slo_us=10000),  # loose
                MfsRequestInfo(request_id=12, ttft_slo_us=10000),  # loose
            ],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[10] = 0
        alloc.request_start_time[11] = 0
        alloc.request_start_time[12] = 0

        red = alloc._compute_red((10, 11, 12))
        # Deadlines: 100, 10000, 10000
        # Largest gap is between 100 and 10000 = 9900
        # tight set = {100}, loose set = {10000, 10000}
        # f = 1/3, RED = (1/3)*100 + (2/3)*10000 = 6700
        assert red == 6700
        # RED should NOT equal the outlier's 100
        assert red > 100
        # RED should be closer to loose min than tight min
        assert red > 5000

    def test_red_missing_slo_returns_none(self):
        """No SLO or no start time → None."""
        ctx = _make_context(
            [MfsTaskInfo(1, 0, None, (10,), 0, MfsStage.EARLY, 0, "collective")],
            [MfsRequestInfo(request_id=10, ttft_slo_us=None)],
        )
        alloc = MfsAllocator(ctx)
        assert alloc._compute_red((10,)) is None

    def test_red_no_request_info_returns_none(self):
        """No request_info at all → None."""
        ctx = _make_context(
            [MfsTaskInfo(1, 0, None, (99,), 0, MfsStage.EARLY, 0, "collective")],
        )
        alloc = MfsAllocator(ctx)
        assert alloc._compute_red((99,)) is None


class TestRedSubPriority:
    """Tests for RED quantile-based queue tiering."""

    def test_lower_red_gets_priority_over_higher_red(self):
        """Two EARLY RLI=0 flows: lower RED goes to urgent queue, gets all bandwidth."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        # Flow 1: tight batch (RED = 1000)
        ctx = _make_context(
            [
                MfsTaskInfo(1, 0, None, (10,), 0, MfsStage.EARLY, 0, "collective"),
                MfsTaskInfo(2, 0, None, (20,), 0, MfsStage.EARLY, 0, "collective"),
            ],
            [
                MfsRequestInfo(request_id=10, ttft_slo_us=1000),
                MfsRequestInfo(request_id=20, ttft_slo_us=100000),
            ],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[10] = 0
        alloc.request_start_time[20] = 0

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100, path=[0, 1])
        result = alloc.allocate([f1, f2], topo, current_time=0)

        # Both EARLY RLI=0, quantile split: flow 1 (RED=1000) → urgent (queue 3),
        # flow 2 (RED=100000) → relaxed (queue 2). Urgent queue served first.
        assert result[1] == 100.0
        assert result[2] == 0.0
        assert result[1] == 100.0
        assert result[2] == 0.0

    def test_red_does_not_affect_p2d_mlu(self):
        """RED sub-priority should not change P2D MLU promotion."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 1.0, 1.0, 0.0))

        ctx = _make_context(
            [
                MfsTaskInfo(1, 0, None, (10,), 0, MfsStage.P2D, 0, "p2d_transfer"),
                MfsTaskInfo(2, 0, None, (20,), 0, MfsStage.EARLY, 0, "collective"),
            ],
            [
                MfsRequestInfo(request_id=10, ttft_slo_us=1000),
                MfsRequestInfo(request_id=20, ttft_slo_us=100000),
            ],
        )
        alloc = MfsAllocator(ctx)
        alloc.request_start_time[10] = 0
        alloc.request_start_time[20] = 0

        # P2D with tight deadline (MLU high) should be promoted
        f1 = _flow(1, 0, 1, 100000, start=0, path=[0, 1])
        f2 = _flow(2, 0, 1, 100, path=[0, 1])
        result = alloc.allocate([f1, f2], topo, current_time=900)

        # P2D at urgent queue 4 > EARLY at queue 2/3
        assert result[1] == 1.0
        assert result[2] == 0.0

    def test_non_early_queue_keeps_fair_share(self):
        """P2D flows in same queue still get fair share, not RED-ordered."""
        topo = NetworkTopology()
        topo.add_link(Link(0, 1, 100.0, 1.0, 0.0))

        # Two P2D flows without SLO → both at p2d_initial_queue
        ctx = _make_context([
            MfsTaskInfo(1, 0, None, (10,), 0, MfsStage.P2D, 0, "p2d_transfer"),
            MfsTaskInfo(2, 0, None, (20,), 0, MfsStage.P2D, 0, "p2d_transfer"),
        ])
        alloc = MfsAllocator(ctx)

        f1 = _flow(1, 0, 1, 100, path=[0, 1])
        f2 = _flow(2, 0, 1, 100, path=[0, 1])
        result = alloc.allocate([f1, f2], topo, current_time=0)

        # Fair share: 50 / 50
        assert result[1] == pytest.approx(50.0)
        assert result[2] == pytest.approx(50.0)

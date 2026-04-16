"""
Tests for critical path analysis module.

Tests CPM algorithm with compute-only tasks, flow tasks,
and mixed workloads. Validates slack computation, critical
task identification, and flow duration estimation.
"""

import pytest

from src.scheduler.critical_path import (
    CriticalPathInfo,
    CriticalPathStrategy,
    TaskTimingInfo,
    _estimate_duration,
    _topological_sort,
    analyze_cpm,
    analyze_critical_path,
)
from src.scheduler.routing_hints import RoutingHints, compute_routing_hints
from src.scheduler.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Job,
    Meta,
    P2PWorkload,
    Task,
    TaskType,
    Phase,
)


# --- Helpers ---


def _make_compute_task(task_id: int, duration_us: int, deps=None, node=0):
    return Task(
        task_id=task_id,
        job_id=0,
        type=TaskType.COMPUTE,
        node=node,
        duration_us=duration_us,
        deps=deps or [],
    )


def _make_flow_task(task_id: int, src: int, dst: int, size_bytes: int, deps=None):
    return Task(
        task_id=task_id,
        job_id=0,
        type=TaskType.FLOW,
        src=src,
        dst=dst,
        size_bytes=size_bytes,
        comm_type=CommType.TP_ALLREDUCE_RING,
        deps=deps or [],
    )


def _make_workload(tasks):
    max_node = 0
    for t in tasks:
        if t.is_flow() and t.src is not None and t.dst is not None:
            max_node = max(max_node, t.src, t.dst)
        elif t.is_compute() and t.node is not None:
            max_node = max(max_node, t.node)
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=max_node + 1),
        tasks=tasks,
    )


def _make_simple_topo():
    """2-node topology: 0 ←400Gbps, 0.5us→ 1"""
    topo = NetworkTopology()
    topo.add_link(Link(src=0, dst=1, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))
    topo.add_link(Link(src=1, dst=0, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))
    return topo


def _make_star_topo():
    """Star: 0,1,2 connect to switch 10 (bidirectional, 400Gbps, 0.5us)."""
    topo = NetworkTopology()
    for leaf in [0, 1, 2]:
        topo.add_link(Link(src=leaf, dst=10, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))
        topo.add_link(Link(src=10, dst=leaf, bandwidth_gbps=400.0, latency_us=0.5, error_rate=0))
    return topo


# ============================================================
# Tests for _estimate_duration
# ============================================================


class TestEstimateDuration:
    """Tests for task duration estimation."""

    def test_compute_task_duration(self):
        """Compute task returns its duration_us."""
        topo = _make_simple_topo()
        hints = RoutingHints()
        task = _make_compute_task(0, duration_us=100)
        assert _estimate_duration(task, topo, hints) == 100

    def test_compute_task_no_duration(self):
        """Compute task without duration_us returns 0."""
        topo = _make_simple_topo()
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.COMPUTE, node=0)
        assert _estimate_duration(task, topo, hints) == 0

    def test_flow_direct_link(self):
        """Flow on direct link: duration = tx_time + latency."""
        topo = _make_simple_topo()
        hints = RoutingHints()
        # 1 byte = 8 bits, 400Gbps → tx = 8 / (400e9) * 1e6 = 0.00002 us
        # + 0.5 us latency → total ~0.5 us, int(0.50002) = 0
        # Use larger size: 50 MB = 50*1024*1024 bytes
        size = 50 * 1024 * 1024  # 50 MiB
        task = _make_flow_task(0, src=0, dst=1, size_bytes=size)
        dur = _estimate_duration(task, topo, hints)
        # tx = 50*1024*1024*8 / (400e9) * 1e6 = ~1048.58 us
        # + 0.5 us latency
        assert dur == pytest.approx(1049, abs=2)

    def test_flow_two_hops(self):
        """Flow through switch: bottleneck is min bandwidth, latency is sum."""
        topo = _make_star_topo()
        hints = RoutingHints()
        # Path 0 → 10 → 1, both links 400Gbps, 0.5us each
        size = 50 * 1024 * 1024
        task = _make_flow_task(0, src=0, dst=1, size_bytes=size)
        dur = _estimate_duration(task, topo, hints)
        # tx = same as direct (bottleneck = 400Gbps)
        # latency = 0.5 + 0.5 = 1.0 us
        # total ≈ 1048.58 + 1.0 ≈ 1050
        assert dur == pytest.approx(1050, abs=2)

    def test_flow_zero_size(self):
        """Flow with zero size_bytes returns 0."""
        topo = _make_simple_topo()
        hints = RoutingHints()
        task = _make_flow_task(0, src=0, dst=1, size_bytes=0)
        assert _estimate_duration(task, topo, hints) == 0

    def test_flow_none_src(self):
        """Flow with src=None returns 0."""
        topo = _make_simple_topo()
        hints = RoutingHints()
        task = Task(task_id=0, job_id=0, type=TaskType.FLOW, dst=1, size_bytes=1000)
        assert _estimate_duration(task, topo, hints) == 0


# ============================================================
# Tests for _topological_sort
# ============================================================


class TestTopologicalSort:
    """Tests for topological sort."""

    def test_empty_tasks(self):
        assert _topological_sort([]) == []

    def test_single_task(self):
        t = _make_compute_task(0, 100)
        assert _topological_sort([t]) == [t]

    def test_linear_chain(self):
        """A → B → C: topological order respects deps."""
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200, deps=[0])
        t2 = _make_compute_task(2, 300, deps=[1])
        result = _topological_sort([t2, t0, t1])  # Pass in shuffled order
        indices = [t.task_id for t in result]
        assert indices.index(0) < indices.index(1) < indices.index(2)

    def test_diamond_dependency(self):
        """Diamond: 0 → {1, 2} → 3."""
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 100, deps=[0])
        t2 = _make_compute_task(2, 100, deps=[0])
        t3 = _make_compute_task(3, 100, deps=[1, 2])
        result = _topological_sort([t3, t1, t0, t2])
        indices = [t.task_id for t in result]
        assert indices.index(0) < indices.index(1)
        assert indices.index(0) < indices.index(2)
        assert indices.index(1) < indices.index(3)
        assert indices.index(2) < indices.index(3)


# ============================================================
# Tests for analyze_critical_path
# ============================================================


class TestAnalyzeCriticalPath:
    """Tests for the main CPM analysis function."""

    def test_empty_workload(self):
        """Empty workload returns empty analysis."""
        topo = _make_simple_topo()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=0, num_nodes=0))
        hints = compute_routing_hints(topo, wl)
        result = analyze_critical_path(wl, topo, hints)
        assert result.task_timings == {}
        assert result.critical_tasks == []
        assert result.makespan_us == 0
        assert result.analysis_method == "cpm"

    def test_single_compute_task(self):
        """Single compute task: all timing fields equal its duration."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, duration_us=100)
        wl = _make_workload([t0])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        assert result.makespan_us == 100
        assert result.critical_tasks == [0]
        timing = result.task_timings[0]
        assert timing.earliest_start_us == 0
        assert timing.earliest_finish_us == 100
        assert timing.latest_start_us == 0
        assert timing.latest_finish_us == 100
        assert timing.slack_us == 0.0

    def test_linear_chain_all_critical(self):
        """A → B → C: all tasks have slack = 0."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200, deps=[0])
        t2 = _make_compute_task(2, 300, deps=[1])
        wl = _make_workload([t0, t1, t2])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        assert result.makespan_us == 600
        assert result.critical_tasks == [0, 1, 2]
        assert result.task_timings[0].earliest_finish_us == 100
        assert result.task_timings[1].earliest_start_us == 100
        assert result.task_timings[1].earliest_finish_us == 300
        assert result.task_timings[2].earliest_start_us == 300
        assert result.task_timings[2].earliest_finish_us == 600

    def test_parallel_branches_longer_is_critical(self):
        """Two branches of different length: only longer branch is critical."""
        topo = _make_simple_topo()
        # Branch 1: 0 → 1 (total 200us)
        # Branch 2: 2 → 3 → 4 (total 300us)
        # Start: task 0 and task 2 have no deps
        # Converge: task 5 depends on [1, 4] (duration 50us)
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 100, deps=[0])
        t2 = _make_compute_task(2, 100)
        t3 = _make_compute_task(3, 100, deps=[2])
        t4 = _make_compute_task(4, 100, deps=[3])
        t5 = _make_compute_task(5, 50, deps=[1, 4])
        wl = _make_workload([t0, t1, t2, t3, t4, t5])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        # Critical path: 2 → 3 → 4 → 5 (300 + 50 = 350us)
        # Non-critical: 0 → 1 (200us), slack = 350 - 200 - 50 = 100us for task 1
        assert result.makespan_us == 350
        assert 2 in result.critical_tasks
        assert 3 in result.critical_tasks
        assert 4 in result.critical_tasks
        assert 5 in result.critical_tasks
        # Task 0 and 1 are NOT critical
        assert result.task_timings[0].slack_us > 0
        assert result.task_timings[1].slack_us > 0

    def test_independent_tasks_all_critical(self):
        """Independent tasks (no deps): all have slack = 0."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200)
        wl = _make_workload([t0, t1])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        # Both start at 0, makespan = 200
        assert result.makespan_us == 200
        # Task 0 finishes at 100 but is on the critical path because
        # it has no dependents → latest_finish = makespan = 200
        # latest_start = 200 - 100 = 100, slack = 100 - 0 = 100
        assert result.task_timings[0].slack_us == 100.0
        assert result.task_timings[1].slack_us == 0.0
        assert result.critical_tasks == [1]

    def test_makespan_equals_max_finish(self):
        """makespan_us == max(earliest_finish_us) over all tasks."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200, deps=[0])
        t2 = _make_compute_task(2, 50, deps=[1])
        wl = _make_workload([t0, t1, t2])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        max_finish = max(t.earliest_finish_us for t in result.task_timings.values())
        assert result.makespan_us == max_finish

    def test_slack_nonnegative(self):
        """All slack values must be >= 0."""
        topo = _make_star_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_flow_task(1, src=0, dst=1, size_bytes=1024, deps=[0])
        t2 = _make_compute_task(2, 200, deps=[1])
        wl = _make_workload([t0, t1, t2])
        hints = compute_routing_hints(topo, wl)

        result = analyze_critical_path(wl, topo, hints)

        for timing in result.task_timings.values():
            assert timing.slack_us >= 0.0

    def test_critical_tasks_have_zero_slack(self):
        """Every task in critical_tasks must have slack_us == 0."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200, deps=[0])
        t2 = _make_compute_task(2, 300, deps=[1])
        wl = _make_workload([t0, t1, t2])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        for tid in result.critical_tasks:
            assert result.task_timings[tid].slack_us == 0.0

    def test_flow_duration_in_chain(self):
        """Flow task in a chain: duration is correctly estimated."""
        topo = _make_simple_topo()
        # 10 MiB flow on 400Gbps link: tx ≈ 0.2us + 0.5us latency ≈ 0.7us → int = 0
        # Use larger: 1 GiB = 1024^3 bytes
        size = 1024 * 1024 * 1024  # 1 GiB
        t0 = _make_compute_task(0, 1000)
        t1 = _make_flow_task(1, src=0, dst=1, size_bytes=size, deps=[0])
        t2 = _make_compute_task(2, 500, deps=[1])
        wl = _make_workload([t0, t1, t2])
        hints = compute_routing_hints(topo, wl)

        result = analyze_critical_path(wl, topo, hints)

        # Flow duration: tx = 1GiB * 8 / 400Gbps * 1e6 ≈ 21474.84us + 0.5us
        flow_dur = result.task_timings[1].earliest_finish_us - result.task_timings[1].earliest_start_us
        assert flow_dur == pytest.approx(21475, abs=2)
        assert result.task_timings[1].earliest_start_us == 1000

    def test_diamond_with_flow(self):
        """Diamond dependency with flow tasks."""
        topo = _make_star_topo()
        # 0: compute (100us)
        # 1: flow 0→1 (via switch 10)
        # 2: flow 0→2 (via switch 10)
        # 3: compute (100us) depends on [1, 2]
        t0 = _make_compute_task(0, 100)
        t1 = _make_flow_task(1, src=0, dst=1, size_bytes=1024 * 1024, deps=[0])  # 1 MiB
        t2 = _make_flow_task(2, src=0, dst=2, size_bytes=1024 * 1024, deps=[0])
        t3 = _make_compute_task(3, 100, deps=[1, 2])
        wl = _make_workload([t0, t1, t2, t3])
        hints = compute_routing_hints(topo, wl)

        result = analyze_critical_path(wl, topo, hints)

        # Both flows have same duration (same path length, same bandwidth)
        flow_dur_1 = result.task_timings[1].earliest_finish_us - result.task_timings[1].earliest_start_us
        flow_dur_2 = result.task_timings[2].earliest_finish_us - result.task_timings[2].earliest_start_us
        assert flow_dur_1 == flow_dur_2
        # All on critical path (parallel branches of equal length)
        assert result.critical_tasks == [0, 1, 2, 3]

    def test_analysis_method_is_cpm(self):
        """Default analysis method is 'cpm'."""
        topo = _make_simple_topo()
        wl = _make_workload([_make_compute_task(0, 100)])
        hints = RoutingHints()
        result = analyze_critical_path(wl, topo, hints)
        assert result.analysis_method == "cpm"

    def test_get_slack_and_is_critical(self):
        """Test CriticalPathInfo accessor methods."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200, deps=[0])
        wl = _make_workload([t0, t1])
        hints = RoutingHints()

        result = analyze_critical_path(wl, topo, hints)

        assert result.get_slack(0) == 0.0
        assert result.get_slack(1) == 0.0
        assert result.is_critical(0)
        assert result.is_critical(1)
        assert result.get_earliest_start(0) == 0
        assert result.get_earliest_start(1) == 100


# ============================================================
# Tests for custom analysis strategy
# ============================================================


class TestCustomAnalysisStrategy:
    """Tests for pluggable critical path analysis strategy."""

    def test_default_is_cpm(self):
        """Without strategy argument, uses CPM."""
        topo = _make_simple_topo()
        wl = _make_workload([_make_compute_task(0, 100)])
        hints = RoutingHints()
        result = analyze_critical_path(wl, topo, hints)
        assert result.analysis_method == "cpm"

    def test_explicit_cpm_strategy(self):
        """Passing analyze_cpm explicitly gives same result."""
        topo = _make_simple_topo()
        t0 = _make_compute_task(0, 100)
        t1 = _make_compute_task(1, 200, deps=[0])
        wl = _make_workload([t0, t1])
        hints = RoutingHints()

        default = analyze_critical_path(wl, topo, hints)
        explicit = analyze_critical_path(wl, topo, hints, analysis_strategy=analyze_cpm)

        assert default.makespan_us == explicit.makespan_us
        assert default.critical_tasks == explicit.critical_tasks

    def test_custom_strategy(self):
        """Custom strategy is invoked and its result returned."""
        topo = _make_simple_topo()
        wl = _make_workload([_make_compute_task(0, 100)])
        hints = RoutingHints()

        def mock_strategy(workload, topology, routing_hints):
            return CriticalPathInfo(
                task_timings={
                    0: TaskTimingInfo(
                        task_id=0,
                        earliest_start_us=0,
                        earliest_finish_us=42,
                        latest_start_us=0,
                        latest_finish_us=42,
                        slack_us=0.0,
                        is_critical=True,
                    )
                },
                critical_tasks=[0],
                makespan_us=42,
                analysis_method="mock",
            )

        result = analyze_critical_path(wl, topo, hints, analysis_strategy=mock_strategy)
        assert result.analysis_method == "mock"
        assert result.makespan_us == 42

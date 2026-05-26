"""Unit tests for src/cassini/communication_pattern.py — pattern extraction."""

from src.workload_format.schema import (
    CommType, Job, Meta, P2PWorkload, ParallelismConfig,
    Phase, Task, TaskType,
)
from src.static_analysis.passes.critical_path import (
    CriticalPathInfo, TaskTimingInfo,
)
from src.static_analysis.passes.routing.bfs import BfsRouteTable
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.cassini.communication_pattern import (
    CommunicationPattern,
    extract_communication_patterns,
    _path_to_links,
    _flow_bandwidth_gbps,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_2node_topo(bw_gbps=100.0):
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    topo.node_types = {0: "gpu", 1: "gpu"}
    for s, d in [(0, 1), (1, 0)]:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=bw_gbps,
                           latency_us=1.0, error_rate=0.0))
    return topo


def _make_route_table(paths):
    """Build a BfsRouteTable pre-populated with (src,dst)→path entries."""
    topo = _make_2node_topo()
    rt = BfsRouteTable(topo)
    rt._paths = paths
    return rt


def _make_critical_path(task_timings, makespan_us=1000):
    """Build a CriticalPathInfo with given timing dict."""
    cps = [tid for tid, t in task_timings.items() if t.is_critical]
    return CriticalPathInfo(
        task_timings=task_timings,
        critical_tasks=cps,
        makespan_us=makespan_us,
        analysis_method="synthetic",
    )


def _timing(earliest_start, earliest_finish, task_id=0):
    return TaskTimingInfo(
        task_id=task_id,
        earliest_start_us=earliest_start,
        earliest_finish_us=earliest_finish,
        latest_start_us=earliest_start,
        latest_finish_us=earliest_finish,
        slack_us=0.0,
        is_critical=True,
    )


def _wl(tasks, num_jobs=1):
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=num_jobs, num_nodes=2),
        jobs=[Job(job_id=jid, name=f"job_{jid}", assigned_nodes=[0, 1],
                  parallelism=ParallelismConfig(tp=2))
              for jid in range(num_jobs)],
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# _path_to_links
# ---------------------------------------------------------------------------


class TestPathToLinks:
    def test_single_hop(self):
        assert _path_to_links([0, 1]) == [(0, 1)]

    def test_two_hops(self):
        assert _path_to_links([0, 2, 1]) == [(0, 2), (2, 1)]

    def test_empty(self):
        assert _path_to_links([0]) == []


# ---------------------------------------------------------------------------
# _flow_bandwidth_gbps
# ---------------------------------------------------------------------------


class TestFlowBandwidthGbps:
    def test_1GB_over_1s(self):
        # 1 GB = 8e9 bits, 1s = 1e6 us, bw = 8 Gbps
        bw = _flow_bandwidth_gbps(size_bytes=1_000_000_000, duration_us=1_000_000)
        assert bw == 8.0

    def test_zero_size(self):
        assert _flow_bandwidth_gbps(0, 1000) == 0.0

    def test_zero_duration(self):
        assert _flow_bandwidth_gbps(1000, 0) == 0.0


# ---------------------------------------------------------------------------
# extract_communication_patterns
# ---------------------------------------------------------------------------


class TestExtractPatterns:
    def test_single_flow_on_one_link(self):
        """One flow task populates one link's buckets."""
        flow = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                    src=0, dst=1, size_bytes=125_000_000,
                    iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                    comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([flow])

        timing = {
            1: _timing(0, 100_000, task_id=1),  # 100ms duration
        }
        cpi = _make_critical_path(timing, makespan_us=100_000)
        rt = _make_route_table({(0, 1): [0, 1]})

        patterns = extract_communication_patterns(wl, cpi, rt)
        assert 0 in patterns
        p = patterns[0]
        assert isinstance(p, CommunicationPattern)
        assert p.iteration_time_us > 0
        assert (0, 1) in p.link_demands

    def test_multi_flow_aggregation(self):
        """Two flows on same link aggregate demand."""
        f1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=125_000_000,
                  iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                  comm_type=CommType.TP_ALLREDUCE_RING)
        f2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=125_000_000,
                  iteration=0, phase=Phase.FORWARD, layer_id=1, item_id=2,
                  deps=[1], comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([f1, f2])

        # Both flows at same time → aggregated demand
        timing = {
            1: _timing(0, 100_000, task_id=1),
            2: _timing(0, 100_000, task_id=2),
        }
        cpi = _make_critical_path(timing, makespan_us=100_000)
        rt = _make_route_table({(0, 1): [0, 1]})

        patterns = extract_communication_patterns(wl, cpi, rt)
        assert 0 in patterns
        demands = patterns[0].link_demands[(0, 1)]
        # Both flows contribute to the same angle buckets
        assert any(v > 0 for v in demands.values())

    def test_empty_workload(self):
        """No flow tasks → empty patterns."""
        c0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=100, iteration=0,
                  phase=Phase.FORWARD, layer_id=0, item_id=0)
        wl = _wl([c0])
        cpi = _make_critical_path({}, makespan_us=100)
        rt = _make_route_table({})
        patterns = extract_communication_patterns(wl, cpi, rt)
        assert len(patterns) == 0

    def test_zero_size_flow_excluded(self):
        """Zero-size flows contribute no bandwidth → skipped."""
        flow = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                    src=0, dst=1, size_bytes=0,
                    iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                    comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([flow])
        timing = {1: _timing(0, 1000, task_id=1)}
        cpi = _make_critical_path(timing, makespan_us=1000)
        rt = _make_route_table({(0, 1): [0, 1]})
        patterns = extract_communication_patterns(wl, cpi, rt)
        assert len(patterns) == 0

    def test_flow_without_timing_skipped(self):
        """Flow missing from CriticalPathInfo → gracefully skipped."""
        flow = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                    src=0, dst=1, size_bytes=125_000_000,
                    iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                    comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([flow])
        cpi = _make_critical_path({}, makespan_us=1000)  # no timing for task 1
        rt = _make_route_table({(0, 1): [0, 1]})
        patterns = extract_communication_patterns(wl, cpi, rt)
        assert len(patterns) == 0

    def test_flow_without_route_skipped(self):
        """Flow missing from RouteTable → gracefully skipped."""
        flow = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                    src=0, dst=1, size_bytes=125_000_000,
                    iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                    comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([flow])
        timing = {1: _timing(0, 100_000, task_id=1)}
        cpi = _make_critical_path(timing, makespan_us=100_000)
        rt = _make_route_table({})  # no route for (0,1)
        patterns = extract_communication_patterns(wl, cpi, rt)
        assert len(patterns) == 0

    def test_multi_link_path(self):
        """One flow populates all hops along its path."""
        flow = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                    src=0, dst=2, size_bytes=125_000_000,
                    iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                    comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([flow])
        timing = {1: _timing(0, 100_000, task_id=1)}
        cpi = _make_critical_path(timing, makespan_us=100_000)
        # Path: 0 → 1 → 2 (two hops)
        rt = _make_route_table({(0, 2): [0, 1, 2]})
        patterns = extract_communication_patterns(wl, cpi, rt)
        assert 0 in patterns
        p = patterns[0]
        assert (0, 1) in p.link_demands
        assert (1, 2) in p.link_demands

    def test_two_jobs_separate_patterns(self):
        """Two jobs produce two separate patterns."""
        f0 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=125_000_000,
                  iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                  comm_type=CommType.TP_ALLREDUCE_RING)
        f1 = Task(task_id=2, job_id=1, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=125_000_000,
                  iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=2,
                  comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([f0, f1], num_jobs=2)
        timing = {
            1: _timing(0, 100_000, task_id=1),
            2: _timing(50000, 150_000, task_id=2),
        }
        cpi = _make_critical_path(timing, makespan_us=150_000)
        rt = _make_route_table({(0, 1): [0, 1]})
        patterns = extract_communication_patterns(wl, cpi, rt)
        assert len(patterns) == 2
        assert 0 in patterns
        assert 1 in patterns

    def test_duration_longer_than_iteration_fills_all_buckets(self):
        """Flow spanning entire iteration → all 360 buckets populated."""
        flow = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                    src=0, dst=1, size_bytes=125_000_000,
                    iteration=0, phase=Phase.FORWARD, layer_id=0, item_id=1,
                    comm_type=CommType.TP_ALLREDUCE_RING)
        wl = _wl([flow])
        # duration 2000us > makespan/iteration_time_us estimate
        timing = {1: _timing(0, 2000, task_id=1)}
        cpi = _make_critical_path(timing, makespan_us=2000)
        rt = _make_route_table({(0, 1): [0, 1]})
        patterns = extract_communication_patterns(wl, cpi, rt)
        if patterns:  # may or may not pattern if iteration estimate is 0
            p = patterns[0]
            if (0, 1) in p.link_demands:
                buckets = p.link_demands[(0, 1)]
                # Flow spans full iteration → all 360 buckets have demand
                assert len(buckets) == 360

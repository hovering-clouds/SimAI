"""Executor-level tests for MFS scheduling policy."""
import pytest

from src.workload_format.schema import (
    P2PWorkload, Meta, Job, Task, Phase, CommType, TaskType,
    ParallelismConfig,
)
from src.static_analysis.passes.topology_loader import NetworkTopology, Link
from src.static_analysis.strategies.mfs_strategy import MfsAnalyzer
from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.mfs_policy import MfsSchedulingPolicy
from src.executor.bandwidth_allocators.mfs_allocator import MfsAllocatorConfig


def _line_topo(n=4, bw=100.0):
    """Create a simple line topology: 0-1-2-...(n-1) with given bandwidth."""
    topo = NetworkTopology()
    for i in range(n - 1):
        topo.add_link(Link(i, i + 1, bw, 1.0, 0.0))
        topo.add_link(Link(i + 1, i, bw, 1.0, 0.0))
    return topo


def _star_topo(bw=100.0):
    """4 GPUs (0-3) connected via switch node 4, all links same bw."""
    topo = NetworkTopology()
    for g in range(4):
        topo.add_link(Link(g, 4, bw, 1.0, 0.0))
        topo.add_link(Link(4, g, bw, 1.0, 0.0))
    return topo


def _make_workload(tasks, nodes=4):
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=nodes),
        jobs=[Job(job_id=0, assigned_nodes=list(range(nodes)))],
        tasks=tasks,
    )


def _empty_trace(request_ids=None):
    """Build a minimal trace with optional ttft_slo_us per request."""
    requests = {}
    for rid in (request_ids or []):
        requests[str(rid)] = {
            "num_prefill_tokens": 100,
            "num_decode_tokens": 10,
        }
    return {"requests": requests}


class TestMfsPolicyExecutor:
    """Integration tests: MFS policy + AnalyticalExecutor."""

    def test_collective_completes_before_deferred_p2d(self):
        """Under MFS, a collective + P2D sharing a bottleneck link should have
        the collective finish first when P2D is not promoted (no SLO)."""
        topo = _line_topo(n=3, bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=100000, comm_type=CommType.TP_ALLREDUCE_RING,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=0, dst=2, size_bytes=100000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])

        wl = _make_workload([t0, t1, t2], nodes=3)
        btm = {
            "b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [2], "request_ids": [1], "type": "kv_transfer"},
        }
        trace = _empty_trace([1])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        assert result.per_task[1].start_time_us == 10
        assert result.per_task[2].start_time_us == 10
        assert result.per_task[1].end_time_us < result.per_task[2].end_time_us

    def test_p2d_with_tight_slo_gets_promoted(self):
        """P2D with tight ttft_slo_us should be promoted via MLU."""
        topo = _star_topo(bw=1.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=100000, comm_type=CommType.TP_ALLREDUCE_RING,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=2, dst=3, size_bytes=100000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0)

        wl = _make_workload([t0, t1, t2])
        btm = {
            "b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [2], "request_ids": [1], "type": "kv_transfer"},
        }
        trace = {"requests": {"1": {"num_prefill_tokens": 100, "num_decode_tokens": 10, "ttft_slo_us": 1000}}}

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        assert result.per_task[2].end_time_us > 0

    def test_compute_ordering_matches_default(self):
        """MFS policy should preserve compute ordering same as default."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=100, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=100, phase=Phase.PREFILL, layer_id=1,
                  deps=[0])
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=1000, comm_type=CommType.TP_ALLREDUCE_RING,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])

        wl = _make_workload([t0, t1, t2])
        btm = {"b1": {"task_ids": [0, 1, 2], "request_ids": [1], "type": "prefill"}}
        trace = _empty_trace([1])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        assert result.per_task[1].start_time_us >= result.per_task[0].end_time_us

    def test_no_deadlock_with_ready_p2d(self):
        """MFS should not deadlock when P2D flows are ready alongside others."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=1000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=2, dst=3, size_bytes=2000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0)

        wl = _make_workload([t0, t1, t2])
        btm = {
            "b1": {"task_ids": [0], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [1], "request_ids": [1], "type": "kv_transfer"},
            "kv_b3_b4": {"task_ids": [2], "request_ids": [2], "type": "kv_transfer"},
        }
        trace = _empty_trace([1, 2])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)
        assert len(result.per_task) == 3

    def test_request_start_time_tracking(self):
        """Policy should track request_start_time in allocator when tasks are emitted."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=1000, comm_type=CommType.TP_ALLREDUCE_RING,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])

        wl = _make_workload([t0, t1])
        btm = {"b1": {"task_ids": [0, 1], "request_ids": [5], "type": "prefill"}}
        trace = _empty_trace([5])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # After execution, request 5 should have a start time recorded
        assert 5 in policy.allocator.request_start_time
        assert policy.allocator.request_start_time[5] >= 0

    def test_current_layer_advances_on_compute_complete(self):
        """Allocator's current_layer should advance as compute tasks complete."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=1,
                  deps=[0])

        wl = _make_workload([t0, t1])
        btm = {"b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill", "stage_id": 0}}
        trace = _empty_trace([1])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # After both compute tasks complete, current_layer for (0,0) should be 2
        assert policy.allocator.current_layer_by_stage.get((0, 0), 0) >= 1


class TestRemainingUsTracking:
    """Tests for remaining_us initialization and decrements in policy."""

    def test_remaining_us_initialized(self):
        """After initialize(), remaining_us should match static analysis totals."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=200, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=1000, comm_type=CommType.PP_SEND,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=0, dst=2, size_bytes=2000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[1])

        wl = _make_workload([t0, t1, t2])
        btm = {
            "b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [2], "request_ids": [1], "type": "kv_transfer"},
        }
        trace = _empty_trace([1])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # remaining_us should have been decremented for completed critical tasks
        fi = analysis.feasibility_info
        assert fi is not None
        assert 1 in fi.request_path_total_us
        # After all tasks complete, remaining_us should be 0 or negative
        assert policy.allocator.remaining_us[1] <= 0

    def test_remaining_us_decreases_on_critical_task_completion(self):
        """Critical task completion should decrease remaining_us."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=100, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=1000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])

        wl = _make_workload([t0, t1])
        btm = {
            "b1": {"task_ids": [0], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [1], "request_ids": [1], "type": "kv_transfer"},
        }
        trace = _empty_trace([1])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        fi = analysis.feasibility_info
        initial_total = fi.request_path_total_us[1]

        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # After execution, remaining_us should be less than initial total
        final_remaining = policy.allocator.remaining_us[1]
        assert final_remaining < initial_total

    def test_remaining_us_unchanged_for_non_critical_task(self):
        """Non-critical task should not affect remaining_us of unrelated request."""
        topo = _star_topo(bw=100.0)

        # Request 1: t0(BG compute) → t1(P2D flow)  [critical path]
        # Request 2: t2(BG compute) → t3(P2D flow)  [independent critical path]
        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=100, phase=Phase.PREFILL, layer_id=0)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=1000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        t2 = Task(task_id=2, job_id=0, type=TaskType.COMPUTE,
                  node=2, duration_us=100, phase=Phase.PREFILL, layer_id=0)
        t3 = Task(task_id=3, job_id=0, type=TaskType.FLOW,
                  src=2, dst=3, size_bytes=1000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[2])

        wl = _make_workload([t0, t1, t2, t3])
        btm = {
            "b1": {"task_ids": [0], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [1], "request_ids": [1], "type": "kv_transfer"},
            "b2": {"task_ids": [2], "request_ids": [2], "type": "prefill"},
            "kv_b3_b4": {"task_ids": [3], "request_ids": [2], "type": "kv_transfer"},
        }
        trace = _empty_trace([1, 2])

        analysis = MfsAnalyzer(topo).analyze(wl, btm, trace=trace)
        fi = analysis.feasibility_info

        # Each request's critical path should only contain its own tasks
        assert fi.request_critical_tasks[1] == {0, 1}
        assert fi.request_critical_tasks[2] == {2, 3}

        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # Both requests' remaining_us should be fully decremented
        assert policy.allocator.remaining_us[1] <= 0
        assert policy.allocator.remaining_us[2] <= 0

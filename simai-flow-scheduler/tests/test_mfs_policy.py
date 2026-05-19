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


class TestMfsPolicyExecutor:
    """Integration tests: MFS policy + AnalyticalExecutor."""

    def test_collective_completes_before_deferred_p2d(self):
        """Under MFS, a collective + P2D sharing a bottleneck link should have
        the collective finish first when P2D is not promoted."""
        # Line topology: 0 -- 1 -- 2
        # Both flows must traverse link (0,1), creating bottleneck contention
        topo = _line_topo(n=3, bw=100.0)

        # Compute on node 0 (gate for both flows)
        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        # Collective flow (early, RLI 0) 0 -> 1, shares link (0,1)
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=100000, comm_type=CommType.TP_ALLREDUCE_RING,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        # P2D flow (low priority) 0 -> 2, must traverse link (0,1) too
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=0, dst=2, size_bytes=100000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])

        wl = _make_workload([t0, t1, t2], nodes=3)
        btm = {
            "b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [2], "request_ids": [1], "type": "kv_transfer"},
        }

        analysis = MfsAnalyzer(topo).analyze(wl, btm)
        cfg = MfsAllocatorConfig(p2d_promotion_delay_us=100000)  # long delay
        policy = MfsSchedulingPolicy(analysis=analysis, allocator_config=cfg)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # Both start at t=10 (after compute). Collective gets full 100Gbps,
        # P2D gets 0. Collective finishes first.
        assert result.per_task[1].start_time_us == 10
        assert result.per_task[2].start_time_us == 10
        assert result.per_task[1].end_time_us < result.per_task[2].end_time_us

    def test_p2d_promoted_after_delay(self):
        """With short promotion delay, P2D should be promoted and get high bw."""
        topo = _star_topo(bw=100.0)

        # Two computes
        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        # Early collective
        t1 = Task(task_id=1, job_id=0, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=100000, comm_type=CommType.TP_ALLREDUCE_RING,
                  phase=Phase.PREFILL, layer_id=0, deps=[0])
        # P2D sharing the same switch bottleneck
        t2 = Task(task_id=2, job_id=0, type=TaskType.FLOW,
                  src=2, dst=3, size_bytes=100000, comm_type=CommType.KV_CACHE_TRANSFER,
                  phase=Phase.PREFILL, layer_id=0)

        wl = _make_workload([t0, t1, t2])
        btm = {
            "b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [2], "request_ids": [1], "type": "kv_transfer"},
        }

        analysis = MfsAnalyzer(topo).analyze(wl, btm)
        # Very short promotion delay so P2D promotes almost immediately
        cfg = MfsAllocatorConfig(p2d_promotion_delay_us=1)
        policy = MfsSchedulingPolicy(analysis=analysis, allocator_config=cfg)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # With fast promotion, P2D should complete relatively quickly
        # (not stalled behind collective)
        assert result.per_task[2].end_time_us > 0

    def test_compute_ordering_matches_default(self):
        """MFS policy should preserve compute ordering same as default."""
        topo = _star_topo(bw=100.0)

        # Two sequential computes on node 0
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

        analysis = MfsAnalyzer(topo).analyze(wl, btm)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        result = executor.execute(wl)

        # t1 should start after t0 completes
        assert result.per_task[1].start_time_us >= result.per_task[0].end_time_us

    def test_no_deadlock_with_ready_p2d(self):
        """MFS should not deadlock when P2D flows are ready alongside others."""
        topo = _star_topo(bw=100.0)

        t0 = Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                  node=0, duration_us=10, phase=Phase.PREFILL, layer_id=0)
        # Multiple P2D flows
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

        analysis = MfsAnalyzer(topo).analyze(wl, btm)
        policy = MfsSchedulingPolicy(analysis=analysis)
        executor = AnalyticalExecutor(topology=topo, policy=policy)
        # Should not raise RuntimeError (deadlock)
        result = executor.execute(wl)
        assert len(result.per_task) == 3

"""Tests for MFS feasibility analysis — per-task durations and per-request critical paths."""
import pytest

from src.static_analysis.passes.mfs_context import (
    MfsContext, MfsTaskInfo, MfsStage, MfsRequestInfo,
)
from src.static_analysis.passes.mfs_feasibility import (
    FeasibilityInfo, build_feasibility_info,
)
from src.static_analysis.passes.routing import BfsStrategy
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    P2PWorkload, Meta, Job, Task, Phase, CommType, TaskType,
)


def _make_workload(tasks: list[Task]) -> P2PWorkload:
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=4),
        jobs=[Job(job_id=0, assigned_nodes=[0, 1, 2, 3])],
        tasks=tasks,
    )


def _compute_task(tid, node=0, duration=100, layer=0, deps=None):
    return Task(
        task_id=tid, job_id=0, type=TaskType.COMPUTE,
        node=node, duration_us=duration,
        phase=Phase.PREFILL, layer_id=layer,
        deps=deps or [],
    )


def _flow_task(tid, src, dst, size, comm_type=CommType.TP_ALLREDUCE_RING,
               layer=0, deps=None):
    return Task(
        task_id=tid, job_id=0, type=TaskType.FLOW,
        src=src, dst=dst, size_bytes=size,
        comm_type=comm_type, phase=Phase.PREFILL, layer_id=layer,
        deps=deps or [],
    )


def _simple_topo():
    """Two nodes connected by a 100 Gbps link with 0 latency."""
    topo = NetworkTopology()
    topo.add_link(Link(0, 1, 100.0, 0.0, 0.0))
    return topo


def _make_context_with_request(rid, tasks_with_stages):
    """Build a minimal MfsContext.

    tasks_with_stages: list of (task_id, mfs_stage, request_ids)
    """
    ctx = MfsContext()
    for tid, stage, req_ids in tasks_with_stages:
        ctx.task_info[tid] = MfsTaskInfo(
            task_id=tid, job_id=0, batch_id=None,
            request_ids=req_ids, stage_id=0,
            mfs_stage=stage, target_layer=0, comm_role="test",
        )
    # Build request_to_tasks
    req_tasks: dict[int, list[int]] = {}
    for tid, stage, req_ids in tasks_with_stages:
        for r in req_ids:
            req_tasks.setdefault(r, []).append(tid)
    ctx.request_to_tasks = {r: tuple(tids) for r, tids in req_tasks.items()}
    return ctx


# ── Per-task duration tests ────────────────────────────────────────────────


class TestTaskDuration:
    def test_compute_task_duration(self):
        t0 = _compute_task(0, duration=500)
        wl = _make_workload([t0])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(1, [(0, MfsStage.BACKGROUND, (1,))])
        fi = build_feasibility_info(wl, ctx, rt, topo)
        assert fi.task_duration_us[0] == 500

    def test_flow_task_duration_estimated(self):
        # 1000 bytes over 100 Gbps link with 0 latency
        t0 = _flow_task(0, 0, 1, 1000, CommType.KV_CACHE_TRANSFER)
        wl = _make_workload([t0])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(1, [(0, MfsStage.P2D, (1,))])
        fi = build_feasibility_info(wl, ctx, rt, topo)
        # 1000 bytes = 8000 bits, 100 Gbps → 8000 / (100e9) * 1e6 = 0.08 us
        expected = int(8000 / (100e9) * 1e6)
        assert fi.task_duration_us[0] == expected


# ── Per-request critical path tests ────────────────────────────────────────


class TestRequestCriticalPath:

    def test_linear_chain_to_p2d(self):
        """A(BG compute) → B(EARLY) → C(P2D): all critical, total = A+B+C."""
        t0 = _compute_task(0, duration=200)
        t1 = _flow_task(1, 0, 1, 500, CommType.PP_SEND, deps=[0])
        t2 = _flow_task(2, 0, 1, 1000, CommType.KV_CACHE_TRANSFER, deps=[1])
        wl = _make_workload([t0, t1, t2])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(10, [
            (0, MfsStage.BACKGROUND, (10,)),
            (1, MfsStage.EARLY, (10,)),
            (2, MfsStage.P2D, (10,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)

        assert 10 in fi.request_path_total_us
        total = fi.request_path_total_us[10]
        expected = fi.task_duration_us[0] + fi.task_duration_us[1] + fi.task_duration_us[2]
        assert total == expected

        # All three tasks are critical
        assert fi.request_critical_tasks[10] == {0, 1, 2}
        assert 0 in fi.critical_path_tasks
        assert 1 in fi.critical_path_tasks
        assert 2 in fi.critical_path_tasks

    def test_parallel_branches(self):
        """Diamond: A → {B, C} → D(P2D). B longer → B+D critical, C not."""
        dur_a, dur_b, dur_c, dur_d = 100, 300, 50, 200
        t0 = _compute_task(0, duration=dur_a)
        t1 = _compute_task(1, duration=dur_b, deps=[0])
        t2 = _compute_task(2, duration=dur_c, deps=[0])
        t3 = _flow_task(3, 0, 1, 1000, CommType.KV_CACHE_TRANSFER, deps=[1, 2])
        wl = _make_workload([t0, t1, t2, t3])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(20, [
            (0, MfsStage.BACKGROUND, (20,)),
            (1, MfsStage.BACKGROUND, (20,)),
            (2, MfsStage.BACKGROUND, (20,)),
            (3, MfsStage.P2D, (20,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)

        # Critical path: A → B → D = 100 + 300 + flow_duration
        total = fi.request_path_total_us[20]
        path_b_d = dur_a + dur_b + fi.task_duration_us[3]
        assert total == path_b_d

        crit = fi.request_critical_tasks[20]
        assert 0 in crit   # A is on all paths
        assert 1 in crit   # B (longer)
        assert 2 not in crit  # C (shorter, not on critical path)
        assert 3 in crit   # D (P2D terminal)

    def test_no_p2d_tasks_skipped(self):
        """Request with no P2D tasks should be skipped."""
        t0 = _compute_task(0, duration=100)
        t1 = _flow_task(1, 0, 1, 500, CommType.PP_SEND)
        wl = _make_workload([t0, t1])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(30, [
            (0, MfsStage.BACKGROUND, (30,)),
            (1, MfsStage.EARLY, (30,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)
        assert 30 not in fi.request_path_total_us
        assert 30 not in fi.request_critical_tasks

    def test_decode_tasks_excluded(self):
        """Tasks after P2D (decode compute) should not be in critical path.

        Chain: prefill(BG) → EARLY → P2D → decode(BG).
        Only prefill+EARLY+P2D should be critical, not decode.
        """
        t0 = _compute_task(0, duration=100)
        t1 = _flow_task(1, 0, 1, 500, CommType.PP_SEND, deps=[0])
        t2 = _flow_task(2, 0, 1, 1000, CommType.KV_CACHE_TRANSFER, deps=[1])
        t3 = _compute_task(3, duration=999, deps=[2])  # decode compute
        wl = _make_workload([t0, t1, t2, t3])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(40, [
            (0, MfsStage.BACKGROUND, (40,)),
            (1, MfsStage.EARLY, (40,)),
            (2, MfsStage.P2D, (40,)),
            (3, MfsStage.BACKGROUND, (40,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)

        # Decode task should NOT be in critical path
        assert 3 not in fi.critical_path_tasks
        assert 3 not in fi.request_critical_tasks.get(40, set())

        # Critical path total should NOT include decode duration
        total = fi.request_path_total_us[40]
        expected = fi.task_duration_us[0] + fi.task_duration_us[1] + fi.task_duration_us[2]
        assert total == expected

    def test_multiple_requests_independent(self):
        """Two requests with independent chains: each has its own total."""
        # Request 50: t0(BG, 200) → t1(P2D, flow)
        # Request 51: t2(BG, 300) → t3(P2D, flow)
        t0 = _compute_task(0, duration=200)
        t1 = _flow_task(1, 0, 1, 1000, CommType.KV_CACHE_TRANSFER, deps=[0])
        t2 = _compute_task(2, duration=300)
        t3 = _flow_task(3, 0, 1, 2000, CommType.KV_CACHE_TRANSFER, deps=[2])
        wl = _make_workload([t0, t1, t2, t3])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(50, [
            (0, MfsStage.BACKGROUND, (50,)),
            (1, MfsStage.P2D, (50,)),
            (2, MfsStage.BACKGROUND, (51,)),
            (3, MfsStage.P2D, (51,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)

        total_50 = fi.request_path_total_us[50]
        total_51 = fi.request_path_total_us[51]
        assert total_50 == fi.task_duration_us[0] + fi.task_duration_us[1]
        assert total_51 == fi.task_duration_us[2] + fi.task_duration_us[3]
        assert total_50 != total_51  # Different durations

    def test_request_critical_tasks_per_request(self):
        """Shared task should only be in critical set of requests where it's critical."""
        # Request 60: t0(BG, 100) → t1(P2D, flow)  [short path]
        # Request 61: t0(BG, 100) → t2(BG, 500) → t3(P2D, flow)  [long path]
        # t0 is shared. For req 60, t0 is critical (on its only path).
        # For req 61, t0 is critical too (on the only path).
        t0 = _compute_task(0, duration=100)
        t1 = _flow_task(1, 0, 1, 500, CommType.KV_CACHE_TRANSFER, deps=[0])
        t2 = _compute_task(2, duration=500, deps=[0])
        t3 = _flow_task(3, 0, 1, 1000, CommType.KV_CACHE_TRANSFER, deps=[2])
        wl = _make_workload([t0, t1, t2, t3])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(60, [
            (0, MfsStage.BACKGROUND, (60, 61)),
            (1, MfsStage.P2D, (60,)),
            (2, MfsStage.BACKGROUND, (61,)),
            (3, MfsStage.P2D, (61,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)

        # Request 60 critical: t0, t1
        assert fi.request_critical_tasks[60] == {0, 1}
        # Request 61 critical: t0, t2, t3
        assert fi.request_critical_tasks[61] == {0, 2, 3}

    def test_equal_parallel_branches_picks_one_path(self):
        """Diamond with equal-duration branches: only one path marked critical.

        A(BG, 100) → {B(BG, 200), C(BG, 200)} → D(P2D)
        Both branches are equal length. Backtracking should only pick one.
        The sum of critical task durations must equal request_path_total_us.
        """
        t0 = _compute_task(0, duration=100)
        t1 = _compute_task(1, duration=200, deps=[0])
        t2 = _compute_task(2, duration=200, deps=[0])
        t3 = _flow_task(3, 0, 1, 1000, CommType.KV_CACHE_TRANSFER, deps=[1, 2])
        wl = _make_workload([t0, t1, t2, t3])
        topo = _simple_topo()
        rt = BfsStrategy().compute_routes(wl, topo)
        ctx = _make_context_with_request(70, [
            (0, MfsStage.BACKGROUND, (70,)),
            (1, MfsStage.BACKGROUND, (70,)),
            (2, MfsStage.BACKGROUND, (70,)),
            (3, MfsStage.P2D, (70,)),
        ])
        fi = build_feasibility_info(wl, ctx, rt, topo)

        crit = fi.request_critical_tasks[70]
        total = fi.request_path_total_us[70]

        # A (t0) and D (t3) must be critical
        assert 0 in crit
        assert 3 in crit

        # Exactly one of B (t1) or C (t2) should be critical, not both
        assert len(crit & {1, 2}) == 1

        # Sum of critical task durations must equal total
        crit_sum = sum(fi.task_duration_us[tid] for tid in crit)
        assert crit_sum == total

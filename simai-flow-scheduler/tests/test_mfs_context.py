"""Tests for MFS context metadata and RLI computation."""
import pytest

from src.workload_format.schema import (
    P2PWorkload, Meta, Job, Task, Phase, CommType, TaskType,
    ParallelismConfig,
)
from src.static_analysis.passes.mfs_context import (
    MfsContext, MfsTaskInfo, MfsStage,
    build_mfs_context,
)
from src.static_analysis.passes.mfs_rli import RliInfo, compute_static_rli


def _make_workload(tasks: list[Task]) -> P2PWorkload:
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=4),
        jobs=[Job(job_id=0, assigned_nodes=[0, 1, 2, 3])],
        tasks=tasks,
    )


def _compute_task(tid, node=0, duration=100, layer=0, phase=Phase.PREFILL):
    return Task(
        task_id=tid, job_id=0, type=TaskType.COMPUTE,
        node=node, duration_us=duration,
        phase=phase, layer_id=layer,
    )


def _flow_task(tid, src, dst, size, comm_type=CommType.TP_ALLREDUCE_RING,
               layer=0, phase=Phase.PREFILL, deps=None):
    return Task(
        task_id=tid, job_id=0, type=TaskType.FLOW,
        src=src, dst=dst, size_bytes=size,
        comm_type=comm_type, phase=phase, layer_id=layer,
        deps=deps or [],
    )


# ── MfsContext tests ──────────────────────────────────────────────────────────


class TestBuildMfsContext:
    """Tests for build_mfs_context."""

    def test_kv_transfer_classified_as_p2d(self):
        """KV_CACHE_TRANSFER flows should be MfsStage.P2D."""
        t0 = _compute_task(0, node=0)
        t1 = _flow_task(1, 0, 2, 1000, CommType.KV_CACHE_TRANSFER)
        wl = _make_workload([t0, t1])
        btm = {"kv_b1_b2": {"task_ids": [1], "request_ids": [10], "type": "kv_transfer"}}
        ctx = build_mfs_context(wl, btm)
        assert ctx.task_info[1].mfs_stage == MfsStage.P2D
        assert ctx.task_info[1].comm_role == "p2d_transfer"

    def test_pp_send_classified_as_early(self):
        """PP_SEND flows should be MfsStage.EARLY."""
        t1 = _flow_task(1, 0, 1, 500, CommType.PP_SEND)
        wl = _make_workload([t1])
        btm = {"pp_b1_b2": {"task_ids": [1], "request_ids": [], "type": "pp_comm"}}
        ctx = build_mfs_context(wl, btm)
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY
        assert ctx.task_info[1].comm_role == "pp_send"

    def test_collective_classified_as_early(self):
        """TP/EP collective flows should be MfsStage.EARLY."""
        for ct in [CommType.TP_ALLREDUCE_RING, CommType.EP_ALLTOALL,
                    CommType.TP_REDUCESCATTER_RING]:
            t1 = _flow_task(1, 0, 1, 500, ct)
            wl = _make_workload([t1])
            btm = {"b1": {"task_ids": [1], "request_ids": [1], "type": "prefill"}}
            ctx = build_mfs_context(wl, btm)
            assert ctx.task_info[1].mfs_stage == MfsStage.EARLY, f"Failed for {ct}"
            assert ctx.task_info[1].comm_role == "collective"

    def test_compute_classified_as_background(self):
        """Compute tasks should be MfsStage.BACKGROUND."""
        t0 = _compute_task(0, node=0)
        wl = _make_workload([t0])
        btm = {"b1": {"task_ids": [0], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)
        assert ctx.task_info[0].mfs_stage == MfsStage.BACKGROUND
        assert ctx.task_info[0].comm_role == "compute"

    def test_unknown_flow_classified_as_background(self):
        """CommType.UNKNOWN flows should be MfsStage.BACKGROUND."""
        t1 = _flow_task(1, 0, 1, 500, CommType.UNKNOWN)
        wl = _make_workload([t1])
        btm = {}
        ctx = build_mfs_context(wl, btm)
        assert ctx.task_info[1].mfs_stage == MfsStage.BACKGROUND
        assert ctx.task_info[1].comm_role == "unknown"

    def test_batch_to_tasks_mapping(self):
        """batch_to_tasks should map batch IDs to their task tuples."""
        t0 = _compute_task(0)
        t1 = _flow_task(1, 0, 1, 100)
        wl = _make_workload([t0, t1])
        btm = {"b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)
        assert ctx.batch_to_tasks["b1"] == (0, 1)

    def test_request_to_tasks_mapping(self):
        """request_to_tasks should include all tasks for a request."""
        t0 = _compute_task(0)
        t1 = _flow_task(1, 0, 2, 1000, CommType.KV_CACHE_TRANSFER)
        wl = _make_workload([t0, t1])
        btm = {
            "b1": {"task_ids": [0], "request_ids": [5], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [1], "request_ids": [5], "type": "kv_transfer"},
        }
        ctx = build_mfs_context(wl, btm)
        assert set(ctx.request_to_tasks[5]) == {0, 1}

    def test_prefill_collective_is_early(self):
        """Collective flows inside a prefill batch should be EARLY."""
        t0 = _compute_task(0, node=0, layer=2)
        t1 = _flow_task(1, 0, 1, 200, CommType.TP_ALLREDUCE_RING, layer=2)
        wl = _make_workload([t0, t1])
        btm = {"b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY

    def test_no_batch_task_map(self):
        """Tasks not in batch_task_map should get None batch_id, empty req_ids."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        ctx = build_mfs_context(wl, {})
        assert ctx.task_info[0].batch_id is None
        assert ctx.task_info[0].request_ids == ()

    def test_stage_id_from_batch_task_map(self):
        """stage_id should come from batch_task_map when present."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        btm = {"b1": {"task_ids": [0], "request_ids": [1], "type": "prefill", "stage_id": 2}}
        ctx = build_mfs_context(wl, btm)
        assert ctx.task_info[0].stage_id == 2


# ── RLI tests ─────────────────────────────────────────────────────────────────


class TestComputeStaticRli:
    """Tests for compute_static_rli."""

    def test_flow_at_layer_0_gets_rli_0(self):
        """Flow at layer 0 with current layer 0 gets RLI 0."""
        t0 = _compute_task(0, node=0, layer=0)
        t1 = _flow_task(1, 0, 1, 100, CommType.TP_ALLREDUCE_RING, layer=0)
        wl = _make_workload([t0, t1])
        btm = {"b1": {"task_ids": [0, 1], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)
        rli = compute_static_rli(wl, ctx)
        assert rli[1].base_rli == 0

    def test_flow_at_layer_2_gets_rli_2(self):
        """Flow at layer 2 with current layer 0 gets RLI 2."""
        t1 = _flow_task(1, 0, 1, 100, CommType.TP_ALLREDUCE_RING, layer=2)
        wl = _make_workload([t1])
        btm = {"b1": {"task_ids": [1], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)
        rli = compute_static_rli(wl, ctx)
        assert rli[1].base_rli == 2

    def test_p2d_gets_large_rli(self):
        """P2D flows should get a large sentinel RLI (not outrank RLI 0)."""
        t1 = _flow_task(1, 0, 2, 1000, CommType.KV_CACHE_TRANSFER, layer=0)
        wl = _make_workload([t1])
        btm = {"kv_b1_b2": {"task_ids": [1], "request_ids": [1], "type": "kv_transfer"}}
        ctx = build_mfs_context(wl, btm)
        rli = compute_static_rli(wl, ctx)
        assert rli[1].base_rli > 100

    def test_early_outranks_p2d(self):
        """RLI 0 collective should outrank P2D in priority."""
        t_early = _flow_task(1, 0, 1, 100, CommType.TP_ALLREDUCE_RING, layer=0)
        t_p2d = _flow_task(2, 0, 2, 1000, CommType.KV_CACHE_TRANSFER, layer=0)
        wl = _make_workload([t_early, t_p2d])
        btm = {
            "b1": {"task_ids": [1], "request_ids": [1], "type": "prefill"},
            "kv_b1_b2": {"task_ids": [2], "request_ids": [1], "type": "kv_transfer"},
        }
        ctx = build_mfs_context(wl, btm)
        rli = compute_static_rli(wl, ctx)
        assert rli[1].base_rli < rli[2].base_rli

    def test_rli_with_current_layer(self):
        """RLI should decrease when current_layer advances."""
        t1 = _flow_task(1, 0, 1, 100, CommType.TP_ALLREDUCE_RING, layer=3)
        wl = _make_workload([t1])
        btm = {"b1": {"task_ids": [1], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)

        rli_0 = compute_static_rli(wl, ctx, current_layer_by_stage={(0, 0): 0})
        rli_2 = compute_static_rli(wl, ctx, current_layer_by_stage={(0, 0): 2})

        assert rli_0[1].base_rli == 3
        assert rli_2[1].base_rli == 1

    def test_rli_non_negative(self):
        """RLI should never go below 0 (even if current_layer > target_layer)."""
        t1 = _flow_task(1, 0, 1, 100, CommType.TP_ALLREDUCE_RING, layer=2)
        wl = _make_workload([t1])
        btm = {"b1": {"task_ids": [1], "request_ids": [1], "type": "prefill"}}
        ctx = build_mfs_context(wl, btm)
        rli = compute_static_rli(wl, ctx, current_layer_by_stage={(0, 0): 5})
        assert rli[1].base_rli == 0

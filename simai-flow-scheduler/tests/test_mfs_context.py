"""Tests for MFS context metadata."""
import pytest

from src.workload_format.schema import (
    P2PWorkload, Meta, Job, Task, Phase, CommType, TaskType,
    ParallelismConfig, BatchTaskInfo, BatchEntryType,
)
from src.static_analysis.passes.mfs_context import (
    MfsContext, MfsTaskInfo, MfsStage, MfsRequestInfo,
    build_mfs_context,
)


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
        btm = [BatchTaskInfo(
            batch_id="kv_b1_b2", task_ids=[1],
            entry_type=BatchEntryType.KV_TRANSFER,
            replica_id=0, stage_id=0, request_ids=[10],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.P2D
        assert ctx.task_info[1].comm_role == "p2d_transfer"

    def test_pp_send_classified_as_early(self):
        """PP_SEND flows should be MfsStage.EARLY."""
        t1 = _flow_task(1, 0, 1, 500, CommType.PP_SEND)
        wl = _make_workload([t1])
        btm = [BatchTaskInfo(
            batch_id="pp_b1_b2", task_ids=[1],
            entry_type=BatchEntryType.PP_COMM,
            replica_id=0, stage_id=0, request_ids=[],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY
        assert ctx.task_info[1].comm_role == "pp_send"

    def test_collective_classified_as_early(self):
        """TP/EP collective flows should be MfsStage.EARLY."""
        for ct in [CommType.TP_ALLREDUCE_RING, CommType.EP_ALLTOALL,
                    CommType.TP_REDUCESCATTER_RING]:
            t1 = _flow_task(1, 0, 1, 500, ct)
            wl = _make_workload([t1])
            btm = [BatchTaskInfo(
                batch_id="b1", task_ids=[1],
                entry_type=BatchEntryType.PREFILL,
                replica_id=0, stage_id=0, request_ids=[1],
            )]
            ctx = build_mfs_context(wl, btm, trace={"requests": {}})
            assert ctx.task_info[1].mfs_stage == MfsStage.EARLY, f"Failed for {ct}"
            assert ctx.task_info[1].comm_role == "collective"

    def test_compute_classified_as_background(self):
        """Compute tasks should be MfsStage.BACKGROUND."""
        t0 = _compute_task(0, node=0)
        wl = _make_workload([t0])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[0].mfs_stage == MfsStage.BACKGROUND
        assert ctx.task_info[0].comm_role == "compute"

    def test_unknown_flow_classified_as_background(self):
        """CommType.UNKNOWN flows should be MfsStage.BACKGROUND."""
        t1 = _flow_task(1, 0, 1, 500, CommType.UNKNOWN)
        wl = _make_workload([t1])
        btm = []
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.BACKGROUND
        assert ctx.task_info[1].comm_role == "unknown"

    def test_batch_to_tasks_mapping(self):
        """batch_to_tasks should map batch IDs to their task tuples."""
        t0 = _compute_task(0)
        t1 = _flow_task(1, 0, 1, 100)
        wl = _make_workload([t0, t1])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0, 1],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.batch_to_tasks["b1"] == (0, 1)

    def test_request_to_tasks_mapping(self):
        """request_to_tasks should include all tasks for a request."""
        t0 = _compute_task(0)
        t1 = _flow_task(1, 0, 2, 1000, CommType.KV_CACHE_TRANSFER)
        wl = _make_workload([t0, t1])
        btm = [
            BatchTaskInfo(
                batch_id="b1", task_ids=[0],
                entry_type=BatchEntryType.PREFILL,
                replica_id=0, stage_id=0, request_ids=[5],
            ),
            BatchTaskInfo(
                batch_id="kv_b1_b2", task_ids=[1],
                entry_type=BatchEntryType.KV_TRANSFER,
                replica_id=0, stage_id=0, request_ids=[5],
            ),
        ]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert set(ctx.request_to_tasks[5]) == {0, 1}

    def test_prefill_collective_is_early(self):
        """Collective flows inside a prefill batch should be EARLY."""
        t0 = _compute_task(0, node=0, layer=2)
        t1 = _flow_task(1, 0, 1, 200, CommType.TP_ALLREDUCE_RING, layer=2)
        wl = _make_workload([t0, t1])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0, 1],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY

    def test_no_batch_task_map(self):
        """Tasks not in batch_task_map should get None batch_id, empty req_ids."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        ctx = build_mfs_context(wl, [], trace={"requests": {}})
        assert ctx.task_info[0].batch_id is None
        assert ctx.task_info[0].request_ids == ()

    def test_stage_id_from_batch_task_map(self):
        """stage_id should come from batch_task_map when present."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=2, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[0].stage_id == 2

    def test_decode_collective_is_background(self):
        """Decode-phase collective flows should be BACKGROUND, not EARLY."""
        t1 = _flow_task(1, 0, 1, 200, CommType.TP_ALLREDUCE_RING, phase=Phase.DECODE)
        wl = _make_workload([t1])
        btm = [BatchTaskInfo(
            batch_id="d1", task_ids=[1],
            entry_type=BatchEntryType.DECODE,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.BACKGROUND
        assert ctx.task_info[1].comm_role == "decode_collective"

    def test_prefill_collective_is_earlier_than_decode(self):
        """Same comm_type, different phase: prefill → EARLY, decode → BACKGROUND."""
        t1 = _flow_task(1, 0, 1, 200, CommType.TP_ALLREDUCE_RING, phase=Phase.PREFILL)
        t2 = _flow_task(2, 0, 1, 300, CommType.TP_ALLREDUCE_RING, phase=Phase.DECODE)
        wl = _make_workload([t1, t2])
        btm = [
            BatchTaskInfo(
                batch_id="p1", task_ids=[1],
                entry_type=BatchEntryType.PREFILL,
                replica_id=0, stage_id=0, request_ids=[1],
            ),
            BatchTaskInfo(
                batch_id="d1", task_ids=[2],
                entry_type=BatchEntryType.DECODE,
                replica_id=0, stage_id=0, request_ids=[1],
            ),
        ]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY
        assert ctx.task_info[2].mfs_stage == MfsStage.BACKGROUND

    def test_decode_pp_send_still_early(self):
        """PP_SEND in decode phase should still be EARLY (unlikely but test boundary)."""
        t1 = _flow_task(1, 0, 1, 200, CommType.PP_SEND, phase=Phase.DECODE)
        wl = _make_workload([t1])
        btm = [BatchTaskInfo(
            batch_id="d1", task_ids=[1],
            entry_type=BatchEntryType.DECODE,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY

    def test_kv_cache_reuse_classified_as_early(self):
        """KV_CACHE_REUSE flows should be MfsStage.EARLY."""
        t1 = _flow_task(1, 4, 0, 1000, CommType.KV_CACHE_REUSE)
        wl = _make_workload([t1])
        btm = [BatchTaskInfo(
            batch_id="kv_reuse_p0", task_ids=[1],
            entry_type=BatchEntryType.KV_REUSE,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.task_info[1].mfs_stage == MfsStage.EARLY
        assert ctx.task_info[1].comm_role == "kv_cache_reuse"


# ── MfsRequestInfo / SLO parsing tests ────────────────────────────────────────


class TestSloParsing:
    """Tests for ttft_slo_us metadata parsing from trace."""

    def test_request_info_with_slo(self):
        """Trace with ttft_slo_us should populate request_info."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=0, request_ids=[5],
        )]
        trace = {
            "requests": {
                "5": {
                    "num_prefill_tokens": 1024,
                    "num_decode_tokens": 64,
                    "ttft_slo_us": 2000000,
                },
            },
        }
        ctx = build_mfs_context(wl, btm, trace=trace)
        assert ctx.request_info[5].ttft_slo_us == 2000000

    def test_trace_without_slo(self):
        """Trace missing ttft_slo_us should produce None."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=0, request_ids=[5],
        )]
        trace = {
            "requests": {
                "5": {
                    "num_prefill_tokens": 1024,
                    "num_decode_tokens": 64,
                },
            },
        }
        ctx = build_mfs_context(wl, btm, trace=trace)
        assert ctx.request_info[5].ttft_slo_us is None

    def test_multiple_requests_with_slos(self):
        """Multiple requests should each get their own SLO info."""
        t0 = _compute_task(0)
        t1 = _compute_task(1, node=1)
        wl = _make_workload([t0, t1])
        btm = [
            BatchTaskInfo(
                batch_id="b1", task_ids=[0],
                entry_type=BatchEntryType.PREFILL,
                replica_id=0, stage_id=0, request_ids=[1],
            ),
            BatchTaskInfo(
                batch_id="b2", task_ids=[1],
                entry_type=BatchEntryType.PREFILL,
                replica_id=0, stage_id=0, request_ids=[2],
            ),
        ]
        trace = {
            "requests": {
                "1": {
                    "num_prefill_tokens": 512,
                    "num_decode_tokens": 32,
                    "ttft_slo_us": 1000000,
                },
                "2": {
                    "num_prefill_tokens": 1024,
                    "num_decode_tokens": 64,
                    "ttft_slo_us": 2000000,
                },
            },
        }
        ctx = build_mfs_context(wl, btm, trace=trace)
        assert ctx.request_info[1].ttft_slo_us == 1000000
        assert ctx.request_info[2].ttft_slo_us == 2000000

    def test_empty_requests_in_trace(self):
        """Trace with empty requests dict should produce empty request_info."""
        t0 = _compute_task(0)
        wl = _make_workload([t0])
        btm = [BatchTaskInfo(
            batch_id="b1", task_ids=[0],
            entry_type=BatchEntryType.PREFILL,
            replica_id=0, stage_id=0, request_ids=[1],
        )]
        ctx = build_mfs_context(wl, btm, trace={"requests": {}})
        assert ctx.request_info == {}

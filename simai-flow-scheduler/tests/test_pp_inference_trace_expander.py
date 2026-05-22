"""
Tests for InferenceTraceExpander with Pipeline Parallelism (pp>1).

Uses a synthetic per-stage trace:
  - 4 layers, tp=2, ep=1, pp=2
  - Each PP stage gets 2 layers
  - Replica 0 (P-node): stage 0 = GPUs [0,1], stage 1 = GPUs [2,3]
  - Replica 1 (D-node): stage 0 = GPUs [4,5], stage 1 = GPUs [6,7]

Trace has:
  - 1 prefill micro-batch traversing 2 stages on P-node
  - 1 decode micro-batch traversing 2 stages on D-node
  - Per-stage KV transfer: P-stage-s → D-stage-s
  - PP inter-stage communication between consecutive stages
"""

import pytest
from src.workload_generator.inference_trace_expander import InferenceTraceExpander
from src.workload_generator.inference_profile import InferenceProfileStore
from src.workload_format.schema import Phase, CommType, TaskType


def _kv_task_ids(btm, dep_id, bid):
    """Collect all KV transfer task_ids for dep_id → bid (aggregated across per-request entries)."""
    prefix = f"kv_{dep_id}_to_{bid}_req_"
    ids: list[int] = []
    for k, v in btm.items():
        if k.startswith(prefix):
            ids.extend(v["task_ids"])
    return ids


# ── Constants ──────────────────────────────────────────────────────────────────

TP = 2
EP = 1
PP = 2
NUM_LAYERS = 4  # 2 per PP stage
LAYERS_PER_STAGE = NUM_LAYERS // PP

PREFILL_PROFILE_KEY = "prefill_bs2_seq192"
DECODE_PROFILE_KEY = "decode_bs2_seq129"

# 4 layers × (attention + mlp)
PREFILL_TSV = (
    "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
    "0\tattention\t1000000\t8192\n"
    "0\tmlp\t2000000\t8192\n"
    "1\tattention\t1000000\t8192\n"
    "1\tmlp\t2000000\t8192\n"
    "2\tattention\t1500000\t8192\n"
    "2\tmlp\t2500000\t8192\n"
    "3\tattention\t1500000\t8192\n"
    "3\tmlp\t2500000\t8192\n"
)

DECODE_TSV = (
    "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
    "0\tattention\t100000\t1024\n"
    "0\tmlp\t200000\t1024\n"
    "1\tattention\t100000\t1024\n"
    "1\tmlp\t200000\t1024\n"
    "2\tattention\t150000\t1024\n"
    "2\tmlp\t250000\t1024\n"
    "3\tattention\t150000\t1024\n"
    "3\tmlp\t250000\t1024\n"
)

# Per-stage KV: total_kv=4096 for req 0, total_kv=2048 for req 1
# Per-stage share = total / pp = 2048 / 1024
PP_TRACE = {
    "version": "1.0",
    "model": "test-pp-model",
    "hidden_size": 64,
    "dtype_bytes": 2,
    "total_layers": NUM_LAYERS,
    "parallelism": {"tp": TP, "ep": EP, "pp": PP},
    "requests": {
        "0": {"num_prefill_tokens": 128, "num_decode_tokens": 4},
        "1": {"num_prefill_tokens": 64, "num_decode_tokens": 2},
    },
    "batches": [
        # Prefill stage 0
        {
            "batch_id": "p0_s0",
            "type": "prefill",
            "replica_id": 0,
            "stage_id": 0,
            "request_ids": [0, 1],
            "num_tokens": [128, 64],
            "kv_cache_bytes": {"0": 2048, "1": 1024},
            "depends_on": [],
        },
        # Prefill stage 1 (depends on stage 0 = PP comm)
        {
            "batch_id": "p0_s1",
            "type": "prefill",
            "replica_id": 0,
            "stage_id": 1,
            "request_ids": [0, 1],
            "num_tokens": [128, 64],
            "kv_cache_bytes": {"0": 2048, "1": 1024},
            "depends_on": ["p0_s0"],
        },
        # Decode stage 0 (depends on prefill stage 0 = KV transfer)
        {
            "batch_id": "d0_s0",
            "type": "decode",
            "replica_id": 1,
            "stage_id": 0,
            "request_ids": [0, 1],
            "num_tokens": [1, 1],
            "kv_cache_seq_lens": [129, 65],
            "kv_cache_bytes": None,
            "depends_on": ["p0_s0"],
        },
        # Decode stage 1 (depends on prefill stage 1 = KV transfer + decode stage 0 = PP comm)
        {
            "batch_id": "d0_s1",
            "type": "decode",
            "replica_id": 1,
            "stage_id": 1,
            "request_ids": [0, 1],
            "num_tokens": [1, 1],
            "kv_cache_seq_lens": [129, 65],
            "kv_cache_bytes": None,
            "depends_on": ["p0_s1", "d0_s0"],
        },
    ],
}


# ── Fixtures ────────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    p_file = tmp_path / "prefill.tsv"
    d_file = tmp_path / "decode.tsv"
    p_file.write_text(PREFILL_TSV)
    d_file.write_text(DECODE_TSV)

    s = InferenceProfileStore()
    s.load(str(p_file), PREFILL_PROFILE_KEY)
    s.load(str(d_file), DECODE_PROFILE_KEY)
    return s


@pytest.fixture
def pp_expander(store):
    return InferenceTraceExpander(store, tp=TP, ep=EP, pp=PP)


@pytest.fixture
def pp_result(pp_expander):
    return pp_expander.expand(PP_TRACE, job_id=0)


# ── Test: Rank mapping ─────────────────────────────────────────────────────────

class TestPPRankMapping:

    def test_prefill_stage0_uses_stage0_ranks(self, pp_result):
        """Stage 0 of replica 0 should use GPUs [0, 1]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        nodes = {
            task_map[tid].node
            for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes == {0, 1}

    def test_prefill_stage1_uses_stage1_ranks(self, pp_result):
        """Stage 1 of replica 0 should use GPUs [2, 3]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        nodes = {
            task_map[tid].node
            for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes == {2, 3}

    def test_decode_stage0_uses_replica1_stage0_ranks(self, pp_result):
        """Stage 0 of replica 1 (D-node) should use GPUs [4, 5]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        nodes = {
            task_map[tid].node
            for tid in btm["d0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes == {4, 5}

    def test_decode_stage1_uses_replica1_stage1_ranks(self, pp_result):
        """Stage 1 of replica 1 (D-node) should use GPUs [6, 7]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        nodes = {
            task_map[tid].node
            for tid in btm["d0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes == {6, 7}


# ── Test: Layer filtering ──────────────────────────────────────────────────────

class TestPPLayerFiltering:

    def test_stage0_only_has_first_half_layers(self, pp_result):
        """Stage 0 should only contain layers 0, 1 (not 2, 3)."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        layers = {
            task_map[tid].layer_id
            for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert layers == {0, 1}

    def test_stage1_only_has_second_half_layers(self, pp_result):
        """Stage 1 should only contain layers 2, 3 (not 0, 1)."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        layers = {
            task_map[tid].layer_id
            for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert layers == {2, 3}

    def test_stage0_compute_count(self, pp_result):
        """Stage 0: 2 layers × 2 sub-ops × 2 ranks = 8 compute tasks."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        compute_count = sum(
            1 for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        )
        assert compute_count == LAYERS_PER_STAGE * 2 * TP

    def test_stage1_compute_count(self, pp_result):
        """Stage 1: 2 layers × 2 sub-ops × 2 ranks = 8 compute tasks."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        compute_count = sum(
            1 for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        )
        assert compute_count == LAYERS_PER_STAGE * 2 * TP


# ── Test: PP inter-stage communication ─────────────────────────────────────────

class TestPPCommunication:

    def test_pp_comm_exists_for_prefill(self, pp_result):
        """There should be a pp_comm entry for p0_s0 → p0_s1."""
        _, btm = pp_result
        pp_key = "pp_p0_s0_to_p0_s1"
        assert pp_key in btm
        assert btm[pp_key]["type"] == "pp_comm"
        assert btm[pp_key]["from_stage"] == 0
        assert btm[pp_key]["to_stage"] == 1

    def test_pp_comm_exists_for_decode(self, pp_result):
        """There should be a pp_comm entry for d0_s0 → d0_s1."""
        _, btm = pp_result
        pp_key = "pp_d0_s0_to_d0_s1"
        assert pp_key in btm
        assert btm[pp_key]["type"] == "pp_comm"

    def test_pp_comm_prefill_src_dst_ranks(self, pp_result):
        """PP comm: stage 0 ranks [0,1] → stage 1 ranks [2,3]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        pp_key = "pp_p0_s0_to_p0_s1"
        pp_tasks = [task_map[tid] for tid in btm[pp_key]["task_ids"]]
        srcs = {t.src for t in pp_tasks}
        dsts = {t.dst for t in pp_tasks}
        assert srcs == {0, 1}
        assert dsts == {2, 3}

    def test_pp_comm_decode_src_dst_ranks(self, pp_result):
        """PP comm: D-stage 0 ranks [4,5] → D-stage 1 ranks [6,7]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        pp_key = "pp_d0_s0_to_d0_s1"
        pp_tasks = [task_map[tid] for tid in btm[pp_key]["task_ids"]]
        srcs = {t.src for t in pp_tasks}
        dsts = {t.dst for t in pp_tasks}
        assert srcs == {4, 5}
        assert dsts == {6, 7}

    def test_pp_comm_uses_pp_send_type(self, pp_result):
        """PP comm tasks should use CommType.PP_SEND."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        pp_key = "pp_p0_s0_to_p0_s1"
        pp_tasks = [task_map[tid] for tid in btm[pp_key]["task_ids"]]
        for t in pp_tasks:
            assert t.type == TaskType.FLOW
            assert t.comm_type == CommType.PP_SEND

    def test_pp_comm_flow_count(self, pp_result):
        """PP comm: tp*ep = 2 flows per stage transition."""
        _, btm = pp_result
        pp_key = "pp_p0_s0_to_p0_s1"
        assert len(btm[pp_key]["task_ids"]) == TP * EP


# ── Test: Per-stage KV transfer ────────────────────────────────────────────────

class TestPPKVTransfer:

    def test_kv_transfer_stage0_exists(self, pp_result):
        """KV transfer from P-stage-0 to D-stage-0 should exist."""
        _, btm = pp_result
        assert any(k.startswith("kv_p0_s0_to_d0_s0_req_") for k in btm)
        assert next(v for k, v in btm.items() if k.startswith("kv_p0_s0_to_d0_s0_req_"))["type"] == "kv_transfer"

    def test_kv_transfer_stage1_exists(self, pp_result):
        """KV transfer from P-stage-1 to D-stage-1 should exist."""
        _, btm = pp_result
        assert any(k.startswith("kv_p0_s1_to_d0_s1_req_") for k in btm)
        assert next(v for k, v in btm.items() if k.startswith("kv_p0_s1_to_d0_s1_req_"))["type"] == "kv_transfer"

    def test_kv_transfer_stage0_src_dst(self, pp_result):
        """KV stage 0: P-stage-0 [0,1] → D-stage-0 [4,5]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_task_ids = _kv_task_ids(btm, "p0_s0", "d0_s0")
        kv_tasks = [task_map[tid] for tid in kv_task_ids]
        srcs = {t.src for t in kv_tasks}
        dsts = {t.dst for t in kv_tasks}
        assert srcs == {0, 1}
        assert dsts == {4, 5}

    def test_kv_transfer_stage1_src_dst(self, pp_result):
        """KV stage 1: P-stage-1 [2,3] → D-stage-1 [6,7]."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_task_ids = _kv_task_ids(btm, "p0_s1", "d0_s1")
        kv_tasks = [task_map[tid] for tid in kv_task_ids]
        srcs = {t.src for t in kv_tasks}
        dsts = {t.dst for t in kv_tasks}
        assert srcs == {2, 3}
        assert dsts == {6, 7}

    def test_kv_transfer_per_stage_count(self, pp_result):
        """Each stage: 2 requests × tp*ep = 2 flows per stage. But per-stage
        KV bytes = total_kv / pp, so stage_size = tp*ep = 2.
        Total per stage: 2 reqs × 2 ranks = 4 flows."""
        _, btm = pp_result
        assert len(_kv_task_ids(btm, "p0_s0", "d0_s0")) == 2 * TP
        assert len(_kv_task_ids(btm, "p0_s1", "d0_s1")) == 2 * TP


# ── Test: Pipeline dependency chain ────────────────────────────────────────────

class TestPPDependencies:

    def test_prefill_stage0_has_no_deps(self, pp_result):
        """First prefill stage has no predecessor tasks."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        p0_s0_ids = set(btm["p0_s0"]["task_ids"])
        first_computes = [
            tid for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in p0_s0_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        for tid in first_computes:
            assert task_map[tid].deps == []

    def test_prefill_stage1_depends_on_pp_comm(self, pp_result):
        """Prefill stage 1 compute tasks should depend on PP comm from stage 0."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        pp_task_ids = set(btm["pp_p0_s0_to_p0_s1"]["task_ids"])
        # First compute tasks of p0_s1
        p0_s1_ids = set(btm["p0_s1"]["task_ids"])
        first_computes = [
            tid for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in p0_s1_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        for tid in first_computes:
            deps = task_map[tid].deps
            assert any(d in pp_task_ids for d in deps), (
                f"p0_s1 first compute {tid} should depend on PP comm tasks"
            )

    def test_decode_stage0_depends_on_kv_transfer_stage0(self, pp_result):
        """Decode stage 0 should depend on KV transfer from P-stage-0."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_task_ids = set(_kv_task_ids(btm, "p0_s0", "d0_s0"))
        d0_s0_ids = set(btm["d0_s0"]["task_ids"])
        first_computes = [
            tid for tid in btm["d0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in d0_s0_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        for tid in first_computes:
            deps = task_map[tid].deps
            assert any(d in kv_task_ids for d in deps), (
                f"d0_s0 first compute {tid} should depend on KV stage-0 transfer"
            )

    def test_decode_stage1_depends_on_kv_and_pp(self, pp_result):
        """Decode stage 1 should depend on both KV transfer from P-stage-1
        and PP comm from decode stage 0."""
        workload, btm = pp_result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_task_ids = set(_kv_task_ids(btm, "p0_s1", "d0_s1"))
        pp_task_ids = set(btm["pp_d0_s0_to_d0_s1"]["task_ids"])
        d0_s1_ids = set(btm["d0_s1"]["task_ids"])
        first_computes = [
            tid for tid in btm["d0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in d0_s1_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        all_pred = kv_task_ids | pp_task_ids
        for tid in first_computes:
            deps = task_map[tid].deps
            assert any(d in all_pred for d in deps), (
                f"d0_s1 first compute {tid} should depend on KV or PP comm tasks"
            )

    def test_task_ids_unique(self, pp_result):
        """All task IDs across all batches must be unique."""
        workload, _ = pp_result
        ids = [t.task_id for t in workload.tasks]
        assert len(ids) == len(set(ids))

    def test_workload_validates(self, pp_result):
        """Generated P2PWorkload should pass validation."""
        workload, _ = pp_result
        errors = workload.validate()
        assert errors == [], f"Validation errors: {errors}"


# ── Test: PP with assigned_nodes ───────────────────────────────────────────────

class TestPPWithAssignedNodes:

    def test_assigned_nodes_override(self, store):
        """When assigned_nodes is provided, ranks should come from that list."""
        expander = InferenceTraceExpander(
            store, tp=TP, ep=EP, pp=PP,
            assigned_nodes=list(range(100, 108)),
        )
        workload, btm = expander.expand(PP_TRACE, job_id=0)
        task_map = {t.task_id: t for t in workload.tasks}

        # Stage 0 of replica 0 → [100, 101]
        nodes_s0 = {
            task_map[tid].node
            for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes_s0 == {100, 101}

        # Stage 1 of replica 0 → [102, 103]
        nodes_s1 = {
            task_map[tid].node
            for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes_s1 == {102, 103}

        # Stage 0 of replica 1 → [104, 105]
        nodes_d_s0 = {
            task_map[tid].node
            for tid in btm["d0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert nodes_d_s0 == {104, 105}

    def test_assigned_nodes_pp_comm_ranks(self, store):
        """PP comm flows should use assigned_nodes ranks."""
        expander = InferenceTraceExpander(
            store, tp=TP, ep=EP, pp=PP,
            assigned_nodes=list(range(100, 108)),
        )
        workload, btm = expander.expand(PP_TRACE, job_id=0)
        task_map = {t.task_id: t for t in workload.tasks}

        pp_key = "pp_p0_s0_to_p0_s1"
        pp_tasks = [task_map[tid] for tid in btm[pp_key]["task_ids"]]
        srcs = {t.src for t in pp_tasks}
        dsts = {t.dst for t in pp_tasks}
        assert srcs == {100, 101}
        assert dsts == {102, 103}


# ── Test: Pipeline overlap (2 micro-batches) ───────────────────────────────────

class TestPPPipelineOverlap:

    """Test with 2 micro-batches to verify pipeline overlap deps.

    MB0: p0_s0 → p0_s1 (stage 0 then stage 1)
    MB1: p1_s0 → p1_s1 (stage 0 then stage 1)

    Dependencies:
      p1_s0 depends on p0_s0 (same-stage sequential: stage 0 processes MB0 before MB1)
      p1_s1 depends on p0_s1 (same-stage sequential: stage 1 processes MB0 before MB1)
                       AND p1_s0 (cross-stage PP dep: MB1 stage 0 → stage 1)
    """

    @pytest.fixture
    def overlap_result(self, store):
        expander = InferenceTraceExpander(store, tp=TP, ep=EP, pp=PP)
        trace = {
            "version": "1.0",
            "model": "test-overlap",
            "hidden_size": 64,
            "dtype_bytes": 2,
            "total_layers": NUM_LAYERS,
            "parallelism": {"tp": TP, "ep": EP, "pp": PP},
            "requests": {
                "0": {"num_prefill_tokens": 128, "num_decode_tokens": 4},
                "1": {"num_prefill_tokens": 64, "num_decode_tokens": 2},
            },
            "batches": [
                # MB0 stage 0
                {
                    "batch_id": "p0_s0",
                    "type": "prefill",
                    "replica_id": 0,
                    "stage_id": 0,
                    "request_ids": [0],
                    "num_tokens": [128],
                    "kv_cache_bytes": {"0": 2048},
                    "depends_on": [],
                },
                # MB0 stage 1
                {
                    "batch_id": "p0_s1",
                    "type": "prefill",
                    "replica_id": 0,
                    "stage_id": 1,
                    "request_ids": [0],
                    "num_tokens": [128],
                    "kv_cache_bytes": {"0": 2048},
                    "depends_on": ["p0_s0"],
                },
                # MB1 stage 0 (same-stage dep on MB0 stage 0)
                {
                    "batch_id": "p1_s0",
                    "type": "prefill",
                    "replica_id": 0,
                    "stage_id": 0,
                    "request_ids": [1],
                    "num_tokens": [64],
                    "kv_cache_bytes": {"1": 1024},
                    "depends_on": ["p0_s0"],
                },
                # MB1 stage 1 (same-stage dep on MB0 stage 1 + cross-stage dep on MB1 stage 0)
                {
                    "batch_id": "p1_s1",
                    "type": "prefill",
                    "replica_id": 0,
                    "stage_id": 1,
                    "request_ids": [1],
                    "num_tokens": [64],
                    "kv_cache_bytes": {"1": 1024},
                    "depends_on": ["p0_s1", "p1_s0"],
                },
            ],
        }
        return expander.expand(trace, job_id=0)

    def test_mb1_stage0_depends_on_mb0_stage0(self, overlap_result):
        """MB1 stage 0 should depend on MB0 stage 0 (same-stage sequential)."""
        workload, btm = overlap_result
        task_map = {t.task_id: t for t in workload.tasks}
        p0_s0_ids = set(btm["p0_s0"]["task_ids"])
        p1_s0_ids = set(btm["p1_s0"]["task_ids"])
        first_computes = [
            tid for tid in btm["p1_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in p1_s0_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        for tid in first_computes:
            deps = task_map[tid].deps
            assert any(d in p0_s0_ids for d in deps), (
                f"p1_s0 first compute {tid} should depend on p0_s0 exit tasks"
            )

    def test_mb1_stage1_depends_on_mb0_stage1_and_mb1_stage0(self, overlap_result):
        """MB1 stage 1 should depend on both:
        - MB0 stage 1 (same-stage sequential dep)
        - PP comm from MB1 stage 0 (cross-stage dep)"""
        workload, btm = overlap_result
        task_map = {t.task_id: t for t in workload.tasks}
        p0_s1_ids = set(btm["p0_s1"]["task_ids"])
        pp_key = "pp_p1_s0_to_p1_s1"
        pp_task_ids = set(btm[pp_key]["task_ids"])
        p1_s1_ids = set(btm["p1_s1"]["task_ids"])

        first_computes = [
            tid for tid in btm["p1_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in p1_s1_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0

        for tid in first_computes:
            deps = task_map[tid].deps
            has_same_stage = any(d in p0_s1_ids for d in deps)
            has_cross_stage = any(d in pp_task_ids for d in deps)
            assert has_same_stage or has_cross_stage, (
                f"p1_s1 first compute {tid} should depend on p0_s1 or PP comm"
            )


# ── Test: Uneven layer split (total_layers % pp != 0) ──────────────────────────

class TestPPUnevenLayers:
    """5 layers with pp=2: stage 0 gets layers 0-1, stage 1 gets layers 2-4."""

    UNEVEN_LAYERS = 5
    UNEVEN_PREFILL_TSV = (
        "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
        "0\tattention\t1000000\t8192\n"
        "0\tmlp\t2000000\t8192\n"
        "1\tattention\t1000000\t8192\n"
        "1\tmlp\t2000000\t8192\n"
        "2\tattention\t1500000\t8192\n"
        "2\tmlp\t2500000\t8192\n"
        "3\tattention\t1500000\t8192\n"
        "3\tmlp\t2500000\t8192\n"
        "4\tattention\t1800000\t8192\n"
        "4\tmlp\t2800000\t8192\n"
    )
    UNEVEN_DECODE_TSV = (
        "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
        "0\tattention\t100000\t1024\n"
        "0\tmlp\t200000\t1024\n"
        "1\tattention\t100000\t1024\n"
        "1\tmlp\t200000\t1024\n"
        "2\tattention\t150000\t1024\n"
        "2\tmlp\t250000\t1024\n"
        "3\tattention\t150000\t1024\n"
        "3\tmlp\t250000\t1024\n"
        "4\tattention\t180000\t1024\n"
        "4\tmlp\t280000\t1024\n"
    )

    UNEVEN_TRACE = {
        "version": "1.0",
        "model": "test-uneven",
        "hidden_size": 64,
        "dtype_bytes": 2,
        "total_layers": UNEVEN_LAYERS,
        "parallelism": {"tp": TP, "ep": EP, "pp": PP},
        "requests": {
            "0": {"num_prefill_tokens": 128, "num_decode_tokens": 4},
            "1": {"num_prefill_tokens": 64, "num_decode_tokens": 2},
        },
        "batches": [
            # Prefill stage 0 layers 0-1: 2/5 of total KV
            {
                "batch_id": "p0_s0",
                "type": "prefill",
                "replica_id": 0,
                "stage_id": 0,
                "request_ids": [0, 1],
                "num_tokens": [128, 64],
                "kv_cache_bytes": {"0": 2048, "1": 1024},
                "depends_on": [],
            },
            # Prefill stage 1 layers 2-4: 3/5 of total KV (more than stage 0!)
            {
                "batch_id": "p0_s1",
                "type": "prefill",
                "replica_id": 0,
                "stage_id": 1,
                "request_ids": [0, 1],
                "num_tokens": [128, 64],
                "kv_cache_bytes": {"0": 3072, "1": 1536},
                "depends_on": ["p0_s0"],
            },
            # Decode stage 0
            {
                "batch_id": "d0_s0",
                "type": "decode",
                "replica_id": 1,
                "stage_id": 0,
                "request_ids": [0, 1],
                "num_tokens": [1, 1],
                "kv_cache_seq_lens": [129, 65],
                "kv_cache_bytes": None,
                "depends_on": ["p0_s0"],
            },
            # Decode stage 1
            {
                "batch_id": "d0_s1",
                "type": "decode",
                "replica_id": 1,
                "stage_id": 1,
                "request_ids": [0, 1],
                "num_tokens": [1, 1],
                "kv_cache_seq_lens": [129, 65],
                "kv_cache_bytes": None,
                "depends_on": ["p0_s1", "d0_s0"],
            },
        ],
    }

    @pytest.fixture
    def uneven_store(self, tmp_path):
        p_file = tmp_path / "prefill_uneven.tsv"
        d_file = tmp_path / "decode_uneven.tsv"
        p_file.write_text(self.UNEVEN_PREFILL_TSV)
        d_file.write_text(self.UNEVEN_DECODE_TSV)
        s = InferenceProfileStore()
        s.load(str(p_file), PREFILL_PROFILE_KEY)
        s.load(str(d_file), DECODE_PROFILE_KEY)
        return s

    @pytest.fixture
    def uneven_result(self, uneven_store):
        expander = InferenceTraceExpander(uneven_store, tp=TP, ep=EP, pp=PP)
        return expander.expand(self.UNEVEN_TRACE, job_id=0)

    def test_stage0_gets_first_two_layers(self, uneven_result):
        """Stage 0: layers 0, 1. Should NOT have layers 2, 3, 4."""
        workload, btm = uneven_result
        task_map = {t.task_id: t for t in workload.tasks}
        layers = {
            task_map[tid].layer_id
            for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert layers == {0, 1}

    def test_stage1_gets_remaining_three_layers(self, uneven_result):
        """Stage 1: layers 2, 3, 4 (the remainder). Should NOT have 0, 1."""
        workload, btm = uneven_result
        task_map = {t.task_id: t for t in workload.tasks}
        layers = {
            task_map[tid].layer_id
            for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert layers == {2, 3, 4}

    def test_stage1_has_more_compute_than_stage0(self, uneven_result):
        """Stage 1 has 3 layers, stage 0 has 2. Stage 1 should have more compute tasks."""
        workload, btm = uneven_result
        task_map = {t.task_id: t for t in workload.tasks}
        s0_compute = sum(
            1 for tid in btm["p0_s0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        )
        s1_compute = sum(
            1 for tid in btm["p0_s1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        )
        # Stage 0: 2 layers × 2 sub-ops × 2 ranks = 8
        # Stage 1: 3 layers × 2 sub-ops × 2 ranks = 12
        assert s0_compute == 2 * 2 * TP, f"Expected 8, got {s0_compute}"
        assert s1_compute == 3 * 2 * TP, f"Expected 12, got {s1_compute}"

    def test_uneven_kv_bytes_per_rank(self, uneven_result):
        """Stage 1 KV bytes per rank > stage 0, because stage 1 has more layers."""
        workload, btm = uneven_result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_tids_s0 = _kv_task_ids(btm, "p0_s0", "d0_s0")
        kv_tids_s1 = _kv_task_ids(btm, "p0_s1", "d0_s1")
        kv_s0 = [task_map[tid] for tid in kv_tids_s0]
        kv_s1 = [task_map[tid] for tid in kv_tids_s1]
        # All tasks for the same request within a stage have equal size
        s0_size = kv_s0[0].size_bytes if kv_s0 else 0
        s1_size = kv_s1[0].size_bytes if kv_s1 else 0
        assert s1_size > s0_size, (
            f"Stage 1 per-rank KV bytes ({s1_size}) should exceed stage 0 ({s0_size})"
            " since stage 1 has more layers"
        )

    def test_uneven_workload_validates(self, uneven_result):
        """Uneven layer split workload must pass validation."""
        workload, _ = uneven_result
        errors = workload.validate()
        assert errors == [], f"Validation errors: {errors}"

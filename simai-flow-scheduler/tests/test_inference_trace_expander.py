"""
Tests for InferenceTraceExpander.

Uses a minimal synthetic trace:
  - 2 requests (req 0: 128 prefill / 4 decode, req 1: 64 prefill / 2 decode)
  - 1 prefill batch (p0) on replica 0
  - 3 decode batches (d0, d1, d2) on replica 1
  - KV transfer between p0 and d0

Model: 2 layers, tp=2, ep=1 (simple, no MoE)
"""

import pytest
from src.workload_generator.inference_trace_expander import InferenceTraceExpander
from src.workload_generator.inference_profile import InferenceProfileStore
from src.workload_format.schema import Phase, CommType, TaskType


# ── Fixtures ──────────────────────────────────────────────────────────────────

TP = 2
EP = 1
NUM_LAYERS = 2

PREFILL_PROFILE_KEY = "prefill_bs2_seq128"
DECODE_PROFILE_KEY = "decode_bs2_seq1"

# TSV content: 2 layers, attention + mlp each
PREFILL_TSV = (
    "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
    "0\tattention\t1000000\t8192\n"   # 1000 us, 8 KB
    "0\tmlp\t2000000\t8192\n"         # 2000 us, 8 KB
    "1\tattention\t1000000\t8192\n"
    "1\tmlp\t2000000\t8192\n"
)

DECODE_TSV = (
    "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
    "0\tattention\t100000\t1024\n"    # 100 us, 1 KB
    "0\tmlp\t200000\t1024\n"          # 200 us, 1 KB
    "1\tattention\t100000\t1024\n"
    "1\tmlp\t200000\t1024\n"
)

TRACE = {
    "version": "1.0",
    "model": "test-model",
    "model_config": {"hidden_size": 64, "num_layers": NUM_LAYERS, "dense_layers": 2},
    "parallelism": {"tp": TP, "ep": EP, "pp": 1},
    "pd_config": {"pd_node_ratio": 0.5, "pd_p2p_comm_bandwidth_gbps": 200},
    "requests": {
        "0": {"num_prefill_tokens": 128, "num_decode_tokens": 4},
        "1": {"num_prefill_tokens": 64,  "num_decode_tokens": 2},
    },
    "batches": [
        {
            "batch_id": "p0",
            "type": "prefill",
            "replica_id": 0,
            "request_ids": [0, 1],
            "num_tokens": [128, 64],
            "kv_cache_bytes": {"0": 4096, "1": 2048},
            "depends_on": [],
        },
        {
            "batch_id": "d0",
            "type": "decode",
            "replica_id": 1,
            "request_ids": [0, 1],
            "num_tokens": [1, 1],
            "kv_cache_bytes": None,
            "depends_on": ["p0"],
        },
        {
            "batch_id": "d1",
            "type": "decode",
            "replica_id": 1,
            "request_ids": [0, 1],
            "num_tokens": [1, 1],
            "kv_cache_bytes": None,
            "depends_on": ["d0"],
        },
        {
            "batch_id": "d2",
            "type": "decode",
            "replica_id": 1,
            "request_ids": [0],
            "num_tokens": [1],
            "kv_cache_bytes": None,
            "depends_on": ["d1"],
        },
    ],
}


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
def expander(store):
    return InferenceTraceExpander(store, tp=TP, ep=EP)


@pytest.fixture
def result(expander):
    return expander.expand(
        TRACE,
        job_id=0,
        prefill_profile_key=PREFILL_PROFILE_KEY,
        decode_profile_key=DECODE_PROFILE_KEY,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExpansionStructure:

    def test_returns_workload_and_map(self, result):
        workload, batch_task_map = result
        assert workload is not None
        assert isinstance(batch_task_map, dict)

    def test_batch_task_map_keys(self, result):
        _, btm = result
        # 4 batches + 1 KV transfer entry
        assert "p0" in btm
        assert "d0" in btm
        assert "d1" in btm
        assert "d2" in btm
        assert "kv_p0_to_d0" in btm

    def test_batch_types(self, result):
        _, btm = result
        assert btm["p0"]["type"] == "prefill"
        assert btm["d0"]["type"] == "decode"
        assert btm["kv_p0_to_d0"]["type"] == "kv_transfer"

    def test_request_ids_preserved(self, result):
        _, btm = result
        assert btm["p0"]["request_ids"] == [0, 1]
        assert btm["d2"]["request_ids"] == [0]

    def test_task_ids_unique(self, result):
        workload, _ = result
        ids = [t.task_id for t in workload.tasks]
        assert len(ids) == len(set(ids)), "Task IDs must be unique"

    def test_workload_validates(self, result):
        workload, _ = result
        errors = workload.validate()
        assert errors == [], f"Validation errors: {errors}"


class TestPhaseAssignment:

    def test_prefill_tasks_have_prefill_phase(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        for tid in btm["p0"]["task_ids"]:
            assert task_map[tid].phase == Phase.PREFILL

    def test_decode_tasks_have_decode_phase(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        for tid in btm["d0"]["task_ids"]:
            assert task_map[tid].phase == Phase.DECODE

    def test_kv_transfer_comm_type(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_tasks = [task_map[tid] for tid in btm["kv_p0_to_d0"]["task_ids"]]
        assert all(t.type == TaskType.FLOW for t in kv_tasks)
        assert all(t.comm_type == CommType.KV_CACHE_TRANSFER for t in kv_tasks)


class TestRankMapping:

    def test_prefill_uses_replica0_ranks(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # Replica 0 with tp=2, ep=1 → ranks [0, 1]
        p0_nodes = {
            task_map[tid].node
            for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert p0_nodes == {0, 1}

    def test_decode_uses_replica1_ranks(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # Replica 1 with tp=2, ep=1 → ranks [2, 3]
        d0_nodes = {
            task_map[tid].node
            for tid in btm["d0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert d0_nodes == {2, 3}

    def test_kv_transfer_src_is_p_ranks_dst_is_d_ranks(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_tasks = [task_map[tid] for tid in btm["kv_p0_to_d0"]["task_ids"]]
        srcs = {t.src for t in kv_tasks}
        dsts = {t.dst for t in kv_tasks}
        assert srcs == {0, 1}   # P-node ranks
        assert dsts == {2, 3}   # D-node ranks


class TestTaskCounts:

    def test_prefill_compute_task_count(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # 2 layers × 2 sub-ops (attention + mlp) × 2 ranks = 8 compute tasks
        compute_count = sum(
            1 for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        )
        assert compute_count == NUM_LAYERS * 2 * TP

    def test_kv_transfer_task_count(self, result):
        _, btm = result
        # 2 requests × tp=2 ranks = 4 KV flows
        assert len(btm["kv_p0_to_d0"]["task_ids"]) == 2 * TP

    def test_allreduce_flow_count_per_batch(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # tp=2, ep=1: AllReduce ring for 2 ranks = 2*(2-1)*2 = 4 flows per sub-op
        # 2 layers × 2 sub-ops × 4 flows = 16 flow tasks
        flow_count = sum(
            1 for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.FLOW
        )
        assert flow_count == NUM_LAYERS * 2 * TP * 2 * (TP - 1)


class TestDependencies:

    def test_decode_first_compute_depends_on_kv_transfer(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}

        kv_task_ids = set(btm["kv_p0_to_d0"]["task_ids"])
        # Only the very first sub-op compute tasks (layer 0, attention) should
        # directly depend on KV transfer. Find them: compute tasks whose deps
        # are all from outside d0 (i.e., from KV transfer).
        d0_task_ids = set(btm["d0"]["task_ids"])
        first_computes = [
            tid for tid in btm["d0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in d0_task_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0, "No first-layer compute tasks found in d0"
        for tid in first_computes:
            task = task_map[tid]
            assert any(dep in kv_task_ids for dep in task.deps), (
                f"First compute task {tid} (rank {task.node}) "
                f"has no dependency on KV transfer tasks"
            )

    def test_d1_depends_on_d0(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}

        d0_task_ids = set(btm["d0"]["task_ids"])
        d1_task_ids = set(btm["d1"]["task_ids"])
        # First sub-op compute tasks of d1 should depend on d0's exit tasks
        first_computes = [
            tid for tid in btm["d1"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in d1_task_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        for tid in first_computes:
            task = task_map[tid]
            assert any(dep in d0_task_ids for dep in task.deps), (
                f"d1 first compute task {tid} has no dependency on d0 tasks"
            )

    def test_prefill_first_compute_has_no_deps(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}

        p0_task_ids = set(btm["p0"]["task_ids"])
        first_computes = [
            tid for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
            and not any(dep in p0_task_ids for dep in task_map[tid].deps)
        ]
        assert len(first_computes) > 0
        for tid in first_computes:
            assert task_map[tid].deps == []

    def test_flow_depends_on_its_compute(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}

        compute_ids = {
            tid for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.COMPUTE
        }
        flow_ids = [
            tid for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.FLOW
        ]
        for tid in flow_ids:
            task = task_map[tid]
            assert any(dep in compute_ids for dep in task.deps), (
                f"Flow task {tid} has no dependency on a compute task"
            )


class TestMoEExpansion:

    def test_moe_layer_uses_ep_alltoall(self, tmp_path):
        """MoE layers should produce EP_ALLTOALL flows, not TP_ALLREDUCE."""
        moe_tsv = (
            "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
            "0\tattention\t1000000\t8192\n"
            "0\tmoe\t3000000\t16384\n"
        )
        f = tmp_path / "moe.tsv"
        f.write_text(moe_tsv)

        store = InferenceProfileStore()
        store.load(str(f), "prefill_moe")

        # tp=2, ep=2 → world_size=4 per replica
        expander = InferenceTraceExpander(store, tp=2, ep=2)

        trace = {
            "version": "1.0",
            "model": "moe-test",
            "parallelism": {"tp": 2, "ep": 2, "pp": 1},
            "requests": {"0": {"num_prefill_tokens": 64, "num_decode_tokens": 2}},
            "batches": [
                {
                    "batch_id": "p0",
                    "type": "prefill",
                    "replica_id": 0,
                    "request_ids": [0],
                    "num_tokens": [64],
                    "kv_cache_bytes": {"0": 1024},
                    "depends_on": [],
                }
            ],
        }

        workload, btm = expander.expand(trace, prefill_profile_key="prefill_moe")
        task_map = {t.task_id: t for t in workload.tasks}

        flow_tasks = [
            task_map[tid] for tid in btm["p0"]["task_ids"]
            if task_map[tid].type == TaskType.FLOW
        ]
        comm_types = {t.comm_type for t in flow_tasks}

        assert CommType.EP_ALLTOALL in comm_types
        assert CommType.TP_ALLREDUCE_RING in comm_types

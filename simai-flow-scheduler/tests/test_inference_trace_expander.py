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


def _kv_task_ids(btm, dep_id, bid):
    """Collect all KV transfer task_ids for dep_id → bid (aggregated across per-request entries)."""
    prefix = f"kv_{dep_id}_to_{bid}_req_"
    ids: list[int] = []
    for k, v in btm.items():
        if k.startswith(prefix):
            ids.extend(v.task_ids)
    return ids


def _kv_reuse_task_ids(btm, bid):
    """Collect all KV reuse task IDs for bid (aggregated across per-request entries)."""
    prefix = f"kv_reuse_{bid}_req_"
    ids: list[int] = []
    for batch_id, entry in btm.items():
        if batch_id.startswith(prefix):
            ids.extend(entry.task_ids)
    return ids


from src.workload_format.schema import Phase, CommType, TaskType, BatchEntryType


# ── Fixtures ──────────────────────────────────────────────────────────────────

TP = 2
EP = 1
NUM_LAYERS = 2

PREFILL_PROFILE_KEY = "prefill_bs2_seq192"
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
            "kv_cache_seq_lens": [129, 65],
            "kv_cache_bytes": None,
            "depends_on": ["p0"],
        },
        {
            "batch_id": "d1",
            "type": "decode",
            "replica_id": 1,
            "request_ids": [0, 1],
            "num_tokens": [1, 1],
            "kv_cache_seq_lens": [130, 66],
            "kv_cache_bytes": None,
            "depends_on": ["d0"],
        },
        {
            "batch_id": "d2",
            "type": "decode",
            "replica_id": 1,
            "request_ids": [0],
            "num_tokens": [1],
            "kv_cache_seq_lens": [131],
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
    wl, btm_list = expander.expand(TRACE, job_id=0)
    btm = {e.batch_id: e for e in btm_list}
    return wl, btm


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestExpansionStructure:

    def test_returns_workload_and_map(self, result):
        workload, batch_task_map = result
        assert workload is not None
        assert isinstance(batch_task_map, dict)

    def test_batch_task_map_keys(self, result):
        _, btm = result
        # 4 batches + per-request KV transfer entries
        assert "p0" in btm
        assert "d0" in btm
        assert "d1" in btm
        assert "d2" in btm
        assert any(k.startswith("kv_p0_to_d0_req_") for k in btm)

    def test_batch_types(self, result):
        _, btm = result
        assert btm["p0"].entry_type == BatchEntryType.PREFILL
        assert btm["d0"].entry_type == BatchEntryType.DECODE
        assert next(v.entry_type for v in btm.values() if v.batch_id.startswith("kv_")) == BatchEntryType.KV_TRANSFER

    def test_request_ids_preserved(self, result):
        _, btm = result
        assert btm["p0"].request_ids == [0, 1]
        assert btm["d2"].request_ids == [0]

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
        for tid in btm["p0"].task_ids:
            assert task_map[tid].phase == Phase.PREFILL

    def test_decode_tasks_have_decode_phase(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        for tid in btm["d0"].task_ids:
            assert task_map[tid].phase == Phase.DECODE

    def test_kv_transfer_comm_type(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_task_ids = _kv_task_ids(btm, "p0", "d0")
        kv_tasks = [task_map[tid] for tid in kv_task_ids]
        assert all(t.type == TaskType.FLOW for t in kv_tasks)
        assert all(t.comm_type == CommType.KV_CACHE_TRANSFER for t in kv_tasks)


class TestRankMapping:

    def test_prefill_uses_replica0_ranks(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # Replica 0 with tp=2, ep=1 → ranks [0, 1]
        p0_nodes = {
            task_map[tid].node
            for tid in btm["p0"].task_ids
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert p0_nodes == {0, 1}

    def test_decode_uses_replica1_ranks(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # Replica 1 with tp=2, ep=1 → ranks [2, 3]
        d0_nodes = {
            task_map[tid].node
            for tid in btm["d0"].task_ids
            if task_map[tid].type == TaskType.COMPUTE
        }
        assert d0_nodes == {2, 3}

    def test_kv_transfer_src_is_p_ranks_dst_is_d_ranks(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        kv_task_ids = _kv_task_ids(btm, "p0", "d0")
        kv_tasks = [task_map[tid] for tid in kv_task_ids]
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
            1 for tid in btm["p0"].task_ids
            if task_map[tid].type == TaskType.COMPUTE
        )
        assert compute_count == NUM_LAYERS * 2 * TP

    def test_kv_transfer_task_count(self, result):
        _, btm = result
        # 2 requests × tp=2 ranks = 4 KV flows
        assert len(_kv_task_ids(btm, "p0", "d0")) == 2 * TP

    def test_allreduce_flow_count_per_batch(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}
        # tp=2, ep=1: AllReduce ring for 2 ranks = 2*(2-1)*2 = 4 flows per sub-op
        # 2 layers × 2 sub-ops × 4 flows = 16 flow tasks
        flow_count = sum(
            1 for tid in btm["p0"].task_ids
            if task_map[tid].type == TaskType.FLOW
        )
        assert flow_count == NUM_LAYERS * 2 * TP * 2 * (TP - 1)


class TestDependencies:

    def test_decode_first_compute_depends_on_kv_transfer(self, result):
        workload, btm = result
        task_map = {t.task_id: t for t in workload.tasks}

        kv_task_ids = set(_kv_task_ids(btm, "p0", "d0"))
        # Only the very first sub-op compute tasks (layer 0, attention) should
        # directly depend on KV transfer. Find them: compute tasks whose deps
        # are all from outside d0 (i.e., from KV transfer).
        d0_task_ids = set(btm["d0"].task_ids)
        first_computes = [
            tid for tid in btm["d0"].task_ids
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

        d0_task_ids = set(btm["d0"].task_ids)
        d1_task_ids = set(btm["d1"].task_ids)
        # First sub-op compute tasks of d1 should depend on d0's exit tasks
        first_computes = [
            tid for tid in btm["d1"].task_ids
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

        p0_task_ids = set(btm["p0"].task_ids)
        first_computes = [
            tid for tid in btm["p0"].task_ids
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
            tid for tid in btm["p0"].task_ids
            if task_map[tid].type == TaskType.COMPUTE
        }
        flow_ids = [
            tid for tid in btm["p0"].task_ids
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
        store.load(str(f), "prefill_bs1_seq64")

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

        workload, btm_list = expander.expand(trace)
        btm = {e.batch_id: e for e in btm_list}
        task_map = {t.task_id: t for t in workload.tasks}

        flow_tasks = [
            task_map[tid] for tid in btm["p0"].task_ids
            if task_map[tid].type == TaskType.FLOW
        ]
        comm_types = {t.comm_type for t in flow_tasks}

        assert CommType.EP_ALLTOALL in comm_types
        assert CommType.TP_ALLREDUCE_RING in comm_types


# ── Stage 1 KV Reuse tests ────────────────────────────────────────────────────

# Trace with stage1_kv_reuse enabled
TRACE_WITH_REUSE = {
    "version": "1.0",
    "model": "test-model",
    "model_config": {"hidden_size": 64, "num_layers": NUM_LAYERS, "dense_layers": 2},
    "total_layers": NUM_LAYERS,
    "parallelism": {"tp": TP, "ep": EP, "pp": 1},
    "pd_config": {"pd_node_ratio": 0.5, "pd_p2p_comm_bandwidth_gbps": 200},
    "stage1_kv_reuse": {
        "num_storage_nodes": 2,
    },
    "requests": {
        "0": {
            "num_prefill_tokens": 128, "num_decode_tokens": 4,
            "kv_reuse_hit_ratio": 0.5, "kv_reuse_storage_node_idx": 0,
        },
        "1": {
            "num_prefill_tokens": 64, "num_decode_tokens": 2,
            "kv_reuse_hit_ratio": 0.3, "kv_reuse_storage_node_idx": 1,
        },
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
            "kv_cache_seq_lens": [129, 65],
            "kv_cache_bytes": None,
            "depends_on": ["p0"],
        },
    ],
}


@pytest.fixture
def reuse_result(store):
    exp = InferenceTraceExpander(store, tp=TP, ep=EP)
    wl, btm_list = exp.expand(TRACE_WITH_REUSE, job_id=0)
    btm = {e.batch_id: e for e in btm_list}
    return wl, btm


class TestStage1KvReuseFlows:

    def test_stage1_reuse_flows_created(self, reuse_result):
        """Stage 1 KV reuse flows should be created when metadata is present."""
        workload, btm = reuse_result
        assert "kv_reuse_p0_req_0" in btm
        assert "kv_reuse_p0_req_1" in btm
        reuse_tids = _kv_reuse_task_ids(btm, "p0")
        # 2 requests × 2 layers × 2 ranks = 8 reuse flows
        assert len(reuse_tids) == 2 * NUM_LAYERS * TP

    def test_stage1_no_reuse_no_flows(self, store):
        """No stage1 metadata → no reuse flows."""
        exp = InferenceTraceExpander(store, tp=TP, ep=EP)
        workload, btm_list = exp.expand(TRACE, job_id=0)
        btm = {e.batch_id: e for e in btm_list}
        assert not any(k.startswith("kv_reuse_") for k in btm)

    def test_stage1_comm_type(self, reuse_result):
        """All reuse flows should have KV_CACHE_REUSE comm_type."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        for tid in _kv_reuse_task_ids(btm, "p0"):
            assert task_map[tid].comm_type == CommType.KV_CACHE_REUSE

    def test_stage1_per_layer_layer_id(self, reuse_result):
        """Each reuse flow should have the correct layer_id."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        reuse_tasks = [task_map[tid] for tid in _kv_reuse_task_ids(btm, "p0")]
        layer_ids = {t.layer_id for t in reuse_tasks}
        assert layer_ids == {0, 1}

    def test_stage1_per_layer_bytes(self, reuse_result):
        """Bytes per flow = kv_cache_bytes * hit_ratio / num_layers / stage_size."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        # Request 0: 4096 * 0.5 / 2 layers / 2 ranks = 512
        req0_flows = [
            task_map[tid] for tid in _kv_reuse_task_ids(btm, "p0")
            if tid in {t.task_id for t in workload.tasks
                       if t.src == 4}  # storage node 0
        ]
        # All flows from storage node 0 should be for request 0
        assert len(req0_flows) > 0
        for f in req0_flows:
            assert f.size_bytes == 4096 * 0.5 // NUM_LAYERS // TP

    def test_stage1_per_request_storage_node(self, reuse_result):
        """Same request's all flows come from same storage node; different requests can differ."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        reuse_tasks = [task_map[tid] for tid in _kv_reuse_task_ids(btm, "p0")]

        # Storage node 0 → rank [0,1], storage node 1 → rank [1]
        # Actually, check that each request's flows have consistent src
        # tp=2, ep=1: dest ranks = [0, 1]
        # Storage node for req 0: idx 0 → actual node = total_gpus + 0 = 4
        # Storage node for req 1: idx 1 → actual node = total_gpus + 1 = 5
        # total_gpus = tp*ep*pp * num_replicas = 2*1*1 * 2 = 4

        # All flows for req 0: src = 4, all flows for req 1: src = 5
        req0_tasks = [t for t in reuse_tasks if t.src == 4]
        req1_tasks = [t for t in reuse_tasks if t.src == 5]
        assert len(req0_tasks) == NUM_LAYERS * TP  # req 0
        assert len(req1_tasks) == NUM_LAYERS * TP  # req 1

    def test_stage1_per_request_hit_ratio(self, reuse_result):
        """Different requests should have different sizes based on their hit_ratio."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        reuse_tasks = [task_map[tid] for tid in _kv_reuse_task_ids(btm, "p0")]

        # Req 0: src=4, hit_ratio=0.5
        req0_sizes = {t.size_bytes for t in reuse_tasks if t.src == 4}
        # Req 1: src=5, hit_ratio=0.3
        req1_sizes = {t.size_bytes for t in reuse_tasks if t.src == 5}

        assert len(req0_sizes) == 1
        assert len(req1_sizes) == 1
        assert req0_sizes != req1_sizes

    def test_stage1_compute_depends_on_reuse(self, reuse_result):
        """First sub-op compute tasks at each layer should depend on reuse flows for that layer."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}

        reuse_tids = set(_kv_reuse_task_ids(btm, "p0"))
        # Find first compute tasks for p0 at layer 0 (attention = first sub-op)
        # These are the compute tasks whose deps include reuse flows but no other p0 tasks
        p0_task_ids = set(btm["p0"].task_ids)
        layer0_first_computes = [
            tid for tid in btm["p0"].task_ids
            if task_map[tid].type == TaskType.COMPUTE
            and task_map[tid].layer_id == 0
            and not any(dep in p0_task_ids for dep in task_map[tid].deps)
        ]
        assert len(layer0_first_computes) > 0, "No first sub-op compute tasks found"
        for tid in layer0_first_computes:
            deps = task_map[tid].deps
            assert any(dep in reuse_tids for dep in deps), (
                f"First compute task {tid} should depend on reuse flows, "
                f"got deps={deps}"
            )

    def test_stage1_batch_task_map(self, reuse_result):
        """batch_task_map should have per-request kv_reuse entries."""
        _, btm = reuse_result
        req0_key = "kv_reuse_p0_req_0"
        req1_key = "kv_reuse_p0_req_1"
        assert req0_key in btm
        assert req1_key in btm
        assert btm[req0_key].entry_type == BatchEntryType.KV_REUSE
        assert btm[req1_key].entry_type == BatchEntryType.KV_REUSE
        assert btm[req0_key].request_ids == [0]
        assert btm[req1_key].request_ids == [1]
        # Each request should have NUM_LAYERS * TP flow IDs
        assert len(btm[req0_key].task_ids) == NUM_LAYERS * TP
        assert len(btm[req1_key].task_ids) == NUM_LAYERS * TP

    def test_stage1_reuse_flows_no_deps(self, reuse_result):
        """All reuse flows should have no dependencies (data already on storage node)."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        for tid in _kv_reuse_task_ids(btm, "p0"):
            assert task_map[tid].deps == []

    def test_stage1_reuse_flows_prefill_phase(self, reuse_result):
        """All reuse flows should have PREFILL phase."""
        workload, btm = reuse_result
        task_map = {t.task_id: t for t in workload.tasks}
        for tid in _kv_reuse_task_ids(btm, "p0"):
            assert task_map[tid].phase == Phase.PREFILL

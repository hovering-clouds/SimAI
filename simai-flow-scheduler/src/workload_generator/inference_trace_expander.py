"""
Inference Trace Expander - expands Vidur inference trace JSON into P2PWorkload.

Trace format (per phase5-taskA-research.md section 9.4):
{
  "version": "1.0",
  "model": "deepseek-671b",
  "model_config": { "hidden_size": 7168, "num_layers": 61, "dense_layers": 3, ... },
  "parallelism": { "tp": 8, "pp": 1, "ep": 8 },
  "pd_config": { "pd_node_ratio": 0.5, "pd_p2p_comm_bandwidth_gbps": 200 },
  "requests": { "0": { "num_prefill_tokens": 128, "num_decode_tokens": 64 }, ... },
  "batches": [
    { "batch_id": "p0", "type": "prefill", "replica_id": 0,
      "request_ids": [0, 1], "num_tokens": [128, 256],
      "kv_cache_bytes": { "0": 123456, "1": 234567 }, "depends_on": [] },
    { "batch_id": "d0", "type": "decode", "replica_id": 4,
      "request_ids": [0, 1], "num_tokens": [1, 1],
      "kv_cache_bytes": null, "depends_on": ["p0"] },
    ...
  ]
}

Expansion rules:
- Prefill/Decode batch → per-layer (COMPUTE + TP AllReduce) for attention/mlp layers,
  per-layer (COMPUTE + EP AlltoAll) for moe layers.
- KV cache transfer → P2P FLOW from each P-node TP rank to corresponding D-node TP rank,
  inserted between a prefill batch and the decode batch that depends on it.
- Inter-batch deps follow the `depends_on` field.

Returns (P2PWorkload, batch_task_map) where batch_task_map is:
  { batch_id: { "task_ids": [...], "request_ids": [...], "type": "prefill"/"decode"/"kv_transfer" } }
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..workload_format.schema import (
    P2PWorkload, Job, Meta, Task, Phase, CommType, TaskType, ParallelismConfig,
)
from .collective_expander import FlowTask, AllReduceExpander, AlltoAllExpander
from .inference_profile import InferenceProfileStore, LayerProfile
from .rank_grouper import RankGrouper


# ── Internal helpers ──────────────────────────────────────────────────────────

@dataclass
class _FlowGroupResult:
    """Tracks expanded flows with a receiver index for O(1) dep lookup."""
    flows: list[FlowTask] = field(default_factory=list)
    receiver_index: dict[int, list[int]] = field(default_factory=dict)

    def add(self, flow: FlowTask):
        self.flows.append(flow)
        self.receiver_index.setdefault(flow.dst, []).append(flow.task_id)


# ── Main class ────────────────────────────────────────────────────────────────

class InferenceTraceExpander:
    """
    Expand a Vidur inference trace JSON into a P2PWorkload.

    Args:
        profile_store: Loaded InferenceProfileStore.
        tp: Tensor parallel size.
        ep: Expert parallel size (default 1).
        pp: Pipeline parallel size (default 1, only pp=1 supported).
        assigned_nodes: All available node IDs for inference replicas.
            Each replica occupies tp*ep*pp consecutive nodes from this list.
            If None, falls back to the default mapping where replica r occupies
            [r*ws, ..., (r+1)*ws - 1].
    """

    def __init__(
        self,
        profile_store: InferenceProfileStore,
        tp: int,
        ep: int = 1,
        pp: int = 1,
        assigned_nodes: Optional[list[int]] = None,
    ):
        if pp != 1:
            raise NotImplementedError("pp > 1 not yet supported for inference expansion")
        self._store = profile_store
        self._tp = tp
        self._ep = ep
        self._pp = pp
        self._assigned_nodes = assigned_nodes
        self._ar = AllReduceExpander()
        self._a2a = AlltoAllExpander()

    # ── Public API ────────────────────────────────────────────────────────────

    def expand(
        self,
        trace: dict,
        job_id: int = 0,
        prefill_profile_key: Optional[str] = None,
        decode_profile_key: Optional[str] = None,
    ) -> tuple[P2PWorkload, dict]:
        """
        Expand trace into P2PWorkload + batch_task_map.

        Args:
            trace: Parsed trace dict (from JSON).
            job_id: Job ID to assign to all generated tasks.
            prefill_profile_key: Key in profile_store for prefill profiling data.
            decode_profile_key: Key in profile_store for decode profiling data.

        Returns:
            (P2PWorkload, batch_task_map) where batch_task_map maps
            batch_id → { task_ids, request_ids, type, replica_id }.
        """
        all_flow_tasks: list[FlowTask] = []
        task_id = 0
        batch_task_map: dict = {}

        # batch_id → {rank: [task_ids]} — "exit" tasks of each batch
        # (the tasks that the next batch's first tasks should depend on)
        batch_exits: dict[str, dict[int, list[int]]] = {}

        batch_lookup = {b["batch_id"]: b for b in trace["batches"]}

        for batch in trace["batches"]:
            bid = batch["batch_id"]
            btype = batch["type"]
            replica_id = batch["replica_id"]

            # ── Collect predecessor exit tasks ────────────────────────────
            prev_exits: dict[int, list[int]] = {}

            for dep_id in batch.get("depends_on", []):
                dep_batch = batch_lookup[dep_id]

                if dep_batch["type"] == "prefill" and btype == "decode":
                    # Insert KV cache transfer between prefill and this decode
                    kv_tasks, kv_exits, task_id = self._expand_kv_transfer(
                        request_ids=batch["request_ids"],
                        kv_cache_bytes=dep_batch.get("kv_cache_bytes") or {},
                        p_replica_id=dep_batch["replica_id"],
                        d_replica_id=replica_id,
                        job_id=job_id,
                        task_id_start=task_id,
                        prev_exits=batch_exits.get(dep_id, {}),
                    )
                    all_flow_tasks.extend(kv_tasks)

                    kv_key = f"kv_{dep_id}_to_{bid}"
                    batch_task_map[kv_key] = {
                        "task_ids": [t.task_id for t in kv_tasks],
                        "type": "kv_transfer",
                        "from_batch": dep_id,
                        "to_batch": bid,
                    }
                    for rank, tids in kv_exits.items():
                        prev_exits.setdefault(rank, []).extend(tids)
                else:
                    for rank, tids in batch_exits.get(dep_id, {}).items():
                        prev_exits.setdefault(rank, []).extend(tids)

            # ── Expand batch ──────────────────────────────────────────────
            profile_key = (
                prefill_profile_key if btype == "prefill" else decode_profile_key
            )
            batch_tasks, exits, task_id = self._expand_batch(
                batch=batch,
                profile_key=profile_key,
                job_id=job_id,
                task_id_start=task_id,
                prev_exits=prev_exits,
            )
            all_flow_tasks.extend(batch_tasks)
            batch_exits[bid] = exits

            batch_task_map[bid] = {
                "task_ids": [t.task_id for t in batch_tasks],
                "request_ids": batch["request_ids"],
                "type": btype,
                "replica_id": replica_id,
            }

        # ── Build P2PWorkload ─────────────────────────────────────────────
        all_ranks: set[int] = set()
        for batch in trace["batches"]:
            all_ranks.update(self._replica_ranks(batch["replica_id"]))

        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(all_ranks)),
            jobs=[Job(
                job_id=job_id,
                assigned_nodes=sorted(all_ranks),
                parallelism=ParallelismConfig(tp=self._tp, ep=self._ep, pp=self._pp),
            )],
            tasks=[ft.to_task() for ft in all_flow_tasks],
        )
        return workload, batch_task_map

    @staticmethod
    def load_trace(path: str) -> dict:
        """Load a trace JSON file."""
        with open(path) as f:
            return json.load(f)

    # ── Rank helpers ──────────────────────────────────────────────────────────

    def _world_size(self) -> int:
        return self._tp * self._ep * self._pp

    def _replica_ranks(self, replica_id: int) -> list[int]:
        """Physical nodes for a replica: tp*ep*pp consecutive nodes from assigned_nodes."""
        ws = self._world_size()
        offset = replica_id * ws
        if self._assigned_nodes is not None:
            return list(self._assigned_nodes[offset:offset + ws])
        return list(range(offset, offset + ws))

    def _grouper(self, replica_id: int) -> RankGrouper:
        return RankGrouper(
            self._replica_ranks(replica_id),
            ParallelismConfig(tp=self._tp, ep=self._ep, pp=self._pp),
        )

    # ── Batch expansion ───────────────────────────────────────────────────────

    def _expand_batch(
        self,
        batch: dict,
        profile_key: Optional[str],
        job_id: int,
        task_id_start: int,
        prev_exits: dict[int, list[int]],
    ) -> tuple[list[FlowTask], dict[int, list[int]], int]:
        """
        Expand one prefill or decode batch into FlowTasks.

        Returns (tasks, exits, next_task_id) where exits maps
        rank → [task_ids] for the last layer's output tasks.
        """
        phase = Phase.PREFILL if batch["type"] == "prefill" else Phase.DECODE
        replica_id = batch["replica_id"]
        grouper = self._grouper(replica_id)
        ranks = self._replica_ranks(replica_id)

        profiles = self._store.get_profile(profile_key) if profile_key else []
        layers = self._group_profiles_by_layer(profiles)

        all_tasks: list[FlowTask] = []
        task_id = task_id_start
        current_exits = dict(prev_exits)  # rank → [task_ids]

        # Layer names in canonical execution order within a layer
        _LAYER_NAME_ORDER = {"attention": 0, "mlp": 1, "moe": 1}

        for layer_id, layer_map in sorted(layers.items()):
            # Iterate sub-operations in order: attention → mlp/moe
            sub_ops = sorted(
                layer_map.items(),
                key=lambda kv: _LAYER_NAME_ORDER.get(kv[0], 99),
            )

            for op_name, profile in sub_ops:
                # ── COMPUTE tasks (one per rank) ──────────────────────────
                compute: dict[int, FlowTask] = {}
                for rank in ranks:
                    ft = FlowTask(
                        task_id=task_id,
                        job_id=job_id,
                        type=TaskType.COMPUTE,
                        node=rank,
                        duration_us=max(profile.comp_time_us, 1),
                        phase=phase,
                        layer_id=layer_id,
                        iteration=0,
                        deps=list(current_exits.get(rank, [])),
                    )
                    compute[rank] = ft
                    all_tasks.append(ft)
                    task_id += 1

                # ── FLOW tasks ────────────────────────────────────────────
                flow_result = _FlowGroupResult()
                comm_size = profile.comm_size_bytes

                if comm_size > 0:
                    if op_name == "moe":
                        # EP AlltoAll: one AlltoAll per TP-rank position
                        for tp_idx in range(self._tp):
                            ep_group = grouper.get_ep_group(0, 0, tp_idx)
                            flows = self._a2a.expand_alltoall(
                                ep_group, comm_size, job_id, task_id, "ep")
                            for fl in flows:
                                fl.phase = phase
                                fl.layer_id = layer_id
                                fl.deps.append(compute[fl.src].task_id)
                                flow_result.add(fl)
                            all_tasks.extend(flows)
                            task_id += len(flows)
                    else:
                        # TP AllReduce: one AllReduce per EP group
                        for ep_idx in range(self._ep):
                            tp_group = grouper.get_tp_group(0, 0, ep_idx)
                            flows = self._ar.expand_allreduce(
                                tp_group, comm_size, "ring", job_id, task_id, "tp")
                            for fl in flows:
                                fl.phase = phase
                                fl.layer_id = layer_id
                                fl.deps.append(compute[fl.src].task_id)
                                flow_result.add(fl)
                            all_tasks.extend(flows)
                            task_id += len(flows)

                # ── Update exits for next sub-op / next layer ─────────────
                if flow_result.flows:
                    current_exits = {
                        rank: flow_result.receiver_index.get(rank, [])
                        for rank in ranks
                    }
                else:
                    current_exits = {rank: [compute[rank].task_id] for rank in ranks}

        return all_tasks, current_exits, task_id

    # ── KV cache transfer ─────────────────────────────────────────────────────

    def _expand_kv_transfer(
        self,
        request_ids: list,
        kv_cache_bytes: dict,
        p_replica_id: int,
        d_replica_id: int,
        job_id: int,
        task_id_start: int,
        prev_exits: dict[int, list[int]],
    ) -> tuple[list[FlowTask], dict[int, list[int]], int]:
        """
        Expand KV cache transfer flows from P-node TP ranks to D-node TP ranks.

        One flow per (request, TP rank pair). Each flow carries
        kv_cache_bytes[req_id] / tp bytes. All flows depend on the prefill
        batch's exit tasks.

        Returns (tasks, exits, next_task_id) where exits maps
        d_rank → [task_ids] for the KV transfer flows received at D-node.
        """
        p_grouper = self._grouper(p_replica_id)
        d_grouper = self._grouper(d_replica_id)

        tasks: list[FlowTask] = []
        task_id = task_id_start
        exits: dict[int, list[int]] = {}  # d_rank → [task_ids]

        for req_id in request_ids:
            req_key = str(req_id)
            total_kv = kv_cache_bytes.get(req_key, 0)
            if total_kv == 0:
                continue

            per_rank_bytes = max(total_kv // self._tp, 1)

            # One flow per TP rank pair (same ep_idx=0, tp_idx varies)
            for tp_idx in range(self._tp):
                p_rank = p_grouper.get_pp_rank(0, 0, 0, tp_idx)
                d_rank = d_grouper.get_pp_rank(0, 0, 0, tp_idx)

                ft = FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=p_rank,
                    dst=d_rank,
                    size_bytes=per_rank_bytes,
                    comm_type=CommType.KV_CACHE_TRANSFER,
                    chunk_id=0,
                    num_chunks=1,
                    phase=Phase.PREFILL,
                    layer_id=0,
                    deps=list(prev_exits.get(p_rank, [])),
                )
                tasks.append(ft)
                exits.setdefault(d_rank, []).append(task_id)
                task_id += 1

        return tasks, exits, task_id

    # ── Profile helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _group_profiles_by_layer(
        profiles: list[LayerProfile],
    ) -> dict[int, dict[str, LayerProfile]]:
        """Group profiles by layer_id → {layer_name: LayerProfile}."""
        layers: dict[int, dict[str, LayerProfile]] = {}
        for p in profiles:
            layers.setdefault(p.layer_id, {})[p.layer_name] = p
        return layers

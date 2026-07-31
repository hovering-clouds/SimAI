"""
Inference Trace Expander - expands Vidur inference trace JSON into P2PWorkload.

Supports both pp=1 (flat batches) and pp>1 (per-stage batches with stage_id).

Trace format (pp=1):
  batches: [{ batch_id, type, replica_id, request_ids, num_tokens, kv_cache_bytes, depends_on }]

Trace format (pp>1, per-stage):
  batches: [{ batch_id, type, replica_id, stage_id, request_ids, num_tokens,
              kv_cache_bytes (per-stage share), depends_on }]
  - Each entry = one micro-batch on one PP stage
  - depends_on encodes same-stage + cross-stage pipeline deps
  - kv_cache_bytes on prefill entries = per-stage KV (proportional to actual layers in that stage)

Expansion rules:
- Per-stage batch → per-layer COMPUTE + TP/EP communication for layers in that stage
- PP inter-stage → P2P PP_SEND flows between consecutive stage GPU sets
- KV cache transfer → per-stage P2P flows from P-stage-s to D-stage-s
- Pipeline overlap: same-stage deps + cross-stage deps recreated from trace depends_on
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..workload_format.schema import (
    P2PWorkload, Job, Meta, Task, Phase, CommType, TaskType,
    ParallelismConfig, BatchTaskInfo, BatchEntryType,
)
from .collective_expander import FlowTask, AllReduceExpander, AlltoAllExpander
from .inference_profile import InferenceProfileStore, LayerProfile
from .rank_grouper import VllmRankGrouper


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

    Supports PP (pipeline parallelism) with per-stage trace entries.
    Each PP stage occupies tp*ep GPUs within a replica's world_size.

    Args:
        profile_store: Loaded InferenceProfileStore.
        tp: Tensor parallel size.
        ep: Expert parallel size (default 1).
        pp: Pipeline parallel size (default 1).
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
        storage_node_ids: list[int] | None = None,
    ) -> tuple[P2PWorkload, list[BatchTaskInfo]]:
        """
        Expand trace into P2PWorkload + batch_task_info list.
        Now internally loops over expand_single_batch() — behaviour unchanged.

        Args:
            trace: Parsed trace dict (from JSON).
            job_id: Job ID to assign to all generated tasks.
            storage_node_ids: Explicit storage node topology IDs for KV reuse.
                When provided, overrides auto-computation from trace metadata.

        Returns:
            (P2PWorkload, batch_task_info) where batch_task_info is a list of
            BatchTaskInfo entries, one per trace batch or sub-operation.
        """
        total_layers = self._get_total_layers(trace)
        batch_lookup = {b["batch_id"]: b for b in trace["batches"]}
        resolved_storage_node_ids, request_reuse_info = self._setup_kv_reuse(
            trace, storage_node_ids)

        all_flow_tasks: list[FlowTask] = []
        task_id = 0
        batch_task_info: list[BatchTaskInfo] = []

        # batch_id → {rank: [task_ids]} — "exit" tasks of each batch
        # (the tasks that the next batch's first tasks should depend on)
        batch_exits: dict[str, dict[int, list[int]]] = {}

        for batch in trace["batches"]:
            tasks, exits, infos, task_id = self.expand_single_batch(
                batch=batch,
                batch_lookup=batch_lookup,
                batch_exits=batch_exits,
                job_id=job_id,
                task_id_start=task_id,
                total_layers=total_layers,
                trace=trace,  # needed by _expand_pp_communication
                request_reuse_info=request_reuse_info,
                resolved_storage_node_ids=resolved_storage_node_ids,
            )
            all_flow_tasks.extend(tasks)
            batch_exits[batch["batch_id"]] = exits
            batch_task_info.extend(infos)

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
                # dp = ep: in the VllmRankGrouper model, EP spans the whole
                # PP stage (dp × tp ranks), so a replica holds dp=ep model copies.
                parallelism=ParallelismConfig(
                    tp=self._tp, dp=self._ep, pp=self._pp, ep=self._ep),
            )],
            tasks=[ft.to_task() for ft in all_flow_tasks],
        )
        return workload, batch_task_info

    # ── Single-batch expansion (for JobExpander in dynamic mode) ───────────

    def expand_single_batch(
        self,
        batch: dict,
        batch_lookup: dict,
        batch_exits: dict[str, dict[int, list[int]]],
        job_id: int,
        task_id_start: int,
        total_layers: int,
        trace: dict,
        request_reuse_info: dict[int, dict],
        resolved_storage_node_ids: list[int],
    ) -> tuple[list[FlowTask], dict[int, list[int]], list[BatchTaskInfo], int]:
        """
        Expand a single trace batch.

        Args:
            batch: The batch dict from the trace.
            batch_lookup: batch_id → batch dict (for resolving deps).
            batch_exits: Accumulated exit tasks from previous batches.
            job_id: Job ID for generated tasks.
            task_id_start: Starting task_id.
            total_layers: Model total layer count.
            request_reuse_info: KV reuse config from _setup_kv_reuse().
            resolved_storage_node_ids: KV storage node IDs.

        Returns:
            (new_flow_tasks, exits, batch_infos, next_task_id)
        """
        bid = batch["batch_id"]
        btype = batch["type"]
        replica_id = batch["replica_id"]
        stage_id = batch.get("stage_id", 0)
        task_id = task_id_start

        new_flow_tasks: list[FlowTask] = []
        batch_infos: list[BatchTaskInfo] = []

        # ── Collect predecessor exit tasks ────────────────────────────────
        prev_exits: dict[int, list[int]] = {}

        for dep_id in batch.get("depends_on", []):
            dep_batch = batch_lookup[dep_id]
            dep_stage_id = dep_batch.get("stage_id", 0)

            # PP inter-stage: dep from a different stage on same replica
            if (dep_stage_id != stage_id
                    and dep_batch.get("replica_id") == replica_id):
                pp_tasks, pp_exits, task_id = self._expand_pp_communication(
                    src_stage_id=dep_stage_id,
                    dst_stage_id=stage_id,
                    replica_id=replica_id,
                    total_tokens=sum(batch["num_tokens"]),
                    job_id=job_id,
                    task_id_start=task_id,
                    prev_exits=batch_exits.get(dep_id, {}),
                    trace=trace,
                )
                new_flow_tasks.extend(pp_tasks)
                batch_infos.append(BatchTaskInfo(
                    batch_id=f"pp_{dep_id}_to_{bid}",
                    task_ids=[t.task_id for t in pp_tasks],
                    entry_type=BatchEntryType.PP_COMM,
                    replica_id=replica_id,
                    stage_id=stage_id,
                    request_ids=[],
                ))
                for rank, tids in pp_exits.items():
                    prev_exits.setdefault(rank, []).extend(tids)
                continue

            # KV transfer: prefill → decode across replicas
            if dep_batch["type"] == "prefill" and btype == "decode":
                kv_tasks, kv_exits, req_task_map, task_id = self._expand_kv_transfer(
                    request_ids=batch["request_ids"],
                    kv_cache_bytes=dep_batch.get("kv_cache_bytes") or {},
                    p_replica_id=dep_batch["replica_id"],
                    d_replica_id=replica_id,
                    src_stage_id=dep_stage_id,
                    dst_stage_id=stage_id,
                    job_id=job_id,
                    task_id_start=task_id,
                    prev_exits=batch_exits.get(dep_id, {}),
                )
                new_flow_tasks.extend(kv_tasks)
                for req_id, tids in req_task_map.items():
                    batch_infos.append(BatchTaskInfo(
                        batch_id=f"kv_{dep_id}_to_{bid}_req_{req_id}",
                        task_ids=tids,
                        entry_type=BatchEntryType.KV_TRANSFER,
                        replica_id=replica_id,
                        stage_id=stage_id,
                        request_ids=[req_id],
                    ))
                for rank, tids in kv_exits.items():
                    prev_exits.setdefault(rank, []).extend(tids)
            else:
                # Same-stage or same-replica dependency
                for rank, tids in batch_exits.get(dep_id, {}).items():
                    prev_exits.setdefault(rank, []).extend(tids)

        # ── Select profile by (phase, bs, seq) ─────────────────────────────
        bs = len(batch["request_ids"])
        if btype == "prefill":
            seq = sum(batch["num_tokens"])
        else:
            kv_lens = batch.get("kv_cache_seq_lens")
            seq = max(kv_lens) if kv_lens else 1
        profiles = self._store.get_profile_for_batch(btype, bs, seq)

        # ── Expand batch (per-stage when pp>1) ────────────────────────────
        stage_layer_range = self._layers_for_stage(stage_id, total_layers, self._pp) \
            if self._pp > 1 else None

        # ── Stage 1 KV reuse flows (prefill batches only) ─────────────────
        reuse_layer_deps = None
        if btype == "prefill" and request_reuse_info:
            batch_req_ids = batch["request_ids"]
            batch_reuse = {
                rid: request_reuse_info[rid]
                for rid in batch_req_ids if rid in request_reuse_info
            }
            if batch_reuse:
                dest_ranks = self._stage_ranks(replica_id, stage_id) \
                    if self._pp > 1 else self._replica_ranks(replica_id)
                stage_layer_ids = list(stage_layer_range) \
                    if stage_layer_range is not None \
                    else list(range(total_layers))
                kv_bytes = batch.get("kv_cache_bytes") or {}
                reuse_flows, reuse_layer_deps, req_task_map, task_id = \
                    self._expand_kv_reuse(
                        storage_node_ids=resolved_storage_node_ids,
                        dest_ranks=dest_ranks,
                        kv_cache_bytes=kv_bytes,
                        stage_layer_ids=stage_layer_ids,
                        request_reuse_info=batch_reuse,
                        job_id=job_id,
                        task_id_start=task_id,
                        prev_exits=prev_exits,
                    )
                new_flow_tasks.extend(reuse_flows)
                for req_key, tids in req_task_map.items():
                    batch_infos.append(BatchTaskInfo(
                        batch_id=f"kv_reuse_{bid}_req_{req_key}",
                        task_ids=tids,
                        entry_type=BatchEntryType.KV_REUSE,
                        replica_id=replica_id,
                        stage_id=stage_id,
                        request_ids=[int(req_key)],
                    ))

        batch_tasks, exits, task_id = self._expand_batch(
            batch=batch,
            profiles=profiles,
            job_id=job_id,
            task_id_start=task_id,
            prev_exits=prev_exits,
            stage_id=stage_id,
            stage_layer_range=stage_layer_range,
            reuse_layer_deps=reuse_layer_deps,
        )
        new_flow_tasks.extend(batch_tasks)
        batch_infos.append(BatchTaskInfo(
            batch_id=bid,
            task_ids=[t.task_id for t in batch_tasks],
            entry_type=BatchEntryType(btype),
            replica_id=replica_id,
            stage_id=stage_id,
            request_ids=batch["request_ids"],
        ))

        return new_flow_tasks, exits, batch_infos, task_id

    # ── KV reuse setup (extracted for reuse by both expand and expand_single_batch) ──

    def _setup_kv_reuse(
        self,
        trace: dict,
        storage_node_ids: list[int] | None,
    ) -> tuple[list[int], dict[int, dict]]:
        """Set up KV reuse configuration.  Returns (resolved_storage_node_ids, request_reuse_info)."""
        resolved_storage_node_ids: list[int] = []
        request_reuse_info: dict[int, dict] = {}
        reuse_cfg = trace.get("stage1_kv_reuse")
        if reuse_cfg and reuse_cfg.get("num_storage_nodes", 0) > 0:
            if storage_node_ids is not None:
                resolved_storage_node_ids = list(storage_node_ids)
            else:
                num_storage = reuse_cfg["num_storage_nodes"]
                if self._assigned_nodes is not None:
                    total_gpus = max(self._assigned_nodes) + 1
                else:
                    num_replicas = len({b["replica_id"] for b in trace["batches"]})
                    total_gpus = self._world_size() * num_replicas
                resolved_storage_node_ids = list(
                    range(total_gpus, total_gpus + num_storage))
            for req_key, req_meta in trace.get("requests", {}).items():
                hit_ratio = req_meta.get("kv_reuse_hit_ratio")
                node_idx = req_meta.get("kv_reuse_storage_node_idx")
                if hit_ratio is not None and hit_ratio > 0 and node_idx is not None:
                    request_reuse_info[int(req_key)] = {
                        "hit_ratio": hit_ratio,
                        "storage_node_idx": node_idx,
                    }
        return resolved_storage_node_ids, request_reuse_info

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

    def _grouper(self, replica_id: int) -> VllmRankGrouper:
        return VllmRankGrouper(
            self._replica_ranks(replica_id),
            ParallelismConfig(tp=self._tp, dp=self._ep, pp=self._pp, ep=self._ep),
        )

    def _stage_ranks(self, replica_id: int, stage_id: int) -> list[int]:
        """GPUs for a specific PP stage within a replica."""
        stage_size = self._tp * self._ep
        offset = replica_id * self._world_size() + stage_id * stage_size
        if self._assigned_nodes is not None:
            return list(self._assigned_nodes[offset:offset + stage_size])
        return list(range(offset, offset + stage_size))

    def _stage_grouper(self, replica_id: int, stage_id: int) -> VllmRankGrouper:
        """VllmRankGrouper for a specific PP stage (dp=ep, pp=1, only tp+ep)."""
        return VllmRankGrouper(
            self._stage_ranks(replica_id, stage_id),
            ParallelismConfig(tp=self._tp, dp=self._ep, pp=1, ep=self._ep),
        )

    @staticmethod
    def _layers_for_stage(stage_id: int, total_layers: int, pp: int) -> range:
        """Layer IDs belonging to a PP stage.

        The last stage takes any remainder from uneven splits
        (e.g. 61 layers with pp=2 → stage 0: 0-29, stage 1: 30-60).
        """
        per_stage = total_layers // pp
        start = stage_id * per_stage
        if stage_id == pp - 1:
            end = total_layers
        else:
            end = start + per_stage
        return range(start, end)

    # ── Batch expansion ───────────────────────────────────────────────────────

    def _expand_batch(
        self,
        batch: dict,
        profiles: list,
        job_id: int,
        task_id_start: int,
        prev_exits: dict[int, list[int]],
        stage_id: int = 0,
        stage_layer_range: Optional[range] = None,
        reuse_layer_deps: Optional[dict[int, dict[int, list[int]]]] = None,
    ) -> tuple[list[FlowTask], dict[int, list[int]], int]:
        """
        Expand one prefill or decode batch into FlowTasks.

        When pp>1, stage_id and stage_layer_range restrict expansion to
        the layers belonging to that PP stage, using per-stage ranks.

        Returns (tasks, exits, next_task_id) where exits maps
        rank → [task_ids] for the last layer's output tasks.
        """
        phase = Phase.PREFILL if batch["type"] == "prefill" else Phase.DECODE
        replica_id = batch["replica_id"]

        if self._pp > 1:
            grouper = self._stage_grouper(replica_id, stage_id)
            ranks = self._stage_ranks(replica_id, stage_id)
        else:
            grouper = self._grouper(replica_id)
            ranks = self._replica_ranks(replica_id)

        layers = self._group_profiles_by_layer(profiles)

        # Filter layers to only those in this PP stage
        if stage_layer_range is not None:
            layers = {lid: lm for lid, lm in layers.items()
                      if lid in stage_layer_range}

        all_tasks: list[FlowTask] = []
        task_id = task_id_start
        current_exits = dict(prev_exits)  # rank → [task_ids]

        # Layer names in canonical execution order within a layer
        _LAYER_NAME_ORDER = {"attention": 0, "mlp": 1, "moe": 1}

        for layer_id, layer_map in sorted(layers.items()):
            # Inject KV reuse flow deps for this layer
            if reuse_layer_deps and layer_id in reuse_layer_deps:
                for rank, reuse_deps in reuse_layer_deps[layer_id].items():
                    current_exits.setdefault(rank, []).extend(reuse_deps)

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
                        # EP AlltoAll (vLLM style): one AlltoAll across ALL
                        # ranks in the PP stage (dp x tp).
                        ep_group = grouper.get_ep_group(0)
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
                            tp_group = grouper.get_tp_group(0, ep_idx)
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

    # ── PP inter-stage communication ──────────────────────────────────────────

    def _expand_pp_communication(
        self,
        src_stage_id: int,
        dst_stage_id: int,
        replica_id: int,
        total_tokens: int,
        job_id: int,
        task_id_start: int,
        prev_exits: dict[int, list[int]],
        trace: dict,
    ) -> tuple[list[FlowTask], dict[int, list[int]], int]:
        """
        Expand PP inter-stage hidden-state transfer as P2P PP_SEND flows.

        One flow per rank pair (src_stage_rank[i] → dst_stage_rank[i]).
        Size = hidden_size / tp * dtype_bytes * total_tokens per rank.

        Returns (tasks, exits, next_task_id).
        """
        src_ranks = self._stage_ranks(replica_id, src_stage_id)
        dst_ranks = self._stage_ranks(replica_id, dst_stage_id)

        # Estimate per-rank PP comm size from trace metadata or defaults
        hidden_size = trace.get("hidden_size", 4096)
        dtype_bytes = trace.get("dtype_bytes", 2)
        per_rank_bytes = max(hidden_size // self._tp * dtype_bytes * total_tokens, 1)

        tasks: list[FlowTask] = []
        task_id = task_id_start
        exits: dict[int, list[int]] = {}

        for src_rank, dst_rank in zip(src_ranks, dst_ranks):
            ft = FlowTask(
                task_id=task_id,
                job_id=job_id,
                type=TaskType.FLOW,
                src=src_rank,
                dst=dst_rank,
                size_bytes=per_rank_bytes,
                comm_type=CommType.PP_SEND,
                chunk_id=0,
                num_chunks=1,
                phase=Phase.PREFILL,
                layer_id=0,
                deps=list(prev_exits.get(src_rank, [])),
            )
            tasks.append(ft)
            exits.setdefault(dst_rank, []).append(task_id)
            task_id += 1

        return tasks, exits, task_id

    # ── KV cache reuse (Stage 1) ────────────────────────────────────────────────

    def _expand_kv_reuse(
        self,
        storage_node_ids: list[int],
        dest_ranks: list[int],
        kv_cache_bytes: dict,
        stage_layer_ids: list[int],
        request_reuse_info: dict,
        job_id: int,
        task_id_start: int,
        prev_exits: dict[int, list[int]] | None = None,
    ) -> tuple[list[FlowTask], dict[int, dict[int, list[int]]], dict[str, list[int]], int]:
        """
        Expand per-layer KV reuse flows from storage nodes to prefill ranks.

        Each request's reuse flows all come from the same storage node.
        Flows are per-layer so RLI can prioritize them correctly.
        Flows depend on prev_exits for their destination rank.

        Args:
            storage_node_ids: Actual topology node IDs for storage nodes.
            dest_ranks: Prefill stage ranks (tp * ep GPUs).
            kv_cache_bytes: {req_id_str: total kv bytes for this stage}.
            stage_layer_ids: Layer IDs in this PP stage.
            request_reuse_info: {req_id: {hit_ratio, storage_node_idx}}.
            job_id: Job ID for tasks.
            task_id_start: Starting task ID.
            prev_exits: {rank: [task_ids]} — reuse flows wait for these.

        Returns:
            (all_flows, reuse_layer_deps, request_task_map, next_task_id)
        """
        all_flows: list[FlowTask] = []
        task_id = task_id_start
        reuse_layer_deps: dict[int, dict[int, list[int]]] = {}
        request_task_map: dict[str, list[int]] = {}

        stage_size = len(dest_ranks)
        num_layers = len(stage_layer_ids)

        for req_id, reuse_cfg in request_reuse_info.items():
            req_key = str(req_id)
            hit_ratio = reuse_cfg["hit_ratio"]
            storage_node = storage_node_ids[reuse_cfg["storage_node_idx"]]
            total_kv = kv_cache_bytes.get(req_key, 0)
            if total_kv == 0 or hit_ratio <= 0:
                continue

            per_rank_per_layer = max(total_kv // num_layers // stage_size, 1)
            reuse_bytes = int(per_rank_per_layer * hit_ratio)
            req_tids: list[int] = []

            for layer_id in stage_layer_ids:
                for rank in dest_ranks:
                    ft = FlowTask(
                        task_id=task_id,
                        job_id=job_id,
                        type=TaskType.FLOW,
                        src=storage_node,
                        dst=rank,
                        size_bytes=reuse_bytes,
                        comm_type=CommType.KV_CACHE_REUSE,
                        chunk_id=0,
                        num_chunks=1,
                        phase=Phase.PREFILL,
                        layer_id=layer_id,
                        deps=list(prev_exits.get(rank, [])) if prev_exits else [],
                    )
                    all_flows.append(ft)
                    req_tids.append(task_id)
                    reuse_layer_deps \
                        .setdefault(layer_id, {}) \
                        .setdefault(rank, []) \
                        .append(task_id)
                    task_id += 1

            request_task_map[req_key] = req_tids

        return all_flows, reuse_layer_deps, request_task_map, task_id

    # ── KV cache transfer ─────────────────────────────────────────────────────

    def _expand_kv_transfer(
        self,
        request_ids: list,
        kv_cache_bytes: dict,
        p_replica_id: int,
        d_replica_id: int,
        src_stage_id: int,
        dst_stage_id: int,
        job_id: int,
        task_id_start: int,
        prev_exits: dict[int, list[int]],
    ) -> tuple[list[FlowTask], dict[int, list[int]], int]:
        """
        Expand KV cache transfer flows from P-node stage ranks to D-node stage ranks.

        When pp>1, uses per-stage ranks and kv_cache_bytes already contains
        the per-stage share (proportional to actual layers in that stage).
        One flow per (request, rank pair).
        Each flow carries kv_cache_bytes[req_id] / stage_size. All flows depend
        on the prefill batch's exit tasks.

        Returns (tasks, exits, request_task_map, next_task_id) where:
        - exits maps d_rank → [task_ids] for the KV transfer flows received at D-node.
        - request_task_map maps req_id → [task_ids] for each request's flows.
        """
        tasks: list[FlowTask] = []
        task_id = task_id_start
        exits: dict[int, list[int]] = {}  # d_rank → [task_ids]
        request_task_map: dict[int, list[int]] = {}

        p_ranks = self._stage_ranks(p_replica_id, src_stage_id)
        d_ranks = self._stage_ranks(d_replica_id, dst_stage_id)
        stage_size = len(p_ranks)  # tp * ep

        for req_id in request_ids:
            req_key = str(req_id)
            total_kv = kv_cache_bytes.get(req_key, 0)
            if total_kv == 0:
                continue

            per_rank_bytes = max(total_kv // stage_size, 1)
            req_tids: list[int] = []

            for p_rank, d_rank in zip(p_ranks, d_ranks):
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
                req_tids.append(task_id)
                exits.setdefault(d_rank, []).append(task_id)
                task_id += 1

            request_task_map[req_id] = req_tids

        return tasks, exits, request_task_map, task_id

    # ── Total layers helper ─────────────────────────────────────────────────

    @staticmethod
    def _get_total_layers(trace: dict) -> int:
        """Extract total number of layers from the trace or profile store."""
        if "total_layers" in trace:
            return trace["total_layers"]
        model_config = trace.get("model_config", {})
        if isinstance(model_config, dict) and "num_layers" in model_config:
            return model_config["num_layers"]
        return 0

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

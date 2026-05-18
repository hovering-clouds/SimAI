"""
Trace Recorder for Vidur inference simulation.

Captures per-batch-stage scheduling decisions during a Vidur simulation run
and outputs a trace JSON file suitable for replay in simai-flow-scheduler.

Supports both pp=1 (one entry per batch) and pp>1 (one entry per batch-stage).
Each trace entry records stage_id, per-stage KV cache bytes, and three-layer
dependency chain: same-stage serial, cross-stage PP, and KV transfer.

Usage:
    # In simulator.py:
    from vidur.trace_recorder import TraceRecorder

    self._trace_recorder = TraceRecorder(replica_config, output_dir)
    self._metric_store.trace_recorder = self._trace_recorder

    # After simulation:
    self._trace_recorder.save()
"""

import json
import os
from math import ceil
from typing import Optional


_DTYPE_BYTES = {
    'float16': 2, 'bfloat16': 2, 'float32': 4,
    'int8': 1, 'int16': 2, 'int32': 4, 'int64': 8,
    'float64': 8,
}


class TraceRecorder:
    """Records batch-stage events from Vidur simulation into a trace JSON."""

    def __init__(
        self,
        replica_config,
        output_dir: str = ".",
        enabled: bool = True,
    ):
        """
        Args:
            replica_config: Vidur ReplicaConfig with model, parallelism, and PD settings.
            output_dir: Directory to write trace JSON.
            enabled: Set False to disable recording.
        """
        self._config = replica_config
        self._output_dir = output_dir
        self._enabled = enabled

        # Accumulated data
        self._requests: dict[str, dict] = {}
        self._batches: list[dict] = []

        # --- Per-stage tracking state ---
        # Vidur batch.id → sequential counter for human-readable batch_id
        self._parent_batch_counter: dict[int, int] = {}
        self._next_counter = 0

        # (replica_id, stage_id) → last entry batch_id (same-stage serial dep)
        self._last_entry_per_stage: dict[tuple[int, int], str] = {}

        # (vidur_batch.id, stage_id) → entry batch_id (cross-stage PP dep)
        self._batch_stage_entries: dict[tuple[int, int], str] = {}

        # (request_id, stage_id) → prefill entry batch_id (KV transfer dep)
        self._request_prefill_stage_map: dict[tuple[int, int], str] = {}

    def record_request(self, request) -> None:
        """Record a request when it first arrives (call from global scheduler)."""
        if not self._enabled:
            return
        self._requests[str(request.id)] = {
            "num_prefill_tokens": request.num_prefill_tokens,
            "num_decode_tokens": request.num_decode_tokens,
        }

    def record_batch_stage(
        self,
        batch,
        batch_stage,
        replica,
        replica_id: int,
        stage_id: int,
    ) -> None:
        """
        Record a completed batch-stage event.

        Called from BatchStageEndEvent.handle_event() after existing logic.

        Args:
            batch: The parent Batch object.
            batch_stage: The BatchStage object that just completed on this stage.
            replica: The Replica object that executed this batch-stage.
            replica_id: ID of the replica.
            stage_id: Pipeline stage index (0-indexed).
        """
        if not self._enabled:
            return

        from vidur.entities.replica import ReplicaType

        # Determine batch type from replica type
        if replica.replica_type == ReplicaType.PREFILL:
            batch_type = "prefill"
        elif replica.replica_type == ReplicaType.DECODE:
            batch_type = "decode"
        else:
            return

        # --- Generate batch_id ---
        if batch.id not in self._parent_batch_counter:
            self._parent_batch_counter[batch.id] = self._next_counter
            self._next_counter += 1
        counter = self._parent_batch_counter[batch.id]
        prefix = "p" if batch_type == "prefill" else "d"
        batch_id = f"{prefix}{counter}_s{stage_id}"

        # --- Compute depends_on ---
        depends_on = []

        # 1. Same-stage serial dependency
        last_in_stage = self._last_entry_per_stage.get((replica_id, stage_id))
        if last_in_stage is not None:
            depends_on.append(last_in_stage)

        # 2. Cross-stage PP dependency (same parent batch, previous stage)
        if stage_id > 0:
            prev_stage_entry = self._batch_stage_entries.get(
                (batch.id, stage_id - 1))
            if prev_stage_entry is not None and prev_stage_entry not in depends_on:
                depends_on.append(prev_stage_entry)

        # 3. KV transfer dependency (decode → prefill, same stage across replicas).
        #    Using pop() so each (request, stage) pair only produces the KV dep once.
        #    Subsequent decode batches reach KV transitively via the same-stage serial chain.
        if batch_type == "decode":
            for req in batch_stage.requests:
                prefill_bid = self._request_prefill_stage_map.pop(
                    (req.id, stage_id), None)
                if prefill_bid is not None and prefill_bid not in depends_on:
                    depends_on.append(prefill_bid)

        # --- Per-stage KV cache bytes (prefill only) ---
        kv_cache_bytes = None
        if batch_type == "prefill":
            kv_cache_bytes = self._compute_stage_kv_bytes(batch_stage, replica, stage_id)

        # --- Build entry ---
        batch_entry = {
            "batch_id": batch_id,
            "type": batch_type,
            "replica_id": replica_id,
            "stage_id": stage_id,
            "request_ids": [req.id for req in batch_stage.requests],
            "num_tokens": list(batch_stage.num_tokens),
            "kv_cache_bytes": kv_cache_bytes,
            "depends_on": depends_on,
        }

        if batch_type == "decode":
            batch_entry["kv_cache_seq_lens"] = [
                req.num_processed_prefill_tokens + req.num_processed_decode_tokens
                for req in batch_stage.requests
            ]

        self._batches.append(batch_entry)

        # --- Update tracking state ---
        self._last_entry_per_stage[(replica_id, stage_id)] = batch_id
        self._batch_stage_entries[(batch.id, stage_id)] = batch_id

        if batch_type == "prefill":
            for req in batch_stage.requests:
                # Record KV dependency immediately when this stage finishes prefill.
                # In PP mode, each stage independently produces its own KV cache
                # for its layers. Stage 0's KV cache is available as soon as stage 0
                # completes, regardless of whether the full prefill (all stages) is done.
                self._request_prefill_stage_map[
                    (req.id, stage_id)] = batch_id

    # ── KV cache computation ───────────────────────────────────────────────

    @staticmethod
    def _compute_stage_kv_bytes(batch_stage, replica, stage_id: int = 0) -> dict[str, int]:
        """
        Compute per-stage KV cache bytes using correct attention dimensions.

        Formula: 2 * head_dim * kv_heads_per_tp * (layers_in_this_stage) * tokens * dtype_bytes
        """
        dtype_str = getattr(replica, 'pd_p2p_comm_dtype', 'float16')
        dtype_bytes = _DTYPE_BYTES.get(dtype_str, 2)

        head_dim = replica.embedding_dim // replica.num_q_heads
        kv_heads = ceil(replica.num_kv_heads / replica.num_tensor_parallel_workers)

        # Compute actual layers assigned to this stage (last stage takes remainder)
        total_layers = replica.num_layers
        pp = replica.num_pipeline_stages
        layers_per_stage = total_layers // pp
        if stage_id == pp - 1:
            num_stage_layers = total_layers - stage_id * layers_per_stage
        else:
            num_stage_layers = layers_per_stage

        kv_per_token = 2 * head_dim * kv_heads * num_stage_layers * dtype_bytes

        kv_bytes = {}
        for req, tokens in zip(batch_stage.requests, batch_stage.num_tokens):
            kv_bytes[str(req.id)] = kv_per_token * tokens
        return kv_bytes

    # ── Legacy method (kept for backward compatibility) ────────────────────

    def record_batch(self, batch, replica) -> None:
        """
        Record a completed batch event (legacy, pp=1 only).

        Prefer record_batch_stage() for pp>1 support.
        """
        if not self._enabled:
            return
        # Delegate to record_batch_stage with stage_id=0
        # Build a minimal batch_stage-like object for compatibility
        self.record_batch_stage(
            batch=batch,
            batch_stage=batch,  # Batch has same interface as BatchStage for our needs
            replica=replica,
            replica_id=replica.id,
            stage_id=0,
        )

    # ── Save ───────────────────────────────────────────────────────────────

    def save(self, filename: str = "inference_trace.json") -> str:
        """
        Write the trace to a JSON file.

        Returns:
            Path to the written file.
        """
        os.makedirs(self._output_dir, exist_ok=True)
        path = os.path.join(self._output_dir, filename)

        trace = self._build_trace()
        with open(path, "w") as f:
            json.dump(trace, f, indent=2)

        print(f"[TraceRecorder] Trace saved to {path} "
              f"({len(self._batches)} batches, {len(self._requests)} requests)")
        return path

    def _build_trace(self) -> dict:
        """Build the full trace dict."""
        cfg = self._config
        model_cfg = getattr(cfg, 'model_config', None)

        model_config = {}
        if model_cfg:
            model_config = {
                "hidden_size": getattr(model_cfg, 'embedding_dim', 0),
                "embedding_dim": getattr(model_cfg, 'embedding_dim', 0),
                "num_layers": getattr(model_cfg, 'num_layers', 0),
                "mlp_hidden_dim": getattr(model_cfg, 'mlp_hidden_dim', 0),
                "num_q_heads": getattr(model_cfg, 'num_q_heads', 0),
                "num_kv_heads": getattr(model_cfg, 'num_kv_heads', 0),
            }
            if hasattr(model_cfg, 'num_experts'):
                model_config["num_experts"] = model_cfg.num_experts
            if hasattr(model_cfg, 'num_experts_per_tok'):
                model_config["moe_topk"] = model_cfg.num_experts_per_tok

        dtype_str = getattr(cfg, 'pd_p2p_comm_dtype', 'float16')

        trace = {
            "version": "1.0",
            "model": cfg.model_name,
            "model_config": model_config,
            "hidden_size": model_config.get("embedding_dim", 0),
            "dtype_bytes": _DTYPE_BYTES.get(dtype_str, 2),
            "total_layers": model_config.get("num_layers", 0),
            "parallelism": {
                "tp": cfg.tensor_parallel_size,
                "pp": cfg.num_pipeline_stages,
                "ep": cfg.expert_model_parallel_size,
            },
            "pd_config": {
                "pd_node_ratio": cfg.pd_node_ratio,
                "pd_p2p_comm_bandwidth_gbps": cfg.pd_p2p_comm_bandwidth,
            },
            "requests": self._requests,
            "batches": self._batches,
        }
        return trace

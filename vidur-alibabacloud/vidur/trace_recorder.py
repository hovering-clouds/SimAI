"""
Trace Recorder for Vidur inference simulation.

Captures per-batch scheduling decisions during a Vidur simulation run
and outputs a trace JSON file suitable for replay in simai-flow-scheduler.

Trace format follows simai-flow-scheduler/docs/phase5-dev/phase5-taskA-research.md section 9.4.

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
from typing import Optional


class TraceRecorder:
    """Records batch events from Vidur simulation into a trace JSON."""

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

        # State for depends_on tracking
        self._last_batch_per_replica: dict[int, str] = {}  # replica_id → batch_id
        self._request_prefill_batch: dict[int, str] = {}    # request_id → prefill batch_id

        # Counter for batch_id generation
        self._prefill_counter = 0
        self._decode_counter = 0

    def record_request(self, request) -> None:
        """Record a request when it first arrives (call from global scheduler)."""
        if not self._enabled:
            return
        self._requests[str(request.id)] = {
            "num_prefill_tokens": request.num_prefill_tokens,
            "num_decode_tokens": request.num_decode_tokens,
        }

    def record_batch(self, batch, replica) -> None:
        """
        Record a completed batch event.

        Call from BatchEndEvent.handle_event() after existing logic.

        Args:
            batch: The Batch object that just completed.
            replica: The Replica object that executed this batch.
        """
        if not self._enabled:
            return

        # Determine batch type from replica type
        from vidur.entities.replica import ReplicaType

        if replica.replica_type == ReplicaType.PREFILL:
            batch_type = "prefill"
            batch_id = f"p{self._prefill_counter}"
            self._prefill_counter += 1
        elif replica.replica_type == ReplicaType.DECODE:
            batch_type = "decode"
            batch_id = f"d{self._decode_counter}"
            self._decode_counter += 1
        else:
            # MIXED or unknown — skip
            return

        # Compute depends_on
        depends_on = []

        # 1. Same-replica serial dependency
        last_on_replica = self._last_batch_per_replica.get(replica.id)
        if last_on_replica is not None:
            depends_on.append(last_on_replica)

        # 2. For decode batches: depend on the prefill batch that contains each request
        if batch_type == "decode":
            for req in batch.requests:
                prefill_bid = self._request_prefill_batch.get(req.id)
                if prefill_bid is not None and prefill_bid not in depends_on:
                    depends_on.append(prefill_bid)

        # Collect per-request KV cache bytes (only for prefill batches)
        kv_cache_bytes = None
        if batch_type == "prefill":
            kv_cache_bytes = {}
            for req, n_tokens in zip(batch.requests, batch.num_tokens):
                # Only record KV for tokens that belong to this request's prefill
                if not req.is_prefill_complete:
                    # Prefill not yet complete — this is a partial prefill
                    pass
                kv_cache_bytes[str(req.id)] = req.pd_p2p_comm_size \
                    if req.pd_p2p_comm_size != float('inf') else 0

        # Build batch entry
        batch_entry = {
            "batch_id": batch_id,
            "type": batch_type,
            "replica_id": replica.id,
            "request_ids": [req.id for req in batch.requests],
            "num_tokens": list(batch.num_tokens),
            "kv_cache_bytes": kv_cache_bytes,
            "depends_on": depends_on,
        }

        self._batches.append(batch_entry)

        # Update tracking state
        self._last_batch_per_replica[replica.id] = batch_id

        if batch_type == "prefill":
            for req in batch.requests:
                if req.is_prefill_complete or req.completed:
                    self._request_prefill_batch[req.id] = batch_id

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
                "hidden_size": getattr(model_cfg, 'hidden_size', 0),
                "num_layers": getattr(model_cfg, 'num_layers', 0),
                "mlp_hidden_dim": getattr(model_cfg, 'mlp_hidden_dim', 0),
            }
            # MoE-specific fields
            if hasattr(model_cfg, 'num_experts'):
                model_config["num_experts"] = model_cfg.num_experts
            if hasattr(model_cfg, 'num_experts_per_tok'):
                model_config["moe_topk"] = model_cfg.num_experts_per_tok

        trace = {
            "version": "1.0",
            "model": cfg.model_name,
            "model_config": model_config,
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

# Phase6 Task1: Vidur TraceRecorder PP 支持

## Context

simai-flow-scheduler 的 `InferenceTraceExpander` 已实现 pp>1 支持（per-stage 展开和依赖链），但 Vidur 的 `TraceRecorder` 当前只在 `BatchEndEvent`（整个 micro-batch 跑完所有 PP stage 后）记录，没有 stage 信息。需要修改 Vidur 使其输出 per-stage trace，供 flow-scheduler 消费。

**核心变更**：将 trace recording hook 从 `BatchEndEvent` 移到 `BatchStageEndEvent`，每个 stage 完成时立即记录一条 trace entry。

## 背景：Vidur PP 事件流

```
ReplicaScheduleEvent
  → BatchStageArrivalEvent (stage 0)
    → ReplicaStageScheduleEvent (stage 0)
      → BatchStageEndEvent (stage 0, is_last_stage=False)
        → BatchStageArrivalEvent (stage 1)       ← PP 通信：stage 0 → stage 1
          → ReplicaStageScheduleEvent (stage 1)
            → BatchStageEndEvent (stage 1, is_last_stage=True)
              → BatchEndEvent                     ← 当前 trace hook 在这里
```

当前 `TraceRecorder` 在 `BatchEndEvent` 记录——此时整个 micro-batch 已完成所有 stage，丢失了 per-stage 信息。

## 文件清单

| 文件 | 改动 |
|------|------|
| `vidur-alibabacloud/vidur/trace_recorder.py` | 新增 `record_batch_stage()`，重写依赖追踪和 KV 计算 |
| `vidur-alibabacloud/vidur/events/batch_stage_end_event.py` | 添加 trace recording hook |
| `vidur-alibabacloud/vidur/events/batch_end_event.py` | 移除旧 trace recording hook |

## 实现步骤

### 步骤 1：重写 TraceRecorder

**文件**：`vidur-alibabacloud/vidur/trace_recorder.py`

#### 1a. 新增追踪状态

替换现有 `_last_batch_per_replica` / `_request_prefill_batch`：

```python
# 父 batch.id → 递增计数器（用于生成人类可读的 batch_id）
self._parent_batch_counter: dict[int, int] = {}
self._next_counter = 0

# (replica_id, stage_id) → 最近一条 entry 的 batch_id（同 stage 串行依赖）
self._last_entry_per_stage: dict[tuple[int, int], str] = {}

# (vidur_batch.id, stage_id) → entry batch_id（跨 stage PP 依赖）
self._batch_stage_entries: dict[tuple[int, int], str] = {}

# (request_id, stage_id) → prefill entry batch_id（KV transfer 依赖）
self._request_prefill_stage_map: dict[tuple[int, int], str] = {}
```

#### 1b. 新方法 `record_batch_stage(batch, batch_stage, replica, replica_id, stage_id)`

替代原 `record_batch()`。每个 stage 完成时由 `BatchStageEndEvent` 调用。

**batch_id 生成**：
```python
# 首次见到该 vidur batch.id → 分配新计数器
if batch.id not in self._parent_batch_counter:
    self._parent_batch_counter[batch.id] = self._next_counter
    self._next_counter += 1
counter = self._parent_batch_counter[batch.id]

prefix = "p" if batch_type == "prefill" else "d"
batch_id = f"{prefix}{counter}_s{stage_id}"
```

**依赖追踪**（三层依赖模型）：
```python
depends_on = []

# 1. 同 stage 串行依赖：该 (replica_id, stage_id) 上的上一条 entry
#    例：p1_s0 depends on p0_s0（stage 0 处理完 MB0 后才处理 MB1）
last_in_stage = self._last_entry_per_stage.get((replica_id, stage_id))
if last_in_stage:
    depends_on.append(last_in_stage)

# 2. 跨 stage PP 依赖：同一父 batch 在 stage_id-1 的 entry
#    例：p0_s1 depends on p0_s0（stage 1 的输入来自 stage 0 的输出）
if stage_id > 0:
    prev_stage_entry = self._batch_stage_entries.get((batch.id, stage_id - 1))
    if prev_stage_entry and prev_stage_entry not in depends_on:
        depends_on.append(prev_stage_entry)

# 3. KV transfer 依赖（decode → prefill，同 stage 跨 replica）
#    例：d0_s0 depends on p0_s0（D-stage-0 需要 P-stage-0 的 KV cache）
if batch_type == "decode":
    for req in batch_stage.requests:
        prefill_bid = self._request_prefill_stage_map.get((req.id, stage_id))
        if prefill_bid and prefill_bid not in depends_on:
            depends_on.append(prefill_bid)
```

**Per-stage KV cache 字节计算**（仅 prefill entries）：

当前 `request.estimate_kv_cache_size()` 使用 `mlp_hidden_dim`（不正确），且只在 `BatchEndEvent` 时调用。我们直接用 attention 维度计算 per-stage KV：

```python
def _compute_stage_kv_bytes(self, batch_stage, replica):
    dtype_map = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'int8': 1}
    dtype_bytes = dtype_map.get(getattr(replica, 'pd_p2p_comm_dtype', 'float16'), 2)

    head_dim = replica.embedding_dim // replica.num_q_heads
    kv_heads = ceil(replica.num_kv_heads / replica.num_tensor_parallel_workers)
    layers_per_stage = replica.num_layers // replica.num_pipeline_stages
    kv_per_token_per_layer = 2 * head_dim * kv_heads * dtype_bytes

    kv_bytes = {}
    for req, tokens in zip(batch_stage.requests, batch_stage.num_tokens):
        kv_bytes[str(req.id)] = kv_per_token_per_layer * tokens * layers_per_stage
    return kv_bytes
```

> **注意**：这比 `estimate_kv_cache_size()` 更准确——使用 `head_dim * kv_heads` 而非 `mlp_hidden_dim`。

**更新追踪状态**：
```python
# 记录该 entry
self._last_entry_per_stage[(replica_id, stage_id)] = batch_id
self._batch_stage_entries[(batch.id, stage_id)] = batch_id

# prefill: 记录 request → prefill stage entry 映射
if batch_type == "prefill":
    for req in batch_stage.requests:
        if req.is_prefill_complete or req.completed:
            self._request_prefill_stage_map[(req.id, stage_id)] = batch_id
```

#### 1c. 更新 `_build_trace()`

添加 simai-flow-scheduler expander `_expand_pp_communication()` 需要的字段：

```python
trace = {
    "version": "1.0",
    "model": cfg.model_name,
    "model_config": model_config,
    "hidden_size": model_config.get("embedding_dim", 0),      # 新增
    "dtype_bytes": dtype_map.get(cfg.pd_p2p_comm_dtype, 2),   # 新增
    "total_layers": model_config.get("num_layers", 0),         # 新增
    "parallelism": { ... },
    "pd_config": { ... },
    "requests": self._requests,
    "batches": self._batches,
}
```

#### 1d. 保留旧 `record_batch()` 方法

保留方法体但从 `BatchEndEvent` 不再调用。向后兼容，防止其他地方直接调用。

### 步骤 2：修改 BatchStageEndEvent

**文件**：`vidur-alibabacloud/vidur/events/batch_stage_end_event.py`

在 `handle_event()` 中 `metrics_store.on_batch_stage_end()` 之后添加：

```python
# --- Per-stage trace recording for simai-flow-scheduler ---
if hasattr(metrics_store, 'trace_recorder'):
    replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
    metrics_store.trace_recorder.record_batch_stage(
        self._batch,
        self._batch_stage,
        replica_scheduler.replica,
        self._replica_id,
        self._stage_id,
    )
```

### 步骤 3：移除 BatchEndEvent 中的旧 hook

**文件**：`vidur-alibabacloud/vidur/events/batch_end_event.py`

删除 lines 141-144：
```python
# 删除这段：
# --- Trace recording for simai-flow-scheduler ---
if hasattr(metrics_store, 'trace_recorder'):
    metrics_store.trace_recorder.record_batch(
        self._batch, replica_scheduler.replica)
```

## pp=1 兼容性

- pp=1 时，`BatchStageEndEvent` 对每个 micro-batch 只触发一次（stage 0, is_last_stage=True）
- trace entry 会包含 `stage_id: 0`
- expander 用 `batch.get("stage_id", 0)` 默认值处理不含 stage_id 的旧 trace
- 行为与之前等价（每个 micro-batch 一条 entry）

## 输出 trace 示例

```json
{
  "version": "1.0",
  "model": "deepseek-671B",
  "hidden_size": 7168,
  "dtype_bytes": 2,
  "total_layers": 61,
  "parallelism": {"tp": 2, "pp": 2, "ep": 4},
  "batches": [
    {"batch_id": "p0_s0", "type": "prefill", "replica_id": 0, "stage_id": 0,
     "request_ids": [0], "num_tokens": [2048],
     "kv_cache_bytes": {"0": 12345678}, "depends_on": []},
    {"batch_id": "p0_s1", "type": "prefill", "replica_id": 0, "stage_id": 1,
     "request_ids": [0], "num_tokens": [2048],
     "kv_cache_bytes": {"0": 12345678}, "depends_on": ["p0_s0"]},
    {"batch_id": "p1_s0", "type": "prefill", "replica_id": 0, "stage_id": 0,
     "request_ids": [1], "num_tokens": [1024],
     "kv_cache_bytes": {"1": 6172839}, "depends_on": ["p0_s0"]},
    {"batch_id": "p1_s1", "type": "prefill", "replica_id": 0, "stage_id": 1,
     "request_ids": [1], "num_tokens": [1024],
     "kv_cache_bytes": {"1": 6172839}, "depends_on": ["p0_s1", "p1_s0"]},
    {"batch_id": "d0_s0", "type": "decode", "replica_id": 1, "stage_id": 0,
     "request_ids": [0], "num_tokens": [1],
     "kv_cache_seq_lens": [2049], "depends_on": ["p0_s0"]},
    {"batch_id": "d0_s1", "type": "decode", "replica_id": 1, "stage_id": 1,
     "request_ids": [0], "num_tokens": [1],
     "kv_cache_seq_lens": [2049], "depends_on": ["p0_s1", "d0_s0"]}
  ]
}
```

## 验证

1. **pp=1 回归**：运行 Vidur pp=1 仿真 → trace 格式应包含 stage_id=0，expander 正常消费
2. **pp=2 验证**：运行 Vidur pp=2 仿真 → trace 应包含 per-stage entries，用 simai-flow-scheduler 的 29 个 PP 测试验证
3. **`uv run pytest tests/ -v`**：确保所有测试通过

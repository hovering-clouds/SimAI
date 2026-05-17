# Phase 6: InferenceTraceExpander PP 流水线并行支持

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 InferenceTraceExpander 支持流水线并行 (pp > 1)，能从 Vidur 生成的 per-stage trace 中重建完整的 pipeline 执行 DAG。

**Architecture:** Vidur 端通过修改 TraceRecorder 在 `BatchStageEndEvent` 处记录 per-stage batch 信息（包含 `stage_id`、依赖链）。simai-flow-scheduler 端的 InferenceTraceExpander 接收新格式 trace，按 stage 分配 GPU 节点、过滤 layer、插入 PP inter-stage 通信流、重建 pipeline overlap 依赖。

**Tech Stack:** Python 3.10+, pytest, dataclass

---

## 目录

1. [背景与调研结论](#1-背景与调研结论)
2. [Trace 格式规范](#2-trace-格式规范)
3. [实现任务](#3-实现任务)
   - [Task 1: Vidur TraceRecorder per-stage 记录](#task-1-vidur-tracerecorder-per-stage-记录)
   - [Task 2: InferenceTraceExpander PP 基础设施](#task-2-inferencetraceexpander-pp-基础设施)
   - [Task 3: Per-stage batch expansion](#task-3-per-stage-batch-expansion)
   - [Task 4: PP inter-stage 通信流](#task-4-pp-inter-stage-通信流)
   - [Task 5: PP-aware KV transfer](#task-5-pp-aware-kv-transfer)
   - [Task 6: Pipeline dependency 链](#task-6-pipeline-dependency-链)
   - [Task 7: 端到端测试](#task-7-端到端测试)

---

## 1. 背景与调研结论

### 1.1 Vidur PP 事件流

Vidur 使用事件驱动模拟器。Pipeline parallelism 的关键组件：

- **BaseReplicaScheduler** (`base_replica_scheduler.py:59-67`)：为每个 PP stage 创建独立的 `ReplicaStageScheduler`
- **on_schedule()** (`base_replica_scheduler.py:139-151`)：一次性发放最多 `num_stages` 个 micro-batch 填满流水线
- **ReplicaStageScheduler** (`replica_stage_schduler.py`)：每个 stage 独立队列，`is_busy` 标志保证同一 stage 串行执行
- **world_size = pp * tp**（`config.py:501`）

事件链：

```
ReplicaScheduleEvent → 发 num_stages 个 MB 到 Stage 0
  → BatchStageArrivalEvent(S0, MBn) → 入队 Stage 0
    → ReplicaStageScheduleEvent(S0) → 执行 MB，is_busy=True
      → BatchStageEndEvent(S0, MBn) @ t + exec
        → on_stage_end() 释放 Stage 0
        → ReplicaStageScheduleEvent(S0)   ← Stage 0 调度下一个 MB
        → BatchStageArrivalEvent(S1, MBn) ← MB 进入 Stage 1（如果不是 last stage）
        或 BatchEndEvent                  ← MB 走完流水线（last stage）
```

**关键特性**：Stage 0 处理 MB1 的同时 Stage 1 可以处理 MB0，实现 overlap：

```
时间线:  t0        t0+E      t0+2E       t0+2E+E'
Stage 0: [MB0-----][MB1-----]
Stage 1:           [MB0-----][MB1-------]
                         ↑ overlap
```

### 1.2 当前 TraceRecorder 的局限

- 只在 `BatchEndEvent`（last stage 完成时）记录，无 stage 信息
- MIXED replica 模式直接跳过（`trace_recorder.py:89-91`）
- 无法区分同一 micro-batch 在不同 stage 上的执行

### 1.3 依赖链建模

用两层依赖重建 pipeline overlap：
- **同 stage 依赖**：MB n-S0 → MB n-1-S0（同一组 GPU 串行）
- **跨 stage 数据依赖**：MB n-S1 → MB n-S0（stage 间前向传播）

AnalyticalExecutor 在依赖满足后会自动并行调度不同 stage 的任务，产生正确的 overlap。

### 1.4 Per-stage KV transfer 设计决策

**真实场景**：每个 PP stage 的 attention 层都会产生自己那部分 layers 的 KV cache。PD disaggregation 下，KV cache 的传输应该是 per-stage 的：

```
P-stage-0 → D-stage-0: 传输 stage 0 的 KV cache (layers 0..L/pp-1)
P-stage-1 → D-stage-1: 传输 stage 1 的 KV cache (layers L/pp..2L/pp-1)
```

D-stage-s **不需要等待其他 P-stages 完成**就能开始 decode——它只需要自己那个 stage 的 KV cache。这允许 KV transfer 与其他 stages 的 prefill 计算并行：

```
P-stage-0 完成 → KV transfer S0→D → D-stage-0 开始 decode
P-stage-1 完成 → KV transfer S1→D → D-stage-1 开始 decode
                      ↑ 两步可并行
```

**对比 Vidur 模型**：Vidur 把 KV transfer 建模为所有 PP stages 完成后的一次性传输（在 `BatchEndEvent` 中处理）。simai-flow-scheduler 采用更接近真实场景的 per-stage 模型，这能更好地研究 KV transfer 流量与训练流量的网络冲撞。

**Trace 格式**：每条 prefill stage entry 都记录自己的 `kv_cache_bytes`（= total_kv / pp）。TraceRecorder 从 model config 计算每 stage 的 KV cache 大小。

---

## 2. Trace 格式规范

### 2.1 PP-aware trace 格式

每条 batch 记录代表一个 micro-batch 在单个 PP stage 上的执行。

```json
{
  "version": "1.0",
  "model": "deepseek-671B",
  "model_config": { "hidden_size": 7168, "num_layers": 61, "dense_layers": 3, "moe_topk": 8 },
  "parallelism": { "tp": 2, "pp": 2, "ep": 4 },
  "pd_config": { "pd_node_ratio": 0.5, "pd_p2p_comm_bandwidth_gbps": 200000000000 },
  "requests": {
    "0": { "num_prefill_tokens": 1024, "num_decode_tokens": 64 },
    "1": { "num_prefill_tokens": 512, "num_decode_tokens": 32 }
  },
  "batches": [
    {
      "batch_id": "p0_s0",
      "type": "prefill",
      "replica_id": 0,
      "stage_id": 0,
      "request_ids": [0, 1],
      "num_tokens": [1024, 512],
      "kv_cache_bytes": { "0": 1152460800, "1": 576230400 },
      "depends_on": []
    },
    {
      "batch_id": "p0_s1",
      "type": "prefill",
      "replica_id": 0,
      "stage_id": 1,
      "request_ids": [0, 1],
      "num_tokens": [1024, 512],
      "kv_cache_bytes": { "0": 1152460800, "1": 576230400 },
      "depends_on": ["p0_s0"]
    },
    {
      "batch_id": "d0_s0",
      "type": "decode",
      "replica_id": 1,
      "stage_id": 0,
      "request_ids": [0, 1],
      "num_tokens": [1, 1],
      "kv_cache_bytes": null,
      "depends_on": ["p0_s0"]
    },
    {
      "batch_id": "d0_s1",
      "type": "decode",
      "replica_id": 1,
      "stage_id": 1,
      "request_ids": [0, 1],
      "num_tokens": [1, 1],
      "kv_cache_bytes": null,
      "depends_on": ["d0_s0", "p0_s1"]
    }
  ]
}
```

### 2.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `batch_id` | string | 唯一标识，建议格式 `{type}{n}_s{stage}` |
| `stage_id` | int | **新增**。PP stage 编号（0-based）。pp=1 时默认为 0 |
| `kv_cache_bytes` | dict/null | **每条 prefill stage entry 记录该 stage 的 KV cache**（= total_kv / pp）。decode batch 为 null。Per-stage KV transfer：P-stage-s → D-stage-s，D-stage-s 拿到 KV 后即可独立开始 decode，无需等待其他 stages |
| `depends_on` | list[str] | 包含两类依赖：同 stage 前一个 MB + 跨 stage 同一个 MB 的前一个 stage |

### 2.3 向后兼容

- 当 `parallelism.pp == 1` 时，`stage_id` 字段可选，默认为 0
- 现有 pp=1 的 trace 无需任何修改即可继续使用
- Expander 通过 `trace["parallelism"]["pp"]` 判断是否启用 PP 逻辑

### 2.4 PP 通信量计算

Stage s → Stage s+1 之间传输 hidden state tensor。每个 TP rank 发送自己的分片：

```python
pp_comm_bytes_per_rank = hidden_size // tp * dtype_bytes * total_tokens_in_batch
# dtype_bytes = 2 (FP16/BF16)
# total_tokens_in_batch = sum(batch["num_tokens"])
```

---

## 3. 实现任务

### Task 1: Vidur TraceRecorder per-stage 记录

**Files:**
- Modify: `vidur-alibabacloud/vidur/trace_recorder.py`
- Modify: `vidur-alibabacloud/vidur/events/batch_stage_end_event.py`

> 注意：此任务在 Vidur 仓库中完成，作为 simai-flow-scheduler 的前置依赖。但 simai-flow-scheduler 的开发和测试可以使用手工编写的 synthetic trace。

- [ ] **Step 1: 修改 TraceRecorder 添加 per-stage 记录方法**

在 `trace_recorder.py` 的 `TraceRecorder` 类中添加以下方法：

```python
def record_batch_stage(self, batch, replica, stage_id: int, is_last_stage: bool) -> None:
    """
    Record a batch-stage completion event (per PP stage).

    Call from BatchStageEndEvent.handle_event().

    Args:
        batch: The Batch object (micro-batch).
        replica: The Replica object.
        stage_id: PP stage index (0-based).
        is_last_stage: Whether this is the last PP stage.
    """
    if not self._enabled:
        return

    from vidur.entities.replica import ReplicaType

    if replica.replica_type == ReplicaType.PREFILL:
        batch_type = "prefill"
    elif replica.replica_type == ReplicaType.DECODE:
        batch_type = "decode"
    else:
        return

    # Generate unique batch_id: {type}{counter}_s{stage_id}
    # Use a combined key to generate IDs
    mb_key = (batch.id, replica.id)
    if mb_key not in self._mb_id_map:
        if batch_type == "prefill":
            self._mb_id_map[mb_key] = f"p{self._prefill_counter}"
            self._prefill_counter += 1
        else:
            self._mb_id_map[mb_key] = f"d{self._decode_counter}"
            self._decode_counter += 1

    mb_prefix = self._mb_id_map[mb_key]
    batch_id = f"{mb_prefix}_s{stage_id}"

    # ── Compute depends_on ──────────────────────────────
    depends_on = []

    # 1. Same-stage dependency: previous MB on the same (replica, stage)
    last_on_stage = self._last_entry_per_stage.get((replica.id, stage_id))
    if last_on_stage is not None:
        depends_on.append(last_on_stage)

    # 2. Cross-stage dependency: same MB on previous stage
    prev_stage_entry = self._mb_prev_stage_entry.get((replica.id, batch.id))
    if prev_stage_entry is not None:
        depends_on.append(prev_stage_entry)

    # ── KV cache bytes (per-stage) ───────────────────────
    # Every prefill stage records its own per-stage KV cache (total_kv / pp)
    kv_cache_bytes = None
    if batch_type == "prefill":
        kv_cache_bytes = {}
        num_stages = self._config.num_pipeline_stages
        for req in batch.requests:
            # Compute total KV, then divide by pp for per-stage share
            total_kv = req.estimate_kv_cache_size()
            per_stage_kv = total_kv // num_stages if num_stages > 1 else total_kv
            kv_cache_bytes[str(req.id)] = per_stage_kv

    # ── Build entry ─────────────────────────────────────
    batch_entry = {
        "batch_id": batch_id,
        "type": batch_type,
        "replica_id": replica.id,
        "stage_id": stage_id,
        "request_ids": [req.id for req in batch.requests],
        "num_tokens": list(batch.num_tokens),
        "kv_cache_bytes": kv_cache_bytes,
        "depends_on": depends_on,
    }

    if batch_type == "decode":
        batch_entry["kv_cache_seq_lens"] = [
            req.num_processed_prefill_tokens + req.num_processed_decode_tokens
            for req in batch.requests
        ]

    self._batches.append(batch_entry)

    # ── Update tracking state ───────────────────────────
    self._last_entry_per_stage[(replica.id, stage_id)] = batch_id
    self._mb_prev_stage_entry[(replica.id, batch.id)] = batch_id

    if batch_type == "prefill":
        for req in batch.requests:
            if req.is_prefill_complete or req.completed:
                self._request_prefill_batch[req.id] = batch_id
```

- [ ] **Step 2: 在 TraceRecorder.__init__ 中初始化新的追踪状态**

在 `TraceRecorder.__init__` 中添加：

```python
# PP-aware tracking state
self._mb_id_map: dict[tuple[int, int], str] = {}           # (batch.id, replica.id) → prefix
self._last_entry_per_stage: dict[tuple[int, int], str] = {}  # (replica_id, stage_id) → batch_id
self._mb_prev_stage_entry: dict[tuple[int, int], str] = {}   # (replica_id, batch.id) → batch_id
```

- [ ] **Step 3: 在 BatchStageEndEvent 中调用 TraceRecorder**

在 `batch_stage_end_event.py` 的 `handle_event` 方法末尾，`return next_events + [...]` 之前添加：

```python
# Record per-stage trace
if hasattr(metrics_store, 'trace_recorder'):
    replica_scheduler = scheduler.get_replica_scheduler(self._replica_id)
    metrics_store.trace_recorder.record_batch_stage(
        self._batch,
        replica_scheduler.replica,
        self._stage_id,
        self._is_last_stage,
    )
```

- [ ] **Step 4: 更新 `_build_trace` 中的 pp 值**

确认 `_build_trace` 已正确写入 `pp` 值（当前已正确：`"pp": cfg.num_pipeline_stages`）。无需改动。

- [ ] **Step 5: 验证**

用 Vidur 运行 pp=2 的推理模拟，检查生成的 `inference_trace.json`：
- batch 记录数 = 微批次总数 × pp
- 每条记录有 `stage_id`
- `depends_on` 正确反映 pipeline 依赖链
- last stage 的 prefill 有 `kv_cache_bytes`（per-stage KV = total_kv / pp），非 prefill stage 也记录

---

### Task 2: InferenceTraceExpander PP 基础设施

**Files:**
- Modify: `simai-flow-scheduler/src/workload_generator/inference_trace_expander.py`
- Test: `simai-flow-scheduler/tests/test_inference_trace_expander.py`

- [ ] **Step 1: 写失败测试 — stage node mapping**

在 `test_inference_trace_expander.py` 中添加：

```python
class TestPPNodeMapping:
    """Test PP stage → GPU node assignment."""

    def test_stage_ranks_pp2(self, store):
        """pp=2, tp=2, ep=1: replica 0 has 4 GPUs, split into 2 stages."""
        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)
        # Stage 0: ranks [0, 1], Stage 1: ranks [2, 3]
        assert expander._stage_ranks(0, 0) == [0, 1]
        assert expander._stage_ranks(0, 1) == [2, 3]

    def test_stage_ranks_pp2_with_assigned_nodes(self, store):
        """pp=2, tp=2, ep=1, assigned_nodes=[10,11,12,13]."""
        expander = InferenceTraceExpander(
            store, tp=2, ep=1, pp=2,
            assigned_nodes=[10, 11, 12, 13],
        )
        assert expander._stage_ranks(0, 0) == [10, 11]
        assert expander._stage_ranks(0, 1) == [12, 13]

    def test_stage_ranks_multi_replica(self, store):
        """2 replicas, each pp=2, tp=2."""
        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)
        # Replica 0: ranks [0,1,2,3], Replica 1: ranks [4,5,6,7]
        assert expander._stage_ranks(0, 0) == [0, 1]
        assert expander._stage_ranks(0, 1) == [2, 3]
        assert expander._stage_ranks(1, 0) == [4, 5]
        assert expander._stage_ranks(1, 1) == [6, 7]

    def test_layers_for_stage(self, store):
        """61 layers, pp=2: stage 0 gets 0-29, stage 1 gets 30-60."""
        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)
        assert expander._layers_for_stage(0, 61) == range(0, 30)
        assert expander._layers_for_stage(1, 61) == range(30, 61)

    def test_layers_for_stage_pp1(self, store):
        """pp=1: all layers in stage 0."""
        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=1)
        assert expander._layers_for_stage(0, 61) == range(0, 61)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py::TestPPNodeMapping -v`
Expected: FAIL (AttributeError: 'InferenceTraceExpander' object has no attribute '_stage_ranks')

- [ ] **Step 3: 实现 `_stage_ranks` 和 `_layers_for_stage`**

在 `inference_trace_expander.py` 的 `__init__` 中：
1. 删除 `if pp != 1: raise NotImplementedError`
2. 在 rank helpers 区域添加以下方法：

```python
def _stage_ranks(self, replica_id: int, stage_id: int) -> list[int]:
    """GPUs for a specific PP stage within a replica.

    Layout: [stage0_tp_ep][stage1_tp_ep]...
    Each stage has tp*ep GPUs.
    """
    stage_size = self._tp * self._ep
    offset = replica_id * self._world_size() + stage_id * stage_size
    if self._assigned_nodes is not None:
        return list(self._assigned_nodes[offset:offset + stage_size])
    return list(range(offset, offset + stage_size))

def _layers_for_stage(self, stage_id: int, total_layers: int) -> range:
    """Layer IDs belonging to a PP stage."""
    per_stage = total_layers // self._pp
    start = stage_id * per_stage
    end = start + per_stage
    return range(start, end)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py::TestPPNodeMapping -v`
Expected: 5 passed

- [ ] **Step 5: 更新 `_grouper` 以支持 PP stage**

当前 `_grouper` 使用整个 replica 的 ranks。PP 下需要 per-stage 的 grouper：

```python
def _stage_grouper(self, replica_id: int, stage_id: int) -> RankGrouper:
    """RankGrouper for a specific PP stage (dp=1, pp=1, only tp+ep)."""
    return RankGrouper(
        self._stage_ranks(replica_id, stage_id),
        ParallelismConfig(tp=self._tp, ep=self._ep, pp=1),
    )
```

- [ ] **Step 6: 运行全部现有测试确认无回归**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py -v`
Expected: 所有现有测试通过（pp=1 路径不受影响）

- [ ] **Step 7: Commit**

```bash
git add simai-flow-scheduler/src/workload_generator/inference_trace_expander.py \
        simai-flow-scheduler/tests/test_inference_trace_expander.py
git commit -m "feat(expander): add PP stage node mapping and layer assignment helpers"
```

---

### Task 3: Per-stage batch expansion

**Files:**
- Modify: `simai-flow-scheduler/src/workload_generator/inference_trace_expander.py`
- Test: `simai-flow-scheduler/tests/test_inference_trace_expander.py`

- [ ] **Step 1: 写失败测试 — per-stage expansion 只展开本 stage 的 layer**

```python
class TestPPBatchExpansion:
    """Test that PP expansion only includes layers for the relevant stage."""

    def test_pp2_expands_correct_layers_per_stage(self):
        """4 layers total, pp=2: stage 0 has layers 0-1, stage 1 has layers 2-3."""
        # Setup: 4-layer profile
        profile_tsv = (
            "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
            "0\tattention\t100000\t1024\n"
            "0\tmlp\t200000\t1024\n"
            "1\tattention\t100000\t1024\n"
            "1\tmlp\t200000\t1024\n"
            "2\tattention\t100000\t1024\n"
            "2\tmlp\t200000\t1024\n"
            "3\tattention\t100000\t1024\n"
            "3\tmlp\t200000\t1024\n"
        )

        store = InferenceProfileStore()
        store.load_from_string(profile_tsv, "prefill_bs1_seq64")

        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)

        trace = {
            "version": "1.0",
            "model": "test-pp",
            "model_config": {"hidden_size": 64, "num_layers": 4},
            "parallelism": {"tp": 2, "ep": 1, "pp": 2},
            "requests": {"0": {"num_prefill_tokens": 64, "num_decode_tokens": 2}},
            "batches": [
                {
                    "batch_id": "p0_s0", "type": "prefill", "replica_id": 0,
                    "stage_id": 0,
                    "request_ids": [0], "num_tokens": [64],
                    "kv_cache_bytes": {"0": 1024},
                    "depends_on": [],
                },
                {
                    "batch_id": "p0_s1", "type": "prefill", "replica_id": 0,
                    "stage_id": 1,
                    "request_ids": [0], "num_tokens": [64],
                    "kv_cache_bytes": None,
                    "depends_on": ["p0_s0"],
                },
            ],
        }

        workload, btm = expander.expand(trace)
        task_map = {t.task_id: t for t in workload.tasks}

        # Stage 0 compute tasks should only be on GPUs 0-1 (layers 0,1)
        s0_compute = [
            t for t in workload.tasks
            if t.type == TaskType.COMPUTE and t.node in {0, 1}
        ]
        s0_layer_ids = {t.layer_id for t in s0_compute}
        assert s0_layer_ids == {0, 1}, f"Stage 0 layers: {s0_layer_ids}"

        # Stage 1 compute tasks should only be on GPUs 2-3 (layers 2,3)
        s1_compute = [
            t for t in workload.tasks
            if t.type == TaskType.COMPUTE and t.node in {2, 3}
        ]
        s1_layer_ids = {t.layer_id for t in s1_compute}
        assert s1_layer_ids == {2, 3}, f"Stage 1 layers: {s1_layer_ids}"
```

> 注意：上面的测试使用 `store.load_from_string()` 加载 profile。当前 `InferenceProfileStore` 的 `load()` 方法接受文件路径。需要在测试中写入临时文件，或给 store 添加一个 `load_from_string` 辅助方法。实际测试代码应使用 `tmp_path` fixture 写入临时文件再加载。

- [ ] **Step 2: 运行测试确认失败**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py::TestPPBatchExpansion -v`
Expected: FAIL

- [ ] **Step 3: 修改 `expand()` 和 `_expand_batch()` 支持 stage**

**修改 `expand()` 方法中的 batch expansion 部分：**

在 `expand()` 中，获取 batch 的 `stage_id`，并将其传递给 `_expand_batch`：

```python
for batch in trace["batches"]:
    bid = batch["batch_id"]
    btype = batch["type"]
    replica_id = batch["replica_id"]
    stage_id = batch.get("stage_id", 0)  # 新增：默认 0（向后兼容）

    # ... (prev_exits 处理逻辑不变) ...

    # ── Select profile and filter for this stage ──────────
    bs = len(batch["request_ids"])
    if btype == "prefill":
        seq = sum(batch["num_tokens"])
    else:
        kv_lens = batch.get("kv_cache_seq_lens")
        seq = max(kv_lens) if kv_lens else 1
    profiles = self._store.get_profile_for_batch(btype, bs, seq)

    # 获取 total_layers（从 model_config 或 profiles 推断）
    total_layers = self._get_total_layers(trace, profiles)
    stage_layer_range = self._layers_for_stage(stage_id, total_layers)

    # ── Expand batch for this stage only ──────────────────
    batch_tasks, exits, task_id = self._expand_batch(
        batch=batch,
        profiles=profiles,
        job_id=job_id,
        task_id_start=task_id,
        prev_exits=prev_exits,
        stage_id=stage_id,                    # 新增
        stage_layer_range=stage_layer_range,  # 新增
    )
    # ... (后续逻辑不变) ...
```

**添加 `_get_total_layers` 辅助方法：**

```python
def _get_total_layers(self, trace: dict, profiles: list) -> int:
    """Get total number of model layers from trace config or profiles."""
    model_config = trace.get("model_config", {})
    if "num_layers" in model_config:
        return model_config["num_layers"]
    # Fallback: infer from profiles
    if profiles:
        return max(p.layer_id for p in profiles) + 1
    return 1
```

**修改 `_expand_batch()` 签名和实现：**

```python
def _expand_batch(
    self,
    batch: dict,
    profiles: list,
    job_id: int,
    task_id_start: int,
    prev_exits: dict[int, list[int]],
    stage_id: int = 0,                # 新增
    stage_layer_range: range = None,   # 新增
) -> tuple[list[FlowTask], dict[int, list[int]], int]:
    phase = Phase.PREFILL if batch["type"] == "prefill" else Phase.DECODE
    replica_id = batch["replica_id"]

    # PP: 使用 per-stage 的 ranks 和 grouper
    ranks = self._stage_ranks(replica_id, stage_id)
    grouper = self._stage_grouper(replica_id, stage_id)

    layers = self._group_profiles_by_layer(profiles)

    # PP: 过滤掉不属于本 stage 的 layer
    if stage_layer_range is not None:
        layers = {
            lid: lmap for lid, lmap in layers.items()
            if lid in stage_layer_range
        }

    all_tasks: list[FlowTask] = []
    task_id = task_id_start
    current_exits = dict(prev_exits)

    _LAYER_NAME_ORDER = {"attention": 0, "mlp": 1, "moe": 1}

    for layer_id, layer_map in sorted(layers.items()):
        sub_ops = sorted(
            layer_map.items(),
            key=lambda kv: _LAYER_NAME_ORDER.get(kv[0], 99),
        )

        for op_name, profile in sub_ops:
            # ── COMPUTE tasks (one per rank in this stage) ──
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

            # ── FLOW tasks ──────────────────────────────────
            flow_result = _FlowGroupResult()
            comm_size = profile.comm_size_bytes

            if comm_size > 0:
                if op_name == "moe":
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

            # ── Update exits ────────────────────────────────
            if flow_result.flows:
                current_exits = {
                    rank: flow_result.receiver_index.get(rank, [])
                    for rank in ranks
                }
            else:
                current_exits = {rank: [compute[rank].task_id] for rank in ranks}

    return all_tasks, current_exits, task_id
```

**注意**：当前代码中 `_expand_batch` 使用 `ranks = self._replica_ranks(replica_id)` 获取所有 ranks。PP 下改为使用 `self._stage_ranks(replica_id, stage_id)` 只获取当前 stage 的 ranks。这样 compute 和 flow task 都自然地只分配到正确的 GPU 上。

- [ ] **Step 4: 运行测试确认通过**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py -v`
Expected: ALL passed

- [ ] **Step 5: Commit**

```bash
git add simai-flow-scheduler/src/workload_generator/inference_trace_expander.py \
        simai-flow-scheduler/tests/test_inference_trace_expander.py
git commit -m "feat(expander): support per-stage batch expansion with PP layer filtering"
```

---

### Task 4: PP inter-stage 通信流

**Files:**
- Modify: `simai-flow-scheduler/src/workload_generator/inference_trace_expander.py`
- Test: `simai-flow-scheduler/tests/test_inference_trace_expander.py`

PP stage 间传输 hidden state，每个 TP rank pair 产生一条 P2P 流。

- [ ] **Step 1: 写失败测试 — PP inter-stage communication**

```python
class TestPPInterStageComm:
    """Test PP inter-stage communication flow tasks."""

    def test_pp_comm_inserted_between_stages(self):
        """Between stage 0 and stage 1, PP_SEND/PP_RECV flows should be inserted."""
        profile_tsv = (
            "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
            "0\tattention\t100000\t1024\n"
            "0\tmlp\t200000\t1024\n"
            "1\tattention\t100000\t1024\n"
            "1\tmlp\t200000\t1024\n"
        )
        store = InferenceProfileStore()
        # ... load profile ...

        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)

        trace = {
            "version": "1.0", "model": "test-pp",
            "model_config": {"hidden_size": 64, "num_layers": 2},
            "parallelism": {"tp": 2, "ep": 1, "pp": 2},
            "requests": {"0": {"num_prefill_tokens": 64, "num_decode_tokens": 2}},
            "batches": [
                {
                    "batch_id": "p0_s0", "type": "prefill", "replica_id": 0,
                    "stage_id": 0, "request_ids": [0], "num_tokens": [64],
                    "kv_cache_bytes": {"0": 1024}, "depends_on": [],
                },
                {
                    "batch_id": "p0_s1", "type": "prefill", "replica_id": 0,
                    "stage_id": 1, "request_ids": [0], "num_tokens": [64],
                    "kv_cache_bytes": None, "depends_on": ["p0_s0"],
                },
            ],
        }

        workload, btm = expander.expand(trace)

        # Should have PP communication flow tasks
        pp_flows = [t for t in workload.tasks
                    if t.type == TaskType.FLOW and t.comm_type in (CommType.PP_SEND, CommType.PP_RECV)]
        assert len(pp_flows) > 0, "Expected PP communication flows"

        # Each PP flow goes from stage 0 rank to stage 1 rank
        for t in pp_flows:
            assert t.src in {0, 1}, f"PP flow src {t.src} not in stage 0"
            assert t.dst in {2, 3}, f"PP flow dst {t.dst} not in stage 1"

        # Stage 1 compute tasks should depend on PP flows, not directly on stage 0 tasks
        s1_compute = [t for t in workload.tasks
                      if t.type == TaskType.COMPUTE and t.node in {2, 3}]
        first_s1 = min(s1_compute, key=lambda t: t.task_id)
        pp_flow_ids = {t.task_id for t in pp_flows}
        assert any(dep in pp_flow_ids for dep in first_s1.deps), \
            "Stage 1 first compute should depend on PP communication flows"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py::TestPPInterStageComm -v`
Expected: FAIL

- [ ] **Step 3: 实现 PP inter-stage communication 插入逻辑**

在 `expand()` 方法中，当检测到跨 stage 依赖时（`depends_on` 中的 batch 的 `stage_id` ≠ 当前 `stage_id`），在 stage 0 的 exit tasks 和 stage 1 的 compute tasks 之间插入 PP 通信流。

在 `expand()` 的 `for dep_id in batch.get("depends_on", [])` 循环中添加跨 stage 处理逻辑：

```python
for dep_id in batch.get("depends_on", []):
    dep_batch = batch_lookup[dep_id]
    dep_stage_id = dep_batch.get("stage_id", 0)

    # ── PP inter-stage: dep is from a different stage ─────
    if dep_stage_id != stage_id and dep_stage_id == stage_id - 1:
        # Insert PP communication between stages
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
        all_flow_tasks.extend(pp_tasks)

        btm_key = f"pp_{dep_id}_to_{bid}"
        batch_task_map[btm_key] = {
            "task_ids": [t.task_id for t in pp_tasks],
            "type": "pp_comm",
            "from_stage": dep_stage_id,
            "to_stage": stage_id,
        }
        for rank, tids in pp_exits.items():
            prev_exits.setdefault(rank, []).extend(tids)
        continue

    # ── Existing KV transfer and same-stage dep logic ─────
    if dep_batch["type"] == "prefill" and btype == "decode":
        # ... (existing KV transfer logic, modified for PP in Task 5) ...
    else:
        for rank, tids in batch_exits.get(dep_id, {}).items():
            prev_exits.setdefault(rank, []).extend(tids)
```

**添加 `_expand_pp_communication` 方法：**

```python
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
    Expand PP inter-stage communication: one P2P flow per TP rank pair.

    src_stage ranks → dst_stage ranks, carrying hidden state tensor shard.
    """
    tasks: list[FlowTask] = []
    task_id = task_id_start
    exits: dict[int, list[int]] = {}

    src_ranks = self._stage_ranks(replica_id, src_stage_id)
    dst_ranks = self._stage_ranks(replica_id, dst_stage_id)

    # Compute comm size: hidden_size / tp * dtype_bytes * total_tokens
    model_config = trace.get("model_config", {})
    hidden_size = model_config.get("hidden_size", 0)
    dtype_bytes = 2  # FP16/BF16
    per_rank_bytes = max(hidden_size // self._tp * dtype_bytes * total_tokens, 1)

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
            phase=Phase.FORWARD,  # PP comm is always forward
            layer_id=0,
            deps=list(prev_exits.get(src_rank, [])),
        )
        tasks.append(ft)
        exits.setdefault(dst_rank, []).append(task_id)
        task_id += 1

    return tasks, exits, task_id
```

- [ ] **Step 4: 运行测试确认通过**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py -v`
Expected: ALL passed

- [ ] **Step 5: Commit**

```bash
git add simai-flow-scheduler/src/workload_generator/inference_trace_expander.py \
        simai-flow-scheduler/tests/test_inference_trace_expander.py
git commit -m "feat(expander): insert PP inter-stage communication flows between pipeline stages"
```

---

### Task 5: PP-aware KV transfer

**Files:**
- Modify: `simai-flow-scheduler/src/workload_generator/inference_trace_expander.py`
- Test: `simai-flow-scheduler/tests/test_inference_trace_expander.py`

PP 下 KV cache 按 layer 分配到各 stage。每个 stage 独立传输自己的 KV cache 分片。D-stage-s 只依赖自己的 KV transfer 完成即可开始 decode，不需要等待其他 stages。

**与 Vidur 的差异**：Vidur 把 KV transfer 建模为所有 PP stages 完成后的一次性传输。simai-flow-scheduler 采用 per-stage 模型，D-stage 可以提前开始 decode，更接近真实系统行为。

- [ ] **Step 1: 写失败测试 — PP per-stage KV transfer**

```python
class TestPPKVTransfer:
    """Test KV transfer with PP: per-stage transfer."""

    def test_kv_transfer_per_stage_pp2(self):
        """pp=2: KV cache split evenly across stages, each stage transfers independently."""
        profile_tsv = (
            "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
            "0\tattention\t100000\t1024\n"
            "0\tmlp\t200000\t1024\n"
            "1\tattention\t100000\t1024\n"
            "1\tmlp\t200000\t1024\n"
        )
        store = InferenceProfileStore()
        # ... load ...

        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)

        trace = {
            "version": "1.0", "model": "test-pp",
            "model_config": {"hidden_size": 64, "num_layers": 2},
            "parallelism": {"tp": 2, "ep": 1, "pp": 2},
            "requests": {"0": {"num_prefill_tokens": 64, "num_decode_tokens": 2}},
            "batches": [
                # P-replica prefill (each stage records per-stage KV = total_kv / pp)
                {"batch_id": "p0_s0", "type": "prefill", "replica_id": 0,
                 "stage_id": 0, "request_ids": [0], "num_tokens": [64],
                 "kv_cache_bytes": {"0": 2048}, "depends_on": []},
                {"batch_id": "p0_s1", "type": "prefill", "replica_id": 0,
                 "stage_id": 1, "request_ids": [0], "num_tokens": [64],
                 "kv_cache_bytes": {"0": 2048}, "depends_on": ["p0_s0"]},
                # D-replica decode (each stage independently depends on its P-stage KV)
                {"batch_id": "d0_s0", "type": "decode", "replica_id": 1,
                 "stage_id": 0, "request_ids": [0], "num_tokens": [1],
                 "kv_cache_bytes": None, "depends_on": ["p0_s0"]},
                {"batch_id": "d0_s1", "type": "decode", "replica_id": 1,
                 "stage_id": 1, "request_ids": [0], "num_tokens": [1],
                 "kv_cache_bytes": None, "depends_on": ["d0_s0", "p0_s1"]},
            ],
        }

        workload, btm = expander.expand(trace)

        # Should have KV transfer tasks for both stages
        kv_tasks = [t for t in workload.tasks
                    if t.comm_type == CommType.KV_CACHE_TRANSFER]

        # Stage 0 KV: P-stage0 [4,5] → D-stage0 [12,13]
        kv_s0 = [t for t in kv_tasks if t.src in {4, 5}]
        assert len(kv_s0) > 0, "Expected KV transfer for stage 0"

        # Stage 1 KV: P-stage1 [6,7] → D-stage1 [14,15]
        kv_s1 = [t for t in kv_tasks if t.src in {6, 7}]
        assert len(kv_s1) > 0, "Expected KV transfer for stage 1"

        # Per-stage KV bytes = total_kv / pp = 4096 / 2 = 2048 (recorded in trace entry)
        # Per-rank bytes = 2048 / (tp*ep) = 2048 / 2 = 1024
        for t in kv_s0:
            assert t.size_bytes == 1024
        for t in kv_s1:
            assert t.size_bytes == 1024
```

- [ ] **Step 2: 运行测试确认失败**

- [ ] **Step 3: 修改 `_expand_kv_transfer` 支持 PP stage**

核心改动：KV transfer 改为 per-stage，每条流从 P-stage-s rank 到 D-stage-s rank。

修改 `expand()` 中 KV transfer 的触发逻辑。当 `btype == "decode"` 且 `dep_batch["type"] == "prefill"` 时：

```python
if dep_batch["type"] == "prefill" and btype == "decode":
    kv_key = (dep_id, replica_id, stage_id)  # 新增 stage_id 到 dedup key
    if kv_key in kv_transfer_done:
        for rank, tids in kv_transfer_done[kv_key].items():
            prev_exits.setdefault(rank, []).extend(tids)
    else:
        # 计算 per-stage KV cache bytes
        total_kv_bytes = dep_batch.get("kv_cache_bytes") or {}
        kv_tasks, kv_exits, task_id = self._expand_kv_transfer(
            request_ids=batch["request_ids"],
            kv_cache_bytes=total_kv_bytes,
            p_replica_id=dep_batch["replica_id"],
            d_replica_id=replica_id,
            src_stage_id=dep_stage_id,      # 新增
            dst_stage_id=stage_id,          # 新增
            job_id=job_id,
            task_id_start=task_id,
            prev_exits=batch_exits.get(dep_id, {}),
        )
        all_flow_tasks.extend(kv_tasks)
        kv_transfer_done[kv_key] = kv_exits
        # ... (btm 更新同前) ...
```

修改 `_expand_kv_transfer` 签名：

```python
def _expand_kv_transfer(
    self,
    request_ids: list,
    kv_cache_bytes: dict,
    p_replica_id: int,
    d_replica_id: int,
    src_stage_id: int,        # 新增
    dst_stage_id: int,        # 新增
    job_id: int,
    task_id_start: int,
    prev_exits: dict[int, list[int]],
) -> tuple[list[FlowTask], dict[int, list[int]], int]:
    tasks: list[FlowTask] = []
    task_id = task_id_start
    exits: dict[int, list[int]] = {}

    stage_size = self._tp * self._ep  # GPUs per stage
    p_ranks = self._stage_ranks(p_replica_id, src_stage_id)
    d_ranks = self._stage_ranks(d_replica_id, dst_stage_id)

    for req_id in request_ids:
        req_key = str(req_id)
        total_kv = kv_cache_bytes.get(req_key, 0)
        if total_kv == 0:
            continue

        # Per-rank bytes: kv_cache_bytes entry is already per-stage (= total_kv / pp)
        # Divide by stage_size (tp*ep) to get per-rank share
        per_rank_bytes = max(total_kv // stage_size, 1)

        for p_rank, d_rank in zip(p_ranks, d_ranks):
            ft = FlowTask(
                task_id=task_id,
                job_id=job_id,
                type=TaskType.FLOW,
                src=p_rank,
                dst=d_rank,
                size_bytes=per_rank_bytes,
                comm_type=CommType.KV_CACHE_TRANSFER,
                chunk_id=0, num_chunks=1,
                phase=Phase.PREFILL,
                layer_id=0,
                deps=list(prev_exits.get(p_rank, [])),
            )
            tasks.append(ft)
            exits.setdefault(d_rank, []).append(task_id)
            task_id += 1

    return tasks, exits, task_id
```

- [ ] **Step 4: 运行测试确认通过**

- [ ] **Step 5: Commit**

```bash
git add simai-flow-scheduler/src/workload_generator/inference_trace_expander.py \
        simai-flow-scheduler/tests/test_inference_trace_expander.py
git commit -m "feat(expander): per-stage KV cache transfer for PP"
```

---

### Task 6: Pipeline dependency 链验证

**Files:**
- Test: `simai-flow-scheduler/tests/test_inference_trace_expander.py`

验证整个 DAG 的依赖结构正确重建了 pipeline overlap。

- [ ] **Step 1: 写测试 — 验证 pipeline overlap 依赖结构**

```python
class TestPPPipelineDependencies:
    """Verify that dependency chain correctly models pipeline overlap."""

    def test_pipeline_overlap_deps(self):
        """
        2 micro-batches, pp=2:
          MB0-S0 → []
          MB0-S1 → [MB0-S0]              (cross-stage)
          MB1-S0 → [MB0-S0]              (same-stage)
          MB1-S1 → [MB1-S0, MB0-S1]      (cross-stage + same-stage)

        Stage 0 and Stage 1 should have overlapping execution windows.
        """
        profile_tsv = (
            "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
            "0\tattention\t100000\t1024\n"
            "0\tmlp\t200000\t1024\n"
            "1\tattention\t100000\t1024\n"
            "1\tmlp\t200000\t1024\n"
        )
        store = InferenceProfileStore()
        # ... load ...

        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)

        trace = {
            "version": "1.0", "model": "test-pp",
            "model_config": {"hidden_size": 64, "num_layers": 2},
            "parallelism": {"tp": 2, "ep": 1, "pp": 2},
            "requests": {
                "0": {"num_prefill_tokens": 64, "num_decode_tokens": 4},
                "1": {"num_prefill_tokens": 32, "num_decode_tokens": 2},
            },
            "batches": [
                # MB0: request 0
                {"batch_id": "p0_s0", "type": "prefill", "replica_id": 0,
                 "stage_id": 0, "request_ids": [0], "num_tokens": [64],
                 "kv_cache_bytes": {"0": 4096}, "depends_on": []},
                {"batch_id": "p0_s1", "type": "prefill", "replica_id": 0,
                 "stage_id": 1, "request_ids": [0], "num_tokens": [64],
                 "kv_cache_bytes": None, "depends_on": ["p0_s0"]},
                # MB1: request 1
                {"batch_id": "p1_s0", "type": "prefill", "replica_id": 0,
                 "stage_id": 0, "request_ids": [1], "num_tokens": [32],
                 "kv_cache_bytes": {"1": 2048}, "depends_on": ["p0_s0"]},
                {"batch_id": "p1_s1", "type": "prefill", "replica_id": 0,
                 "stage_id": 1, "request_ids": [1], "num_tokens": [32],
                 "kv_cache_bytes": None, "depends_on": ["p1_s0", "p0_s1"]},
            ],
        }

        workload, btm = expander.expand(trace)
        errors = workload.validate()
        assert errors == [], f"Validation errors: {errors}"

        # MB1-S1 first compute depends on BOTH MB1-S0 (via PP comm) and MB0-S1 (via same-stage)
        s1_compute_mb1 = [t for t in workload.tasks
                          if t.type == TaskType.COMPUTE and t.node in {2, 3}
                          and t.layer_id == 0]  # first layer of stage 1

        # There should be compute tasks on stage 1 ranks from both MB0 and MB1
        assert len(s1_compute_mb1) >= 2

        # Check that workload is valid DAG (no cycles) — already checked by validate()
        # Check that all deps exist
        task_ids = {t.task_id for t in workload.tasks}
        for t in workload.tasks:
            for dep in t.deps:
                assert dep in task_ids, f"Task {t.task_id} has dangling dep {dep}"

    def test_no_cross_stage_dep_without_pp_comm(self):
        """Stage 1 tasks should NOT directly depend on Stage 0 compute tasks.
        They should go through PP communication flows."""
        # ... similar setup ...
        workload, btm = expander.expand(trace)

        pp_comm_ids = {t.task_id for t in workload.tasks
                       if t.comm_type in (CommType.PP_SEND, CommType.PP_RECV)}
        s0_compute_ids = {t.task_id for t in workload.tasks
                          if t.type == TaskType.COMPUTE and t.node in {0, 1}}
        s1_compute_ids = {t.task_id for t in workload.tasks
                          if t.type == TaskType.COMPUTE and t.node in {2, 3}}

        for s1_tid in s1_compute_ids:
            s1_task = next(t for t in workload.tasks if t.task_id == s1_tid)
            for dep in s1_task.deps:
                assert dep not in s0_compute_ids, \
                    f"Stage 1 compute {s1_tid} directly depends on Stage 0 compute {dep}, " \
                    f"should go through PP comm"
```

- [ ] **Step 2: 运行测试确认通过**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py::TestPPPipelineDependencies -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add simai-flow-scheduler/tests/test_inference_trace_expander.py
git commit -m "test(expander): verify PP pipeline overlap dependency structure"
```

---

### Task 7: 端到端测试

**Files:**
- Test: `simai-flow-scheduler/tests/test_inference_trace_expander.py`

使用完整的 synthetic trace（包含 prefill + decode + KV transfer + PP）运行 expander，然后通过 AnalyticalExecutor 执行，验证 pipeline overlap 效果。

- [ ] **Step 1: 写端到端测试**

```python
class TestPPEndToEnd:
    """End-to-end: PP trace → expand → validate workload."""

    def test_full_pp2_inference_workload(self):
        """
        Full inference with pp=2, tp=2, ep=1, 2 replicas (P=0, D=1).
        2 requests, 1 prefill batch, 2 decode batches.
        """
        profile_tsv = (
            "layer_id\tlayer_name\tcomp_time\tcomm_size\n"
            "0\tattention\t100000\t1024\n"
            "0\tmlp\t200000\t1024\n"
            "1\tattention\t100000\t1024\n"
            "1\tmlp\t200000\t1024\n"
        )
        store = InferenceProfileStore()
        # ... load prefill and decode profiles ...

        expander = InferenceTraceExpander(store, tp=2, ep=1, pp=2)

        trace = {
            "version": "1.0", "model": "test-pp",
            "model_config": {"hidden_size": 64, "num_layers": 2},
            "parallelism": {"tp": 2, "ep": 1, "pp": 2},
            "requests": {
                "0": {"num_prefill_tokens": 64, "num_decode_tokens": 4},
                "1": {"num_prefill_tokens": 32, "num_decode_tokens": 2},
            },
            "batches": [
                # P-replica prefill (per-stage KV = total_kv / pp)
                {"batch_id": "p0_s0", "type": "prefill", "replica_id": 0,
                 "stage_id": 0, "request_ids": [0, 1], "num_tokens": [64, 32],
                 "kv_cache_bytes": {"0": 2048, "1": 1024}, "depends_on": []},
                {"batch_id": "p0_s1", "type": "prefill", "replica_id": 0,
                 "stage_id": 1, "request_ids": [0, 1], "num_tokens": [64, 32],
                 "kv_cache_bytes": {"0": 2048, "1": 1024}, "depends_on": ["p0_s0"]},
                # D-replica decode batch 0
                {"batch_id": "d0_s0", "type": "decode", "replica_id": 1,
                 "stage_id": 0, "request_ids": [0, 1], "num_tokens": [1, 1],
                 "kv_cache_bytes": None, "depends_on": ["p0_s0"],
                 "kv_cache_seq_lens": [65, 33]},
                {"batch_id": "d0_s1", "type": "decode", "replica_id": 1,
                 "stage_id": 1, "request_ids": [0, 1], "num_tokens": [1, 1],
                 "kv_cache_bytes": None, "depends_on": ["d0_s0", "p0_s1"],
                 "kv_cache_seq_lens": [65, 33]},
                # D-replica decode batch 1
                {"batch_id": "d1_s0", "type": "decode", "replica_id": 1,
                 "stage_id": 0, "request_ids": [0, 1], "num_tokens": [1, 1],
                 "kv_cache_bytes": None, "depends_on": ["d0_s0"],
                 "kv_cache_seq_lens": [66, 34]},
                {"batch_id": "d1_s1", "type": "decode", "replica_id": 1,
                 "stage_id": 1, "request_ids": [0, 1], "num_tokens": [1, 1],
                 "kv_cache_bytes": None, "depends_on": ["d1_s0", "d0_s1"],
                 "kv_cache_seq_lens": [66, 34]},
            ],
        }

        workload, btm = expander.expand(trace)

        # 1. Valid workload
        errors = workload.validate()
        assert errors == [], f"Validation errors: {errors}"

        # 2. Correct node count: 2 replicas × (tp*ep*pp) = 2 × 4 = 8 nodes
        all_nodes = set()
        for job in workload.jobs:
            all_nodes.update(job.assigned_nodes)
        assert len(all_nodes) == 8

        # 3. P-replica uses nodes 0-3 (stage 0: [0,1], stage 1: [2,3])
        #    D-replica uses nodes 4-7 (stage 0: [4,5], stage 1: [6,7])
        p_nodes = {0, 1, 2, 3}
        d_nodes = {4, 5, 6, 7}
        compute_nodes = {t.node for t in workload.tasks if t.type == TaskType.COMPUTE}
        assert compute_nodes == (p_nodes | d_nodes)

        # 4. PP comm flows between stages
        pp_flows = [t for t in workload.tasks
                    if t.comm_type in (CommType.PP_SEND, CommType.PP_RECV)]
        assert len(pp_flows) > 0

        # 5. KV transfer flows from P to D
        kv_flows = [t for t in workload.tasks
                    if t.comm_type == CommType.KV_CACHE_TRANSFER]
        assert len(kv_flows) > 0

        # 6. batch_task_map has all expected entries
        assert "p0_s0" in btm
        assert "p0_s1" in btm
        assert "d0_s0" in btm
        assert "d0_s1" in btm
        assert "d1_s0" in btm
        assert "d1_s1" in btm
```

- [ ] **Step 2: 运行测试确认通过**

Run: `cd simai-flow-scheduler && uv run pytest tests/test_inference_trace_expander.py::TestPPEndToEnd -v`
Expected: PASS

- [ ] **Step 3: 运行全部测试确认无回归**

Run: `cd simai-flow-scheduler && uv run pytest tests/ -v`
Expected: ALL passed（除已知的 2 个 visualizer 预存问题）

- [ ] **Step 4: Commit**

```bash
git add simai-flow-scheduler/tests/test_inference_trace_expander.py
git commit -m "test(expander): add PP end-to-end inference workload test"
```

---

## 附录

### A. Profile 文件的 PP 适配

当前 profile 文件名包含 `pp1`：`vidur-DeepSeek-671B-world_size8-tp2-pp1-ep4-bs1-seq1024-prefill.csv`

Profile 文件记录的是 per-layer 的计算和通信时间。**PP 不影响 per-layer 性能**，只影响每个 stage 处理哪些 layer。因此：

- 现有 pp=1 的 profile 文件可以直接用于 pp>1 的场景
- Expander 通过 `_layers_for_stage()` 过滤出当前 stage 的 layers
- 不需要为 pp>1 生成新的 profile 文件

### B. PP 通信量估算

```
pp_comm_bytes_per_rank = hidden_size / tp * dtype_bytes * total_tokens

# DeepSeek-671B 示例:
# hidden_size = 7168, tp = 2, dtype = FP16
# prefill (1024 tokens): 7168/2 * 2 * 1024 = 7,340,032 bytes ≈ 7 MB/rank
# decode (1 token):      7168/2 * 2 * 1    = 7,168 bytes ≈ 7 KB/rank
```

### C. 未来扩展

以下内容不在 Phase 6 范围内，记录为后续工作：

1. **PP 反向传播**：推理场景只有 forward pass，暂不支持 PP backward
2. **EP + PP 组合**：EP AlltoAll 和 PP 的交互需要更仔细的 rank 映射验证

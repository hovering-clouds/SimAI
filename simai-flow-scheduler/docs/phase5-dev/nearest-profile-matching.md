# 推理 Trace 展开的就近 Profile 匹配

**日期：** 2026-05-11
**Commits：** `dcf11bd`, `3aa05b1`

## 背景

此前 `InferenceTraceExpander` 要求调用者显式指定 `prefill_profile_key` 和 `decode_profile_key`（例如 `"prefill_bs1_seq512"`、`"decode_bs1_seq1"`），整个模拟过程所有 prefill batch 共用同一个 profile，所有 decode batch 共用同一个 profile——无论每个 batch 的实际参数（batch size、序列长度）如何变化。

实际上 Vidur 调度产生的各 batch 参数差异较大：
- **Prefill batch**：总 token 数 = 各请求 `num_tokens` 之和
- **Decode batch**：batch size = 并发请求数，KV cache 序列长度随 decode 步数增长

对所有 batch 使用固定 profile 会引入明显误差，尤其是 batch size 跨度较大时。

## 合理性分析：seq_length 就近匹配的可行性

Decode 阶段，seq_length 对不同操作的影响：

| 操作 | GEMM 的 m 维度 | 是否受 seq_length 影响 | 就近匹配是否可行 |
|---|---|---|---|
| MLP up/down | `batch_size` (≈1) | 不影响 | 可行 |
| MoE expert | `batch_size` (≈1) | 不影响 | 可行 |
| Attention (FlashMLA) | `batch_size` (≈1) | **影响** — KV cache 越长，内存读取越多 | 有误差但误差不大 |

Prefill 阶段：所有 GEMM 的 m 维度 = `seq_length`，计算时间线性增长，不能用就近匹配。

**结论**：Decode 阶段可以安全使用就近匹配（bs 优先），Prefill 阶段需要精确匹配 seq。

## 改动概览

### 1. `InferenceProfileStore` — 目录批量加载 + 就近匹配查询

**改动前**：手动逐个加载 CSV 文件，显式指定 key：
```python
store = InferenceProfileStore()
store.load("prefill_bs1_seq512.csv", "prefill_bs1_seq512")
store.load("decode_bs1_seq1.csv", "decode_bs1_seq1")
```

**改动后**：从目录批量加载，按并行配置过滤，按 (phase, bs, seq) 就近查询：
```python
store = InferenceProfileStore(tp=2, ep=4, pp=1)
store.load_directory("inputs/vidur-csv/deepseek-tp2-pp1-ep4/")
profiles = store.get_profile_for_batch("decode", bs=8, seq=1024)
```

新增 API：
- `load_directory(dir_path)` — 扫描目录下所有 `*.csv` 文件，从文件名解析 (phase, bs, seq, tp, ep, pp)，按 store 的并行配置过滤后加载。
- `get_profile_for_batch(phase, bs, seq)` — 使用就近匹配策略返回最合适的 profile。
- `_parse_filename(stem)` — 支持两种文件名格式：
  - 新格式：`prefill_bs4_seq4096_tp2_ep1_pp1.csv`
  - Vidur 格式：`vidur-DeepSeek-671B-world_size8-tp2-pp1-ep4-bs4-seq4096-prefill.csv`

**就近匹配优先级**（`get_profile_for_batch`）：
1. phase + bs + seq 完全精确匹配
2. phase + bs 相同，seq 最接近
3. phase 相同，bs 最接近（不限制 seq）

匹配策略的合理性依据见上方「合理性分析」章节：decode 阶段计算时间主要由 GEMM 的 m=batch_size 维度决定，因此 bs 的匹配优先级高于 seq。

### 2. `InferenceTraceExpander` — 逐 batch 动态选择 profile

**改动前**：所有 batch 使用固定的 profile key：
```python
expander.expand(trace, prefill_profile_key="prefill_bs1_seq512", decode_profile_key="decode_bs1_seq1")
```

**改动后**：每个 batch 独立选择最合适的 profile：
```python
expander.expand(trace)  # 不再需要指定 profile key
```

`expand()` 内部对每个 batch：
- **Prefill**：`seq = sum(batch["num_tokens"])`，`bs = len(batch["request_ids"])`
- **Decode**：`seq = max(batch["kv_cache_seq_lens"])`，`bs = len(batch["request_ids"])`

然后调用 `store.get_profile_for_batch(btype, bs, seq)` 查找最接近的 profile。

### 3. `TraceRecorder` — 记录逐请求 KV cache 序列长度

Decode batch 的 profile 选择需要知道每个请求当前的 KV cache 长度，用于确定 attention kernel 的有效序列长度。Vidur 的 `TraceRecorder` 新增记录：

```python
# trace JSON 中 decode batch 新增字段：
"kv_cache_seq_lens": [129, 65, ...]  # 逐请求，= 已处理的 prefill tokens + 已处理的 decode tokens
```

该字段取自 `req.num_processed_prefill_tokens + req.num_processed_decode_tokens`。

## 变更文件

| 文件 | 改动 |
|---|---|
| `src/workload_generator/inference_profile.py` | 新增 `load_directory()`、`get_profile_for_batch()`、`_parse_filename()`、`_parse_keys_for_phase()`；构造函数接受 tp/ep/pp 过滤参数 |
| `src/workload_generator/inference_trace_expander.py` | 移除 `prefill_profile_key`/`decode_profile_key` 参数；每个 batch 通过 `get_profile_for_batch()` 动态选择 profile |
| `vidur-alibabacloud/vidur/trace_recorder.py` | Decode batch 记录 `kv_cache_seq_lens` |
| `scripts/run_mixed_e2e.py` | 改用 `load_directory()` 替代手动 `load()`；从 trace JSON 读取并行配置 |
| `inputs/traces/deepseek_sample.json` | Decode batch 添加 `kv_cache_seq_lens` 字段 |
| `tests/test_inference_profile.py` | 新增 `load_directory()` 测试：新格式、Vidur 格式、混合格式、无过滤、空目录 |
| `tests/test_inference_trace_expander.py` | 适配新的 `expand()` API（不再传 profile key）；trace fixture 添加 `kv_cache_seq_lens` |

## 迁移指南

1. 将手动 `store.load()` 替换为 `store.load_directory()`：
   ```python
   # 旧写法
   store = InferenceProfileStore()
   store.load("prefill.csv", "prefill_bs1_seq512")
   store.load("decode.csv", "decode_bs1_seq1")

   # 新写法
   store = InferenceProfileStore(tp=2, ep=4, pp=1)
   store.load_directory("inputs/vidur-csv/deepseek-tp2-pp1-ep4/")
   ```

2. 从 `expander.expand()` 中移除 `prefill_profile_key` / `decode_profile_key`：
   ```python
   # 旧写法
   expander.expand(trace, prefill_profile_key="...", decode_profile_key="...")

   # 新写法
   expander.expand(trace)
   ```

3. 如果从 Vidur 生成 trace，确保 decode batch 包含 `kv_cache_seq_lens`（使用更新后的 `TraceRecorder` 会自动记录）。

# Phase 5 开发记录：Stage 1 KV-Cache Reuse Flow Support

## 概述

Phase 5 在 Phase 1-4 的基础上添加了 MFS Stage 1 支持：KV-cache 重用流（`KV_CACHE_REUSE`），即 prefill 之前从专用存储节点获取可复用 KV cache 的通信流。这完成了 MFS 三阶段模型的最后一个拼图：Stage 1（KV reuse）→ Stage 2（collective）→ Stage 3（P2D transfer）。

**核心设计：**
- 存储节点是普通拓扑节点（无 TP/EP/PP 结构），不是计算 replica
- Reuse flow 按逐模型层粒度生成，每层有独立 flow，使 RLI 机制能正确控制优先级
- Per-request 分配：同一 request 的所有层、所有 rank 的 reuse flow 都从同一 storage node 获取
- Trace 只指定存储节点数量和 per-request 的 `hit_ratio` / `storage_node_idx`，具体节点映射由 expander 负责

**最终结果：**
- 全部 598 个测试通过（含 13 个 Phase 5 新测试），无回归
- Vidur trace_recorder 新增 Stage 1 元数据生成（per-request hit_ratio + storage_node_idx）
- Expander 新增 `_expand_kv_reuse()`，per-layer flow 创建 + dependency 注入
- MFS context 分类 `KV_CACHE_REUSE` 为 `MfsStage.EARLY`，RLI 自动控制优先级

---

## 开发顺序与具体内容

### Task 15: Vidur trace_recorder 添加 Stage 1 元数据

**目标：** 在 Vidur trace JSON 中生成 Stage 1 KV reuse 元数据。

**Trace 格式设计：**

顶层：
```json
{
  "stage1_kv_reuse": {
    "num_storage_nodes": 2
  }
}
```

Per-request：
```json
{
  "0": {
    "num_prefill_tokens": 1024,
    "kv_reuse_hit_ratio": 0.5,
    "kv_reuse_storage_node_idx": 0
  }
}
```

**实现：**

1. `__init__()` 新增配置参数：
   - `stage1_kv_reuse_enable: bool` — 总开关
   - `stage1_kv_reuse_hit_ratio: float` — 统一 hit ratio
   - `stage1_num_storage_nodes: int` — 存储节点数量

2. `record_request()` 中：当 enable 时，为每个 request 添加 `kv_reuse_hit_ratio` 和 `kv_reuse_storage_node_idx`（round-robin 分配）。

3. `_build_trace()` 中：输出顶层 `stage1_kv_reuse: {num_storage_nodes}`。

**设计决策：**

- **Trace 不指定具体节点 ID**：`kv_reuse_storage_node_idx` 是逻辑索引（`range(num_storage_nodes)`），expander 根据拓扑映射到实际节点 ID。这解耦了 trace 生成和拓扑配置。
- **Per-request hit_ratio**：每个 request 可以有独立的 `kv_reuse_hit_ratio`，提供精细控制。Vidur 生成时使用统一值，但 trace 可以手动修改为不同值。
- **存储节点数量只在顶层指定**：所有 request 共享同一组存储节点，无需 per-request 重复。

### Task 16: Expander 扩展 + MFS 分类

**目标：** 将 trace 中的 Stage 1 元数据扩展为 P2P flow task，并正确集成到 MFS 调度框架。

#### 16a: Expander 新增 `_expand_kv_reuse()`

**方法签名：**
```python
def _expand_kv_reuse(
    self,
    storage_node_ids: list[int],
    dest_ranks: list[int],      # prefill stage ranks
    kv_cache_bytes: dict,       # req_id → total kv bytes for this stage
    stage_layer_ids: list[int], # layer IDs in this PP stage
    request_reuse_info: dict,   # req_id → {hit_ratio, storage_node_idx}
    job_id: int,
    task_id_start: int,
) -> tuple[list[FlowTask], dict[int, dict[int, list[int]]], dict[str, list[int]], int]:
```

**Flow 创建逻辑：**

```
对每个 req_id (有 reuse 配置):
    storage_node = storage_node_ids[storage_node_idx]  # per-request 分配
    对每个 layer_id:
        对每个 rank:
            bytes = kv_cache_bytes[req_id] / num_layers / stage_size * hit_ratio
            flow(storage_node → rank, layer_id=layer_id, deps=[])
```

**返回值：**
- `all_flows`：所有 reuse flow task
- `reuse_layer_deps`：`{layer_id: {rank: [flow_task_ids]}}`（跨 request 聚合，用于依赖注入）
- `request_task_map`：`{req_id_str: [flow_task_ids]}`

**关键设计：**

- **Flow deps = `[]`**：存储节点已有 KV cache 数据，无需等待任何前置 task。
- **Per-layer `layer_id`**：每个 flow 的 `layer_id` 设置为对应模型层编号，使 RLI = `max(layer_id - current_layer, 0)` 正确工作。
- **同一 request 的所有 flow 来自同一 storage node**：per-request 分配，round-robin 跨 request。

#### 16b: `_expand_batch()` 修改支持 reuse deps

新增可选参数 `reuse_layer_deps: dict[int, dict[int, list[int]]] | None = None`。

在每个 layer 的处理开始处注入依赖：

```python
for layer_id, layer_map in sorted(layers.items()):
    if reuse_layer_deps and layer_id in reuse_layer_deps:
        for rank, reuse_deps in reuse_layer_deps[layer_id].items():
            current_exits.setdefault(rank, []).extend(reuse_deps)
    # ... 原有 compute/flow 创建不变
```

**效果：** `compute[layer_N][rank_R]` 的依赖 = 上一层的 flow exits + layer N 的 reuse flows。

#### 16c: `expand()` 集成

1. 读取 trace 顶层 `stage1_kv_reuse`，计算 `storage_node_ids = range(total_gpus, total_gpus + num_storage_nodes)`
2. 从 trace `requests` 提取 per-request 的 `kv_reuse_hit_ratio` 和 `kv_reuse_storage_node_idx`
3. 对每个 prefill batch：调用 `_expand_kv_reuse()` → 传入 `_expand_batch()` 的 `reuse_layer_deps`
4. `batch_task_map` 新增 `kv_reuse_{bid}` entry

#### 16d: MFS context 分类

在 `_classify_task()` 中新增：

```python
if ct == CommType.KV_CACHE_REUSE:
    return MfsStage.EARLY, "kv_cache_reuse"
```

**RLI 优先级机制：**

- Layer 0 reuse flow: RLI = 0（最高优先级，compute 从 layer 0 开始就需要）
- Layer N reuse flow: RLI = max(N - current_layer, 0)
- 当 compute 推进到 layer N 时，RLI 降为 0，自动获得最高优先级

这意味着：
- 所有层的 reuse flows 在初始时就变为 ready（deps=[]）
- Allocator 根据当前 compute 进度动态调整各层 reuse flow 的优先级
- 即将需要的层获得高优先级，远期层暂时等待
- 这是 RLI 机制的自然工作方式，无需额外代码

#### 16e: Schema 扩展

在 `CommType` 枚举中新增 `KV_CACHE_REUSE = "kv_cache_reuse"`，位于 `KV_CACHE_TRANSFER` 之后。

---

## 文件清单

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `simai-flow-scheduler/src/workload_format/schema.py` | 新增 `CommType.KV_CACHE_REUSE` |
| `vidur-alibabacloud/vidur/trace_recorder.py` | 新增 `stage1_*` 配置参数，`record_request()` 输出 per-request reuse 元数据，`_build_trace()` 输出顶层配置 |
| `simai-flow-scheduler/src/workload_generator/inference_trace_expander.py` | 新增 `_expand_kv_reuse()` 方法，`_expand_batch()` 新增 `reuse_layer_deps` 参数，`expand()` 集成 Stage 1 流程 |
| `simai-flow-scheduler/src/static_analysis/passes/mfs_context.py` | `_classify_task()` 新增 `KV_CACHE_REUSE` → `MfsStage.EARLY` 分类 |
| `simai-flow-scheduler/tests/test_inference_trace_expander.py` | 新增 `TestStage1KvReuseFlows`（12 tests） |
| `simai-flow-scheduler/tests/test_mfs_context.py` | 新增 `test_kv_cache_reuse_classified_as_early` |

---

## 设计决策

1. **存储节点是普通拓扑节点（无 TP/EP/PP）：** 存储节点在网络拓扑中是独立节点，只有网络连接的概念。不像计算 replica 有 TP/EP/PP 的并行结构。Trace 只指定数量，expander 映射到 `range(total_gpus, total_gpus + num_storage_nodes)`。

2. **Per-layer 粒度生成 flow：** 每个 (request, layer, rank) 产生一个独立的 flow task。这使得 RLI 机制能为每层的 reuse flow 独立计算优先级。Layer 0 的 reuse flow RLI=0（最高优先级），Layer N 的 RLI=max(N - current_layer, 0)，随着 compute 推进自然升高。

3. **Per-request storage node 分配：** 同一 request 的所有层、所有 rank 的 reuse flow 都从同一 storage node 获取。这在 trace 的 request 元数据中指定（`kv_reuse_storage_node_idx`），逻辑索引由 expander 映射到实际拓扑节点。

4. **Reuse flow 无依赖（deps=[]）：** 存储节点上的 KV cache 数据是 pre-existing 的，不需要等待任何前置 task。所有层的 reuse flows 同时变为 ready，由 RLI 机制根据 compute 进度动态调整带宽分配优先级。

5. **依赖注入在 `_expand_batch()` 内部完成：** 通过 `reuse_layer_deps` 参数在每层开始处理时注入 reuse flow IDs 到 `current_exits`。这样 compute task 自然依赖 reuse flows，无需修改 `_expand_batch()` 的核心逻辑。向后兼容：`reuse_layer_deps=None` 时不注入任何依赖。

6. **`KV_CACHE_REUSE` 分类为 EARLY 而非 P2D：** Reuse flow 是 prefill 计算的前置通信（Stage 1），不是 prefill-to-decode 的 transfer（Stage 3）。作为 EARLY traffic，它参与 RLI 优先级队列，且与 collective flows 共享 EARLY 优先级空间。

---

## 与 Phase 1-4 的关系

| Phase | 内容 | Stage 1 涉及 |
|-------|------|-------------|
| Phase 1 | RMLQ + RLI 基础框架 | Reuse flow 作为 EARLY 自动获得 RLI 优先级 |
| Phase 2 | MLU + TTFT SLO | Reuse flow 提前完成有助于减小 P2D 的 MLU |
| Phase 3 | RED 量化分级 | Reuse flow 作为 EARLY RLI=0 参与 RED 分级 |
| Phase 4 | Feasibility safeguard | Reuse flow 时长纳入 critical path 计算 |
| **Phase 5** | **Stage 1 KV Reuse** | **新增 reuse flow 类型，集成到完整 MFS 流程** |

Stage 1 reuse flow 作为 EARLY traffic 自然融入 Phase 1-4 构建的调度框架，无需修改 allocator 或 policy 的核心逻辑。唯一的框架修改是 `_expand_batch()` 接受 `reuse_layer_deps` 参数用于依赖注入。

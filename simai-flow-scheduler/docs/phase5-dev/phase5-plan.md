# Phase 5: 混合训练+推理流量模拟

## 目标

在 simai-flow-scheduler 中支持推理流量模拟（含 PD 分离），并允许训练和推理 job 共享集群网络、竞争带宽资源。

## 核心挑战

训练流量是**静态的** —— 整个 DAG 预先生成，固定迭代次数和通信模式。推理流量有几个本质区别：

1. **请求动态到达**（Poisson、trace 等），batch 大小随时间变化
2. **通信量与 batch 组成有关** —— TP AllReduce 大小 = hidden_size × num_tokens × dtype_size
3. **PD 分离** —— P 节点做 prefill，D 节点做 decode，中间有 KV cache 的 P2P 传输
4. **多迭代循环** —— decode 阶段每个 token 都是一次完整前向传播

## 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 推理流量来源 | 从 Vidur 抓 trace，trace-driven replay | Vidur 已有完整的推理调度逻辑（请求生成、batch 调度、PD 分离），无需自建 |
| Compute/comm 时间获取 | CSV profiling 查表 | AICB 已有 per-layer profiling 数据（CSV），精度和灵活性的平衡 |
| Decode 展开粒度 | 完全展开（per-token DAG） | 最精确，充分利用 flow scheduler 的 P2P 建模能力 |
| Batch 间时序 | DAG 依赖驱动（非 Timer） | 网络拥塞导致的推迟会自然传播，无需修改 executor |

## 架构设计

### 核心方案：Trace-driven Replay

采用 **trace-driven replay** 方案，分三步：

1. **抓 trace**：在 Vidur 中以理想条件（独占网络、无拥塞）运行推理模拟，插桩捕获每个 batch 的调度决策和组成信息
2. **展开**：在 simai-flow-scheduler 中，用 InferenceTraceExpander 将 trace 展开为 P2P 任务 DAG
3. **Replay**：与训练 workload 合并，送入 AnalyticalExecutor 执行，网络拥塞效应由执行器自然模拟

**优点：**
- 最大限度复用现有代码，无需修改 executor
- Vidur 负责推理调度逻辑，simai-flow-scheduler 负责网络级模拟，职责清晰
- Trace 格式是 per-batch 粒度（不是 per-layer 或 per-token），文件紧凑
- 执行时的网络拥塞效果自然反映在执行结果中

**局限：**
- Trace 中记录的 batch 时序是 Vidur 理想条件下的结果，实际 replay 时由于网络拥塞，执行时间会不同
- 这意味着 replay 结果中的 batch 开始/结束时间与 Vidur 的不同，但**这正是我们想要的** —— 我们要观察拥塞对推理延迟的影响
- 不能模拟"根据执行进度动态调整 batch 调度"的自适应策略（需要修改 executor 支持动态任务注入，属于后续工作）

### 整体流程

```
Vidur (理想条件) ──→ 推理 trace JSON ──→ InferenceTraceExpander
                                              ↑
                                        InferenceProfileStore
                                        (per-layer CSV 数据)

训练 AICB ──→ AicbParser ──→ WorkloadBuilder ──→ training_workload

                              JobMerger.merge(training_wl, inference_wl)
                                              ↓
                                  merged P2PWorkload (含多 job)
                                              │
                                  ┌───────────┴───────────┐
                                  StaticAnalysis → TaskSerializer
                                  → AnalyticalExecutor
                                  → ExecutionResult
```

### Vidur 插桩与 Trace 格式

#### 插桩点

在 Vidur 的推理调度流程中插入事件捕获代码：

1. **BatchScheduler** —— 当一个 batch 形成时，记录 batch 组成（哪些 request、各多少 token）
2. **PD 分配决策** —— 记录 P/D node 分配和并行策略
3. **不记录执行时间** —— 因为 simai-flow-scheduler 会重新计算

#### Trace 格式设计

Per-batch 粒度，不记录 per-layer 或 per-token 细节。

> **注意：以下格式仅为初始参考设计，负责 Phase E 的开发 agent 应根据 Vidur 项目的实际调度逻辑和数据结构进行调整。**

```json
{
  "version": "1.0",
  "model": "deepseek-671b",
  "parallelism": {"tp": 8, "dp": 1, "pp": 1, "ep": 8},
  "requests": {
    "0": {"num_input_tokens": 128, "num_output_tokens": 64},
    "1": {"num_input_tokens": 256, "num_output_tokens": 32},
    "2": {"num_input_tokens": 64, "num_output_tokens": 128}
  },
  "batches": [
    {
      "batch_id": 0,
      "type": "prefill",
      "p_nodes": [0, 1, 2, 3, 4, 5, 6, 7],
      "request_ids": [0, 1],
      "total_prefill_tokens": 384,
      "kv_cache_bytes": 12345678
    },
    {
      "batch_id": 0,
      "type": "decode",
      "d_nodes": [8, 9, 10, 11, 12, 13, 14, 15],
      "request_ids": [0, 1],
      "num_iterations": 1,
      "kv_cache_bytes": 12345678,
      "depends_on": "batch_0_prefill"
    },
    {
      "batch_id": 1,
      "type": "decode",
      "d_nodes": [8, 9, 10, 11, 12, 13, 14, 15],
      "request_ids": [0, 1],
      "num_iterations": 1,
      "depends_on": "batch_0_decode"
    },
    {
      "batch_id": 2,
      "type": "prefill",
      "p_nodes": [0, 1, 2, 3, 4, 5, 6, 7],
      "request_ids": [2],
      "total_prefill_tokens": 64,
      "depends_on": "batch_0_prefill",
      "kv_cache_bytes": ...
    }
  ]
}
```

**格式要点：**
- `requests` 是全局字典，每个 request 的详情（token 数等）只记录一次。batch 中通过 `request_ids` 引用，同一 request 可出现在多个 batch 中（如 prefill batch 和后续 decode batch）
- 每个 decode batch 记录当前 iteration 包含的 `request_ids`，`num_iterations` 为该 batch 实际执行的 decode 步数（通常为 1，支持 per-token 调度场景下每个 decode step 作为独立 batch）
- `depends_on` 声明 batch 间的执行顺序，精确到具体 batch + type（如 `"batch_0_prefill"`），expander 据此连接 DAG 依赖
- `kv_cache_bytes` 由 Vidur 计算（= 2 × num_tokens × d_kv × num_layers × dtype_size）
- 不包含执行时间（由 simai-flow-scheduler 的 executor 重新计算）

### 新增组件

#### 1. InferenceProfileStore (`src/workload_generator/inference_profile.py`)

加载 AICB per-layer CSV profiling 数据，按 (model, phase, batch_size, seq_length) 查表获取 compute 时间和 comm 大小。

**CSV 格式** 已存在于：
- `aicb/scripts/inference_configs/`（模型配置）
- Vidur 的 aicb backend（per-layer profiling 数据：layer_id, layer_name, comp_time_ns, comm_size_bytes）

```python
@dataclass
class LayerProfile:
    layer_id: int
    layer_name: str       # "attention", "mlp", "moe"
    comp_time_us: int     # 计算时间（微秒）
    comm_size_bytes: int  # 通信数据量（字节）

class InferenceProfileStore:
    def load(self, csv_path: str) -> None: ...
    def get_profile(self, phase: str, batch_size: int, seq_length: int) -> list[LayerProfile]: ...
```

#### 2. InferenceTraceExpander (`src/workload_generator/inference_trace_expander.py`)

核心组件，将 trace 中的每个 batch 展开为 P2P 任务 DAG。

```python
class InferenceTraceExpander:
    def __init__(self, profile_store: InferenceProfileStore): ...

    def expand(self, trace: InferenceTrace) -> P2PWorkload:
        """将整个 trace 展开为 P2PWorkload。

        对每个 batch entry:
          1. prefill → 展开为 per-layer (compute + TP AllReduce P2P flows)
          2. kv_cache_transfer → P→D P2P flows
          3. decode → 展开 num_iterations 个 decode 迭代
        根据 depends_on 连接 batch 间的依赖。
        """
```

**展开逻辑：**

Prefill batch 展开为一个 forward pass（与训练的 forward 阶段结构相同）：
- Per-layer: COMPUTE(duration from profile) + N 个 P2P flow（TP AllReduce Ring 展开）
- MoE layer: 额外 EP AlltoAll P2P flows
- 复用现有 `CollectiveExpander` 进行集合通信展开

Decode 迭代展开（per-token 完全展开）：
- 每次迭代 = 所有 layer 的 compute + TP AllReduce（seq_length=1，通信量更小）
- `decode_iter[i+1]` 依赖 `decode_iter[i]` 的最后一个 task

KV cache 传输：
- P→D 直连 P2P flow，每个 TP rank pair 一个
- `kv_transfer_tasks` 依赖对应 prefill 的最后一个 task
- decode 迭代 `iter[0]` 依赖 `kv_transfer_tasks`

#### 3. 端到端脚本 (`scripts/run_mixed_e2e.py`)

编排训练 + 推理 workload 的合并逻辑直接放在脚本中（与现有 `scripts/run_e2e.py` 风格一致），不单独设计模块：

```python
# 1. 训练 workload（现有 pipeline）
header, items = AicbParser().parse(training_aicb_file)
training_wl = WorkloadBuilder().build_from_aicb(header, items, training_job)

# 2. 推理 workload（trace → P2P tasks）
trace = load_trace(inference_trace_file)
inference_wl = InferenceTraceExpander(profile_store).expand(trace)

# 3. 合并
merged_wl = JobMerger().merge([training_wl, inference_wl])

# 4. 现有 pipeline（不变）
analysis = WorkloadAnalyzer(topology).analyze(merged_wl)
plan = TaskSerializer().serialize(merged_wl, analysis)
result = AnalyticalExecutor(topology, analysis.routing_hints).execute(merged_wl, plan)
```

### Batch 间时序：依赖驱动

Batch 间的执行顺序完全由 DAG 依赖驱动，不使用 Timer Task：

```
prefill_0 ──→ kv_transfer_0 ──→ decode_0[0] → decode_0[1] → ... → decode_0[N]
                                     │
prefill_1 ──→ kv_transfer_1 ──→ decode_1[0] → decode_1[1] → ... → decode_1[M]
```

- 同一 P-node 上的 prefill 串行：`prefill_1` 依赖 `prefill_0` 的最后一个 task
- Decode 依赖对应的 KV transfer：`decode_batch_i[0]` 依赖 `kv_transfer_i` 的 task
- 同一 D-node 上的 decode 串行：`decode_iter[i+1]` 依赖 `decode_iter[i]`

**网络拥塞效果**：如果 prefill_0 因训练流量竞争而延迟，prefill_1 的开始时间自动推迟（通过依赖链传播），不需要任何额外机制。

### 现有组件修改

#### schema.py
- Phase 枚举：+PREFILL, +DECODE
- CommType 枚举：+KV_CACHE_TRANSFER

### Decode 完全展开

每个 output token 生成完整的 P2P DAG：
- Per-layer: COMPUTE(duration) + TP AllReduce Ring 展开的 P2P flows
- 60 层模型 tp=8: ~960 tasks/iteration
- 512 tokens → ~490K tasks（executor 可处理）

## 开发阶段

### Task A: Vidur 调研 + Trace 格式确定
**调研任务：**
- 深入阅读 Vidur 的 batch 调度流程，确认可获取的 per-batch 信息
- 确定插桩点和可捕获的数据字段
- 根据实际可用信息，确定最终的 trace JSON 格式
- 输出：一份 trace 格式规范文档

**说明：** 这一阶段需要先做，因为 trace 格式决定了后续 InferenceTraceExpander 的设计

### Task B: Schema 扩展
**修改文件：**
- `src/workload_format/schema.py` — Phase +PREFILL/DECODE, CommType +KV_CACHE_TRANSFER
- `src/workload_format/validator.py` — 更新 JSON schema（phase 枚举增加新值）
- `src/workload_format/writer.py` — 序列化新枚举值

### Task C: InferenceProfileStore
**新增文件：**
- `src/workload_generator/inference_profile.py` — CSV profiling 数据加载和查表

**说明：** 独立的数据组件，不依赖 executor 或现有 pipeline

### Task D: InferenceTraceExpander
**新增文件：**
- `src/workload_generator/inference_trace_expander.py`

**核心逻辑：**
- 读取 trace JSON，解析 batch 列表和依赖关系
- `expand_prefill_batch()` — 使用 ProfileStore 获取参数，调用 CollectiveExpander 生成 P2P 任务
- `expand_decode_iterations()` — per-token 完全展开，迭代间依赖链
- `expand_kv_cache_transfer()` — P→D P2P flow 任务
- 根据 `depends_on` 连接 batch 间的依赖

### Task E: Vidur 插桩 + Trace 抓取
**修改文件（Vidur 侧）：**
- 根据 Task A 确定的格式和插桩点，在 Vidur 中插入 trace 捕获逻辑
- 输出符合规范的 trace 文件

### Task F: 端到端集成脚本
**新增文件：**
- `scripts/run_mixed_e2e.py` — 端到端混合模拟脚本（编排训练+推理合并逻辑）

## 验证方式

每个阶段独立测试：
- **Task A:** 输出 trace 格式规范文档，并用 Vidur 跑小场景验证可获取的信息
- **Task B:** 验证新 Phase/CommType 枚举值的序列化和反序列化
- **Task C:** 单元测试 ProfileStore 的查表和插值
- **Task D:** 构造测试 trace JSON，验证展开后的 DAG 结构和依赖关系
- **Task E:** 用 Vidur 跑一个小推理场景，检查 trace 输出的正确性
- **Task F:** 端到端运行混合 workload，对比纯训练 vs 混合场景的训练 iteration 时间（应有增大）和推理 batch 延迟（反映网络拥塞影响）

## 调研参考

### AICB 推理流量生成
- **脚本：** `aicb/scripts/inference_workload_with_aiob.sh`
- **生成器：** `aicb/workload_generator/SimAI_inference_workload_generator.py`
- **模型配置：** `aicb/scripts/inference_configs/`（deepseek_default.json 等）
- **输出格式：** 与训练相同的 HYBRID_TRANSFORMER_FWD_IN_BCKWD，但 `mode: 1`，无 backward pass
- **支持模型：** DeepSeek-671B, Qwen3-MoE-235B, Qwen3-Next-80B

### Vidur-Alibabacloud 推理模拟
- **入口：** `vidur-alibabacloud/vidur/main.py`
- **请求生成：** `vidur-alibabacloud/vidur/request_generator/synthetic_request_generator.py`
- **PD 分离：** `vidur-alibabacloud/vidur/scheduler/global_scheduler/splitwise_global_scheduler.py`
- **KV cache 传输：** 通过 DummyLink 建模，大小 = 2 × num_tokens × mlp_hidden_dim × num_layers × dtype_size
- **执行时间预测：** `vidur-alibabacloud/vidur/execution_time_predictor/`，支持 aicb/simai_simulation/simai_analytical/vidur 多种 backend
- **TP 通信：** `vidur-alibabacloud/vidur/execution_time_predictor/communication_time_predictor.py`，AllReduce bytes = hidden_size × num_tokens × tensor_size

### 现有 simai-flow-scheduler 架构
- **Workload 格式：** `src/workload_format/schema.py`（P2PWorkload, Job, Task, Phase, CommType）
- **AICB 解析：** `src/workload_generator/aicb_parser.py`
- **Workload 构建：** `src/workload_generator/workload_builder.py`（build_from_aicb）
- **集合通信展开：** `src/workload_generator/collective_expander.py`（AllReduce, AllGather, ReduceScatter, AlltoAll）
- **多 Job 合并：** `src/workload_generator/job_merger.py`（ID 重映射，依赖更新）
- **分析执行器：** `src/executor/analytical.py`（事件驱动，FairShareAllocator）

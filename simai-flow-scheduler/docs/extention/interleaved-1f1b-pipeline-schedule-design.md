# Interleaved 1F1B Pipeline Schedule 扩展方案

> 状态：隔离式第一阶段已实现并验证（2026-07-23）。Default/Hermod/Puppeteer analyzer 和既有
> 实验入口已恢复到流水线扩展前版本；本策略只通过独立 builder、独立 analyzer/serializer 和
> `scripts/run_e2e_interleaved_1f1b.py` 使用。旧 overlay 已删除。
>
> 参考：[`1f1b-pipeline-schedule-design.md`](./1f1b-pipeline-schedule-design.md)。获批后，本文取代
> `advanced-pipeline-schedules-plan.md` 中 Interleaved 1F1B 的后续实施部分；旧记录仍保留作历史审计。
>
> 隔离调整：本文早期关于接入 Hermod static/dynamic runner 的段落仅保留为未执行的历史方案；
> 当前实现范围以第 13 节为准。

## 1. 背景与目标

现有 `interleaved_1f1b` 已经具备 compute serializer 和“生成普通 PP workload 后再删除、重建”
的 workload overlay。该实现能够构造可执行 DAG，但有两个结构性问题：

1. PP flow 先按普通单向 pipeline 生成，再由 overlay 推断消息大小、删除旧边并重建。
2. serializer 根据 `layer_id` 再次推断 model chunk，workload 展开与调度分析各自维护一套推导逻辑。

重新规划后的目标是：

- 在 AICB task 第一次展开时直接生成 VPP logical-stage PP flow，不再依赖“先生成再删除”。
- 精确保持 `compute/TP output -> PP activation/gradient -> receiver compute` 因果关系。
- model chunk、logical stage、boundary 等策略字段只进入 analyzer-owned sidecar。
- serializer 读取 sidecar 排序，不修改 `Task`、`P2PWorkload` 或公共 schema。
- 保持 GPipe、普通 1F1B、推理 workload 和默认 `WorkloadBuilder` 行为不变。

本文中的 Interleaved 1F1B 是 task-level Megatron 风格近似；不要求复刻 CUDA stream、异步
P2P buffer、显存生命周期或框架内部 schedule table 的每个实现细节。

---

## 2. 当前实现分析

### 2.1 当前执行链

```text
WorkloadBuilder
  -> 生成普通 stage0 -> stage1 -> ... -> stage{pp-1} PP flow
  -> InterleavedOneFOneBWorkloadOverlay
       删除原 PP flow
       按 layer_id 划分 chunk
       重建 pp * chunks - 1 个 logical boundary
  -> InterleavedOneFOneBSerializer
       再次按 layer_id 推断 chunk
       生成 preferred compute order
       通过全局 DAG 拓扑序合法化
```

### 2.2 可以保留的部分

- `WorkloadBuilder` 的 compute、TP/DP/EP collective 展开能力。
- `ItemTasks`、`FlowGroupResult.completion_index` 和 receiver-based completion 语义。
- `AdvancedPipelineSerializer` 的 DAG legalization 和 `ExecutionPlan` 校验。
- 迁移期间使用过 overlay 黄金测试对照 direct builder；迁移收口后旧测试已删除。
- static/dynamic runner 已有 pipeline 模式选择和 metadata 输出结构。

### 2.3 需要改进的部分

- PP 消息大小应直接来自 `AicbHeader.pp_comm_size`，不应从旧 PP task 反向推断。
- chunk ownership 应在 workload 展开阶段只计算一次。
- 通用 `_wire_forward_chain()` 会错误建立
  `chunk c/local last -> chunk c+1/local first` 本地边；真实 VPP 边界应经过 logical PP flow。
- workload 与 serializer 不应各自重新计算 `chunk_id`。
- dynamic task-ID 重映射时必须同步重映射 sidecar key。

---

## 3. 设计边界

### 3.1 不修改的通用类型

| 类型/模块 | 约束 |
|---|---|
| `Task` / `P2PWorkload` / JSON schema | 不增加 chunk、logical stage、slot 字段 |
| `TaskSerializer` | 不改变公共接口和默认校验语义 |
| `WorkloadBuilder` 默认实例 | GPipe/1F1B 展开结果保持不变 |
| `JobExpander` | 推理与普通训练展开保持不变 |
| executor policy / bandwidth allocator | 继续消费 DAG 和 `ExecutionPlan` |

### 3.2 策略专用实现

新增 `InterleavedPipelineWorkloadBuilder(WorkloadBuilder)`。它复用公共 task/collective
生成逻辑，只覆盖下列策略相关步骤：

- `_wire_dependencies()`：按 chunk 建立局部层链，避免错误跨 chunk 本地边。
- `_generate_pp_flows()`：直接生成 logical-stage activation/gradient flow。
- `_wire_pp_dependencies()`：连接 chunk boundary 的 producer 和 consumer。

默认 `WorkloadBuilder` 不接收新参数，也不增加 pipeline 分支。

---

## 4. Sidecar 设计

### 4.1 数据结构

建议在新文件 `src/static_analysis/strategies/interleaved_pipeline_strategy.py` 中定义：

```python
@dataclass
class InterleavedPipelineTaskInfo:
    task_id: int
    job_id: int
    task_role: str
    microbatch_id: int
    model_chunk_id: int
    physical_stage_id: int
    logical_stage_id: int
    logical_boundary_id: int | None
    peer_physical_stage_id: int | None
    peer_logical_stage_id: int | None
    direction: str
    preferred_slot: int | None = None
    final_local_order: int | None = None
```

其中：

- compute task：记录 chunk、physical/logical stage。
- PP flow：额外记录 boundary、peer stage、activation/gradient role。
- `preferred_slot` 和 `final_local_order` 由 serializer/analyzer 回填。

### 4.2 所有权与传递

`InterleavedOneFOneBAnalyzer` 持有：

```python
self.task_info: dict[int, InterleavedPipelineTaskInfo]
```

推荐调用链：

```text
analyzer 创建空 task_info
  -> InterleavedPipelineWorkloadBuilder 接收该 dict 的引用并写入 expansion metadata
  -> analyzer 调用 serializer，serializer 只读 sidecar 并返回 schedule metadata
  -> analyzer 把 preferred/final order 回填到自己持有的 task_info
  -> runner 输出 analyzer.task_info
```

dynamic 模式由 `PipelineJobExpander` 建立 `old_task_id -> global_task_id` 映射，同时重映射
task deps 和 sidecar key。不得把 sidecar 写入 `Task`。

---

## 5. PP Workload 直接展开方案

### 5.1 Chunk 划分

令：

- `local_layers = items_per_ga`
- `num_chunks = pipeline_vpp`

第一阶段只支持每个 chunk 至少一个本地 layer。划分规则采用连续、确定性分区：

```text
chunk(layer_index) = floor(layer_index * num_chunks / local_layers)
```

如果不能整除，前后 chunk 的层数最多相差 1。sidecar 记录最终映射，serializer 不再重新推导。

### 5.2 Logical stage

```text
logical_stage = chunk_id * pp + physical_stage
logical_stage_count = pp * num_chunks
```

对每个 micro-batch 和 `(dp, ep, tp)` lane：

```text
logical 0 -> logical 1 -> ... -> logical {pp*num_chunks-1}
```

物理边包括：

- chunk 内：`stage s/chunk c -> stage s+1/chunk c`
- chunk 间回绕：`stage pp-1/chunk c -> stage 0/chunk c+1`

### 5.3 Forward activation DAG

每个 logical boundary 建立：

```text
source chunk last-layer F compute
  -> source F collective completion（如果存在）
  -> PP_SEND activation
  -> destination chunk first-layer F compute
```

PP=1 且 logical boundary 位于同一 rank 时不创建网络 flow，直接建立 producer → consumer 依赖。

### 5.4 Backward gradient DAG

Backward 使用 forward logical chain 的完全逆序：

```text
destination chunk first-layer B compute
  -> destination B collective completion（如果存在）
  -> PP_SEND gradient
  -> source chunk last-layer B compute
```

每个 forward boundary 必须且只能有一个对应的 reverse gradient boundary。

### 5.5 局部依赖

`InterleavedPipelineWorkloadBuilder._wire_dependencies()` 只在同一 chunk 内调用：

- forward layer 正序链。
- backward-input layer 逆序链。
- 同 layer `B -> W`。
- 每个 micro-batch 自己的 `F -> B` bridge。

不得建立：

- chunk c 最后一层 F → 同 rank chunk c+1 第一层 F。
- chunk c+1 第一层 B → 同 rank chunk c 最后一层 B。

跨 chunk 关系只能来自 logical PP boundary。

### 5.6 Pre/Post 与 DP

第一阶段保留 AICB 的 pre/post task 和 DP collective，不改变其 task 类型与大小：

- pre task 仍在训练 token 前执行。
- post/optimizer 必须等待所有 micro-batch 的 W/DP completion。
- DP/optimizer 不参与 logical PP chain。

如果现有 AICB post task 为零时长，仍保留 DAG barrier，避免仅靠 compute order 隐式保证正确性。

---

## 6. Serializer 与 Analyzer 方案

### 6.1 Serializer 输入

修改现有 `InterleavedOneFOneBSerializer`，使其接收只读 sidecar：

```python
InterleavedOneFOneBSerializer(
    task_info: Mapping[int, InterleavedPipelineTaskInfo],
    interleave_group_size: int | None,
)
```

serializer 不再调用 `_chunk_by_layer()` 推断 chunk。

### 6.2 调度序列

每个 physical stage 使用 sidecar 中的 `(microbatch, chunk)` 生成：

1. stage-specific warmup。
2. `F -> B` steady state。
3. backward chunk 逆序 cooldown。

preferred sequence 只表达单 GPU compute resource 顺序；通信到达时机由 workload DAG 决定。
最终仍使用全局 stable topological legalization，任何 preferred order 与 DAG 冲突时以 DAG 为准。

### 6.3 Analyzer

新增 `InterleavedOneFOneBAnalyzer`，职责：

- 创建并持有 sidecar。
- 调用策略 workload builder。
- 调用 serializer。
- 校验 sidecar 覆盖所有训练 F/B/W compute 和新 PP flow。
- 输出 `ExecutionPlan` 和可审计 metadata。

Default/Puppeteer/Hermod 只消费同一份 workload 和 `ExecutionPlan`，不各自推导 chunk。

---

## 7. 文件修改计划

### 7.1 新增文件

| 文件 | 内容 |
|---|---|
| `src/workload_generator/interleaved_pipeline_builder.py` | chunk-aware 依赖和 PP flow 直接展开 |
| `src/static_analysis/strategies/interleaved_pipeline_strategy.py` | sidecar、analyzer、构建与分析编排 |
| `tests/test_interleaved_pipeline_builder.py` | workload DAG 黄金测试 |
| `tests/test_interleaved_pipeline_strategy.py` | serializer/analyzer/sidecar 测试 |

### 7.2 修改文件

| 文件 | 修改 |
|---|---|
| `src/static_analysis/passes/pipeline_task_serializers.py` | Interleaved serializer 改为读取 sidecar |
| `src/executor/pipeline_job_expander.py` | 选择策略 builder，并重映射动态 sidecar key |
| `scripts/run_hermod_e2e.py` | 通过 analyzer 构建 interleaved workload，不再调用 overlay |
| `scripts/run_hermod_dynamic_e2e.py` | static/dynamic 共用策略 state |
| package `__init__.py` | 导出新增策略类 |

### 7.3 已删除的旧实现

`pipeline_workload_overlay.py` 中的旧 Interleaved overlay 在 direct builder 完成静态、动态和
真实 AICB 验证后已删除。sidecar dataclass 已归位到 `interleaved_pipeline_builder.py`。

---

## 8. 实施阶段

### 阶段 I1：直接展开器

- 新增策略 builder 和 sidecar。
- 实现 chunk-aware local dependencies。
- 直接生成 activation/gradient logical PP flow。
- 不接入 runner。

批准后验收：builder 单测通过，默认 `WorkloadBuilder` 输出逐 task 不变。

### 阶段 I2：Serializer 对齐

- serializer 改为读取 sidecar。
- 建立 preferred slot 与 final local order。
- 对完整 DAG 执行 legalization。

批准后验收：固定 `pp/ga/chunks` golden schedule 与 DAG 均通过。

### 阶段 I3：Static/Dynamic 接入

- static runner 切换到策略 builder。
- dynamic task-ID 与 sidecar 同步重映射。
- Hermod metadata 使用 sidecar boundary，不根据 layer 二次猜测。

批准后验收：static/dynamic 同一输入生成同构 DAG 和一致 compute order。

### 阶段 I4：旧 overlay 收口

- [x] 直接展开完成 task/edge 验证。
- [x] static/dynamic runner 确认无 overlay 引用。
- [x] 删除 overlay 实现、工厂、导出和只验证旧实现的测试。

---

## 9. 验证方案

### 9.1 DAG 黄金测试

| 场景 | 验证 |
|---|---|
| PP=1, chunks=2 | 无网络 PP flow，存在直接 logical boundary |
| PP=2/4, chunks=2/4 | 每个 micro-batch、lane 有 `pp*chunks-1` 对双向边 |
| 多 TP lane | PP 发送依赖 sender 的 TP completion |
| 非连续 rank | src/dst 按 `assigned_nodes`，不按 rank 数值 |
| 不整除 layer/chunk | chunk 大小最多相差 1，边界唯一 |
| 多 micro-batch | 不发生跨 micro-batch 数据依赖 |
| pre/post/DP | optimizer 等待全部 W/DP completion |

### 9.2 Serializer 测试

- 每个 compute task 恰好出现一次。
- sidecar chunk 与 schedule token 完全一致。
- backward chunk 顺序与 forward logical chain 相反。
- 合并 DAG 与 per-GPU resource edge 后无环。
- 不支持的 chunk 配置显式失败，不回退。

### 9.3 回归与 E2E

- `tests/test_workload_builder.py` 全部通过，证明默认 PP 展开未变。
- static/dynamic 的 task 数、PP flow 数、边集合和 makespan 一致。
- 使用真实 AICB 验证无缺失依赖、无死锁。
- Chrome Trace 中应出现 chunk 间回绕通信和交错 F/B，而不是全 F → 全 B。

---

## 10. 兼容性、风险与回退

- 默认路径不实例化策略 builder，旧 GPipe/1F1B 不变。
- 推理 job 继续使用 `InferenceTraceExpander`。
- 直接展开器只支持 training AICB；其他输入显式拒绝。
- 最大风险是重写 local chunk edge 时遗漏 producer/consumer，当前通过 direct builder 逐边黄金
  测试和真实 AICB static/dynamic E2E 控制。

---

## 11. 批准点

I1、I2、I3、I4 均已完成。删除旧 overlay 未修改公共 schema，也未改变默认
`WorkloadBuilder` 行为。

---

## 13. 实施记录（2026-07-23）

已完成：

- 新增 `src/workload_generator/interleaved_pipeline_builder.py`，在首次 AICB 展开时直接生成
  `pp * vpp` logical-stage activation/gradient 边界。
- chunk wrap 与普通边界都使用
  `compute/TP completion -> PP_SEND -> receiver compute`，PP 大小直接取 header。
- 独立 analyzer 拥有并与 builder/serializer 共享 `dict[int, InterleavedPipelineTaskInfo]`；serializer 不再优先
  根据 `layer_id` 重猜 chunk。
- 新增隔离验证入口 `scripts/run_e2e_interleaved_1f1b.py`；不向现有实验或通用 analyzer 注册模式。
- 新增 `tests/test_interleaved_pipeline_builder.py`，覆盖每条 logical boundary、wrap 因果、
  非连续 rank、`pp=1` 本地 chunk 边界和 serializer。

独立 E2E 使用 GPT-7B AICB、`tp=2, pp=2, ga=8, vpp=2` 和 16-GPU AlibabaHPN topology：

- 生成 96 条 PP flow；
- 3,352 个 task 全部执行完成；
- makespan 为 4,197,756 us。

该 makespan 只证明执行闭环，不表示优于其他 pipeline。当前 schedule 是 Megatron 风格的
task-level 近似，尚不模拟框架内部通信 batch、kernel fusion 或 activation memory。

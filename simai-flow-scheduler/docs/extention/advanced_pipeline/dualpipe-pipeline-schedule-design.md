# DualPipe 流水线扩展设计

> 状态：核心实现与 E2E 已完成
>
> 日期：2026-07-24
>
> 目标模式名：`dualpipe`

本文规划在 `simai-flow-scheduler` 中新增独立的 DeepSeek-V3 DualPipe
流水线模式。实现将参考 DeepSeek-V3 技术报告和 DeepSeek-AI 公开的
DualPipe 代码，优先保证：

1. 双向流水线的模型副本放置和 micro-batch 归属正确；
2. forward、backward-input、backward-weight、PP/EP/TP/DP 通信之间的
   DAG 因果关系正确；
3. 通信参与者、方向、次数和字节数正确；
4. 官方八阶段 GPU 本地调度顺序尽可能精确；
5. 计算与通信的可重叠窗口能够被模拟器表达并审计；
6. 不向公共 `Task`、schema、executor 或 baseline analyzer 添加
   DualPipe 字段。

实现遵循本设计的隔离路径。当前完成的是 macro-operation/flow-DAG
精度的 DualPipe；DeepSeek 图 4 的逐 attention/dispatch/MLP/combine
strict profile 仍属于后续增强，详见第 14 节。

## 1. 实现范围

### 1.1 本阶段实现

计划实现的是原始 DualPipe，而不是 DualPipeV，也不是把现有 Chimera
模式重命名：

```text
现有 bidirectional -> 基础 Chimera，继续保留
新增 dualpipe      -> DeepSeek-V3 风格 DualPipe
```

目标能力包括：

- 每个物理 PP rank 保存两个镜像 model shard；
- 一半 micro-batch 从左端进入，另一半从右端进入；
- forward activation 和 backward gradient 沿各自 pipeline 的相反方向传输；
- backward 明确拆为输入梯度 `B` 和权重梯度 `W`；
- 按官方实现建立 deferred-W FIFO；
- 按官方实现生成八阶段本地 schedule；
- 对 `F&B` 配对建立显式 overlap sidecar 和可校准的重叠时间模型；
- 正确展开 PP、TP、EP、DP/副本梯度同步流量；
- 提供独立 analyzer、serializer、单元测试和 E2E 脚本。

### 1.2 不在本阶段承诺

以下内容不会被描述为“完全复现”：

- DeepSeek 内部 HAI-LLM runtime；
- CUDA stream、event、kernel launch 和逐 SM 调度；
- DeepEP 的 IB/NVLink 分层转发实现；
- token 级 MoE 路由不均衡和动态 expert capacity；
- FP8 dispatch/combine 的真实 kernel；
- DeepSeek 未公开的生产训练 profile 与自动参数；
- activation 显存生命周期和显存 OOM；
- DualPipeV；
- 将 DualPipe 注册进 Default、Puppeteer 或 Hermod。

这些限制不应影响本阶段的核心目标：DAG、网络流量、计算—通信先后关系
和主要重叠窗口必须可解释、可验证。

## 2. 参考实现中的关键语义

### 2.1 双向模型副本

设：

```text
P = 物理 PP rank 数
M = 一个 iteration 的 micro-batch 数，即 AICB GA
D = 外部 DP degree
E = EP degree
T = TP degree
L = D * E * T，每个 PP stage 的 lane 数
```

物理 rank `r` 保存两个模块：

```text
module replica 0 -> model shard r
module replica 1 -> model shard P - 1 - r
```

因此同一物理 rank 同时属于两条方向相反的完整 pipeline：

```text
pipeline 0 / down:
  physical 0 -> 1 -> ... -> P-1

pipeline 1 / up:
  physical P-1 -> ... -> 1 -> 0
```

前 `M/2` 个 micro-batch 交给 down pipeline，后 `M/2` 个交给 up
pipeline。每个 micro-batch 只经过其中一条完整 pipeline，不在中途换向。

### 2.2 官方实现的输入约束

DeepSeek-AI 公开实现要求：

```text
P 为偶数
M 为正偶数
M >= 2 * P
```

它不要求 `M % P == 0`。第一版实现计划遵循公开代码的约束，不先放宽，
避免在 schedule 尾部引入未经验证的特殊情况。

### 2.3 local phase 与全局方向

公开代码中的 `phase 0/1` 是本地调度 phase。后半段 rank 会交换它与
module replica 的映射：

```text
is_second_half = r >= P / 2
module_replica_id = local_phase XOR is_second_half
```

全局方向由交换后的 module replica 决定：

```text
module replica 0 -> down
module replica 1 -> up
```

sidecar 必须同时记录 `local_phase` 和 `module_replica_id`，不能只记录
一个模糊的 `pipeline_id`，否则无法逐 rank 对照官方 schedule。

## 3. 官方八阶段调度

对物理 rank `r` 定义：

```text
H = P / 2
h = min(r, P - 1 - r)
N = M / 2
```

官方 `dualpipe.py` 的八个阶段为：

| 阶段 | 重复次数 | 每次执行的操作 |
|---|---:|---|
| 1 | `2 * (H-h-1)` | `F0` |
| 2 | `h+1` | `F0, F1` |
| 3 | `H-h-1` | `B1(defer W), W, F1` |
| 4 | `N-P+h+1` | `F0&B1, F1&B0` |
| 5 | `H-h-1` | `B1(full), F1&B0` |
| 6 | `h+1` | `B1, B0`，中途切换到 defer-W |
| 7 | `H-h-1` | `W, B0(defer W)` |
| 8 | `h+1` | `W` |

其中：

- `F0` 表示 local phase 0 的 forward；
- `B1` 表示 local phase 1 的 backward；
- `F0&B1` 表示一对互相重叠的 forward 和 full backward；
- `B(full)` 包含输入梯度和权重梯度；
- `B(defer W)` 只推进输入梯度，把权重梯度放入 FIFO；
- 独立 `W` 从 FIFO 中按先入先出顺序取出一个权重梯度任务。

阶段 4 的第一次迭代在中间两个 rank 上有一个官方特例：为减小 bubble，
第一对 `F0` 和 `B1` 不做 `F&B` 联合执行，而是按公开代码给出的
send/recv 次序分开执行。该特例会进入 schedule token 和黄金测试，
不会被一般化规则覆盖。

### 3.1 共享 schedule token 生成器

builder 和 serializer 不应分别重新实现八阶段公式。计划在
DualPipe 专用模块中定义一个纯逻辑 token 生成器：

```python
@dataclass(frozen=True)
class DualPipeScheduleToken:
    physical_stage_id: int
    schedule_step: int
    loop_index: int
    local_phase: int
    operation: str
    microbatch_id: int
    recv_forward: bool
    recv_backward: bool
    send_forward: bool
    send_backward: bool
    defer_weight: bool
    overlap_pair_id: int | None
    middle_rank_special_case: bool
```

同一份 token：

- 供 workload builder 建立 schedule-sensitive DAG 和通信提交点；
- 供 `DualPipeSerializer` 生成同 GPU compute order；
- 供测试直接与官方 `P=8, M=20` schedule 对比；
- 写入 sidecar 供 trace 审计。

这样可避免“serializer 看起来像 DualPipe，但 workload DAG 仍是另一种
pipeline”的问题。

## 4. Workload 展开设计

### 4.1 独立 builder

新增：

```text
src/workload_generator/dualpipe_pipeline_builder.py
```

其中：

```python
class DualPipePipelineWorkloadBuilder(WorkloadBuilder):
    ...
```

builder 可以复用 `WorkloadBuilder` 的集合通信和 rank-group 辅助方法，
但会直接构造 DualPipe 的 layer/phase/PP DAG。不会先生成普通 PP 或
Chimera DAG，再用 overlay 删除和重接。

现有：

- `WorkloadBuilder`
- `InterleavedPipelineWorkloadBuilder`
- `ZeroBubblePipelineWorkloadBuilder`
- `BidirectionalPipelineWorkloadBuilder`

的默认行为保持不变。

### 4.2 strategy-specific sidecar

所有 DualPipe 字段保存在 analyzer-owned：

```python
dict[int, DualPipeTaskInfo]
```

计划字段如下：

```python
@dataclass(frozen=True)
class DualPipeTaskInfo:
    task_id: int
    job_id: int
    task_role: str
    component: str
    microbatch_id: int
    physical_stage_id: int
    model_shard_id: int
    module_replica_id: int
    local_phase: int
    direction: str
    schedule_step: int | None
    schedule_loop_index: int | None
    defer_weight: bool
    weight_queue_index: int | None
    overlap_pair_id: int | None
    overlap_role: str | None
    overlap_model: str
    logical_boundary_id: int | None
    peer_physical_stage_id: int | None
    preferred_slot: int | None
    final_local_order: int | None
```

不会修改公共 `Task` 或 `Phase`。`Phase.FORWARD`、
`Phase.BACKWARD_INPUT`、`Phase.BACKWARD_WEIGHT` 和现有 `CommType`
已经足够表达逻辑语义。

### 4.3 每个 micro-batch 的基础 DAG

对每个 pipeline、model shard 和 micro-batch，逻辑关系保持：

```text
forward:
  F_attention
    -> dispatch_F
    -> F_mlp
    -> combine_F
    -> next layer / PP activation

backward:
  incoming gradient
    -> B_attention_input / B_mlp_input
    -> previous layer / PP gradient

weight gradient:
  corresponding B_input
    -> W_attention / W_mlp
    -> replica+DP gradient synchronization
    -> optimizer
```

具体 attention/MLP 的前后顺序将按输入 AICB item 的语义建立，不能只依靠
task ID。反向 critical path 只能等待 `B`，不能误等 deferred `W`：

```text
B(layer n) -> B(layer n-1)
B(layer n) -> W(layer n)

禁止：
W(layer n) -> B(layer n-1)
```

### 4.4 PP activation 与 gradient

对每个 micro-batch、每个物理 PP boundary、每个 `(dp, ep, tp)` lane：

```text
down activation: p -> p+1
down gradient:   p+1 -> p

up activation:   p+1 -> p
up gradient:     p -> p+1
```

producer/flow/consumer 关系为：

```text
forward terminal
  -> PP activation flow
  -> receiver forward entry

backward-input terminal
  -> PP gradient flow
  -> receiver backward-input entry
```

PP gradient 不得依赖 W 或 gradient synchronization。

若每条 PP tensor 为 `S` bytes，则：

```text
PP flow 数 = 2 * M * (P - 1) * D * E * T
PP 网络字节数 = PP flow 数 * S
```

DualPipe 不会因为保存两份模型就把每个 micro-batch 的 PP 流量再乘 2；
两条 pipeline 各处理 `M/2` 个 micro-batch，总数仍为 `M`。

### 4.5 TP 和 EP 通信

DualPipe 改变 TP/EP collective 的时间和重叠对象，不应凭空改变输入定义的
tensor 字节数。

对于每个 AICB collective：

- participant group 继续由 `RankGrouper` 决定；
- `forward_comm` 属于 forward component；
- `backward_comm` 属于 backward-input component；
- 明确属于权重梯度的通信才进入 backward-weight；
- `EP_ALLTOALL` 的 dispatch/combine 角色通过 DualPipe sidecar 标注；
- 原 flow 的 `src/dst/size_bytes/chunk_id/deps` 继续由已有 collective
  expander 生成。

当前 `AlltoAllExpander` 中，`data_size=A` 表示每 rank 的总输入，
每个有向 peer flow 为 `floor(A/E)` bytes。对一个大小为 `E` 的 EP group：

```text
flow 数 = E * (E - 1)
网络字节数 = E * (E - 1) * floor(A / E)
```

测试会同时验证 flow 数和实际逐 flow 字节数，不只验证 aggregate。

#### AICB 信息不足时的处理

DeepSeek 的一个 chunk 明确包含：

```text
attention
all-to-all dispatch
MLP
all-to-all combine
```

但普通 AICB 行不一定能区分两次 All-to-All。实现不会把一次输入
All-to-All 静默复制成 dispatch 和 combine，因为这会错误增加流量。

计划提供 DualPipe 专用 component profile：

```text
dualpipe_component_profile.json
```

它只由 DualPipe builder/analyzer 使用，可补充：

- attention/MLP 的 F/B/W duration；
- dispatch/combine 的 collective 类型和 bytes；
- `F&B` 实测重叠时长；
- PP tensor bytes 覆盖。

运行模式：

```text
strict:
  需要能明确识别 dispatch/combine；缺失时失败

generic:
  完全保留 AICB 已有 collective，不合成缺失流；
  sidecar 标记 component_source=aicb_fallback
```

论文语义验证使用 `strict`，普通 GPT smoke 可以使用 `generic`，两者的
结果不能混为同一种精度。

### 4.6 两个 module replica 的梯度同步

同一 model shard `s` 位于：

```text
replica 0: physical stage s
replica 1: physical stage P - 1 - s
```

两条 pipeline 处理不同 micro-batch，因此 optimizer 前必须合并两个
replica 的梯度。公开示例也在 step 后把镜像 module 的梯度相加。

若外部 DP degree 为 `D`，每个
`(model_shard, ep_lane, tp_lane)` 的同步 group 为：

```text
R = 2 * D
```

第一版计划继续使用已有 Ring `DP_ALLREDUCE` 表达该同步。若单个 model
shard 的梯度输入为 `G_s` bytes：

```text
单 group Ring 网络字节数 = 2 * (R - 1) * G_s
```

所有 shard 的总量为：

```text
sum_s [E * T * 2 * (2D - 1) * G_s]
```

#### 避免重复 DP 流量

DualPipe builder 不会先保留普通 `grad_param_comm` 的 DP collective，
再额外附加一次 replica sync。它会把该语义直接映射成包含两个 model
replica 和外部 DP ranks 的 group。

也就是说：

```text
错误：
  普通 DP sync
    -> 再做 replica sync

计划实现：
  每个 model shard 一次 replica-aware DP sync
```

如果 AICB 同时包含语义不同的 ZeRO-1 reduce-scatter/optimizer-state
通信，必须先明确其 bytes 和 participant 语义；无法判定时显式拒绝，
不猜测。

### 4.7 pre/post 与 optimizer barrier

- pre task 在两条 pipeline 注入前完成；
- post/optimizer 不能只等待最后一个 B；
- optimizer 必须等待本 rank 保存的两个 model shard 的全部 W terminal；
- 若有 replica-aware DP sync，optimizer 等待本 rank 对应 collective 的
  completion；
- deferred-W FIFO 在 iteration 结束前必须清空。

最终 DAG 要满足：

```text
all local B complete
all deferred W drained
all replica-aware gradient sync complete
  -> optimizer/post
```

## 5. 计算和通信重叠模型

### 5.1 模拟器已经能自然表达的重叠

现有 analytical executor 把 GPU compute 和 network flow 作为不同资源。
只要 DAG 不添加多余 barrier，以下重叠能够自然出现：

```text
PP send      || 另一 micro-batch 的 compute
EP AlltoAll  || 另一方向的 compute
TP/DP flow   || 不依赖该 flow 的 compute
```

因此 builder 的关键职责是只添加真实依赖，不通过全局 fork/join
误杀可重叠窗口。

### 5.2 `F&B` 不能只靠 serializer 表达

当前一个 GPU 的 compute task 默认串行。`TaskSerializer` 可以决定：

```text
F -> B
```

却不能单独表达两个 kernel 同时占用不同 SM：

```text
F & B
```

如果简单把 F、B 顺序排列并保留原 duration，会把 `F&B` 计算成
`F+B`，低估 DualPipe 的重叠收益。反过来直接取 `max(F,B)` 又可能
高估真实硬件。

### 5.3 非侵入式 overlap envelope

计划用 DualPipe 专用 component DAG 和 sidecar 表达 overlap envelope，
不修改公共 executor：

1. token 生成器确定一个 `F&B` pair；
2. pair 的两个输入都 ready 后才能进入 overlap window；
3. builder 把 F/B 拆成 attention、MLP、B-input、W 和通信组件；
4. 网络组件保持真实 FLOW task 和真实 bytes；
5. compute component 仍使用普通 COMPUTE task；
6. 通过 profile 校准该 envelope 内 compute component 的有效 duration；
7. F 和 B 分别有独立 completion marker，下游只等待自己的真实完成点；
8. sidecar 记录 `overlap_pair_id`、角色和 profile 来源。

支持三种显式模型：

| 模型 | pair 时长 | 用途 |
|---|---|---|
| `profiled` | profile 给出的 `T_F&B` | 推荐的实验模式 |
| `ideal` | `max(T_F, T_B)` | 只用于上界分析 |
| `conservative` | `T_F + T_B` | 无 profile 时的安全回退 |

不会在没有记录的情况下默认假设完美重叠。E2E summary 和 metadata
必须写出当前 overlap model。

这种方法模拟的是 flow-DAG 可观察的 wall time 和 completion timing，
不是 CUDA SM 的逐周期并发。文档和实验结论会明确这一点。

### 5.4 DeepSeek 图 4 的 component 窗口

strict/profiled 模式计划至少表达下列相对次序：

```text
dispatch(F) || MLP(B-input)
dispatch(B) || MLP(W)
MLP(F)
barrier
combine(F)  || ATTN(B-input)
PP          || ATTN(W)
combine(B)  || ATTN(F)
```

Transformer block 边界在该窗口中并不对齐。实现会以 sidecar 的
`component` 和 `overlap_pair_id` 追踪跨 block 配对，不把同一个
`layer_id` 强行当作 F/B pair。

任何 network overlap 都必须同时满足：

- flow 的 producer 已完成；
- receiver/consumer 的数据依赖正确；
- schedule token 未要求通信 barrier；
- 链路和带宽分配器允许并发。

## 6. Serializer 与 analyzer

### 6.1 `DualPipeSerializer`

在高级流水线 serializer 模块中新增：

```python
class DualPipeSerializer(AdvancedPipelineSerializer):
    schedule_name = "dualpipe"
```

它继承 `TaskSerializer`，职责为：

- 消费共享八阶段 schedule token；
- 根据 sidecar 匹配 local phase、micro-batch、F/B/W 和 overlap pair；
- 生成每个 GPU 的 preferred compute order；
- 通过现有 DAG legalization 保证最终顺序合法；
- 把 `preferred_slot` 和 `final_local_order` 回写到 analyzer-owned
  sidecar。

serializer 不负责：

- 新增/删除网络 flow；
- 猜测 gradient bytes；
- 改写公共 Task 字段；
- 越过 workload DAG 强行执行未 ready 的 task。

### 6.2 `DualPipePipelineAnalyzer`

新增隔离 analyzer：

```python
class DualPipePipelineAnalyzer(_AdvancedPipelineAnalyzer):
    ...
```

它持有：

```text
dict[int, DualPipeTaskInfo]
dict[int, PipelineTaskInfo]
```

并组合：

- `DualPipeSerializer`
- 现有 BFS route pass
- metadata 导出

该 analyzer 不从 baseline strategy package 导出，不进入 Default、
Puppeteer 或 Hermod runner。

## 7. 计划修改的文件

### 7.1 新增文件

| 文件 | 作用 |
|---|---|
| `src/workload_generator/dualpipe_pipeline_builder.py` | 双副本 workload、八阶段 token、B/W FIFO、PP/EP/DP DAG 和 sidecar |
| `tests/test_dualpipe_pipeline_builder.py` | DAG、流量、端点、字节数、同步与输入约束 |
| `scripts/run_e2e_dualpipe.py` | 独立真实 AICB E2E 入口 |
| `docs/extention/dualpipe-pipeline-schedule-design.md` | 本设计文档 |

如 strict component profile 需要独立解析器，优先放在
`dualpipe_pipeline_builder.py` 或同目录的 DualPipe 专用模块中，
不修改通用 AICB schema。

### 7.2 修改文件

| 文件 | 计划改动 |
|---|---|
| `src/static_analysis/passes/pipeline_task_serializers.py` | 新增 `DualPipeSerializer` 和 `dualpipe` 高级模式 |
| `src/static_analysis/strategies/advanced_pipeline_strategies.py` | 新增隔离 analyzer |
| `src/executor/pipeline_job_expander.py` | 动态 job 选择 DualPipe builder，并重映射 sidecar task ID |
| `scripts/pipeline_e2e_common.py` | 接入独立模式、profile 参数和 DualPipe summary |
| `tests/test_pipeline_task_serializers.py` | 官方八阶段黄金序列与 DAG legalization |
| `docs/extention/pipeline-schedule-extensions-implementation.md` | 实现完成后加入第四种扩展 |
| `AGENTS.md`、`CLAUDE.md` | 实现并验证后更新进度，不提前宣称完成 |

### 7.3 明确不修改

| 文件/类型 | 原因 |
|---|---|
| `src/workload_format/schema.py` / `Task` / `CommType` | 现有字段足够，策略字段放 sidecar |
| `src/executor/analytical.py` | overlap 用策略专用 component DAG 表达 |
| Default/Puppeteer/Hermod analyzer | DualPipe 维持隔离 |
| 现有 baseline E2E 脚本和配置 | 不改变既有实验 |
| `bidirectional_pipeline_builder.py` | Chimera 继续作为独立策略 |

## 8. 分阶段实施顺序

得到批准后按以下顺序实施。

### 阶段 A：官方 schedule token

1. 实现输入约束和 rank/phase 映射；
2. 实现八阶段 token 生成；
3. 实现 micro-batch counter 和 deferred-W FIFO；
4. 用 `P=8, M=20` 对照官方序列；
5. 覆盖中间 rank 第一次 step-4 特例。

完成标准：每个 rank 上 F/B/W 数量、micro-batch ID、local phase 和
八阶段边界与官方代码一致。

### 阶段 B：双向 workload DAG

1. 直接展开两个镜像 module replica；
2. 建立每个 micro-batch 的 F/B/W DAG；
3. 建立 down/up PP activation 和 gradient；
4. 确保 gradient 由 B 而不是 W 触发；
5. 建立 optimizer/post barrier；
6. 通过 `P2PWorkload.validate()`。

完成标准：DAG 无环，producer/flow/consumer 闭包正确。

### 阶段 C：集合通信和流量

1. 保留 AICB TP/EP collective 的真实 group 和 bytes；
2. 支持 strict component profile 的 dispatch/combine；
3. 建立 replica-aware DP gradient sync；
4. 移除 DualPipe 路径中重复的普通 DP sync；
5. 校验非连续 rank 和多 DP/EP/TP lane。

完成标准：flow 数、每条端点、每条 bytes、aggregate bytes 与公式一致。

### 阶段 D：serializer 与重叠

1. 新增 `DualPipeSerializer`；
2. 消费共享 schedule token；
3. 建立 overlap pair sidecar；
4. 实现 profiled/ideal/conservative envelope；
5. 验证网络 flow 与 compute 的实际时间区间重叠；
6. 导出最终 local order 和 overlap metadata。

完成标准：最终 compute order 不破坏 DAG，profiled pair 的 wall time
和完成点符合输入 profile。

### 阶段 E：动态展开和 E2E

1. 接入 `PipelineJobExpander` 的独立 `dualpipe` 模式；
2. 验证 task-ID/deps/sidecar 全量 remap；
3. 新增 `run_e2e_dualpipe.py`；
4. 输出 workload、execution result、trace、metadata 和 summary；
5. 更新总览和仓库协作文档。

完成标准：真实 AICB 可以完成构图、路由、序列化和解析式执行闭环。

## 9. 验证方案

### 9.1 schedule 黄金测试

至少覆盖：

| P | M | 验证点 |
|---:|---:|---|
| 2 | 4 | 最小合法配置 |
| 4 | 8 | 多 rank 的八阶段边界 |
| 8 | 20 | 与官方 README/代码示例逐 rank 对照 |
| 4 | 10 | 偶数但不整除 P，验证不错误要求 `M % P == 0` |

拒绝：

- 奇数 P；
- 奇数 M；
- `M < 2P`；
- 非正 PP bytes；
- strict 模式缺失必要 component/bytes；
- 无法判定语义的 ZeRO/FSDP row。

### 9.2 DAG 测试

验证：

- 每个 micro-batch 只属于一个方向；
- down/up forward 路径正确；
- gradient 路径严格反向；
- PP flow 等待 sender 的 TP/EP completion；
- receiver compute 等待 PP flow；
- B 链不等待 W；
- 每个 W 恰好由对应 B 产生；
- deferred-W FIFO 顺序正确；
- optimizer 等待两个 local shard 的全部 W 和 gradient sync；
- 合并 compute resource edge 后仍无环。

### 9.3 通信量测试

分别统计：

- `PP_SEND` activation flows；
- `PP_SEND` gradient flows；
- `EP_ALLTOALL` dispatch flows；
- `EP_ALLTOALL` combine flows；
- TP collective；
- replica-aware `DP_ALLREDUCE`；
- 每种通信的 network bytes。

测试必须验证单条 flow 的：

```text
src
dst
size_bytes
comm_type
phase
microbatch
deps
```

而不是只比较总 task 数。

### 9.4 重叠测试

构造小型 profiled component DAG，检查：

- `F&B` 两侧输入未 ready 时 overlap window 不启动；
- dispatch/combine/PP flow 不早于 producer；
- flow 与指定 compute component 的时间区间确实相交；
- barrier 后 consumer 不会提前；
- `profiled` pair wall time 等于 profile；
- `ideal` 与 `conservative` 只在显式选择时启用；
- summary 记录 overlap model 和 profile source。

### 9.5 E2E 输出

`scripts/run_e2e_dualpipe.py` 计划输出：

```text
workload.json
execution_result.json
pipeline_task_metadata.json
summary.json
trace.json（可选）
```

summary 至少包含：

```text
total/compute/flow task count
PP activation/gradient flow count and bytes
EP dispatch/combine flow count and bytes
replica gradient-sync flow count and bytes
overlap pair count
overlap model/profile source
makespan
DAG validation result
```

E2E makespan 只作为回归信号，不直接当作 DualPipe 性能结论。

## 10. 与现有 Chimera 的区别

| 特性 | 现有 Chimera | 计划 DualPipe |
|---|---|---|
| 双向 pipeline | 是 | 是 |
| 每设备两个镜像 shard | 是 | 是 |
| micro-batch 从两端注入 | 是 | 是 |
| 本地 schedule | 两条普通 1F1B 交替 | 官方八阶段 |
| B/W 拆分与 deferred-W FIFO | 否 | 是 |
| `F&B` 配对 | 否 | 是 |
| attention/dispatch/MLP/combine component DAG | 否 | strict/profiled 模式支持 |
| 计算—通信重叠 profile | 否 | 是 |
| 副本梯度同步 | 是 | 是，但避免重复 DP |
| 模式名 | `bidirectional` | `dualpipe` |

因此不会复用 `bidirectional` 名称，也不会用一个参数把 Chimera
serializer 偷换成 DualPipe。

## 11. 风险和取舍

### 11.1 普通 AICB 缺少 DeepSeek component 粒度

处理方式：strict/profiled 模式要求策略专用 profile；generic 模式不合成
不存在的通信，并明确降级。

### 11.2 单 compute resource 无法逐 SM 并发

处理方式：用 profile-calibrated overlap envelope 模拟 flow-DAG 可观察的
wall time，不修改公共 executor，也不声称逐 SM 复现。

### 11.3 schedule edge 与数据 edge 可能冲突

处理方式：共享 token 生成器，builder 负责数据/通信 DAG，serializer
只提供 preferred order，最终通过稳定拓扑 legalization。

### 11.4 DP/ZeRO 语义可能重复

处理方式：DualPipe 路径直接构造 replica-aware group；无法判定输入语义时
失败，不把 PP bytes 或 compute time 当作 gradient bytes。

### 11.5 profile 结果可能被误解

处理方式：metadata 和 summary 强制记录 `profiled/ideal/conservative`，
不同模式的 makespan 不在没有说明时横向比较。

## 12. 批准点

批准后将按阶段 A 到 E 实施。默认设计决策为：

1. 新增独立模式 `dualpipe`，保留 Chimera；
2. 第一版遵循官方 `P` 偶数、`M` 偶数且 `M >= 2P`；
3. workload builder 直接展开 DualPipe DAG，不使用 overlay；
4. 策略字段全部放 analyzer-owned sidecar；
5. DP 梯度通信直接做 replica-aware group，避免重复流量；
6. `F&B` 使用显式 overlap model；无 profile 时不静默假设完美重叠；
7. 不接入 Default/Puppeteer/Hermod。

## 13. 参考资料

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- [DeepSeek-AI DualPipe 官方仓库](https://github.com/deepseek-ai/DualPipe)
- [官方 `dualpipe.py`](https://github.com/deepseek-ai/DualPipe/blob/main/dualpipe/dualpipe.py)
- [官方 DualPipe 示例](https://github.com/deepseek-ai/DualPipe/blob/main/examples/example_dualpipe.py)
- [DeepSeek 公开 training profile data](https://github.com/deepseek-ai/profile-data)
- [现有流水线扩展实现总览](./pipeline-schedule-extensions-implementation.md)
- [现有 Chimera 设计](./bidirectional-pipeline-schedule-design.md)
- [现有 Zero Bubble 设计](./zero-bubble-pipeline-schedule-design.md)

## 14. 实现结果（2026-07-24）

### 14.1 已完成

- 新增独立模式 `dualpipe`，未改变 `bidirectional` Chimera；
- 实现官方八阶段 schedule token；
- 实现后半 rank 的 local-phase/module-replica XOR 映射；
- 实现 deferred-W FIFO 和中间 rank 的 step-4 首次非重叠特例；
- 复用双向 PP 端点生成，但不使用 overlay；
- PP activation/gradient 保持 `producer -> flow -> consumer`；
- PP gradient 的依赖闭包不包含 W；
- `grad_param_comm` 不再先展开普通 DP，再追加副本同步；
- model replica 与外部 DP 合并为 replica-aware Ring AllReduce；
- 保留独立的 `grad_gather` pre-step 语义；
- 新增 `DualPipeSerializer`、`DualPipePipelineAnalyzer` 和动态 job remap；
- 新增 conservative/ideal/profiled macro overlap envelope；
- sidecar 记录原始/有效 compute duration、八阶段位置、FIFO 和 overlap pair；
- 新增 `scripts/run_e2e_dualpipe.py`。

### 14.2 当前重叠精度

当前实现以一个 F 与一个 full-B macro 为 overlap envelope：

```text
conservative = F + B
ideal        = max(F, B)
profiled     = overlap_factor * (F + B)
```

compute task 仍使用公共 schema。DualPipe builder 按 envelope 目标时长
重新分配 pair 内各 compute task 的有效 duration，同时保留各自 completion
点和全部真实 network flow。网络可以继续与不依赖它的计算重叠。

当前尚未解析单独的 component-profile JSON，也没有把一个 MoE chunk
进一步重建成 DeepSeek 图 4 的：

```text
attention -> dispatch -> MLP -> combine
```

因此：

- 官方八阶段宏观顺序、PP/TP/EP/DP flow DAG 和字节数已建模；
- `EP_ALLTOALL` 完全保留 AICB 输入的次数和 bytes；
- 不会把一次 All-to-All 擅自复制为 dispatch 与 combine；
- 图 4 的逐组件 SM 分配和 barrier 仍未声称复现。

### 14.3 验证结果

高级流水线定向回归：

```text
46 passed
```

排除工作区缺失的 Spectrum-X 外部 topology：

```text
774 passed, 3 skipped, 18 deselected
```

完整测试的其余 18 个错误仍全部来自缺失：

```text
astra-sim-alibabacloud/inputs/topo/
Spectrum-X_8g_8gps_400Gbps_H100
```

默认 GPT-7B、`tp=2, pp=2, dp=1, ep=1, ga=8` E2E：

| overlap model | tasks | PP flow | replica-sync flow | makespan |
|---|---:|---:|---:|---:|
| conservative | 3,304 | 32 | 16 | 3,938,319 us |
| ideal | 3,304 | 32 | 16 | 2,944,279 us |
| profiled (`overlap_factor=0.7`) | 3,304 | 32 | 16 | 3,221,018 us |

两种模式的 replica-gradient-sync network bytes 均为：

```text
42,748,346,368 bytes
```

`ideal` 只是完美 F&B overlap 上界，不是 DeepSeek-V3 实测性能。

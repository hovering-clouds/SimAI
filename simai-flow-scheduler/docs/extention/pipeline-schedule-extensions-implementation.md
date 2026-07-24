# 流水线调度扩展实现总览

本文面向希望理解、使用或继续扩展 `simai-flow-scheduler` 流水线调度能力的开发者，集中说明
以下四种高级训练流水线：

- Interleaved 1F1B
- Zero Bubble（当前为 ZB1P 风格）
- Chimera Bidirectional Pipeline（兼容模式名 `bidirectional`）
- DeepSeek-V3 DualPipe（模式名 `dualpipe`）

四种扩展均已具备独立 workload builder、`TaskSerializer` 子类、analyzer sidecar、单元测试
和端到端验证脚本。它们目前是隔离的研究路径，不会改变 GPipe、普通 1F1B、
Default、Puppeteer 或 Hermod 的既有行为。

详细设计分别见：

- [Interleaved 1F1B 设计](./interleaved-1f1b-pipeline-schedule-design.md)
- [Zero Bubble 设计](./zero-bubble-pipeline-schedule-design.md)
- [Chimera 设计](./bidirectional-pipeline-schedule-design.md)
- [DualPipe 设计与实现结果](./dualpipe-pipeline-schedule-design.md)
- [普通 1F1B 基础设计](./1f1b-pipeline-schedule-design.md)

## 1. 为什么不能只修改 compute 顺序

流水线策略的可观察结果由三部分共同决定：

1. workload DAG：决定数据什么时候真正 ready。
2. GPU compute order：决定同一 GPU 上 ready task 的执行次序。
3. 网络执行：决定 PP、TP、DP、EP flow 的传输和竞争。

如果只修改 compute order，而没有同步修改 PP producer、flow 和 receiver 的依赖，可能得到
一个可以执行但语义错误的模拟：

```text
错误近似：
  compute order 看起来像新流水线
  通信仍然沿用普通 PP DAG

当前实现：
  strategy-specific workload builder
    -> 精确生成 compute/collective/PP 数据依赖
  strategy-specific TaskSerializer
    -> 只决定同 GPU compute resource 顺序
  analytical executor
    -> 按 DAG、compute order 和网络带宽执行
```

本项目把 DAG 和计算—通信因果关系放在第一优先级。serializer 的偏好顺序如果与数据依赖冲突，
最终必须服从 DAG。

## 2. 共同架构

### 2.1 执行链

```text
AICB training workload
        |
        v
strategy-specific WorkloadBuilder
  - 复用通用 compute/TP/DP/EP 展开
  - 生成策略专属 PP flow
  - 建立 producer -> flow -> consumer
  - 写 expansion sidecar
        |
        v
P2PWorkload.validate()
        |
        v
strategy-specific Analyzer
  - 持有 dict[int, Info]
  - 调用 TaskSerializer 子类
  - 生成 BFS route table
        |
        v
ExecutionPlan.compute_order
        |
        v
AnalyticalExecutor + DefaultSchedulingPolicy
```

动态 job 路径使用 `PipelineJobExpander`。它在分配全局 task ID 后同时重映射：

- `Task.task_id`
- `Task.deps`
- sidecar 的 key
- sidecar 中的 `task_id`

### 2.2 非侵入式 sidecar

策略字段不进入公共 `Task`、schema 或 executor，而由 analyzer 维护：

```python
task_info: dict[int, StrategyTaskInfo]
```

典型字段包括：

```text
task_role
microbatch_id
model_chunk_id
pipeline_id
direction
physical_stage_id
logical_stage_id
preferred_slot
final_local_order
```

这样可以在不污染通用 IR 的情况下记录 VPP chunk、Zero Bubble 的 B/W 配对、Chimera 的
model replica 等信息。

### 2.3 Serializer 的职责边界

四种 serializer 都继承 `TaskSerializer`，最终只输出：

```python
ExecutionPlan(
    compute_order={
        node_id: [compute_task_id, ...],
    }
)
```

`AdvancedPipelineSerializer` 会把策略给出的 preferred order 与完整 task DAG 合并，执行一次
stable topological legalization。其结果满足：

- 所有 compute task 恰好出现一次；
- 原 workload DAG 不被破坏；
- 每个 GPU 的投影可以安全转化为 compute resource edge；
- flow 一旦 ready 会优先释放，不会被 compute preference 无谓阻塞。

## 3. Interleaved 1F1B

### 3.1 模型

设：

```text
P = physical PP stage 数
V = 每个 physical stage 的 virtual model chunk 数
M = microbatch/GA 数
L = dp * ep * tp lane 数
S = pp_comm_size
```

每个物理 stage 上的 AICB layer item 被切分为 `V` 个 chunk。逻辑流水线包含：

```text
P * V 个 logical stages

logical_stage = model_chunk_id * P + physical_stage_id
```

例如 `P=2,V=2`：

```text
chunk 0 / stage 0
  -> chunk 0 / stage 1
  -> chunk 1 / stage 0    # wrap communication
  -> chunk 1 / stage 1
```

builder 只在同一 chunk 内建立本地 layer chain；chunk 边界必须经过逻辑 PP 边，避免留下错误的
本地直连依赖。

### 3.2 通信 DAG

每个逻辑边界都有 activation 和反向 gradient：

```text
sender F/TP completion
  -> PP activation
  -> receiver first F

sender B/TP completion
  -> PP gradient
  -> receiver B
```

当 `P>1` 时：

```text
PP flow 数 = 2 * M * (P * V - 1) * L
PP payload = flow 数 * S
```

当 `P=1` 时不生成网络 PP flow，chunk 边界退化为同一 GPU 上的本地数据依赖。

TP、DP、EP collective 的总量不会因为 VPP 切分而重复；变化主要来自新增的 logical PP
boundary。

### 3.3 调度

`InterleavedOneFOneBSerializer` 根据 expansion sidecar 中的 chunk ownership 生成交错
F/B/W preferred order。serializer 不应再次根据 `layer_id` 猜测 chunk。

当前实现是 Megatron 风格的 task-level 近似，不模拟：

- CUDA stream 和异步 P2P buffer；
- activation memory 生命周期；
- kernel fusion；
- 框架内部完整 schedule table。

## 4. Zero Bubble

### 4.1 模型

当前实现目标是 ZB1P 风格。其核心不是改变 PP 路径或通信字节数，而是把 backward 明确拆成：

```text
B = BACKWARD_INPUT
W = BACKWARD_WEIGHT
```

其中 B 位于 pipeline gradient 的关键路径，W 可以延迟到合适的空闲位置。

### 4.2 DAG

每个 microbatch/layer 保持：

```text
F -> B -> W
```

但不同 layer 的 backward input 链不能等待 W：

```text
B(layer n) -> B(layer n-1)
B(layer n) -> W(layer n)

不存在：
W(layer n) -> B(layer n-1)
```

反向 PP gradient 只能由 B 或其 TP completion 触发，依赖闭包不得包含 W 或 DP：

```text
B/TP completion
  -> PP gradient
  -> previous stage B
```

optimizer/post entry 则必须等待所有 microbatch、所有 layer 的：

- W compute；或
- W 后的 DP collective completion。

### 4.3 通信量

Zero Bubble 不改变普通 PP 的参与者和 tensor 大小：

```text
PP flow 数 = 2 * M * (P - 1) * L
PP payload = flow 数 * S
```

已有 TP/DP collective 的数据量也保持不变。Zero Bubble 改变的是 B、W 和通信发生的时间及其
关键路径关系，而不是模型参数或 activation 的字节数。

### 4.4 调度边界

`ZeroBubbleSerializer` 使用 sidecar 中的 B/W 配对生成 ZB1P 风格 preferred order，并通过
DAG legalization 得到最终顺序。

当前未实现：

- 论文中的自动 schedule search；
- duration-aware 最优排程；
- B/F fused kernel；
- 运行时根据网络完成时间动态改写 schedule；
- 严格保证任意 profile 下 bubble 为零。

注意：本策略与 DeepSpeed ZeRO/FSDP 是不同概念。检测到 ZeRO/FSDP 专用 AICB 行格式时会
显式拒绝，避免走错展开路径。

## 5. Chimera Bidirectional Pipeline

### 5.1 名称

代码和脚本继续使用兼容模式名：

```text
bidirectional
```

但当前语义明确是基础 Chimera，不是 DeepSeek DualPipe。它不实现 DualPipe 八阶段调度、
F&B fused overlap 或 MoE 内部的计算通信重排。

### 5.2 双模型副本放置

设物理 PP 深度为偶数 `P`。物理 stage `p` 保存两个镜像 stage replica：

```text
module replica 0 -> model shard p
module replica 1 -> model shard P-1-p
```

偶数个 microbatch 均分为：

```text
前半 microbatch -> down pipeline
后半 microbatch -> up pipeline
```

路径为：

```text
down activation: physical 0 -> 1 -> ... -> P-1
down gradient:   physical P-1 -> ... -> 1 -> 0

up activation:   physical P-1 -> ... -> 1 -> 0
up gradient:     physical 0 -> 1 -> ... -> P-1
```

每个 microbatch 仍只穿过一条完整的 `P`-stage pipeline，因此 PP 总量与普通流水线相同：

```text
PP flow 数 = 2 * M * (P - 1) * L
PP payload = flow 数 * S
```

Chimera 改变的是 flow 方向、发生时序及链路竞争。

### 5.3 镜像副本梯度同步

两个 pipeline replica 处理不同 microbatch，但必须保持同一模型。因此 model shard `s` 的
两个副本需要在 optimizer 前同步：

```text
replica 0: physical stage s
replica 1: physical stage P-1-s
```

如果外部 DP degree 为 `D`，每个 `(model_shard, ep_lane, tp_lane)` 的同步组大小为：

```text
R = 2 * D
```

当前使用现有 Ring `DP_ALLREDUCE` 展开。若单个 stage gradient 输入大小为 `G`，当前 Ring
flow 模型的总网络字节数为：

```text
单个同步组网络字节数 = 2 * (R - 1) * G
全部同步组网络字节数 = P * ep * tp * 2 * (2D - 1) * G
```

collective 的首轮 flow 等待对应 replica 的所有 W/DP terminal，post/optimizer task 等待本 rank
参与的两个 replica-gradient collective 完成。

`G` 不会从 PP tensor 或计算时长猜测：

1. 优先由 `gradient_sync_bytes` 显式传入；
2. 否则读取语义明确且为正的 `grad_param_comm.dp_comm_size`；
3. 两者都不存在时显式失败。

### 5.4 调度

两条 pipeline 各自构建普通同步 1F1B：

```text
warmup -> 1F1B steady state -> cooldown
```

然后在每个物理 GPU 上交替合并：

- 前半 physical stage 优先 down；
- 后半 physical stage 优先 up；
- 完整 DAG legalization 决定最终合法顺序。

当前是 unit-operation preferred merge，尚未实现 Chimera 论文中的 eager-sync、
eager-sync-opt 或根据异构 F/B duration 自动搜索最优 schedule。

## 6. DeepSeek-V3 DualPipe

DualPipe 是独立的 `dualpipe` 模式，不替换 Chimera。它保持两个镜像
module replica，并按照 DeepSeek-AI 公开实现生成八阶段本地序列：

```text
nF0
-> nF0F1
-> nB1W1F1
-> nF0B1F1B0
-> nB1F1B0
-> nB1B0
-> nWB0
-> nW
```

builder 直接生成 down/up PP DAG，backward-input 不等待 deferred W。
`grad_param_comm` 被转换为包含两个 module replica 和外部 DP ranks 的
replica-aware Ring AllReduce，避免普通 DP 加副本同步的重复流量。

`F&B` 使用显式 overlap model：

```text
conservative: F+B
ideal:        max(F,B)
profiled:     overlap_factor*(F+B)
```

该 envelope 校准 flow-DAG 可观察的 wall time，不模拟逐 SM 并发。
EP/TP flow 的参与者、次数和 bytes 完全保留 AICB 输入；普通 AICB
无法区分 dispatch/combine 时不会合成额外 All-to-All。

## 7. 文件结构

| 文件 | 职责 |
|---|---|
| `src/workload_generator/interleaved_pipeline_builder.py` | VPP chunk、logical PP boundary 和 Interleaved sidecar |
| `src/workload_generator/zero_bubble_pipeline_builder.py` | B/W/DP/optimizer DAG 和 Zero Bubble sidecar |
| `src/workload_generator/bidirectional_pipeline_builder.py` | Chimera 双向 PP、镜像 replica 和 gradient sync |
| `src/workload_generator/dualpipe_pipeline_builder.py` | DualPipe 八阶段 token、B/W FIFO、双向 DAG、overlap sidecar 和 replica sync |
| `src/static_analysis/passes/pipeline_task_serializers.py` | 四种 `TaskSerializer` 子类及 DAG legalization |
| `src/static_analysis/strategies/advanced_pipeline_strategies.py` | 隔离 analyzer 和 sidecar ownership |
| `src/executor/pipeline_job_expander.py` | 动态 job 的策略 builder 选择与 task-ID remap |
| `scripts/pipeline_e2e_common.py` | 四种隔离 E2E 的公共执行链和 summary |

四个策略没有注册到通用 analyzer package，也没有加入 Hermod 的 `gpipe/1f1b` CLI。

## 8. 使用方式

在 `simai-flow-scheduler/` 下运行：

```bash
python scripts/run_e2e_interleaved_1f1b.py
python scripts/run_e2e_zero_bubble.py
python scripts/run_e2e_bidirectional.py
python scripts/run_e2e_dualpipe.py
```

指定输入：

```bash
python scripts/run_e2e_interleaved_1f1b.py \
  --aicb <training-workload.txt> \
  --topo <topology-file> \
  --vpp 2 \
  --output outputs/interleaved_e2e

python scripts/run_e2e_zero_bubble.py \
  --aicb <training-workload.txt> \
  --topo <topology-file> \
  --output outputs/zero_bubble_e2e

python scripts/run_e2e_bidirectional.py \
  --aicb <training-workload.txt> \
  --topo <topology-file> \
  --output outputs/chimera_e2e

python scripts/run_e2e_dualpipe.py \
  --overlap-model conservative \
  --output outputs/dualpipe_e2e
```

每个输出目录包含：

```text
workload.json
execution_result.json
pipeline_task_metadata.json
summary.json
```

其中 `pipeline_task_metadata.json` 同时保存 expansion sidecar 和 serializer schedule sidecar，
可用于审计：

- task 属于哪个 microbatch/chunk/replica；
- physical/logical stage 映射；
- PP flow 的 boundary 和方向；
- preferred slot 与最终本地顺序。

## 9. 输入约束

| 策略 | 主要约束 |
|---|---|
| Interleaved 1F1B | training AICB；`vpp >= 2`；每个 GA 至少有一个 layer/chunk；`pp>1` 时 PP size 为正 |
| Zero Bubble | 普通 training AICB；`ga >= 1`；存在独立 B/W；不接受 DeepSpeed ZeRO/FSDP 专用行 |
| Chimera | 偶数 `pp`；偶数 `ga`；`pp_comm_size > 0`；存在明确的 stage gradient bytes |
| DualPipe | 偶数 `pp`；偶数 `ga`；`ga >= 2*pp`；正 PP bytes；明确的 gradient bytes |

四个策略当前都只支持 training AICB direct expansion。推理 workload 继续使用
`InferenceTraceExpander`。

## 10. 测试与验证

定向测试：

```bash
python -m pytest \
  tests/test_interleaved_pipeline_builder.py \
  tests/test_zero_bubble_pipeline_builder.py \
  tests/test_bidirectional_pipeline_builder.py \
  tests/test_dualpipe_pipeline_builder.py \
  tests/test_pipeline_task_serializers.py -q
```

当前结果：

```text
46 passed
```

排除工作区缺失的外部 Spectrum-X topology 集成用例后：

```bash
python -m pytest -q -k "not SpectrumX"
```

当前结果：

```text
774 passed, 3 skipped, 18 deselected
```

默认 GPT-7B AICB smoke run：

| 策略 | 完成任务数 | makespan |
|---|---:|---:|
| Interleaved 1F1B | 3,352 | 4,197,756 us |
| Zero Bubble | 3,288 | 4,272,394 us |
| Chimera | 3,304 | 4,219,863 us |
| DualPipe conservative | 3,304 | 3,938,319 us |
| DualPipe ideal | 3,304 | 2,944,279 us |
| DualPipe profiled (`factor=0.7`) | 3,304 | 3,221,018 us |

Chimera smoke 中包含：

```text
32 条 PP flow
16 条镜像 gradient-sync flow
42,748,346,368 bytes gradient-sync network traffic
```

这些数字只证明当前输入能够完成 DAG 校验、路由、调度和执行闭环。不同策略的 workload
语义和附加通信并不完全相同，因此不能把 smoke makespan 直接解释为论文性能收益。

## 11. 如何继续扩展

新增流水线策略时建议遵循：

1. 先确定模型数据依赖和实际通信参与者。
2. 新建策略专属 `WorkloadBuilder` 子类。
3. 使用 analyzer-owned `dict[int, Info]` 保存策略字段。
4. 直接生成正确 flow，不先生成错误 flow 再 overlay 删除。
5. 新建 `TaskSerializer` 子类，只表达 GPU compute resource 顺序。
6. 为 flow 数、端点、字节数、producer、consumer 和依赖闭包编写黄金测试。
7. 验证非连续 rank、`pp=1/2` 边界、多 TP/DP lane 和动态 task-ID remap。
8. 最后运行真实 AICB E2E，并把 makespan 仅作为回归信号。

不要通过修改公共 `Task` 增加 `chunk_id`、`pipeline_id`、`replica_id` 等策略字段，也不要为了
接入新策略改变 Default、Puppeteer、Hermod 或普通 `WorkloadBuilder` 的默认行为。

## 12. 参考资料

- Megatron-LM pipeline parallelism：
  https://arxiv.org/abs/2104.04473
- Zero Bubble Pipeline Parallelism：
  https://arxiv.org/abs/2401.10241
- Chimera：
  https://arxiv.org/abs/2107.06925
- Chimera 官方实现：
  https://github.com/shigangli/chimera
- DeepSeek-V3 Technical Report：
  https://arxiv.org/abs/2412.19437
- DeepSeek-AI DualPipe：
  https://github.com/deepseek-ai/DualPipe

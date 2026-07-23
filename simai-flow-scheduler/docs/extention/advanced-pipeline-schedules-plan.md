# 高级流水线调度阶段性实施计划

## 1. 目标与实现边界

本轮在不修改 `Task`、`P2PWorkload`、workload schema 和 executor 资源模型的前提下，
新增三类训练 compute schedule：

- `interleaved_1f1b`：Megatron 风格的虚拟流水段（VPP）交错 1F1B。
- `zero_bubble`：ZB1P 风格的 `F/B/W` 拆分调度，其中 `B` 对应
  `BACKWARD_INPUT`，`W` 对应 `BACKWARD_WEIGHT`。
- `bidirectional`：DualPipe/Chimera 风格的双向流水线本地执行序列，把 micro-batch
  分给两个相反方向的逻辑 pipeline 后合并到同一 GPU 时间线。

实现只通过 `ExecutionPlan.compute_order` 约束每个 GPU 上的 compute task 顺序。
策略专用信息放在 sidecar：`dict[int, PipelineTaskInfo]`，key 为 `task_id`；不会向基础
`Task` 增加 VPP chunk、方向或 slot 字段。

需要区分“compute schedule 支持”和“训练 runtime 完整复现”：

- AICB builder 在每个物理 PP stage 上生成相同的 layer task，且原始 `PP_SEND` 按单向
  stage 链建立。高级策略通过深拷贝后的 workload overlay 重建 interleaved/bidirectional
  PP DAG，但 chunk/module ownership 仍是从 AICB 局部序列推导的投影。
- `zero_bubble` 可以利用现有独立的 `BACKWARD_INPUT`/`BACKWARD_WEIGHT` task 延后 W；
  optimizer post-validation、动态 profile/search 和通信计算重叠不在本轮范围内，因此
  “zero bubble”是 ZB1P compute-order 模式名，不保证任意 workload 的实测 bubble 严格为零。
- 双向模式同时表达 DualPipe 的两方向 GPU 本地合并顺序和镜像 PP DAG；serializer 本身不改写
  `PP_SEND src/dst`，static/dynamic runner 会先应用 strategy-owned overlay。它精确到 task-level
  compute—PP communication 因果关系，但不复刻 kernel overlap、异步 buffer 和双 module 参数生命周期。

这些限制会写入 API docstring、sidecar metadata 和测试，不把近似模式表述成论文完整复现。

## 2. 参考语义

### 2.1 Interleaved 1F1B

采用虚拟流水并行度 `vpp >= 2`。每个 GPU 的局部 layer 按连续区间划为 `vpp` 个
model chunk；schedule table 按 micro-batch group 构造：组内先遍历 chunk，再遍历
micro-batch。物理 stage `s` 的 warmup micro-step 数为：

```text
min(ga * vpp, 2 * (pp - s - 1) + (vpp - 1) * group_size)
```

之后进入 1F1B steady state，最后排空剩余 backward。Backward 以 chunk 逆序执行。

### 2.2 Zero Bubble

采用内存占用与 1F1B 同阶的 ZB1P 风格序列。关键约束是：

```text
F(m) -> B(m) -> W(m)
B(m, layer) -> B(m, previous_layer)
```

`B` 优先沿 pipeline 传播输入梯度，`W` 在合法的后续 slot 中延迟执行。serializer 不创建
新 task，也不删除原 DAG 的 `B -> W` 依赖。

### 2.3 Bidirectional Pipeline

要求 `pp` 和 GA/micro-batch 数为偶数，且 GA 至少为 `2 * pp`。micro-batch 平分到 down
和 up 两条逻辑 pipeline；up pipeline 使用镜像 stage 编号。GPU 本地顺序按 DualPipe 的
warmup、双向 steady state、cooldown 八段结构生成，并在可延迟的 backward 上拆分 B/W。
不满足前置条件时显式抛出 `ValueError`，不静默回退到 1F1B。

## 3. 非侵入式结构

新增 `src/static_analysis/passes/pipeline_task_serializers.py`：

```text
PipelineTaskInfo
  task_id, job_id, stage_id, logical_stage_id
  microbatch_id, model_chunk_id, pipeline_id, direction
  op(F/B/W/PRE/POST), schedule_slot, schedule_name

PipelineScheduleResult
  execution_plan: ExecutionPlan
  task_info: dict[int, PipelineTaskInfo]

AdvancedPipelineSerializer(TaskSerializer)
  公共任务分组、GA/chunk 推断、sidecar 写入、DAG 校验

InterleavedOneFOneBSerializer
ZeroBubbleSerializer
BidirectionalPipelineSerializer
```

保持 `TaskSerializer.serialize(workload) -> ExecutionPlan` 接口兼容；每个新 serializer 在
实例的 `task_info` 属性保留 sidecar，并额外提供 `serialize_with_metadata()` 返回组合结果。

新增集中式 factory，统一注册名称和参数。Default、Puppeteer、Hermod 只调用 factory，
避免在各 analyzer 复制 `if/elif` 和 stage 推导逻辑。

## 4. 阶段计划

### 阶段 A：调度内核与 sidecar

- 新增公共 serializer 基类和 sidecar dataclass。
- 按 job/task 推导 stage，避免用单个全局 `pp` 覆盖多 job 配置。
- 实现 pre/post、推理 task 和 `pp=1` 的稳定兼容行为。
- 所有输出运行 `TaskSerializer.validate()`，发现合并 DAG 成环立即失败。

完成标准：三个 serializer 均能对纯 compute synthetic workload 生成完整、无重复、无遗漏的
`compute_order`，且 sidecar 与 task 一一对应。

### 阶段 B：三种策略

- Interleaved 1F1B：VPP chunk、grouped schedule table、stage-specific warmup/steady/cooldown。
- Zero Bubble：B/W 拆分、W 延迟队列、ZB1P stage sequence。
- Bidirectional：down/up micro-batch 划分、镜像 logical stage、DualPipe 八段序列与参数校验。

完成标准：以固定 `pp/ga/vpp` 的 golden sequence 测试验证每个 stage 的操作序列；Zero Bubble
额外验证每个 W 都在同 micro-batch 的 B 之后。

### 阶段 C：analyzer 与入口接入

- factory 支持 `gpipe`、`1f1b` 和三个新名称。
- Default、Puppeteer、Hermod 共用同一 serializer 选择逻辑。
- Hermod static/dynamic runner 增加 pipeline choices，以及 `vpp`、interleave group 参数。
- metadata 输出中保留 schedule sidecar，便于 trace 审计。

完成标准：同一 pipeline 配置下 Default/Puppeteer/Hermod 的 compute order 一致；旧配置不变。

### 阶段 D：验证与文档收口

- 单元测试：边界参数、非连续 node、多个 job、DAG cycle、sidecar 内容、序列 golden cases。
- 集成测试：现有 1F1B dynamic 测试和 Hermod 配置解析回归。
- 运行相关测试后运行完整 `pytest tests -v`；运行 Ruff 检查改动文件。
- 更新 Hermod implementation 文档，明确新模式的输入条件与复现边界。

## 5. 兼容性与回退规则

- `gpipe` 继续使用 `CppReferenceSerializer`；`1f1b` 继续使用现有
  `OneFOneBSerializer`，不改变其排序结果。
- 不修改调用者传入的 workload、基础 schema、executor policy 和 bandwidth allocator。
  需要更精确 DAG 的策略通过深拷贝生成 strategy-owned workload overlay，
  仅在该副本中替换 task/deps。
- 不满足新策略前置条件时报告具体配置错误；不自动选择另一策略。
- 推理 `PREFILL/DECODE` task 保持 task-id 顺序，不套用训练 pipeline 规则。
- sidecar 是 analyzer/serializer 所有，不写回 `Task`，也不影响其他策略。

## 6. 后续阶段（不计入本轮完成声明）

- strategy-specific workload overlay：为普通 AICB 生成真实的 VPP chunk ownership、chunk 间
  PP flow 和双向 `PP_SEND`，仍通过复制/overlay 而不修改通用 schema。
- 基于 task duration、通信延迟和显存上限的 Zero Bubble 自动 schedule search。
- optimizer post-validation、F/B 通信计算重叠和 activation memory 生命周期模型。

参考实现与论文：

- [Megatron-LM pipeline schedules](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/pipeline_parallel/schedules.py)
- [Zero Bubble Pipeline Parallelism](https://arxiv.org/abs/2401.10241)
- [Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines](https://arxiv.org/abs/2107.06925)
- [DeepSeek DualPipe reference implementation](https://github.com/deepseek-ai/DualPipe)

## 7. 实施记录（2026-07-22）

- [x] 阶段 A：新增独立 serializer 模块、`PipelineTaskInfo` sidecar、按 job 的 stage 推导和
  DAG-safe stable topological projection。
- [x] 阶段 B：实现 `InterleavedOneFOneBSerializer`、`ZeroBubbleSerializer`、
  `BidirectionalPipelineSerializer` 及参数前置条件。
- [x] 阶段 C：统一 factory 已接入 Default、Puppeteer、Hermod 和 static/dynamic runner；
  static 输出 `pipeline_task_metadata.json`，dynamic 将同一字段合并进 task metadata。
- [x] 阶段 D：新增 12 个 serializer/analyzer 测试；排除缺失的外部 Spectrum-X topology
  fixture 后，仓库测试为 740 passed、3 skipped、18 deselected。

真实 AICB smoke 使用 GPT-13B、`tp=4, pp=4, dp=1, ga=8`、16-GPU AlibabaHPN。三种模式
均对 7,392 个 task 完成静态 executor 运行；dynamic runner 额外验证了 `zero_bubble` 与
`bidirectional`。本记录只证明入口、DAG 和 executor 路径可运行，不作为论文吞吐收益结论。

## 8. DAG 精度提升计划（2026-07-22）

本阶段优先保证“计算产生什么通信、通信解锁什么计算”的因果关系，
不以完整复制 Megatron runtime 的 stream/buffer 实现为目标。

### 阶段 E1：Interleaved 1F1B workload overlay

- 深拷贝原 workload，不修改通用 `WorkloadBuilder`和调用者持有的原对象。
- 保留原 compute、TP/DP/EP task 和层内依赖；只替换该 job 的原物理
  `PP_SEND` 及跨 VPP chunk 的错误本地直连。
- 将每个物理 stage 的局部层分成 `vpp` 个连续 chunk，建立
  `logical_stage = chunk_id * pp + physical_stage`。
- 对每个 microbatch 和每个 `(dp, ep, tp)` lane，建立 `pp*vpp-1`
  条前向 activation 边和对称的反向 gradient 边，包括
  `last_physical_stage/chunk_c -> first_physical_stage/chunk_{c+1}` 的回绕通信。
- PP 发送依赖发送端 chunk 边界层的 compute/TP 完成；接收端 chunk
  首个 compute 依赖 PP flow 完成。反向按完全逆路径建边。
- overlay 专用字段写入 `dict[int, InterleavedPipelineTaskInfo]` sidecar，
  不向 `Task` 添加字段。

完成标准：`P2PWorkload.validate()` 通过；逐条验证 PP flow 的 producer/consumer；
验证原跨 chunk 本地边已移除；验证 `PP=4,VPP=2` 时存在
`stage3/chunk0 -> stage0/chunk1` 的前向边和对称反向边。

### 阶段 E2：Serializer 与 DAG 对齐

- steady state 改为 Megatron 经典的 `F -> B` 计算顺序，而非当前的
  `B -> F` 投影。
- 使用 overlay 后的真实 DAG 进行 stable topological legalization；sidecar 同时记录
  preferred slot 和最终 local order。
- 对每个 GPU 验证 compute order 与所有前向/反向 PP 边相容。

### 阶段 E3：入口与动态执行

- static runner 在 Hermod metadata、routing 和 serializer 之前生成 overlay，保证新 PP flow
  同时进入路由、带宽竞争与 Hermod coflow 分析。
- dynamic runner 使用 strategy-specific `JobExpander` 包装器，在全局 task-id
  分配后应用同一 overlay，并重算 entry/terminal task。
- 输出 workload sidecar 与 DAG 统计，便于审计每个 logical boundary。

### 阶段 E4：验证与后续复用

- synthetic golden DAG：覆盖 `PP=1/2/4`、`VPP=2/4`、多 TP lane、非连续 rank、
  多 microbatch 和非整除 chunk 大小。
- 使用真实 AICB 检查 task/flow 数、DAG、execution plan、死锁和 makespan；
  makespan 只作为回归信号，不作为性能结论。
- E1-E3 的 chunk ownership、PP boundary 和 sidecar 将作为 Zero Bubble 与
  Bidirectional Pipeline DAG overlay 的共用基础。

### 实施记录（E1-E3）

- [x] E1：新增 `pipeline_workload_overlay.py`。overlay 深拷贝输入，保留 compute 与非 PP
  flow，替换原物理 PP 边和跨 chunk 本地边；生成前向 activation 与反向 gradient 的完整
  logical-stage chain。
- [x] E2：Interleaved serializer steady state 已改为 `F -> B`，sidecar 的
  `logical_stage_id` 已包含 chunk offset，并继续通过全局 DAG 拓扑投影生成安全 GPU 顺序。
- [x] E3：static runner 在 metadata/routing 前应用 overlay；dynamic runner 使用
  `PipelineJobExpander` 通过同一全局 allocator 预留新增 task ID，并重算 entry/terminal task。
- [x] 定向验证：20 个 pipeline serializer/overlay 测试通过；扩展到 workload builder、
  Hermod priority/policy/allocator/metadata 与 dynamic 1F1B 后共 100 tests passed。
- [x] 全仓回归：排除仓库外缺失的 Spectrum-X topology fixture 后，748 passed、3 skipped、
  18 deselected。
- [x] AICB smoke：GPT-13B `tp=4,pp=2,dp=1,ga=2,vpp=2` 生成 48 条 PP flow；
  static Default、dynamic Default/Hermod 均完成，单 iteration makespan 均为 514,621 us。

尚未完成：从真实模型配置读取 chunk ownership、activation memory/buffer 生命周期、CUDA
stream 与异步收发重叠；这些不影响本阶段已经建立的 task-level compute/communication 因果边，
但会影响更细的 runtime overlap 与显存结论。

## 9. Bidirectional Pipeline DAG 精确扩展计划（2026-07-22）

本阶段把 `bidirectional` 从“仅调整 GPU 本地 compute 顺序”提升为“双向 PP 通信 DAG 与
compute schedule 一致”。目标不是逐行复刻 DualPipe runtime，而是优先保证会直接影响网络
竞争和实验结果的三类事实：通信方向、通信生产者、通信消费者。

### 阶段 F1：双向逻辑 pipeline 与 sidecar

- GA micro-batch 均分为 `down` 和 `up` 两组；保持 serializer 的约定：前半组走 down，后半组走 up。
- 物理 stage `s` 对 down pipeline 的逻辑 stage 为 `s`，对 up pipeline 的逻辑 stage 为
  `pp - 1 - s`。每个 GPU 因而持有两个镜像 module replica 的计算投影。
- 新增 `BidirectionalPipelineTaskInfo` sidecar，以 `task_id` 为 key 记录 pipeline ID、方向、
  物理/逻辑 stage、边界和 task role；不向公共 `Task` 增加字段。
- 显式拒绝奇数 `pp`、奇数 GA 和 `GA < 2 * pp`，与 compute serializer 使用相同前置条件，
  不静默回退为普通 1F1B。

### 阶段 F2：双向 activation/gradient DAG overlay

- 深拷贝输入 workload，删除目标 job 原有的单向 `PP_SEND/PP_RECV` 及其依赖，只在副本中重建 PP 边。
- 对 down micro-batch 建立：
  `stage s forward output -> activation(s -> s+1) -> stage s+1 forward input`，以及完全反向的
  `stage s+1 backward-input output -> gradient(s+1 -> s) -> stage s backward-input`。
- 对 up micro-batch 建立镜像链：
  `stage s forward output -> activation(s -> s-1) -> stage s-1 forward input`，以及完全反向的
  `stage s-1 backward-input output -> gradient(s-1 -> s) -> stage s backward-input`。
- PP flow 的生产者必须是边界层真正完成的输出：存在 TP/EP 通信时依赖该通信在发送 rank 上的
  completion，否则依赖 compute；接收端的第一个相关 compute 必须依赖 PP flow。
- 每个 `(job, microbatch, dp, ep, tp)` lane 恰有 `pp-1` 条 activation 和 `pp-1` 条 gradient PP flow；
  非连续 rank 只按 `assigned_nodes` 的 `[PP][DP][EP][TP]` 布局解析，不能按 rank 数值推断方向。

### 阶段 F3：静态、动态与 Hermod 接入

- static runner 在 Hermod metadata、routing 和 serializer 之前应用 overlay，使新 flow 进入路由、
  链路竞争和带宽分配。
- dynamic runner 通过 `PipelineJobExpander` 和同一个全局 task-ID allocator 应用 overlay，并重算
  entry/terminal task，保证多 job/多 iteration 不碰撞。
- Hermod PP coflow ID 增加 bidirectional pipeline 方向，避免同一 MID/phase 中的 down/up flow
  被错误合并；方向依据 job stage 布局和 phase 推导，不依据 GPU 编号大小。
- static/dynamic 输出合并 DAG sidecar 与 serializer sidecar，保留每条双向边的审计信息。

### 阶段 F4：验收与明确近似

必须通过的 DAG 验收：

- `PP=2/4`、多 TP lane、非连续 rank 的 flow 数量、端点、消息大小、phase 和依赖均正确。
- 同一 micro-batch 的 forward activation 与 backward gradient 路径严格互逆；down/up 两组方向相反。
- 每条新 PP flow 至少有一个本地 compute/collective producer，并且至少解锁一个接收端 compute。
- overlay 后 `P2PWorkload.validate()`、serializer DAG legalization、静态和动态 executor 均无环、无死锁。

本阶段允许保留的近似：executor 仍把单 GPU compute 投影为串行，不复刻
`overlapped_forward_backward` kernel；不建模异步 `isend/irecv` wait/buffer 生命周期；两个 module
replica 的参数、DP 梯度和 optimizer task 仍复用现有物理 stage 投影。这些近似会影响 overlap、显存和
部分 DP 时间，但不会改变本阶段建立的 PP compute—communication 因果 DAG。

### 实施记录（F1-F4）

- [x] F1：新增 `BidirectionalPipelineTaskInfo`，为 F/B/W compute 和新 PP flow 记录 down/up、
  pipeline ID、物理/逻辑 stage、边界与 peer stage；未修改公共 `Task`。
- [x] F2：新增 `BidirectionalPipelineWorkloadOverlay`。前半 GA 建立 stage 递增 activation DAG，
  后半 GA 建立 stage 递减 activation DAG；两组 gradient 均严格反向。PP producer 使用发送 rank
  的边界 compute/TP completion，PP consumer 是接收 rank 的边界 compute。
- [x] F3：static runner、`PipelineJobExpander` 和 Hermod dynamic analyzer 已接入。Hermod coflow
  ID 带 `pdown`/`pup`，方向由 `assigned_nodes` 的 stage 布局和 phase 推导。
- [x] F4 synthetic：新增双向定向/参数 case，覆盖 `PP=2/4`、多 TP lane、非连续 rank、
  端点互逆、producer/consumer、serializer DAG legalization、Hermod coflow 和动态全局 task ID。
- [x] F4 回归：pipeline/workload/Hermod 定向集合 106 passed；排除仓库外缺失的 Spectrum-X
  topology fixture 后，全仓 756 passed、3 skipped、18 deselected。Ruff 未安装，compileall 通过。
- [x] F4 AICB E2E：GPT-13B `tp=4,pp=4,dp=1,ga=8` 在 16-GPU AlibabaHPN 上生成 192 条
  PP flow；static Default/Hermod 与 dynamic Default/Hermod 均无死锁，单 iteration makespan
  均为 1,040,799 us。该数值仅作 DAG/executor 回归，不是性能收益结论。

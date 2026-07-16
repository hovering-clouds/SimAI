# Hermod §4.1 优先级排序复现计划

> 日期：2026-07-15  
> 论文：`论文/Coflow Scheduling for LLM Training.pdf`，§4.1（含 §4.1.1–§4.1.3）  
> 实现位置：`SimAI/simai-flow-scheduler`  
> 状态：计划，尚未开始实现

## 1. 目标与边界

本复现只实现 Hermod 的 **inter-coflow strict-priority policy**：为训练 DAG
中的 coflow 建立稳定、可比较的优先级，并让共享链路的带宽按该优先级分层分配。
目标是在现有离散事件模拟器中观察到：高优先级 coflow 在竞争链路上优先获得带宽，
同一优先级内保持公平共享。

明确不在本轮范围内：

- §4.2 的 matching-based optimal intra-coflow scheduling、流匹配和 EP 专用流量分配；
- 改路由、拓扑感知放置、作业准入、作业间 time shift；
- 训练框架或交换机侧的实际部署/DSCP 编码；
- 论文完整端到端性能复现及其生产集群参数校准。

因此，本计划把一个 coflow 展开后的 P2P flows 视作同一优先级组；只改变它们竞争
带宽时的服务次序，不改变 DAG 依赖、路径或 collective 展开算法。

## 2. 论文规则的可执行定义

每个 coflow 使用三个模型因子：

| 因子 | 方向 | 在本项目中的来源/目标元数据 |
| --- | --- | --- |
| MID（microbatch ID） | 小者优先 | 必须显式提供，不能把 `Task.iteration`（GA step）误当 MID |
| CType（coflow type） | EP、PP 优先于 DP | 由 `CommType` 分类 |
| LID（layer ID） | 小者优先 | 逻辑模型层；需验证现有 `Task.layer_id` 是否符合该语义 |

论文的三个主要竞争场景为：

| 场景 | 竞争关系 | 主优先因子 |
| --- | --- | --- |
| Case I | 同一 iteration 内 EP/PP，跨 microbatch | MID |
| Case II | BP 中 EP、PP、DP 重叠 | CType，`EP = PP > DP`；EP/PP 内仍按 MID |
| Case III | 跨 iteration 的前层 EP/PP 与后层 DP | LID，小者优先 |

冲突处理必须按 §4.1.2 编码并有单测：

1. `MID + CType`：DP 只在最后一个 microbatch 后出现，正常训练 DAG 中无该冲突；
   将此作为输入不变量验证。
2. `MID + LID`：非 DP 竞争时 MID 优先；涉及 DP 的跨 iteration 竞争时 LID 优先。
3. `CType + LID` 和三因子同时冲突：论文认为在合法训练 DAG 中不可达；实现应在
   debug/strict 模式报告不可达元数据组合，而不是悄然猜测。
4. 常规 1F1B 的 CType 次序为 `EP = PP > DP`；交错 1F1B 只在 Case II 使用
   `EP > PP > DP`。其余规则不变。

论文将结论概括为默认因子优先顺序 `MID > CType > LID`，但上述 DP 特例不能被简化
为一个对所有输入都正确的固定三元组排序。因此实现采用“先识别候选 coflow 的竞争
场景、再产生全序 priority key”的纯函数；对无法归类的输入采取配置化的保守 fallback
并记录诊断。最终 key 必须稳定，最后以 `coflow_id` 打破平局。

## 3. 现有代码与接入点

可直接复用：

- `src/workload_format/schema.py`：已有 `Task.phase`、`layer_id`、`iteration`、
  `item_id`、`comm_type`；其中 `iteration` 注释为 GA step，不能当作 MID。
- `src/workload_generator/workload_builder.py`：`_expand_comm_all_groups()` 将一项
  collective 展开为多个 P2P flow，是写入 coflow 关联元数据的唯一合适位置。
- `src/static_analysis/passes/routing/`：保持 BFS/现有路由，不随本工作修改。
- `src/executor/analytical.py`：每次 active-flow 集合变化均调用
  `SchedulingPolicy.allocate_bandwidth()`，无需改事件循环。
- `src/executor/bandwidth_allocators/mfs_allocator.py`：可借鉴“跨队列严格优先、队列内
  fair share、逐链路剩余容量”的实现结构，但不复用 MFS 的 RLI/请求语义。
- `src/executor/policies/default_policy.py`：复用计算任务串行化、路由表和生命周期接口。

建议新增而非侵入现有策略：

```text
src/static_analysis/passes/hermod_priority.py       # 元数据校验、分类、比较器/priority key、诊断
src/static_analysis/strategies/hermod_strategy.py   # 路由 + compute order + Hermod priority analysis
src/executor/bandwidth_allocators/hermod_allocator.py
src/executor/policies/hermod_policy.py
tests/test_hermod_priority.py
tests/test_hermod_allocator.py
tests/test_hermod_policy.py
tests/test_hermod_e2e.py
scripts/run_hermod_e2e.py                           # 基线与 Hermod 的可重复对比
```

## 4. 分阶段实施计划

### Phase 0：输入语义审计与复现契约

1. 选定一个包含 EP、PP、DP、常规 1F1B 的最小 AICB/JSON workload，画出其
   `task_id → (phase, item_id, iteration, layer_id, comm_type)` 表。
2. 核实 trace 能否恢复每个 collective 的真实 MID 和逻辑 LID；特别检查 backward
   顺序、PP send、DP bucket 和跨 iteration 的语义。
3. 写下可复现契约：支持的训练 trace 格式、常规/交错 1F1B 的判定来源、DP 的定义，
   以及缺失 MID/LID 时的行为。

验收：不写调度代码前，已有一份机器可读的小型 fixture，且每个 flow 的目标因子均可
人工核对。

### Phase 1：补齐 coflow 和模型因子元数据

1. 为 `Task` 及其 JSON schema 增加可选的、向后兼容字段：
   `coflow_id`、`microbatch_id`、`logical_layer_id`。保留现有字段语义，不复用
   `iteration` 代替 `microbatch_id`。
2. 在 collective 展开处给同一次 collective invocation 的全部 P2P flows 写同一个
   `coflow_id`；不同并行子组是否组成同一个 coflow需在 Phase 0 固化。默认以一次原始
   collective invocation 为一个 coflow，并为每个 subgroup 生成可区分的 coflow ID，
   以免把无共享成员的 collective 强行绑成一个调度实体。
3. 从 trace/AICB 的明确字段传播 MID 和逻辑 LID；若原始输入不具备这些字段，新增
   sidecar mapping（而非基于 task ID 猜测）。
4. 在 workload validator 或 Hermod analysis 前执行完整性检查：Hermod 模式下所有
   可调度 EP/PP/DP flow 都必须有三项元数据和可识别 CType。

验收：序列化/反序列化保留元数据；一个 collective 展开出的所有 flows 获得一致的
priority metadata；缺失或矛盾数据会产生清晰错误。

### Phase 2：实现 §4.1 优先级分析

1. 新建不可变的 `HermodCoflowInfo`（`coflow_id`、flow task IDs、MID、CType、LID、
   iteration/schedule context）和 `HermodPriorityAnalysisResult`。
2. 实现 `classify_coflow_type()`：EP 映射 EP，`PP_SEND`/`PP_RECV` 映射 PP，所有
   `DP_*` 映射 DP；TP/未知通信默认不进入 Hermod 范围，并由配置选择 fair-share
   background 或 fail-fast。
3. 实现带配置的 comparator/key builder：`schedule_variant = conventional_1f1b |
   interleaved_1f1b`，并逐条落实第 2 节的 Case 和冲突规则。输出应同时包含可读的
   `reason`/`case`，便于实验审计。
4. 对 comparator 做全序与一致性检查：候选集合排序不依赖输入列表顺序；不存在
   非法循环；同优先级 coflow 的 tie-break 可复现。

验收：纯单元测试可直接断言 Case I/II/III、DP 特例和交错 1F1B 的排序，无需启动模拟器。

### Phase 3：严格优先级带宽分配与策略接入

1. 实现 `HermodAllocator`：每次重分配时按 coflow priority 从高到低处理；对每个
   优先级层用逐链路剩余带宽进行 max-min/fair-share 分配，再将余量交给下一层。
2. 同一 coflow 的所有 active P2P flows 使用同一层级；同层不同 coflow、同一 collective
   的多条 flow 均公平共享，避免偷偷实现 §4.2 的“匹配”。
3. 实现 `HermodSchedulingPolicy`：compute 准入、路由表和回调沿用 Default policy；
   flow 全准入，优先级只作用于 allocator。这样已就绪的低优先级 flow 不会造成 DAG
   死锁，但在共享链路上可获得 0 带宽，直到高层释放容量。
4. 新建 `HermodAnalyzer/Strategy`，复用 BFS route table 与 C++ compute serializer，
   并把 priority analysis 传给 policy。

验收：两个共享单链路的 active flows 中，高层取得全部容量、低层为 0；高层完成后低层
自动接管；同层流公平共享；不共享链路的流可并行取得各自链路容量。

### Phase 4：端到端验证、基线与文档

1. 新增最小合成 DAG：覆盖 Case I、Case II、Case III 和交错 1F1B Case II；每个 fixture
   同时以 Default 与 Hermod 运行。
2. 增加一个真实生成 trace 的 smoke test，确认元数据传播、路由和执行器重分配无死锁。
3. `scripts/run_hermod_e2e.py` 输出 workload 摘要、coflow priority 表、每条 flow 的
   start/end/bandwidth 时间线及 JCT；默认保存 JSON，便于比较。
4. 文档记录使用命令、已支持输入、未覆盖机制和与默认 fair-share 基线的预期差异。

验收：`pytest tests/test_hermod_*.py -v` 通过，且已有基线对比可以证明 priority ordering
确实影响共享链路的带宽与完成顺序，而非只产生元数据。

## 5. 测试矩阵

| 测试 | 输入竞争 | 期望 |
| --- | --- | --- |
| MID 单调性 | 同 CType/LID、MID 1 vs 2 | MID 1 优先 |
| 常规 Case II | EP、PP、DP 同时活跃 | EP/PP 同层且均优先 DP |
| 交错 Case II | EP、PP、DP 同时活跃 | EP > PP > DP |
| Case III | 前层 EP/PP vs 后层 DP | 较小 LID 优先 |
| MID/LID 非 DP | 不同 MID、不同 LID 的 EP/PP | MID 优先 |
| MID/LID 含 DP | 跨 iteration 的 DP 与 EP/PP | LID 优先 |
| 链路隔离 | 高低优先级不共享链路 | 两者不互相限速 |
| 同层公平 | 多条同优先级 P2P flow 共享链路 | 各 flow 得到公平份额 |
| 元数据失败 | 缺 MID/LID/coflow ID | Hermod 模式 fail-fast，错误可定位 |
| 回归 | Default/MFS/Puppeteer 既有测试 | 行为不变 |

## 6. 风险与决策点

1. **MID 并非现有 `Task.iteration`。** 当前 IR 把它定义为 GA step；若 AICB trace 没有
   microbatch 标识，无法忠实实现 §4.1，必须先扩展 trace 或提供经过验证的 sidecar。
2. **`layer_id` 可能是 trace 中的扁平项序号。** 只有确认它与论文 LID 的“数据更早被消费”
   顺序一致后才可直接使用，否则新增 `logical_layer_id` 映射。
3. **coflow 边界必须明确。** P2P 展开会把一个 collective 拆成很多 flow；错误合并或拆分
   会改变公平共享粒度。Phase 0 的 fixture 是这一决策的依据。
4. **严格优先可能使低优先级暂时为零。** 这符合本文所复现的 §4.1 抽象；不引入老化、
   配额或饥饿缓解机制，除非后续另行扩展。
5. **论文比较器依赖合法训练 DAG。** 对论文声明“不可能”的组合，不应伪造排名来声称
   忠实复现；应记录诊断并由配置决定报错或保守 fallback。

## 7. 完成定义

本复现完成的条件是：Hermod 模式可从带完整模型因子的训练 workload 生成 coflow
优先级；常规与交错 1F1B 的 §4.1 排序均由测试覆盖；模拟器在共享链路上严格按该排序
分层带宽、层内公平共享；全套新增和既有回归测试通过；文档明确声明 §4.2 及其他非范围
机制未实现。

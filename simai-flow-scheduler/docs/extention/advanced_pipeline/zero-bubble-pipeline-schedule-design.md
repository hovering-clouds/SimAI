# Zero Bubble Pipeline Schedule 扩展方案

> 状态：隔离式 ZB1P 第一阶段已实现并验证（2026-07-23）。Default/Hermod/Puppeteer 和既有
> 实验不注册本模式；验证入口为 `scripts/run_e2e_zero_bubble.py`。本文仍不把当前实现表述为
> 论文级 Zero Bubble 完整复现。
>
> 参考：[`1f1b-pipeline-schedule-design.md`](./1f1b-pipeline-schedule-design.md)。获批后，本文取代
> `advanced-pipeline-schedules-plan.md` 中 Zero Bubble 的后续实施部分；旧记录保留作历史审计。
>
> 隔离调整：本文早期关于接入现有 static/dynamic 实验的段落未执行；当前实现范围以第 13 节为准。

## 1. 背景与目标

Zero Bubble Pipeline Parallelism 的核心不是改变 activation/gradient 的 PP 方向，而是把传统
backward 拆成：

- `B`：输入梯度计算，位于 pipeline critical path。
- `W`：权重梯度计算，可在 B 已完成后延迟到空闲 slot。

simai-flow-scheduler 的 AICB 展开已经把两者表示为：

```text
B = Phase.BACKWARD_INPUT
W = Phase.BACKWARD_WEIGHT
```

现有 `ZeroBubbleSerializer` 能改变本地 compute 顺序，但尚未把 Zero Bubble 需要的 workload
语义单独建模和审计。重新规划后的目标是：

- 在策略 workload 展开阶段明确建立 `F -> B -> W`、B 链、PP gradient 和 DP/optimizer 依赖。
- 保证 backward PP gradient 只依赖 B completion，不等待 W。
- 保证 W/DP completion 在 optimizer 前完成。
- 使用 analyzer-owned `dict[int, ZeroBubbleTaskInfo]` 记录 B/W、slot 和关键路径信息。
- serializer 基于 sidecar 生成 ZB1P 风格顺序，不修改公共 `Task` 或通用 executor。

“Zero Bubble”在本方案中是策略名称。第一阶段不承诺任意异构 duration、任意 GA 和任意通信
配置下实测 bubble 严格为零。

---

## 2. 当前实现分析

### 2.1 已有正确基础

通用 `WorkloadBuilder` 已具备：

- 每个 layer 的独立 `BACKWARD_INPUT` 和 `BACKWARD_WEIGHT` compute task。
- 同 layer `B -> W`。
- B 按 layer 逆序传播。
- backward PP flow 从上一 logical stage 的 B output 发出。
- W 后可连接 DP collective。

现有 `ZeroBubbleSerializer` 也已经具备：

- warmup。
- steady state 中优先 B、延迟 W。
- 每个 W 排在同 micro-batch B 之后。

### 2.2 当前不足

- B/W 角色由 serializer 根据 `Phase` 临时推断，没有 workload expansion sidecar。
- PP flow 在通用 builder 中生成，无法审计“gradient 是否只依赖 B”。
- optimizer/post task 主要依赖 compute order，DAG 上缺少完整的 W/DP completion barrier。
- 当前 serializer 使用固定公式，不利用实际 DAG 中哪些 B/W 已 ready。
- “DeepSpeed ZeRO workload”与“Zero Bubble schedule”名字相近但语义不同，当前缺少显式输入边界。

---

## 3. 设计边界

### 3.1 支持范围

第一阶段支持：

- 普通 training AICB。
- `pp >= 1`、`ga >= 1`。
- 已有独立 B/W duration 的 workload。
- PP、TP、DP 的 task-level DAG。
- ZB1P 风格、单 GPU compute 串行执行。

第一阶段不支持：

- `is_zero_workload()` 识别出的 DeepSpeed ZeRO/FSDP 专用行格式。
- 自动 profile search、整数规划或论文中的全部 schedule family。
- B/F fused kernel 或通信计算 overlap。
- activation memory 的字节级生命周期。

DeepSpeed ZeRO 输入必须显式报错，不得误走 Zero Bubble 路径。

### 3.2 不修改的通用类型

不修改：

- `Task`、`P2PWorkload`、schema。
- `TaskSerializer` 公共接口。
- 默认 `WorkloadBuilder` 和 `_build_zero_from_aicb()`。
- 默认 GPipe/1F1B serializer。
- executor policy 和 bandwidth allocator。

新增 `ZeroBubblePipelineWorkloadBuilder(WorkloadBuilder)`，只对被明确选择的策略生效。

---

## 4. Sidecar 设计

建议新增：

```python
@dataclass
class ZeroBubbleTaskInfo:
    task_id: int
    job_id: int
    task_role: str             # F / B / W / PP_ACT / PP_GRAD / DP / OPT
    microbatch_id: int
    physical_stage_id: int
    layer_id: int
    b_task_id: int | None
    w_task_id: int | None
    critical_path: bool
    preferred_slot: int | None = None
    final_local_order: int | None = None
```

sidecar 归 `ZeroBubbleAnalyzer` 所有：

```python
self.task_info: dict[int, ZeroBubbleTaskInfo]
```

workload builder 在生成/连接任务时写入角色和配对关系；serializer 只读该映射并返回 slot/order，
再由 analyzer 回填自己持有的 sidecar。PP、DP flow 也进入 sidecar，从而可以审计 B/W 与通信的关系。

---

## 5. Workload 展开与 DAG 方案

### 5.1 策略 Builder

新增 `ZeroBubblePipelineWorkloadBuilder(WorkloadBuilder)`，复用通用 compute/collective 展开，只覆盖：

- `_wire_dependencies()`：建立 ZB 所需 barrier。
- `_generate_pp_flows()`：复用标准 PP 端点，同时在创建时记录 PP sidecar。
- `_wire_pp_dependencies()`：显式保证 PP gradient 依赖 B completion。

普通 PP 的 src/dst 不需要改变：

```text
Forward:  stage s -> stage s+1
Backward: stage s+1 -> stage s
```

修改 PP 展开层的目的不是重定向 flow，而是让 flow 在第一次生成时获得可审计角色，并确保
producer/consumer 不通过 W 间接连接。

### 5.2 每个 micro-batch/layer 的 DAG

```text
F(layer)
  -> F collective completion
  -> 后续 F layer / forward PP

F terminal
  -> B(last layer)
  -> B collective completion
  -> B(previous layer)
  -> backward PP gradient

B(layer)
  -> W(layer)
  -> W/DP collective completion
```

关键约束：

1. `B(m,l) -> W(m,l)` 必须存在。
2. `W(m,l)` 不得成为 `B(m,l-1)` 的前置依赖。
3. backward PP gradient 不得依赖任何 W 或 DP task。
4. receiver B compute 必须依赖 backward PP gradient。

### 5.3 PP producer/consumer

Forward：

```text
last local F compute/TP completion
  -> PP activation
  -> next stage first F compute
```

Backward：

```text
first local B compute/TP completion
  -> PP gradient
  -> previous stage last B compute
```

PP flow 大小直接来自 `AicbHeader.pp_comm_size`。

### 5.4 Optimizer barrier

策略 builder 为每个 rank 计算训练 terminal：

```text
all W compute/DP completion across all micro-batches
  -> optimizer/post update
```

如果一个 layer 没有 DP flow，则 terminal 是 W compute；如果有 DP flow，则 terminal 是
该 rank 的 DP completion。optimizer 不得仅依赖最后一个 B 或仅依赖 compute order。

该 barrier 只加入 Zero Bubble 策略 workload，不修改通用 builder 的 pre/post 语义。

### 5.5 激活数量的近似约束

第一阶段不建立新的 memory task，但 analyzer 在 sidecar 中维护每个 micro-batch 的 activation
状态：

```text
F 完成 -> activation live
B 完成 -> activation released
```

serializer 使用配置项 `max_inflight_microbatches` 限制未执行 B 的 F 数量。默认值采用普通 1F1B
warmup 上界；该值只影响 compute order，不写入 `Task`。

---

## 6. Serializer 方案

### 6.1 输入

```python
ZeroBubbleSerializer(
    task_info: Mapping[int, ZeroBubbleTaskInfo],
    max_inflight_microbatches: int | None = None,
)
```

serializer 不再只通过 `Phase` 猜测 B/W，而是使用 builder/analyzer 已确认的 sidecar。

### 6.2 Preferred schedule

使用受约束的 ZB1P list scheduling：

1. 先满足 stage-specific warmup。
2. 在所有 DAG-ready 本地 operation 中使用优先级：
   - B：最高，优先传播 pipeline gradient。
   - F：activation budget 未满时次高。
   - W：填充没有 ready B/F 的 slot。
3. cooldown 中先排空 B，再排空 W。

该列表只产生 preferred compute order。最终顺序仍通过全局 DAG stable topological
legalization，避免本地启发式与通信依赖合并后成环。

### 6.3 不追求的细节

- 不根据实际网络完成时间动态改变已经生成的 `ExecutionPlan`。
- 不实现 B/F kernel fusion。
- 不声明得到论文最优 schedule。

这些限制不影响 `B -> PP gradient` 和 `B -> W -> DP/optimizer` 的 DAG 正确性。

---

## 7. Analyzer 方案

新增 `ZeroBubbleAnalyzer`：

- 验证输入不是 DeepSpeed ZeRO/FSDP 专用 workload。
- 创建并持有 sidecar。
- 使用 `ZeroBubblePipelineWorkloadBuilder` 直接展开 workload。
- 验证每个 B 恰好配对一个 W；允许显式零时长 W，但不允许缺失 metadata。
- 验证 PP gradient producer 的依赖闭包不包含 W/DP。
- 调用 `ZeroBubbleSerializer`。
- 输出每个 task 的 role、critical-path 标记和最终 local order。

Default/Puppeteer/Hermod 应共享 analyzer 生成的同一 workload 和 compute plan，不能分别构造
不同 ZB 顺序。

---

## 8. 文件修改计划

### 8.1 新增文件

| 文件 | 内容 |
|---|---|
| `src/workload_generator/zero_bubble_pipeline_builder.py` | B/W、PP、DP、optimizer DAG |
| `src/static_analysis/strategies/zero_bubble_strategy.py` | sidecar 与 analyzer |
| `tests/test_zero_bubble_pipeline_builder.py` | DAG 黄金测试 |
| `tests/test_zero_bubble_strategy.py` | schedule、sidecar、输入校验 |

### 8.2 修改文件

| 文件 | 修改 |
|---|---|
| `src/static_analysis/passes/pipeline_task_serializers.py` | Zero Bubble serializer 改为读取 sidecar/list scheduling |
| `src/executor/pipeline_job_expander.py` | dynamic ZB builder 与 sidecar task-ID 重映射 |
| `scripts/run_hermod_e2e.py` | 使用 ZB analyzer 构建 workload |
| `scripts/run_hermod_dynamic_e2e.py` | static/dynamic 共用 ZB strategy state |
| package `__init__.py` | 导出新增类 |

### 8.3 保留文件

- 默认 `workload_builder.py` 不改。
- `_build_zero_from_aicb()` 不改；它属于 DeepSpeed ZeRO/FSDP，不是本策略。
- 现有 `ZeroBubbleSerializer` 测试先保留，作为迁移基线。

---

## 9. 实施阶段

### 阶段 Z1：DAG 与 sidecar

- 新增策略 builder。
- 建立 B/W pairing、PP producer/consumer 和 optimizer barrier。
- 暂不修改 serializer。

验收：逐边测试通过，默认 builder 输出不变。

### 阶段 Z2：Serializer

- serializer 改为 sidecar 驱动。
- 实现 B > F > W 的受约束 list scheduling。
- 增加 activation budget。

验收：golden sequence、W 延迟、无环验证通过。

### 阶段 Z3：Static/Dynamic 接入

- static/dynamic runner 使用同一策略 state。
- dynamic remap sidecar。
- trace 输出 B/W/PP/DP role。

验收：static/dynamic DAG 和 compute order 一致，无死锁。

### 阶段 Z4：性能与边界审计

- 比较普通 1F1B 和 ZB1P 的 bubble、makespan。
- 记录近似条件，不选择性报告收益。
- 决定是否需要 duration-aware schedule search；需另行批准。

---

## 10. 验证方案

### 10.1 DAG 测试

| 测试 | 验证 |
|---|---|
| F→B→W | 每个 micro-batch/layer 配对正确 |
| B reverse chain | 下一层 B 不等待 W |
| backward PP | producer 为 B/TP completion，不含 W/DP |
| receiver B | 依赖正确 PP gradient |
| W→DP | DP flow 依赖 W completion |
| optimizer | 等待所有 micro-batch W/DP terminal |
| PP=1 | 不生成 PP flow，B/W DAG 仍正确 |
| 非连续 rank | stage 由 `assigned_nodes` 推导 |

### 10.2 Serializer 测试

- 每个 W 位于配对 B 之后。
- 只要存在 ready B，preferred order 不选择 W。
- activation budget 不被超过。
- 所有 compute task 恰好出现一次。
- DAG legalization 后无环。
- GA 很小、PP 大于 GA 时显式处理，不产生负 region 长度。

### 10.3 输入与回归

- DeepSpeed ZeRO/FSDP 输入显式拒绝并给出清楚错误。
- 默认 GPipe/1F1B workload builder 测试全部通过。
- static/dynamic E2E 无死锁。
- Chrome Trace 能看到 B 优先传播、W 填充尾部/空闲 slot。

---

## 11. 兼容性与风险

- 最大风险是 optimizer barrier 与现有 post task 语义重复；测试需验证只增加必要依赖且不成环。
- duration-independent list scheduling 可能不能消除全部 bubble，因此结果只称“ZB1P 风格”。
- activation budget 是 analyzer sidecar 状态，不进入公共 task。
- 旧 Zero Bubble serializer 在迁移完成前保留 feature flag 回退路径。

---

## 12. 批准点

本文获批后才开始 Z1。Z1 完成并展示 DAG 后，再进入 Z2；不会在未批准时修改默认
`WorkloadBuilder`、DeepSpeed ZeRO 路径或公共 schema。

---

## 13. 实施记录（2026-07-23）

已完成：

- 新增 `src/workload_generator/zero_bubble_pipeline_builder.py`，保留公共 builder 的精确
  `F -> B -> W` 和普通 PP 端点，只增加策略专属的 W/DP-to-post barrier。
- backward PP gradient 的 producer 是 B/TP completion，依赖闭包不包含 W 或 DP。
- post/optimizer entry 对每个 rank 等待所有 microbatch/layer 的 W compute 或 DP completion。
- `ZeroBubbleTaskInfo` 记录 F/B/W、PP activation/gradient、DP、B/W 配对、critical-path、
  preferred slot 和最终本地顺序；不修改公共 `Task`。
- `ZeroBubbleSerializer` 从 expansion sidecar 读取 B/W 角色，并把 preferred/final order
  回填到独立 analyzer-owned sidecar。
- 新增隔离验证入口 `scripts/run_e2e_zero_bubble.py`；DeepSpeed ZeRO/FSDP 专用 AICB 行格式会显式拒绝。
- 新增 `tests/test_zero_bubble_pipeline_builder.py`，覆盖 B/W 配对、B reverse chain、
  gradient 闭包、DP completion barrier、sidecar serializer 与动态 task-ID remap。

独立 E2E 使用 GPT-7B AICB、`tp=2, pp=2, ga=8` 和 16-GPU AlibabaHPN topology：

- 生成 32 条 PP flow；
- 3,288 个 task 全部执行完成；
- makespan 为 4,272,394 us。

当前实现是 duration-independent 的 ZB1P preferred order，并通过全局 DAG stable topological
legalization 保证安全；它未实现论文中的自动 schedule search、B/F fusion 或运行时动态重排。

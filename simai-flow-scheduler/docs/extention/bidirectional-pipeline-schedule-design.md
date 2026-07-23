# Bidirectional Pipeline Schedule 扩展方案

> 状态：隔离式 task-level 第一阶段已实现并验证（2026-07-23）。Default/Hermod/Puppeteer 和
> 既有实验不注册本模式；验证入口为 `scripts/run_e2e_bidirectional.py`。实现采用 DualPipe
> 风格本地顺序和 Chimera/DualPipe 共有的双向逻辑 pipeline，不承诺逐行复刻任一框架 runtime。
>
> 参考：[`1f1b-pipeline-schedule-design.md`](./1f1b-pipeline-schedule-design.md)。获批后，本文取代
> `advanced-pipeline-schedules-plan.md` 中 Bidirectional Pipeline 的后续实施部分；旧记录保留作历史审计。
>
> 隔离调整：本文早期关于 Hermod metadata、static/dynamic runner 接入的段落未执行；当前实现
> 范围以第 15 节为准。

## 1. 背景与目标

Bidirectional Pipeline 在同一组物理 GPU 上放置两条方向相反的逻辑 pipeline：

```text
down: stage 0 -> stage 1 -> ... -> stage pp-1
up:   stage pp-1 -> stage pp-2 -> ... -> stage 0
```

不同 micro-batch 被分配给 down/up pipeline；每条 pipeline 的 gradient 都沿其 activation 路径反向传播。

现有实现已经有：

- `BidirectionalPipelineSerializer` 的本地 DualPipe 风格 compute sequence。
- `BidirectionalPipelineWorkloadOverlay`，在普通 PP workload 生成后删除旧 PP flow 并重建双向边。

重新规划后的目标是：

- 在 AICB 第一次展开时直接生成 down/up activation 和 gradient PP flow。
- PP 大小直接使用 header，端点直接来自 `RankGrouper`。
- workload expansion 与 serializer 共享 analyzer-owned sidecar，不重复推导 direction/logical stage。
- 精确保证 PP producer/consumer 和 B/W/DP/optimizer 的 task-level 因果关系。
- 不修改 `Task`、schema、默认 `WorkloadBuilder` 或通用 executor。

---

## 2. 当前实现分析

### 2.1 当前链路

```text
WorkloadBuilder
  -> 对全部 micro-batch 生成 down 方向普通 PP flow
  -> BidirectionalPipelineWorkloadOverlay
       删除原 PP flow
       前半 GA 重建 down
       后半 GA 重建 up
  -> BidirectionalPipelineSerializer
       再次按 GA 前后半推断 direction
       生成八段本地 sequence
```

### 2.2 当前实现可以保留

- 前半 GA = down、后半 GA = up 的确定性分配规则。
- `logical_stage = physical_stage` 或 `pp-1-physical_stage`。
- serializer 的八段 preferred sequence。
- 全局 DAG stable topological legalization。
- 迁移期间使用过 overlay 黄金测试和真实 AICB E2E 对照 direct builder；旧测试现已删除。

### 2.3 需要改进

- 不再生成无用的原单向 PP flow。
- 不再从旧 flow 推断 `pp_comm_size`。
- direction、pipeline ID 和 logical stage 只计算一次。
- Hermod coflow 不再根据 src/dst/phase 二次猜测 direction。
- dynamic task-ID 重映射必须同步更新 sidecar。
- module replica 的 W/DP/optimizer 关系需要明确记录，即使第一阶段仍采用聚合近似。

---

## 3. 支持边界

### 3.1 第一阶段约束

- `pp` 为偶数。
- `ga` 为偶数。
- `ga >= 2 * pp`，保证现有八段 schedule 区域非负。
- training AICB 且 `pp_comm_size > 0`。
- 每个 micro-batch 只属于一个 pipeline。

不满足约束时显式报错，不回退到普通 1F1B。

### 3.2 不修改的通用机制

| 模块 | 约束 |
|---|---|
| `Task` / `P2PWorkload` / schema | 不增加 direction、replica、logical stage |
| 默认 `WorkloadBuilder` | 原单向 PP 展开保持不变 |
| `TaskSerializer` | 公共接口不变 |
| `JobExpander` | 普通训练/推理路径不变 |
| executor/policy/allocator | 不感知双向策略字段 |

新增 `BidirectionalPipelineWorkloadBuilder(WorkloadBuilder)`，利用现有虚方法调用覆盖 PP
生成和 wiring，不给通用 builder 增加策略分支。

---

## 4. Sidecar 设计

建议在 `src/static_analysis/strategies/bidirectional_pipeline_strategy.py` 定义：

```python
@dataclass
class BidirectionalPipelineTaskInfo:
    task_id: int
    job_id: int
    task_role: str
    microbatch_id: int
    pipeline_id: int             # 0=down, 1=up
    direction: str               # down / up
    module_replica_id: int
    physical_stage_id: int
    logical_stage_id: int
    logical_boundary_id: int | None
    peer_physical_stage_id: int | None
    peer_logical_stage_id: int | None
    preferred_slot: int | None = None
    final_local_order: int | None = None
```

覆盖范围：

- F/B/W compute。
- PP activation/gradient flow。
- 能明确归属 micro-batch 的 TP/DP flow。
- optimizer/post task 可以使用 `pipeline_id=-1` 表示两个 replica 的汇合点。

`BidirectionalPipelineAnalyzer` 持有：

```python
self.task_info: dict[int, BidirectionalPipelineTaskInfo]
```

builder 写 expansion metadata；serializer 只读 direction/replica 并返回 slot/order；analyzer 再回填
自己持有的 sidecar。Task 本身不携带这些字段。

---

## 5. PP Workload 直接展开

### 5.1 Micro-batch 分组

```text
half_ga = ga // 2

microbatch [0, half_ga)  -> pipeline 0 / down
microbatch [half_ga, ga) -> pipeline 1 / up
```

该规则由 builder 写入 sidecar；serializer 不再根据 iteration 自己切分。

### 5.2 Logical/Physical stage 映射

down：

```text
logical_stage = physical_stage
physical(logical) = logical
```

up：

```text
logical_stage = pp - 1 - physical_stage
physical(logical) = pp - 1 - logical
```

对两条 pipeline，logical stage 都沿 `0 -> pp-1` 前进，因此 serializer 和 metadata 可以使用统一
logical boundary 编号。

### 5.3 Activation flow

对每个 micro-batch、logical boundary 和 `(dp, ep, tp)` lane：

```text
logical s local last F compute/TP completion
  -> PP activation
  -> logical s+1 local first F compute
```

端点：

```text
down: physical s     -> physical s+1
up:   physical pp-1-s -> physical pp-2-s
```

### 5.4 Gradient flow

每条 activation boundary 有完全反向的 gradient：

```text
logical s+1 local first B compute/TP completion
  -> PP gradient
  -> logical s local last B compute
```

端点：

```text
down: physical s+1   -> physical s
up:   physical pp-2-s -> physical pp-1-s
```

PP size 直接来自 `AicbHeader.pp_comm_size`，不从 task 列表推断。

### 5.5 Local compute DAG

每个 micro-batch 在每个 physical GPU 上保留：

- F layer 正序链。
- B layer 逆序链。
- 同 layer `B -> W`。
- 本 micro-batch 的 `F -> B` bridge。

同一 GPU 上 down/up micro-batch 不增加数据依赖；它们的互斥只由 serializer 的单 compute resource
顺序表达。这样不会把 schedule preference 错写成模型数据依赖。

### 5.6 Module replica 近似

每个 GPU 对 down/up 分别记录 `module_replica_id=0/1`。第一阶段：

- 不复制 compute task。
- down/up micro-batch 使用同一 AICB duration profile。
- W 和 micro-batch 内 DP flow 按 direction 归属对应 replica。
- optimizer/post task 等待两个 replica 的全部 W/DP terminal。

这能保持 aggregate compute/communication DAG，但不用于推导参数显存。若后续需要两个 replica
不同参数大小或 DP volume，必须增加新的 AICB/profile 输入，不能凭 sidecar 猜测。

### 5.7 Strategy Builder

新增 `BidirectionalPipelineWorkloadBuilder(WorkloadBuilder)`，覆盖：

- `_generate_pp_flows()`：按 micro-batch direction 直接生成 flow。
- `_wire_pp_dependencies()`：使用 logical source/destination 连接 producer/consumer。
- `_wire_dependencies()`：只增加两个 replica 汇合到 optimizer 的 barrier；保留局部 F/B/W 链。

返回策略专用 `BidirectionalPPFlowResult`，至少包含：

```text
activation[(microbatch, logical_boundary)][sender_rank]
gradient[(microbatch, logical_boundary)][sender_rank]
all_flows
```

---

## 6. Serializer 方案

### 6.1 输入

```python
BidirectionalPipelineSerializer(
    task_info: Mapping[int, BidirectionalPipelineTaskInfo],
)
```

serializer 不再自行执行：

- GA 前后半 direction 划分。
- mirrored logical stage 推导。
- module replica 推断。

### 6.2 Preferred sequence

保留现有 DualPipe 八段结构作为 preferred compute sequence：

- 双向 warmup。
- down/up steady state。
- 可延迟 W。
- cooldown。

token 必须通过 sidecar 选择 task，而不是仅使用 `(iteration, phase)`。

### 6.3 DAG 合法化

最终仍采用全局 stable topological legalization：

- ready PP flow 优先释放。
- compute task 按 preferred slot 排序。
- 每个 node 的投影作为 `ExecutionPlan.compute_order`。

如果八段 sequence 与真实通信 DAG 冲突，DAG 优先，并在 sidecar 中同时保留
`preferred_slot` 与 `final_local_order`，方便定位偏差。

---

## 7. Analyzer 与 Runner

### 7.1 Analyzer

新增 `BidirectionalPipelineAnalyzer`：

- 验证 pp/ga 前置条件。
- 创建并持有 sidecar。
- 调用策略 builder 直接展开 AICB。
- 验证每个 micro-batch 的 activation/gradient 路径互逆。
- 调用 serializer。
- 输出 task role、direction、replica 和最终 order。

### 7.2 Hermod metadata

Hermod PP coflow direction 直接读取 sidecar：

```text
j{job}:i{mid}:{phase}:pp:pdown
j{job}:i{mid}:{phase}:pp:pup
```

不再根据 rank 数值或 src/dst 比较推断方向。非连续/cyclic placement 下仍以 sidecar 的 logical
stage 为准。

### 7.3 Dynamic

`PipelineJobExpander` 使用策略 builder 生成局部 task ID 后：

1. 从全局 allocator 申请连续范围。
2. 建立 `old_id -> new_id`。
3. 重映射 task ID、deps 和 sidecar key。
4. 重算 entry/terminal task。
5. 将同一 sidecar store 交给 dynamic analyzer。

普通 `JobExpander` 不修改。

---

## 8. 文件修改计划

### 8.1 新增文件

| 文件 | 内容 |
|---|---|
| `src/workload_generator/bidirectional_pipeline_builder.py` | 双向 PP 直接展开与 optimizer barrier |
| `src/static_analysis/strategies/bidirectional_pipeline_strategy.py` | sidecar 与 analyzer |
| `tests/test_bidirectional_pipeline_builder.py` | 双向 DAG 黄金测试 |
| `tests/test_bidirectional_pipeline_strategy.py` | serializer/analyzer/sidecar 测试 |

### 8.2 修改文件

| 文件 | 修改 |
|---|---|
| `src/static_analysis/passes/pipeline_task_serializers.py` | Bidirectional serializer 改为读取 sidecar |
| `src/static_analysis/passes/hermod_metadata.py` | 优先使用 sidecar direction |
| `src/executor/pipeline_job_expander.py` | 策略 builder 与 sidecar ID remap |
| `scripts/run_hermod_e2e.py` | analyzer 直接构建 bidirectional workload |
| `scripts/run_hermod_dynamic_e2e.py` | 共享策略 state |
| package `__init__.py` | 导出新增类 |

### 8.3 暂时保留

旧 `BidirectionalPipelineWorkloadOverlay` 已在 direct builder 完成验证、确认无运行引用后删除。

---

## 9. 实施阶段

### 阶段 B1：直接 PP 展开

- 新增策略 builder 和 sidecar。
- 直接生成 down/up activation/gradient。
- 建立 optimizer 汇合 barrier。
- 不切换 runner。

验收：新 builder 的逐边测试通过。

### 阶段 B2：Serializer sidecar 化

- serializer 读取 direction/replica/logical stage。
- 保留八段 preferred sequence。
- 记录 preferred/final order 差异。

验收：每个 GPU golden sequence 和全局 DAG legalization 通过。

### 阶段 B3：Static/Dynamic 接入

- static runner 切换直接 builder。
- dynamic remap task/sidecar。
- Hermod 使用 sidecar direction。

验收：static/dynamic DAG 同构、无死锁、task metadata 一致。

### 阶段 B4：迁移收口

- [x] 完成 flow 数、端点、deps、makespan 验证。
- [x] 更新实现文档和限制。
- [x] 删除旧 overlay 实现、工厂、导出和旧实现专用测试。

---

## 10. 验证方案

### 10.1 DAG 黄金测试

| 场景 | 验证 |
|---|---|
| PP=2/4 | 每个 micro-batch 有 `pp-1` activation 和 `pp-1` gradient |
| down micro-batch | activation stage 递增，gradient 递减 |
| up micro-batch | activation stage 递减，gradient 递增 |
| 多 TP lane | PP 依赖 sender TP completion |
| 非连续 rank | 端点按 `assigned_nodes` stage 布局 |
| B/W | gradient 不等待 W；W 依赖 B |
| DP/optimizer | 两个 replica terminal 汇合后才能 optimizer |
| 多 job | sidecar 和 task ID 不碰撞 |

### 10.2 不变量

对每个 PP activation `a` 必须找到唯一 gradient `g`：

```text
g.src == a.dst
g.dst == a.src
g.logical_boundary == a.logical_boundary
g.pipeline_id == a.pipeline_id
```

每条 PP flow：

- 至少一个 compute/collective producer。
- 至少一个接收端 compute consumer。
- producer/consumer 属于相同 job、micro-batch、pipeline。

### 10.3 Serializer

- sidecar direction 与 schedule token 一致。
- 每个 F/B/W task 恰好一次。
- up pipeline 的 logical stage 是 physical stage 镜像。
- preferred order 与 DAG 合并后无环。
- 奇数 pp/ga、GA 太小、缺失 PP size 时显式失败。

### 10.4 E2E

- static/dynamic 同一 AICB 生成相同 flow count 和边集合。
- Default/Puppeteer/Hermod 共享同一 compute plan。
- 真实 `pp=4, ga=8` AICB 无死锁。
- Trace 中同时出现相反方向 PP 流量，并能通过 sidecar 区分。
- makespan 只作为回归信号，不作为论文收益结论。

---

## 11. 风险与回退

- module replica 的参数/DP volume 仍是 profile 投影，不可用于显存结论。
- 八段 preferred sequence 不建模 fused kernel 和异步 send/recv wait。
- 默认 `WorkloadBuilder` 和普通 pipeline 模式始终可作为回退。

---

## 12. 批准点

B1 至 B4 均已完成。删除旧 overlay 未修改公共 schema，也未改变默认 PP workload 展开结果。

---

## 15. 实施记录（2026-07-23）

已完成：

- 新增 `src/workload_generator/bidirectional_pipeline_builder.py`。前半 microbatch 直接展开为
  physical stage 递增的 down pipeline，后半展开为递减的 up pipeline；gradient 严格反向。
- 每条边保持 `compute/TP completion -> PP_SEND -> receiver compute`，不再先生成普通单向
  PP 再删除重建。
- `BidirectionalPipelineTaskInfo` 记录 pipeline ID、down/up direction、physical/logical stage
  和 peer/boundary；serializer 用同一 sidecar 校验并生成方向一致的 compute order。
- 独立 analyzer 持有 sidecar；既有 analyzer/runner 已恢复到扩展前版本，旧 overlay 已删除。
- 输入约束显式化：偶数 `pp`、偶数 `ga`、`ga >= 2 * pp`、正 `pp_comm_size`。
- 新增 `tests/test_bidirectional_pipeline_builder.py`，覆盖双向端点、activation/gradient
  计算通信因果、非连续 rank、serializer-sidecar 一致性和形状拒绝。

独立 E2E 使用 GPT-7B AICB、`tp=2, pp=2, ga=8` 和 16-GPU AlibabaHPN topology：

- 生成 32 条 PP flow，后半 microbatch 的 activation 为 stage 1 -> 0；
- 3,288 个 task 全部执行完成；
- makespan 为 3,930,525 us。

该性能数值不代表论文复现或公平性能结论。当前模型未表达同 GPU 上的双 module replica
内存占用、通信 batch、kernel fusion 和框架 runtime 的精细 overlap；已保证的边界是 task-level
DAG、PP/TP/DP 计算通信因果和单 GPU compute 串行顺序。

# 1F1B Pipeline Schedule 实现方案

## 1. 背景与动机

当前 simai-flow-scheduler 的 training workload 调度采用 **GPipe 风格**的全 forward → 全 backward 模式。对于 Pipeline Parallelism (PP) 场景，这会导致较大的 pipeline bubble，尤其是 GA (gradient accumulation) 数量较大时。

**1F1B (One-Forward-One-Backward)** 是 Megatron-LM 等主流框架使用的 pipeline 调度策略。它通过在 warmup 阶段之后交替执行 forward 和 backward，显著缩小 pipeline bubble。

本方案将 1F1B 引入 simai-flow-scheduler 的模拟框架。

---

## 2. 当前实现分析

### 2.1 三层调度机制

当前系统有三层机制共同决定执行行为：

| 层级 | 模块 | 作用 | GPipe 特征 |
|------|------|------|-----------|
| DAG 依赖图 | `workload_builder.py` | 建立 task 间的数据依赖 | bridge 只在最后一个 GA 组；post items 形成全局 fwd→bwd barrier |
| compute_order | `task_serializer.py` | 每个 GPU 上 compute task 的发射顺序 | direction (FORWARD=0 → BACKWARD=1) 为最高优先级排序键 |
| Policy 准入 | `default_policy.py` | 严格按 compute_order 串行化 | `compute_cursor` 控制只发射 cursor 位置的 task |

### 2.2 GPipe 的 DAG 结构

```
pre.fwd chain
    ↓ (Forward Fork: pre → 所有 GA)
GA0.fwd chain →┐
GA1.fwd chain →┤ (Forward Join: 所有 GA → post)
...             │
GA{ga-1}.fwd →─┘
    ↓
post.fwd chain
    ↓ (bridge: post[-1].fwd → post[-1].ig)
post.ig chain (逆序)
    ↓ (Backward Fork: post[0].ig → 所有 GA[-1].ig)  ← ⚠️
GA{ga-1}.ig chain (逆序)
GA1.ig chain (逆序)
GA0.ig chain (逆序)
    ↓ (Backward Join: GA[0].ig → pre[-1].ig)
pre.ig chain (逆序)
```

### 2.3 关键问题：GA 被当作并行单位而不是 pipeline micro-batch

当前代码中，GA 组通过 fork-join 并行化：

```python
# workload_builder.py:498-531 - Fork-Join 模式
for ga_group in ga_groups:
    pre[-1].fwd → ga_group[0].fwd     # Fork: pre → 所有 GA 并行
    ga_group[-1].fwd → post[0].fwd     # Join: 所有 GA → post
    post[0].ig → ga_group[-1].ig       # Fork: post → 所有 GA backward 并行
    ga_group[0].ig → pre[-1].ig        # Join: 所有 GA → pre
```

这隐含地假设所有 GA 步可以完全并行执行——**在真正的 pipeline 中这是错误的**。GA 步 = pipeline micro-batches，它们之间应该**顺序**流动，由激活内存约束和 PP 数据传输驱动。

### 2.4 Astra-sim 参考实现的分析

通过查阅 `astra-sim-alibabacloud/astra-sim/workload/Workload.cc`，确认：

- astra-sim 将所有 items 视为扁平的列表，**没有 pre/post/GA 的概念**——这些分类是 simai-flow-scheduler 自创的
- 执行模式是简单的 **全 forward（0→SIZE-1）→ 全 backward（SIZE-1→0）**
- GA 步的顺序执行由 flat list 的自然遍历保证，不是通过 fork-join
- pre items（grad_gather 等）在文件开头、post items（cross_entropy/optimizer）在末尾，**它们只是位置上的概念，不是语义上的类别**

### 2.5 PP 实现的简化

当前 PP 实现中，所有 PP stage 执行完全相同的 items（pre、GA layers、post 全部相同），差异只来源于 PP flow 的跨 stage 依赖（`stage_k → stage_{k+1}` 的激活传输和反向梯度传输）。

这是设计文档承认的简化（`extend-pp-flows.md:305-311`）。

---

## 3. 1F1B 设计方案

### 3.1 核心思想

将 AICB header 中的 `ga`（gradient accumulation steps）重新解释为 **pipeline micro-batch 数量**。GA 组从"可并行单位"改为"顺序流水处理的 micro-batch"。

### 3.2 经典 1F1B 的时间线

以 pp=4, ga=8 为例（每个 stage k 有不同长度的 warmup）：

```
Stage 0 (k=0): F0 F1 F2 F3 B0 F4 B1 F5 B2 F6 B3 B4 B5 B6 B7
Stage 1 (k=1):    F0 F1 F2 B0 F3 B1 F4 B2 F5 B3 F6 B4 B5 B6 B7
Stage 2 (k=2):       F0 F1 B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 B6 B7
Stage 3 (k=3):          F0 B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 B6 B7
                    ↑ warmup ↑     ↑ steady state     ↑ cooldown ↑
```

公式：
- `warmup(k) = pp - 1 - k`：stage k 初始只做 forward 的 micro-batch 数量
- `steady_count = ga - warmup(k)`：1F1B 交替执行的步数
- `cooldown(k) = warmup(pp-1-k)`：cooldown 阶段只有 backward

---

### 3.3 需要修改的模块

#### 修改 1：DAG 依赖图 — per-GA bridge + 删除 Backward Fork

**文件**：`src/workload_generator/workload_builder.py:_wire_dependencies()`

**改动 a**：将 bridge 从"最后一个操作"改为"每个 GA 组独立 bridge"

```python
# 当前：只连接最后一个 GA 组（或 post items 的最后一个操作）
if post_items:
    bridge_item = post_items[-1]
elif ga_groups:
    bridge_item = ga_groups[-1][-1]
...

# 改为：每个 GA 组独立 bridge（1F1B 需要每个 GA 步的 fwd→bwd 数据依赖）
for ga_group in ga_groups:
    last_item = ga_group[-1]
    self._wire_per_node_phase_transition(
        src_result=last_item.fwd_result,
        src_computes=last_item.fwd_computes,
        dst_computes=last_item.ig_computes)

if post_items:
    last_post = post_items[-1]
    self._wire_per_node_phase_transition(
        src_result=last_post.fwd_result,
        src_computes=last_post.fwd_computes,
        dst_computes=last_post.ig_computes)
```

**改动 b**：删除 Backward Fork

```python
# 当前：post[0].ig → ga_group[-1].ig (Backward Fork — 错误的依赖方向)
#       GA 的 backward 被 post 的 backward 所 gate，但 post 的 backward 是空操作

# 改为：删除这段代码。GA 的 backward 只依赖 per-GA bridge（GA.fwd → GA.ig）
```

**原理**：参考 astra-sim 的实现，post items（cross_entropy, optimizer）的 backward 全部是 `compute=0, comm=NONE`。实际上在 astra-sim 的 flat-list 逆序遍历中，post backward **先于** GA backward 执行——只是因为它们全是空操作，不影响结果。post items 的 backward 不对 GA 的 backward 构成数据依赖，GA 的 backward 也不对 post backward 构成依赖。

**Post backward 的时序只由 post bridge（post[-1].fwd → post[-1].ig）和 compute_order 保证**，不需要额外的 GA→post 或 post→GA 依赖。

> **对 GPipe 的影响**：per-GA bridge 和 Backward Fork 删除**不影响 GPipe**。GPipe 下 compute_order 的 direction 优先级确保所有 forward 后所有 backward，DAG 的额外边不会改变执行顺序。

#### 修改 2：compute_order — 新增 OneFOneBSerializer

**文件**：`src/static_analysis/passes/task_serializer.py`

新增 `OneFOneBSerializer` 类，**不修改 `CppReferenceSerializer`**：

```python
class OneFOneBSerializer(TaskSerializer):
    """
    1F1B 排序：按逻辑时间轴位置排列，每个 task 的位置由
    pp 和 stage_id 决定。
    
    1F1B 排序规则：
    - 对 Stage k (0-indexed)，warmup = pp - 1 - k
    - 排序键将每个 task 投射到逻辑时间轴上的位置：
        forward GA[i] → 位置 i
        backward GA[i] → 位置 warmup + i
    - 然后按位置升序排列
    
    pre items（iteration=-1）排在所有 GA 之前
    post items（iteration=ga）排在所有 GA 之后
    """
    
    def __init__(self, pp: int = 1, node_to_stage: dict[int, int] | None = None):
        self.pp = pp
        self._node_to_stage = node_to_stage or {}
    
    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        ...
        # 遍历 workload.tasks，按 (task.node → stage) 分组
        # 对每个 stage，按 1F1B 排序键排列
        ...
    
    def _compute_sort_key(self, task: Task, warmup: int) -> tuple:
        if task.iteration == -1:
            return (0, task.layer_id, task.item_id)
        if task.iteration == self.pp:
            return (2, task.layer_id, task.item_id)
        
        is_bwd = task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT)
        pos = (warmup + task.iteration) if is_bwd else task.iteration
        return (1, pos, -task.layer_id if is_bwd else task.layer_id, ...)
```

#### 修改 3：新增 OneFOneBAnalyzer

**文件**：`src/static_analysis/strategies/default_strategy.py`

新增 `OneFOneBAnalyzer` 类，**不修改 `DefaultAnalyzer` / `LightweightAnalyzer` / `CassiniAnalyzer`**：

```python
class OneFOneBAnalyzer:
    """与 LightweightAnalyzer 相同，但使用 OneFOneBSerializer。"""

    def __init__(self, pp: int, node_to_stage: dict[int, int]):
        self._serializer = OneFOneBSerializer(pp=pp, node_to_stage=node_to_stage)

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        plan = self._serializer.serialize(workload)
        return DefaultAnalysisResult(
            route_table=BfsRouteTable(None),
            execution_plan=plan,
        )
```

`node_to_stage` 从 Job 的 `assigned_nodes` 和 `parallelism` 推导：

```python
node_to_stage = {}
stage_size = dp * ep * tp
for stage_id in range(pp):
    for i in range(stage_size):
        node = assigned_nodes[stage_id * stage_size + i]
        node_to_stage[node] = stage_id
```

#### 修改 4：新增 1F1B 端到端脚本

**文件**：`scripts/run_cassini_dynamic_e2e_1f1b.py`

不修改现有脚本，从 `run_cassini_dynamic_e2e.py` 复制一份并替换关键行：

```python
# 原有 GPipe:
analyzer = LightweightAnalyzer()

# 改为 1F1B:
analyzer = OneFOneBAnalyzer(
    pp=job.parallelism.pp,
    node_to_stage=...,
)
```

---

### 3.4 不修改的文件

| 文件 | 原因 |
|------|------|
| `_generate_pp_flows()` | GA 步间已独立生成 PP 流，结构无需改动 |
| `_wire_pp_dependencies()` | PP 跨 stage 的通信依赖已正确建立（last fwd → PP → first fwd） |
| `executor/policies/` | Policy 受 compute_order 驱动，不需改动 |
| `CppReferenceSerializer` | 保留，维持 GPipe 行为不变 |
| `DefaultAnalyzer` / `LightweightAnalyzer` / `CassiniAnalyzer` | 保留，不修改现有接口 |
| 所有现有 e2e 脚本 | 保留，不修改参数 |

---

### 3.5 改动清单

#### 修改的文件

| 文件 | 改动 | 风险 |
|------|------|------|
| `workload_builder.py:_wire_dependencies()` | bridge 改为 per-GA；删除 Backward Fork | 对 GPipe 行为无影响（已验证） |

#### 新增的文件

| 文件 | 内容 | 职责 |
|------|------|------|
| `task_serializer.py` | `OneFOneBSerializer` 类 | 1F1B compute_order 排序 |
| `default_strategy.py` | `OneFOneBAnalyzer` 类 | 使用 OneFOneBSerializer 的分析器 |
| `run_cassini_dynamic_e2e_1f1b.py` | 新 e2e 脚本 | 1F1B 模拟入口 |

#### Forward Join / Backward Join / Post bridge

这三个结构**全部保留**，它们表达的是真实的阶段间数据依赖。Backward Fork 是唯一被删除的。

---

## 4. 验证方案

### 4.1 DAG 验证

| 测试 | 文件 | 验证内容 |
|------|------|---------|
| Per-GA bridge | `test_workload_builder.py` | 每个 GA 组都有 fwd→ig 依赖 |
| Backward Fork 删除 | `test_workload_builder.py` | post.ig 不 gate GA.ig |
| 1F1B compute_order | `test_task_serializer.py` | 排序结果符合 1F1B 规律 |
| 无环验证 | `test_task_serializer.py` | 合并 DAG + compute_order 后无环 |

### 4.2 执行验证

对以下配置做 E2E 测试，比较 GPipe vs 1F1B 的 completion time 和 bubble 大小：

| pp | ga | 期望 | 原因 |
|----|----|------|------|
| 1 | N | 两者相同 | 无 PP 时 GPipe = 1F1B |
| 2 | 1 | 两者相同 | ga=1 时无 warmup/cooldown |
| 4 | 16 | 1F1B 显著更快 | 大量 GA 步时 1F1B 的 bubble 更小 |
| 2 | 2 | 1F1B 略快 | 小规模验证 |

### 4.3 可视化

使用现有的 `visualizer.py` 绘制 Gantt 图，1F1B 下应看到：

```
Stage 0: [F0][F1][F2][B0][F3][B1][F4][B2][B3]
Stage 1:      [F0][F1][B0][F2][B1][F3][B2][F4][B3]
                    ↖ 交叉重叠而非全 F→全 B
```

---

## 5. 与 astra-sim 的差异说明

| 特征 | astra-sim (C++) | simai-flow-scheduler (当前) | 1F1B 实现后 |
|------|-----------------|---------------------------|-------------|
| pre/post items 处理 | 无概念，全部 flat list | 显式分类，fork-join | 保留分类但删除 Backward Fork |
| GA 组关系 | 顺序遍历 flat list | 并行 fork-join | 顺序（通过 compute_order + per-GA bridge） |
| 执行顺序 | 全 F → 全 B | 全 F → 全 B (GPipe) | 1F1B 交替 |
| PP 通信 | 纯公式计算 | P2P PP_SEND 流 | 保留 |
| pre items 的 DP 通信 | 各 rank 独立执行 | 各 rank 独立执行 | 保留 |

---

## 6. 后续可扩展方向

1. **Interleaved 1F1B**：当 `vpp > 1` 时，每个 stage 可以在不同 micro-batch 的不同 layer 上交替执行
2. **V-shape 调度**：Pipeline schedule 的另一种变体
3. **激活内存模型**：显式建模 activation 内存占用，使 1F1B 的约束从"compute_order"变为"内存容量"
4. **Dynamic 1F1B**：在动态 executor 中，以 job 粒度调度 1F1B 模式的 training iterations

---

## 7. 附录：术语表

| 术语 | 定义 |
|------|------|
| GA (Gradient Accumulation) | 梯度累积步数；在 PP 上下文中等价于 pipeline micro-batch 数量 |
| vpp (Virtual Pipeline Parallel) | 每个 PP stage 承担的层数 = 总层数 / pp |
| PP (Pipeline Parallelism) | 流水线并行，将网络层切分到不同设备 |
| 1F1B | One Forward One Backward：每个 micro-batch 的 forward 后立即执行其 backward |
| GPipe | 经典 pipeline 调度：所有 micro-batch forward 完后才执行 backward |
| Warmup | 1F1B 初始阶段，只执行 forward（向 pipeline 中填充数据） |
| Steady state | 1F1B 稳态，交替执行 forward 和 backward |
| Cooldown | 1F1B 收尾阶段，只执行 backward（排出 pipeline 中剩余数据） |
| Per-GA bridge | 每个 GA 组内部 fwd→ig 的数据依赖（backprop 需要 activation） |
| Backward Fork | 当前代码中 post.ig → GA.ig 的依赖（应当删除） |

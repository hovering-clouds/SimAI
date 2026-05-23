# Cassini 调度策略解读

本文档用于梳理论文 *Cassini: Network-Aware Job Scheduling in Machine Learning Clusters*
(NSDI '24) 的核心机制，帮助理解哪些方法值得在 `simai-flow-scheduler` 中复现，
以及如何与现有系统架构集成。

本文不是逐段翻译论文，而是回答三个更实际的问题：

1. Cassini 到底在调度什么？
2. 它的几何抽象和 time-shift 方法具体怎么做？
3. 哪些机制是收益的核心，哪些是辅助性设计？

---

## 1. Cassini 的总体思路

Cassini 的基本立场是：DNN 训练作业的通信模式是**周期性**的——每个 iteration
都重复相同的 Up（高带宽）和 Down（低带宽）阶段。如果多个作业的 Up 阶段在时间上
重叠，它们共享的链路就会发生争用。

Cassini 的核心想法是：既然通信模式是周期性的，就可以把时间"卷"成一个圆，
通过**旋转圆**来找到一种时间偏移（time-shift），让不同作业的 Up 阶段相互错开。

这与传统做法截然不同：

| 传统调度 | Cassini |
|----------|---------|
| 按资源需求量做分配 | 按通信模式的时间分布做对齐 |
| 静态分配带宽份额 | 动态调整作业的时间偏移 |
| 不感知通信的阶段特征 | 显式建模 Up/Down 相位 |
| 多作业共存时各自为政 | 主动交错通信以减少争用 |

---

## 2. 输入假设和建模边界

### 2.1 作业必须有可预测的周期性通信

Cassini 假设每个作业的 iteration time 可以提前估计（通过 profiling 或历史运行），
且通信模式在每个 iteration 中保持一致。这与 AI 训练作业的特性高度吻合。

### 2.2 只关注共享链路的争用

Cassini 不解决单作业内部的网络优化（那是 Puppeteer 做的事），而是解决
**多作业之间**的网络争用。这意味着它天然是一个多作业调度器，需要知道每个
作业经过哪些链路。

### 2.3 迭代层级的控制

Cassini 不以单个 flow 为控制粒度，而是以 **iteration time-shift** 为控制手段。
它不改变作业内部的 DAG 和 flow 顺序，只改变作业下一轮 iteration 的开始时间。

---

## 3. 核心方法 #1：通信模式提取

### 3.1 目标

给定一个作业的完整训练执行 trace（或 P2PWorkload），提取它的周期性通信特征：

- **Iteration time**：一轮 iteration 的总时长（从第一个 task 开始到最后一个 task 结束）
- **Up 阶段**：有网络通信的时间段（flow task 活跃）
- **Down 阶段**：只有计算的时间段（仅有 compute task）

### 3.2 方法

从 `P2PWorkload` 出发：

1. 按 `iteration` 字段对 task 分组
2. 利用 `CriticalPathInfo` 获得每个 task 的 `earliest_start_us` 和 `earliest_finish_us`
3. 对每个 flow task，通过 `RouteTable.get_path()` 获知它经过哪些链路
4. 汇总成每个作业在每条链路上的带宽需求时间序列

### 3.3 输出数据

```
CommunicationPattern:
  job_id: int
  iteration_time_us: int
  link_demands: dict[link_id, list[(start_us, end_us, bw_gbps)]]
```

---

## 4. 核心方法 #2：几何圆抽象

### 4.1 核心思想

"把时间卷成一个圆"：

- 圆的**周长** = job 的 iteration time
- 每个角度 α 对应一个时刻 t = α / 2π × perimeter
- 角度 α 上的**带宽需求** = 该时刻作业在此链路上通信所需的带宽
- Up 阶段 → 高带宽需求的弧段
- Down 阶段 → 低/零带宽需求的弧段

### 4.2 形式化定义

```
CircleAbstraction:
  perimeter: int                # = iteration_time_us
  bw_demand: dict[int, float]  # 角度 α (0-359) → 带宽需求 (Gbps)
```

`bw_demand[α]` 的取值方式：将时间轴离散化成 360 等份（或更细粒度），
每个角度对应的时间 `t = α / 360 × perimeter`，取该时刻所有活跃 flow
在该链路上的带宽需求之和。

### 4.3 旋转变换

旋转圆 = 对作业施加 time-shift ∆：

```
bw_demand_after_shift[α] = bw_demand[(α - ∆) mod 360]
```

物理含义：将作业的下一轮 iteration 推迟 ∆/360 × perimeter 微秒开始。

---

## 5. 核心方法 #3：统一圆（LCM Unified Circle）

### 5.1 问题

当两个作业的 iteration time 不同时（如 40ms 和 60ms），不能直接用相同的圆周长。

### 5.2 方法

1. 计算所有 job iteration time 的 **LCM**（最小公倍数）
   - 例：LCM(40, 60) = 120
2. 用 LCM 作为统一圆的周长
3. 每个 job 在统一圆上出现 `LCM / iteration_time` 次
   - 40ms 的 job 出现 3 次（每 120ms 有 3 轮 iteration）
   - 60ms 的 job 出现 2 次
4. 在统一圆上叠加所有 job 的带宽需求

### 5.3 接口

```python
def build_unified_circle(
    patterns: list[CommunicationPattern],
    link_capacity: float,
) -> tuple[int, list[CircleAbstraction]]:
    """返回 (lcm_perimeter, 每个job在统一圆上的表示)"""
```

---

## 6. 核心方法 #4：成对兼容性优化

### 6.1 问题

给定共享同一条链路的两个 job，找到最优 time-shift ∆ 使它们同时运行时
链路争用最小。

### 6.2 兼容性得分定义

```
demand_sum(α) = job1.unified_circle.demand_at(α - ∆1) 
                + job2.unified_circle.demand_at(α - ∆2)

Excess(α) = max(0, demand_sum(α) - link_capacity)

score = 1 - (Σ_α Excess(α)) / (num_angles × link_capacity)
```

- score = 1：完全兼容（零争用）
- score = 0：平均争用刚好占满链路容量
- score < 0：高度不兼容

### 6.3 优化问题

| 成分 | 内容 |
|------|------|
| **输入** | 共享同一条链路的一组 job 的 unified circles |
| **变量** | 每个 job 的旋转角 ∆_j |
| **约束** | 0 ≤ ∆_j < 360 / r_j（r_j = 该job在统一圆上的重复次数） |
| **目标** | 最大化 compatibility score |

### 6.4 实现方法

**方法一：网格搜索**。对于小规模（2-3 个 job），可以枚举所有可能的 time-shift
组合（步长 1° 或 5°），计算每种组合的 score，取最优。

**方法二：对 q 个 jobs，固定 q-1 个，优化第 q 个**。Cassini 论文中描述的方法，
通过图遍历逐步固定 time-shift。

---

## 7. 核心方法 #5：二分亲和图与图遍历

### 7.1 问题

一对 job 在同一链路上的优化，扩展到整个集群拓扑中 N 个 job 和 M 条链路。

### 7.2 亲和图

Cassini 构建一个**二分图**：

- 左侧顶点：所有需要调度的 Jobs
- 右侧顶点：所有被多个 job 共享的 Links
- 边：Job 经过某条 Link（从 route table 获知）

### 7.3 图遍历算法

1. 对每条 link，收集经过它的所有 job
2. 按 link 的争用严重程度排序（经过的 job 数越多、总带宽需求越大，优先级越高）
3. 从最争用的 link 开始：
   a. 对此 link 上的所有 job 求解联合兼容性优化
   b. 固定这些 job 的 time-shift
   c. 将已固定的 time-shift 传播到相邻 link（同个 job 在所有 link 上共用同一个 time-shift）
4. 重复直到所有 job 的 time-shift 都被确定

### 7.4 实现要点

每个 job 在所有链路上只有**一个**全局 time-shift（不是每条链路一个）。
这是约束也是简化：它保证了调度的一致性，但可能无法在所有链路上同时达到最优。

```python
def compute_cluster_time_shifts(
    patterns: dict[int, CommunicationPattern],  # job_id → pattern
    job_links: dict[int, set[tuple[int, int]]], # job_id → set of link_ids
    topology: NetworkTopology,
) -> dict[int, int]:  # job_id → time-shift (微秒)
```

---

## 8. 核心方法 #6：与调度器的集成

### 8.1 集成方式

Cassini 论文将其设计为**可插拔模块**（约 1000 行代码），将 time-shift 值输出给
现有的调度器（Themis、Pollux），不改变调度器的核心逻辑。

在 `simai-flow-scheduler` 中，Cassini 作为 `SchedulingPolicy` 的实现：

1. **分析阶段**（`CassiniStrategy`）：从多 job workload 中提取通信模式，
   计算每个 job 的 time-shift
2. **执行阶段**（`CassiniPolicy`）：在 executor 运行时，按 time-shift 偏移每个
   job 的 iteration 起始时间

### 8.2 Time-shift 的执行方式

在 `emit_ready_tasks()` 中：

- 维护每个 job 的 iteration 计数器 + 当前 iteration 的预期开始时间
- 如果当前时间 < 该 job 的下一个预期开始时间（含 time-shift），
  则不释放该 job 下一轮 iteration 的 task
- 一旦到达预期时间，释放该 job 下一轮所有 ready task

### 8.3 与 Puppeteer 的协同

Cassini 和 Puppeteer 解决不同层次的问题：

| 维度 | Puppeteer | Cassini |
|------|-----------|---------|
| 目标 | 单作业内 flow 级的 route + rate 优化 | 多作业间 iteration 级的 time-shift |
| 控制粒度 | 每条 flow 的路径和带宽 | 每轮 iteration 的起始时间 |
| 解决的问题 | 关键路径上的通信瓶颈 | 多作业共存时的链路争用 |
| 实现位置 | `static_analysis/` + `policies/` | `cassini/` 新模块 |

两者可以结合：先由 Cassini 决定 time-shift，再由 Puppeteer 做单作业内的
细粒度调度。

---

## 9. 哪些是 Cassini 最值得复现的核心策略

### 第一层：必须抓住的核心

1. **通信模式提取** — 从 P2PWorkload 中提取 Up/Down 相位
2. **几何圆抽象 + 旋转** — 将时间映射到圆上，支持 time-shift
3. **成对兼容性优化** — 找到最优 time-shift 使争用最小

没有这三点，就无法体现 Cassini 的方法论。

### 第二层：强烈建议有

4. **统一圆 (LCM)** — 支持不同 iteration time 的 job
5. **亲和图 + 图遍历** — 扩展到多 job 多 link 的集群级别

这一层决定了方案能否真正解决多作业场景的问题。

### 第三层：更像论文，但成本更高

6. **与现有调度器的可插拔集成** — 不影响现有 DefaultPolicy 和 PuppeteerPolicy
7. **大规模集群的近似求解优化** — 当 job 数量较大时，网格搜索不再可行

---

## 10. 从复现角度看，最容易误解的地方

### 10.1 Cassini 不是带宽分配器

Cassini 不做带宽分配——它通过 time-shift 来减少多 job 的通信叠加，
带宽分配仍由下层的策略（fair share 或 TTE-aware）决定。

### 10.2 Time-shift 不改变作业内 DAG

Time-shift 只是推迟整轮 iteration 的起始时间，不改变作业内部的 task 依赖关系
和执行顺序。它也不会引入新的跨作业依赖边。

### 10.3 一个 job 只有一个全局 time-shift

同一 job 在不同链路上的 time-shift 必须相同。这是图遍历传播时的硬约束，
也意味着优化可能无法在所有链路上同时达到局部最优。

### 10.4 与同通信压缩/并行策略正交

Cassini 不影响 TP size、PP size、gradient accumulation steps 等训练超参数的选择。
它是在这些参数已经确定的基础上，优化多作业共存时的网络争用。

---

## 11. 对 `simai-flow-scheduler` 的启发

### 11.1 可以复用的现有模块

| 现有模块 | Cassini 中的用途 |
|----------|-----------------|
| `workload_format/schema.py` | `P2PWorkload` 作为分析输入 |
| `static_analysis/passes/critical_path.py` | 提供每个 task 的起止时间 |
| `static_analysis/passes/routing/` | `RouteTable.get_path()` 获知 flow 经过哪些 link |
| `static_analysis/passes/topology_loader.py` | `NetworkTopology.get_link()` 获取链路容量 |
| `workload_generator/job_merger.py` | 合并多 job workload |
| `executor/policies/base_policy.py` | `SchedulingPolicy` 接口 |
| `executor/analytical.py` | 离散事件模拟器 |
| `executor/bandwidth_allocators/fair_share_allocator.py` | 带宽分配（Cassini 不替代此项） |

### 11.2 新建模块的定位

```
src/cassini/                       ← Cassini 核心算法
├── communication_pattern.py       ← 通信模式提取
├── circle_abstraction.py          ← 几何圆抽象 + 统一圆 (LCM)
├── pair_compatibility.py          ← 成对兼容性优化
└── affinity_graph.py              ← 二分亲和图 + 图遍历

src/executor/policies/
└── cassini_policy.py              ← Cassini 调度策略

src/static_analysis/strategies/
└── cassini_strategy.py            ← Cassini 分析管线
```

---

## 12. 一句话总结

Cassini 的核心贡献是：**用几何圆抽象来表达作业的周期性通信模式，通过圆旋转
（time-shift）将多个作业的通信阶段在时间上错开，减少链路争用**。

对 `simai-flow-scheduler` 来说，Cassini 填补了一个空白：
当前系统可以高效调度单个作业（Puppeteer），也可以公平共享带宽（DefaultPolicy），
但缺少一种**感知多作业通信模式的协调机制**。Cassini 正好补上这一层。

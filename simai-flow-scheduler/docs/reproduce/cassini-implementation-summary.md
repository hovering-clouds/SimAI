# Cassini 策略复现总结

> 基于论文 *Cassini: Network-Aware Job Scheduling in Machine Learning Clusters* (NSDI '24)
> 与 `simai-flow-scheduler` 当前架构的对比及实现规划。

---

## 整体评估

| 维度 | 状态 | 说明 |
|------|------|------|
| 通信模式提取 (Pattern Extraction) | ❌ 未实现 | 需新增，从 P2PWorkload + CriticalPathInfo 提取 |
| 几何圆抽象 (Circle Abstraction) | ❌ 未实现 | 新增 `cassini/circle_abstraction.py` |
| 统一圆 (LCM Unified Circle) | ❌ 未实现 | 新增，支持异周期 job |
| 成对兼容性优化 (Pairwise Compatibility) | ❌ 未实现 | 新增 `cassini/pair_compatibility.py` |
| 二分亲和图 (Affinity Graph) | ✅ 已实现 | `cassini/affinity_graph.py` |
| Time-Shift 集成到 Policy | ✅ 已实现 | `cassini_policy.py` |
| Cassini 分析管线 | ✅ 已实现 | `cassini_strategy.py` |
| 端到端多作业模拟 | ⚠️ 已有基础 | `JobMerger` 已支持合并，但未接入 Cassini |

---

## 1. 通信模式提取

### 论文描述

Cassini 需要将 DNN 训练作业的通信特征表达为**周期性的 Up/Down 相位**：
- Up 阶段：网络通信活跃，flow task 在传输
- Down 阶段：仅有计算，无网络活动（或低网络活动）
- 整个模式在每个 iteration 中重复

### 实现规划

**文件：** `src/cassini/communication_pattern.py`

**核心逻辑：**

1. 从 `P2PWorkload` 中按 `job_id` 和 `iteration` 分组 task
2. 利用 `CriticalPathInfo` 获取每个 task 的 `earliest_start_us` 和 `earliest_finish_us`
3. 对每个 flow task，查 `RouteTable.get_path()` 获知其经过哪些 link
4. 将时间轴离散化，计算每个离散点上每条 link 的带宽需求

**输出 dataclass：**
```python
@dataclass
class CommunicationPattern:
    job_id: int
    iteration_time_us: int
    # link_id → 离散化带宽需求 (角度 → Gbps)
    link_demands: dict[tuple[int, int], dict[int, float]]
```

**依赖：** `P2PWorkload`, `CriticalPathInfo`, `RouteTable`

---

## 2. 几何圆抽象

### 论文描述

Cassini 将时间"卷"到圆上，圆的周长等于 job 的 iteration time。
每个角度 α 对应时间 t = α / 2π × perimeter，带宽需求 `bw(α)` 表示该时刻
作业在给定链路上的通信速率。旋转圆 ∆ 度等价于施加 time-shift。

### 实现规划

**文件：** `src/cassini/circle_abstraction.py`

**核心接口：**
```python
class CircleAbstraction:
    perimeter: int                           # iteration time (us)
    bw_demand: dict[int, float]              # 角度 α (0-359) → 带宽需求 (Gbps)
    
    def rotate(self, shift_deg: int) -> "CircleAbstraction":
        """返回旋转后的新圆"""
    
    def demand_at(self, angle: int) -> float:
        """角度 α 处的带宽需求"""
    
    @staticmethod
    def build_unified(
        patterns: list[CommunicationPattern],
        link_id: tuple[int, int],
    ) -> tuple[int, list["CircleAbstraction"]]:
        """构建统一圆 (LCM)，返回 (lcm_perimeter, circles)"""
```

**离散化方案：** 将 360° 等分（步长 1°），每度对应 `perimeter / 360` 微秒。

---

## 3. 统一圆 (LCM)

### 论文描述

当作业的 iteration time 不同时（例如 40ms 和 60ms），不能直接放到同一个圆上。
Cassini 用 LCM 作为统一圆的周长，每个作业按 `LCM / iteration_time` 次重复出现。

### 实现规划

**文件：** 同上 `circle_abstraction.py`（作为 `CircleAbstraction.build_unified`）

**算法：**
```python
def _lcm_perimeter(times: list[int]) -> int:
    lcm = 1
    for t in times:
        lcm = lcm * t // math.gcd(lcm, t)
    return lcm
```

**限制：** LCM 可能非常大（如 101ms × 103ms = 10403ms），需要设 `max_perimeter` 上限。

---

## 4. 成对兼容性优化

### 论文描述

给定共享同一条链路的两个 job，Cassini 求解最优旋转角 ∆ 以最大化兼容性得分：

```
score = 1 - Σ_α Excess(α) / (|A| × C^l)

Excess(α) = max(0, sum(bw_j(α - ∆_j)) - C^l)

变量: ∆_j (每个 job 的旋转角)
约束: 0 ≤ ∆_j < 360 / r_j (r_j = 在统一圆上的重复次数)
目标: max score
```

### 实现规划

**文件：** `src/cassini/pair_compatibility.py`

```python
@dataclass
class CompatibilityResult:
    score: float
    time_shifts_us: dict[int, int]  # job_id → time-shift (us)
    
def pairwise_optimize(
    circles: dict[int, CircleAbstraction],
    link_capacity: float,
    step_deg: int = 5,
) -> CompatibilityResult:
    """网格搜索最优 time-shift"""
```

**搜索策略：**
- 2 job：固定 job[0] 在 0°，搜索 job[1] 的旋转角，步长 5°
- 3+ job：贪心策略，每次固定一个 job 的 time-shift

---

## 5. 二分亲和图与图遍历

### 论文描述

Cassini 构建二分图：jobs ↔ links，边表示 job 经过该 link。
按 link 的争用严重程度排序，从最严重的开始逐步求解并传播 time-shift。

### 实现规划

**文件：** `src/cassini/affinity_graph.py`

```python
@dataclass
class AffinityGraph:
    job_links: dict[int, set[tuple[int, int]]]   # job_id → {link_ids}
    link_jobs: dict[tuple[int, int], set[int]]   # link_id → {job_ids}
    link_capacity: dict[tuple[int, int], float]
```

**图遍历算法：**
```
1. 计算每条 link 的争用权重：w = num_jobs × avg_demand
2. 按 w 降序排列 links
3. 从最争用的 link 开始，对其上的 jobs 联合求解
4. 固定 time-shift，传播到相邻 link（同一 job 在全部 link 上共用同一个 shift）
5. 重复直到所有 job 的 time-shift 确定
```

---

## 6. Cassini 分析管线

**文件：** `src/static_analysis/strategies/cassini_strategy.py`

```python
@dataclass
class CassiniAnalysisResult:
    route_table: RouteTable
    critical_path: CriticalPathInfo
    communication_patterns: dict[int, CommunicationPattern]
    time_shifts: dict[int, int]        # job_id → time-shift (us)
    execution_plan: ExecutionPlan

class CassiniAnalyzer:
    """管线流程: 路由 → 关键路径 → 模式提取 → 亲和图 → time-shift"""
```

---

## 7. Cassini 调度策略

**文件：** `src/executor/policies/cassini_policy.py`

```python
class CassiniSchedulingPolicy(SchedulingPolicy):
    """在 emit_ready_tasks() 中根据 time-shift 控制 job iteration 的释放时机"""
```

**关键设计：**
- Time-shift 以 iteration 为单位执行，不改变作业内 DAG
- 复用 `FairShareAllocator` 做带宽分配
- 复用 `CppReferenceSerializer` 做 compute ordering

---

## 8. 测试计划

| 测试文件 | 测试内容 |
|----------|----------|
| `tests/test_cassini_communication_pattern.py` | 通信模式提取 |
| `tests/test_cassini_circle.py` | 圆抽象 + 旋转 + 统一圆 |
| `tests/test_cassini_compatibility.py` | 成对兼容性优化 |
| `tests/test_cassini_affinity_graph.py` | 亲和图 + 图遍历 |
| 集成测试 | CassiniPolicy vs DefaultPolicy 对比 |

---

## 9. 实现文件索引

| 文件 | 角色 | 对应论文部分 |
|------|------|-------------|
| `src/cassini/communication_pattern.py` | 通信模式提取 | §3.1 Communication Patterns |
| `src/cassini/circle_abstraction.py` | 几何圆抽象 + 统一圆 | §3.2 / §4.1 Unified Circle |
| `src/cassini/pair_compatibility.py` | 成对兼容性优化 | §4.2 Compatibility Score |
| `src/cassini/affinity_graph.py` | 二分亲和图 + 图遍历 | §4.3 Affinity Graph |
| `src/executor/policies/cassini_policy.py` | Cassini 调度策略 | §3 System Design |
| `src/static_analysis/strategies/cassini_strategy.py` | 分析管线 | §3.1 Overview |
| `tests/test_cassini_*.py` | 测试 | 验证 |

---

## 10. 与 Puppeteer 的关系

| 维度 | Puppeteer | Cassini |
|------|-----------|---------|
| 目标 | 单作业内 flow 级 route/rate 优化 | 多作业间 iteration 级 time-shift |
| 控制粒度 | 每条 flow 的路径和带宽 | 每轮 iteration 的起始时间 |
| 路由 | 贪心路由 (Greedy) | 复用 BFS |
| 带宽分配 | TTE-aware | Fair Share（可替换） |

两者正交可叠加：Cassini 决定 time-shift，Puppeteer 做 flow 级细粒度调度。

---

## 11. 推荐实施顺序

**Phase 1（核心算法）：**
1. `communication_pattern.py` — 通信模式提取
2. `circle_abstraction.py` — 圆抽象 + 统一圆
3. `pair_compatibility.py` — 成对兼容性优化

**Phase 2（系统集成）：**
4. `cassini_strategy.py` — 分析管线
5. `cassini_policy.py` — 调度策略
6. 集成测试：CassiniPolicy vs DefaultPolicy

**Phase 3（完整能力）：**
7. `affinity_graph.py` — 多 job 多 link 集群级调度
8. 端到端脚本 + 大规模测试

---

## 12. Phase 3 详细实现计划：亲和图遍历

### 12.1 问题诊断

当前 `cassini_strategy.py:195-234` 的 `_aggregate_shifts()` 方法使用简化启发式：
每个 job **独立地**从其带宽需求最高的链路上选取 time-shift。这存在两个问题：

1. **违反全局约束**：Cassini 论文要求一个 job 在所有链路上共享**同一个**全局 time-shift，
   但当前方法在各链路上独立优化后简单"投票"，不保证一致性。
2. **缺乏传播机制**：在链路 A 上为 job X 选定的 shift，不会影响链路 B 上
   job X 与 job Y 的联合优化，导致次优解。

### 12.2 论文算法回顾（§4.3 Affinity Graph）

Cassini 将集群调度建模为二分图上的遍历问题：

```
二分图:  Jobs (左侧顶点) ↔ Links (右侧顶点)
边:      Job 经过某条 Link（从 RouteTable 获取）
```

**图遍历算法：**

```
1. 计算每条链路的争用权重: w = num_jobs_on_link × avg_demand_on_link
2. 按 w 降序排列链路
3. 从最争用的链路开始:
   a. 收集该链路上尚未固定 time-shift 的 job 集合 U
   b. 对 U ∪ (该链路上已固定的 job) 联合求解兼容性优化
      — 已固定 job 的 shift 不可变（作为搜索约束）
      — 仅搜索 U 中 job 的 shift 空间
   c. 将 U 中 job 的优化结果写入全局 fixed_shifts
   d. 传播: 同一 job 在其他链路上也使用此固定值
4. 重复直到所有 job 的 time-shift 确定
5. 未出现在任何争用链路上的 job → shift = 0
```

**关键约束:** 每个 job 只有一个全局 time-shift，在图遍历中一旦固定就不再改变。
这保证了调度的一致性，但意味着优化结果可能无法在所有链路上同时达到局部最优。

### 12.3 实现文件

#### 12.3.1 新建 `src/cassini/affinity_graph.py`

**数据结构：**

```python
@dataclass
class AffinityGraph:
    """二分亲和图: jobs ↔ links."""
    job_links: dict[int, set[tuple[int, int]]]     # job → 经过的链路
    link_jobs: dict[tuple[int, int], set[int]]     # 链路 → 经过的 job
    link_capacities: dict[tuple[int, int], float]  # 链路容量 (Gbps)
    patterns: dict[int, CommunicationPattern]       # job 通信模式
```

**公开接口：**

```python
def build_affinity_graph(
    patterns: dict[int, CommunicationPattern],
    job_links: dict[int, set[tuple[int, int]]],
    topology: NetworkTopology,
) -> AffinityGraph:
    """从已有数据构建二分亲和图。"""

def compute_cluster_time_shifts(
    graph: AffinityGraph,
    step_deg: int = 5,
) -> dict[int, int]:
    """图遍历主算法，返回 job_id → time_shift_us。"""
```

**内部辅助函数：**

```python
def _link_contention_weight(
    graph: AffinityGraph, link_id: tuple[int, int]
) -> float:
    """计算链路争用权重: num_jobs × avg_demand."""
    job_ids = graph.link_jobs.get(link_id, set())
    if not job_ids:
        return 0.0
    total_demand = 0.0
    for jid in job_ids:
        pattern = graph.patterns.get(jid)
        if pattern and link_id in pattern.link_demands:
            d = pattern.link_demands[link_id]
            total_demand += sum(d.values()) / len(d) if d else 0.0
    return len(job_ids) * total_demand / len(job_ids)


def _optimize_with_fixed(
    circles: dict[int, CircleAbstraction],   # all circles on this link
    fixed_shifts: dict[int, int],             # job_idx → fixed shift (deg)
    link_capacity: float,
    step_deg: int,
) -> CompatibilityResult:
    """部分固定、部分搜索的兼容性优化。

    与 optimize_link_compatibility 的区别:
    - fixed_shifts 中的 circle 不参与搜索，其 shift 值固定
    - 仅搜索未被固定的 circle 的 shift 空间
    - 2 job 时: 若 1 个已固定，退化为 1D 搜索
    - 3+ job 时: 贪心搜索仅遍历未固定维度
    """
```

**实现要点：**

- `_optimize_with_fixed` 复用 `pair_compatibility.py` 中的 `_search_angles`、`_pre_sample`、`compute_score`、`_deg_to_us`
- 对于 2 job 场景，若 1 个已固定则为 O(72) 搜索（步长 5°），若都未固定则退化为现有的 `_optimize_two`
- 对于 3+ job 场景，贪心搜索中跳过已固定维度，只对未固定维度做 grid search
- 图遍历的主循环需要处理：某条链路上的所有 job 都已固定时，直接跳过

#### 12.3.2 修改 `src/static_analysis/strategies/cassini_strategy.py`

**删除：**
- `_aggregate_shifts()` 方法（第 196-234 行）
- `_avg_demand_on_link()` 方法（第 236-247 行）

**修改 `_compute_time_shifts()`：**

```python
def _compute_time_shifts(
    self,
    patterns: dict[int, CommunicationPattern],
    route_table: RouteTable,
    workload: P2PWorkload,
) -> dict[int, int]:
    if len(patterns) < 2:
        return {jid: 0 for jid in patterns}

    job_links = self._build_job_links(workload, route_table)
    graph = build_affinity_graph(patterns, job_links, self.topology)
    return compute_cluster_time_shifts(graph, self.step_deg)
```

**新增 import：**
```python
from ...cassini.affinity_graph import (
    AffinityGraph,
    build_affinity_graph,
    compute_cluster_time_shifts,
)
```

#### 12.3.3 修改 `src/cassini/__init__.py`

新增导出：

```python
from .affinity_graph import (
    AffinityGraph,
    build_affinity_graph,
    compute_cluster_time_shifts,
)
```

并在 `__all__` 和模块 docstring 中加入对应条目。

### 12.4 与 Phase 2 的关系

Phase 2 的 `cassini_policy.py` 和 `CassiniAnalysisResult` 无需修改——亲和图只替换
time-shift 的**计算方式**，不改变 time-shift 的**使用方式**。`CassiniSchedulingPolicy`
通过 `analysis.time_shifts` 获取结果，对底层算法透明。

### 12.5 待确认的设计问题

1. **已固定 job 的 shift 在后续链路上是否需要微调？** 论文中每个 job 只有一个全局 shift，
   图遍历中固定后不再改变。但如果后续链路在固定 shift 下兼容性得分极低（如 < 0），
   是否需要回退机制？当前计划按论文原意：固定即不变。

2. **LCM 溢出时（max_perimeter 限制），亲和图的精度损失？** `build_unified` 在 LCM 溢出
   时使用 max perimeter 近似，这对亲和图中不同链路的一致性是否有影响？需要在实现中
   记录警告。

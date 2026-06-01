# Cassini 策略复现总结

> 基于论文 *Cassini: Network-Aware Job Scheduling in Machine Learning Clusters* (NSDI '24)
> 与 `simai-flow-scheduler` 当前实现 (Phase 1-3) 的详细比较。

---

## 整体评估

| 维度 | 状态 | 说明 |
|------|------|------|
| 通信模式提取 (Pattern Extraction) | ✅ 完全实现 | §3.1，从 P2PWorkload + CPM timing 提取 per-link 带宽需求 |
| 几何圆抽象 (Circle Abstraction) | ✅ 完全实现 | §3.2，圆周长 = iteration time，360° 角度桶，支持 rotate |
| LCM 统一圆 (Unified Circle) | ✅ 完全实现 | §3.2，LCM tiling + max-pool；overflow 时降级到 time-proportional mapping |
| 成对兼容性优化 (Pairwise Compatibility) | ✅ 完全实现 | §3.3，网格搜索 + 0.02 阈值防无效 shift |
| 二分亲和图 (Affinity Graph) | ✅ 完全实现 | §4.1-4.3，BFS 遍历 + 争用权重排序 + 全局 shift 传播 |
| Time-Shift 集成到 Policy | ✅ 完全实现 | one-shot 门控，跨 iteration DAG 依赖维持周期性对齐 |
| Cassini 分析管线 | ✅ 完全实现 | BFS 路由 → CPM → 模式提取 → 亲和图 → 执行计划 |
| 端到端多作业模拟 | ✅ 完全实现 | `run_cassini_e2e.py` 支持 4 种模式 + 2 种 placement + 多 job 合并 |
| GPU Placement 策略 | ✅ 完全实现 | contiguous（无争用） + contention-spread（强制跨集群争用） |
| 迭代扩展 (Iteration Expansion) | ✅ 完全实现 | GA 合并 + 跨迭代 dep 链 + CPM timing 修正 |

---

## 1. 通信模式提取 (Communication Pattern)

### 论文描述

Cassini 将 DNN 训练作业的通信特征表达为周期性的 Up/Down 相位。每个作业的通信模式以 iteration 为周期重复。

### 当前实现 (`communication_pattern.py`)

**提取流程：**

1. 按 `job_id` 和 `iteration` 分组 task
2. 利用 `CriticalPathInfo` 获取每个 task 的 `earliest_start_us` / `earliest_finish_us`
3. 对每个 flow task，查 `RouteTable.get_path()` 获知其经过哪些 link
4. 对每条链路，将 flow 的传输时间离散化到 360 个角度桶中，带宽叠加
5. **链路速率修正**：CPM 的 timing 假设 NVLink 速率（~2880 Gbps），实际链路（如 200 Gbps ASW）会延长传输时间、降低有效带宽，避免高估瓶颈链路上的争用

**迭代时间估算：**

- 从 CPM timing 构建逻辑迭代窗口（pre/periodic/post）
- 跳过 warmup iteration 0，取剩余窗口的**中位数**作为稳态迭代时间
- 支持 Cassini 三标签格式 `[-1, 0, 1]` 的窗口识别

### 差异分析

| 差异点 | 论文 | 当前实现 | 评估 |
|--------|------|----------|------|
| Pattern 离散化 | 未明确具体分辨率 | 360 个角度桶 | ✅ 充分精细（~1° = 1/360 迭代时间） |
| 链路速率修正 | 未明确 | 显式按瓶颈链路带宽 cap | ✅ 更严谨，防止高估慢链路争用 |
| 迭代时间估计 | 未详细说明方法 | CPM timing median | ✅ 合理 |

**结论：** ✅ 完全实现。提取流程对齐论文 §3.1，且在链路速率修正上做了更严谨的处理。

---

## 2. 几何圆抽象 (Circle Abstraction)

### 论文设计

Cassion 将时间"卷"到一个圆上：
- 周长 = job iteration time
- 每个角度 α (0-359) 映射到带宽需求 bw(α)
- 旋转圆 ∆° 等价于给作业施加一个时间偏移

### 当前实现 (`circle_abstraction.py`)

**核心 dataclass：**
```python
@dataclass
class CircleAbstraction:
    perimeter: int                # 圆周长 (μs) = iteration time
    bw_demand: dict[int, float]   # 角度 0-359 → 带宽需求 (Gbps)
```

**主要方法：**

| 方法 | 功能 |
|------|------|
| `rotate(shift_deg)` | 顺时针旋转，返回新 circle |
| `demand_at(angle)` | 查询某个角度的带宽需求 |
| `from_pattern(pattern, link_id)` | 从 CommunicationPattern 构建 circle |
| `build_unified(patterns, link_id)` | 构建 LCM 统一圆 |

**LCM 统一圆算法：**

```python
# 1. 计算所有 iteration time 的 LCM
lcm = lcm(perimeter_1, perimeter_2, ...)

# 2. 如果 LCM ≤ max_perimeter (50s) → integer tiling
r = lcm / perimeter_i
# 每个 unified 桶 = max-pool over r 个原始桶

# 3. 如果 LCM overflow (>50s) → time-proportional mapping
# 每个 unified 桶精确映射到原始时间位置
angle_orig = (time_us * 360 / perimeter_i) % 360
```

### 差异分析

| 差异点 | 论文 | 当前实现 | 评估 |
|--------|------|----------|------|
| LCM 溢出处理 | 未明确 | time-proportional mapping | ✅ 合法近似 |
| max-pool tiling | 未明确 | 用 max 聚合避免低估峰值 | ✅ 合理保守 |
| overflow 阈值 | 未指定 | 50s（可配置） | ✅ 适配典型场景 |

**结论：** ✅ 完全实现，且在 LCM overflow 时提供了 time-proportional fallback，覆盖了论文未明确说明的边缘情况。

---

## 3. 成对兼容性优化 (Pairwise Compatibility)

### 论文公式

```
Excess(α) = max(0, Σ_j bw_j(α − ∆_j) − C_link)
score     = 1 − Σ_α Excess(α) / (360 · C_link)
```

### 当前实现 (`pair_compatibility.py`)

**优化策略：**

| 场景 | 算法 | 复杂度 |
|------|------|--------|
| 2 个 job，全自由 | 固定 job 0 在 0°，网格搜索 job 1 | O(72) (step=5°) |
| 3+ 个 job，全自由 | 迭代贪婪：每轮逐个坐标搜索，收敛后停止 | O(iter × n × 72) |
| 1 个固定，1+ 自由 | 锁定固定维度，仅搜索自由维度 | 同上，但搜索空间减小 |

**核心设计决策：**

- **0.02 阈值 (`_MIN_COMPAT_IMPROVEMENT`)**：如果旋转某个 job 带来的 score 提升 < 0.02，则放弃该 shift——因为延迟一个 job 的启动时间对 makespan 的伤害大于争用缓解带来的收益。这一判断在所有搜索路径（2-job、multi、partial-fix）中一致。
- **Pre-sampling**：搜索前先采样所有 circle 的 360 个角度值，搜索过程中直接查表而不是重复调用 `demand_at()`，降低 72× 的计算开销。

**兼容性矩阵：** `print_compatibility_report()` 输出所有 job 对之间的兼容性分数，可用于指导 job placement（score < 0.6 的配对应避免放在同一链路）。

### 差异分析

| 差异点 | 论文 | 当前实现 | 评估 |
|--------|------|----------|------|
| 搜索策略 | 未指定具体算法 | 网格搜索 + 迭代贪婪 | ✅ 对齐 §3.3 |
| 0.02 阈值 | 未提及 | 实现添加 | ✅ 防止无效 shift |
| step_deg | 未指定 | 5°（72 个候选点） | ✅ 合理精度 |

**结论：** ✅ 完全实现。公式完全对齐论文 §3.3，且增加了实用性改进（阈值、pre-sampling、兼容性矩阵）。

---

## 4. 二分亲和图 (Affinity Graph)

### 论文设计 (Section 4.1-4.3)

Cassini 的核心贡献：将集群级的 time-shift 调度建模为二分图问题。
- 左顶点：Jobs
- 右顶点：Links
- 边：Job 经过 Link

每个作业必须在所有共享链路上使用**同一个全局 time-shift**——这个约束使问题不能分解为独立的 per-link 优化。

### 当前实现 (`affinity_graph.py`)

**算法流程（per connected component）：**

```
1. 选跟节点：通信总需求量最大的 job
2. 设定 shift = 0（根节点不延迟）
3. BFS 遍历：
   a. 从当前 job 出发，找到所有共享链路
   b. 按争用权重排序（权重 = job 数 × 平均带宽需求）
   c. 对每条链路：
      - 已固定的 job：记录在 fixed_shifts_deg
      - 未固定的 job：调用 optimize_link_compatibility 优化
      - 新固定的 job 入队，继续 BFS
4. 从不出现争用链路的 job → shift = 0
```

**关键约束：** 每个 job 只有**一个全局 time-shift**。当 BFS 在一个新的链路上遇到已经固定的 job 时，不会重新优化该 job——它从第一个遇到的链路获得 shift，后续链路以 fixed 身份参与优化。

### 差异分析

| 差异点 | 论文 | 当前实现 | 评估 |
|--------|------|----------|------|
| 图遍历 | 多候选 acyclic-graph 选择 | BFS + 争用权重排序 | ✅ 等效简化 |
| 根节点选择 | 未明确 | 最高通信总需求 | ✅ 合理启发式 |
| 循环防护 | 通过 acyclic graph 保证 | visited 集合防止回环 | ✅ 等价 |
| 连通分量 | 未明确 | 独立处理 | ✅ 正确 |

**结论：** ✅ 完全实现。BFS 遍历 + visited 集合的回环防止是论文 acyclic-graph 选择的等效实现，且争用权重排序在多候选时提供了确定性选择策略。

---

## 5. Time-Shift 集成到调度策略

### 论文设计

Cassini 的 time-shift 是 per-job 级别的：每个 job 延迟 `t_shift` 微秒后再开始执行。一旦开始，DAG 的跨迭代依赖链自然维持周期性对齐，不需要在每条链路或每个 iteration 上反复调整。

### 当前实现 (`cassini_policy.py`)

**执行逻辑：**

```python
def emit_ready_tasks(self, current_time, ready_tasks):
    for task in ready_tasks:
        if not self._job_cleared(task.job_id, current_time):
            continue          # shift 还没到，不发出
        if task.is_flow():
            emitted.append(task.task_id)
        elif task.is_compute() and self._is_next_compute(task):
            emitted.append(task.task_id)
```

**`_job_cleared` 机制：**

- 每个 job 有一个 `_job_started[job_id]` 标记
- 初始为 False，所有 task 被阻塞
- 当 `current_time >= time_shift[job_id]` 时标记变为 True
- 一旦 cleared，该 job 的所有 task 不再受时间门控
- 这是一个**one-shot gate**（一次性闸门），不是每个 iteration 都检查

**调度正交性：**

Cassini Policy 与底层带宽分配和路由解耦：
- `cassini-default`: Cassini + BFS + FairShare
- `cassini-puppeteer`: Cassini + Greedy k-path + TTE-weighted

### 差异分析

| 差异点 | 论文 | 当前实现 | 评估 |
|--------|------|----------|------|
| Shift 生效方式 | 未明确运行时细节 | one-shot gate | ✅ 合理 |
| 与底层调度器关系 | 独立层 | 策略组合（4 种模式） | ✅ 更灵活 |
| 多 iteration | 周期性对齐 | 跨迭代 DAG dep 链 | ✅ 一致 |

**结论：** ✅ 完全实现。One-shot gate 机制简洁有效，4 种组合模式验证了 Cassini 与不同 base planner 的正交性。

---

## 6. GPU Placement 策略

### 当前实现 (`placement.py`)

**Contiguous（默认）：** 每个 job 获得一段连续的 GPU 范围。同一 server 内的 job 没有跨 spine 流量——无争用，但也不是 Cassini 需要优化的场景。

**Contention-Spread：** 显式将每个 job 的 DP replica 分散到不同集群，制造跨 fabric 的争用流量，让 Cassini 的 time-shift 机制发挥作用。

算法：
1. 将 server 划分为 N 个集群
2. 每个 job 的第 k 个 DP replica 分配到第 `(job_id + k) % N` 个集群
3. 在集群内轮询分配 server
4. 最终 GPU 列表按 RankGrouper 的 `[PP][DP][EP][TP]` 格式输出

### 用途

| 策略 | 适用场景 |
|------|----------|
| `contiguous` | 基线对比，验证 Cassini 在无争用时不会产生副作用 |
| `contention-spread` | 验证 Cassini 在有争用时能否降低 makespan |

---

## 7. 迭代扩展 (Iteration Expansion)

### 当前实现 (`iteration_expansion.py`)

**`merge_ga_to_one_iteration(workload, ga)` — GA 合并：**

AICB 文件中 `ga > 1`（gradient accumulation > 1）时，每个 GA 步被编码为独立 iteration 标签。Cassini 需要所有 GA 步合并为一个周期性 iteration。

```
输入 (ga=3): iteration 0, 1, 2, 3, ...
输出:         iteration -1, 0, 0, 0, 1, ...
```

**`replicate_with_cross_iteration_deps(workload, num_iters)` — 多迭代复制：**

将 workload 复制 N 份，每份 iteration 标签 stride = 3:
```
副本 0: [-1, 0, 1]
副本 1: [ 2, 3, 4]
...
```
跨迭代依赖 `post_k → pre_{k+1}` 确保迭代按顺序执行。

**`patch_iteration_time_us(analysis, workload)` — 时间修正：**

Cassini 内置的迭代时间估算会跳过 iteration 0 作为 warmup。当复制份数少时这个启发式不可靠。此函数用 CPM timing 直接计算 periodic iteration 的 span 中位数，写回 `CommunicationPattern.iteration_time_us`。

---

## 8. 复现模式对照表

| 模式 | 路由 | 带宽分配 | Cassini | 说明 |
|------|------|----------|---------|------|
| `default` | BFS | FairShare | ❌ | 基线 |
| `puppeteer` | Greedy k-path | TTE-weighted | ❌ | 对比策略 |
| `cassini-default` | BFS | FairShare | ✅ | **核心 Cassini** |
| `cassini-puppeteer` | Greedy k-path | TTE-weighted | ✅ | Cassini + Puppeteer |

---

## 9. 实现文件索引

### `src/cassini/` — 核心算法

| 文件 | 角色 | 对应论文部分 |
|------|------|-------------|
| `communication_pattern.py` | 通信模式提取 | §3.1 |
| `circle_abstraction.py` | 几何圆抽象 + LCM 统一圆 | §3.2 |
| `pair_compatibility.py` | 成对兼容性网格搜索优化 | §3.3 |
| `affinity_graph.py` | 二分亲和图 + BFS time-shift 求解 | §4.1-4.3 |
| `iteration_expansion.py` | GA 合并 + 多迭代复制 + 时间修正 | —（实验工具） |
| `task_serializer_patch.py` | 跨迭代 DAG 的执行序补丁 | —（兼容性） |
| `job_placement.py` | GPU placement 策略 | —（实验工具） |
| `diagnostics.py` | 指标提取 + 格式化报告 + 可视化保存 | —（实验工具） |

### `src/static_analysis/strategies/` — 分析管线

| 文件 | 角色 |
|------|------|
| `cassini_strategy.py` | Cassini 分析管线（路由 → CPM → 模式提取 → 亲和图 → 执行计划） |

### `src/executor/policies/` — 调度策略

| 文件 | 角色 |
|------|------|
| `cassini_policy.py` | Time-shift 驱动的多 job 迭代对齐 |

### `scripts/` — 实验脚本

| 文件 | 角色 |
|------|------|
| `run_cassini_e2e.py` | 端到端比较脚本（4 种模式 + 2 种 placement） |
| `run_cassini_demo.py` | 最小化 demo（两 job 单链路易验证） |

### 测试

| 文件 | 测试内容 | 用例数 |
|------|----------|--------|
| `test_cassini_communication_pattern.py` | 模式提取 | 10 |
| `test_cassini_circle_abstraction.py` | 圆抽象 + LCM 统一 | 19 |
| `test_cassini_pair_compatibility.py` | 兼容性优化 | 19 |
| `test_cassini_affinity_graph.py` | 亲和图 | 11 |
| `test_cassini_policy.py` | Policy + E2E 集成 | 12 |
| `test_cassini_iteration_expansion.py` | 迭代扩展 | 9 |
| `test_cassini_placement.py` | GPU placement | 2 |

**合计：82 个测试，全部通过。**

---

## 10. 代码整理 (Code Cleanup)

本次提交前对 Cassini 代码做了系统性整理：

### 循环依赖修复

之前 `cassini/__init__.py` 通过 `__getattr__` 延迟加载 `affinity_graph`，`cassini_strategy.py` 在函数体内延迟 import `affinity_graph`。实际 `communication_pattern.py` 对 `static_analysis` 的引用全是 `TYPE_CHECKING`（运行时无依赖），两个 lazy import 均可移除：

- `cassini/__init__.py`: 用普通 import 替代 `__getattr__`
- `cassini_strategy.py`: `affinity_graph` 提到文件顶部的 top-level import

### 函数合并

- `_search_one_fixed` → 合并到 `_optimize_two`（1D 搜索通用化 + 补上漏掉的 0.02 阈值）
- `_search_multi_fixed` → 合并到 `_optimize_multi`（same logic, same fix）
- 0.02 → `_MIN_COMPAT_IMPROVEMENT` 模块级常量

### 死代码清理

| 文件 | 删除内容 |
|------|----------|
| `communication_pattern.py` | `_add_flow_to_link_buckets` 中 4 处重复的 get+add 抽为 `_inc` 局部函数 |
| `communication_pattern.py` | `task_window_starts` 去掉不必要的 `Optional` |
| `affinity_graph.py` | 删除未使用的 `import math` |
| `affinity_graph.py` | `_build_link_circles` 三段变两段，去除不可能到达的 fallback |
| `circle_abstraction.py` | `build_unified` 返回值从 `(lcm_perimeter, list)` 简化为 `list` |
| `iteration_expansion.py` | `_boundary_by_job_node` 从嵌套函数提取为模块级 `_find_boundary_tasks` |

### 文件重命名

| 旧名 | 新名 | 原因 |
|------|------|------|
| `workload_transform.py` | `iteration_expansion.py` | 更精确描述模块职责 |

### 脚本模块化

`run_cassini_e2e.py` 从 ~980 行精简到 ~460 行，抽取两个新模块：
- `src/cassini/placement.py` — GPU placement 算法
- `src/cassini/diagnostics.py` — 指标提取、报告生成、可视化保存

---

## 一句话总结

Cassini 的**完整算法管线**（通信模式提取 → 几何圆抽象 → 成对兼容性优化 → 亲和图 BFS 求解 → 调度策略集成）已全部实现并通过 82 个测试。整理后的代码消除了循环依赖 hack、合并了重复的搜索函数、清理了死代码，并将臃肿的实验脚本拆分为可维护的模块。

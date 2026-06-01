# Cassini 复现代码说明

---

## 1. 论文核心思想回顾

Cassini 要解决的问题：在共享 GPU 集群中同时运行多个分布式训练作业时，不同作业的集合通信（AllReduce、AllGather 等）会在共享链路上产生拥塞，拖慢所有人的迭代速度。

解决方案分三层：

1. **几何圆抽象 (Section 3.1-3.2)**：把每个作业的通信模式投射到一个圆上，周长 = 迭代时间，角度上的值 = 带宽需求。旋转圆 = 给作业加一个时间偏移。
2. **两两兼容性 (Section 3.3)**：对共享链路上的两个作业，网格搜索找最优旋转角使总带宽需求尽量不超链路容量。输出一个兼容性分数。
3. **亲和图 (Section 4.1-4.3)**：构建二分图（作业 ↔ 链路），按链路争用权重排序 BFS 遍历，每个作业得到一个**全局时间偏移** — 作业在一条链路上的偏移要传播到它经过的所有链路。

---

## 2. `src/cassini/` — 实现论文算法

### 2.1 `communication_pattern.py` — 通信模式提取（论文 Section 3.1）

**核心用途**：给定一个 P2PWorkload，提取出一个作业的周期性通信模式。

**关键数据结构**：

```python
@dataclass
class CommunicationPattern:
    job_id: int
    iteration_time_us: int                              # 一轮迭代的时间
    link_demands: dict[tuple[int, int], dict[int, float]]  # (src,dst) → {角度→Gbps}
```

`link_demands` 是一个两层字典：对每条链路，存一个 360 维的向量，每个元素是该角度上的总带宽需求。

**外部入口** — `extract_communication_patterns()`：

```python
def extract_communication_patterns(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,  # CPM 提供每个 task 的时间
    route_table: RouteTable,          # 提供每个 flow task 的路径
) -> dict[int, CommunicationPattern]:
```

处理流程：
1. 按 `job_id` 分组所有 task
2. 对每个 job 调用 `_build_job_pattern()`

**迭代时间估计** — `_estimate_iteration_time()`：

```
- 从 CPM timing 构建逻辑迭代窗口（pre/periodic/post）
- 排除 iteration=0（因为通信模式可能还不稳定），取剩余窗口的中位数
- 这直接决定了圆的周长
```

**论文对应**：Section 3.1 的 "iteration time T"。

**带宽离散化** — `_add_flow_to_link_buckets()`：

对于每个 flow task：
1. 从 CPM 获取它的开始/结束时间
2. 计算带宽 `bw = size_bytes * 8 / (duration_us * 1000)` (Gbps)
3. 将时间映射到角度：`angle = (time % iteration_time) * 360 / iteration_time`
4. 在覆盖的角度区间内累加带宽
5. **链路速率修正**：对慢于 NVLink 的链路（如 200 Gbps ASW）延长传输时间、降低有效带宽

- 如果 flow 时长 < 迭代周期，按覆盖的角度区间累加
- 如果 flow 时长 >= 迭代周期（跨周期 flow），则填充全部 360°

**论文对应**：Section 3.1 的 "bw(α) — bandwidth demand at angle α"。

---

### 2.2 `circle_abstraction.py` — 几何圆抽象（论文 Section 3.2）

**核心数据结构**：

```python
@dataclass
class CircleAbstraction:
    perimeter: int               # 周长 = 迭代时间 (us)
    bw_demand: dict[int, float]  # {0..359 → Gbps}
```

**旋转操作** — `rotate(shift_deg)`：

```python
def rotate(self, shift_deg: int) -> "CircleAbstraction":
    return CircleAbstraction(
        perimeter=self.perimeter,
        bw_demand={(a + shift_deg) % 360: bw for a, bw in self.bw_demand.items()},
    )
```

顺时针旋转 `shift_deg` 度 = 将作业的迭代开始时间延迟 `shift_deg/360 * perimeter` 微秒。

**论文对应**：Section 3.2 的 "rotating the circle by ∆ degrees"。

**从 CommunicationPattern 构建圆** — `from_pattern()`：

```python
@staticmethod
def from_pattern(pattern: CommunicationPattern, link_id: tuple) -> CircleAbstraction:
    raw = pattern.link_demands.get(link_id, {})
    return CircleAbstraction(
        perimeter=pattern.iteration_time_us,
        bw_demand={a: raw.get(a, 0.0) for a in range(360)},
    )
```

从全局的 `CommunicationPattern` 中抽出一条特定链路的圆。

**论文对应**：Section 3.2 的 "per-link circle representation"。

**LCM 统一圆** — `build_unified()`：

这是处理**不同迭代时间**的作业的方法。如果 job A 迭代 100ms，job B 迭代 150ms，它们的圆周长不同，就不能直接在同一个圆上比较。

处理策略：
- **正常情况**（LCM ≤ 50s）：计算 LCM 周长，将每个作业的 pattern 在统一圆上平铺 `r = LCM/T` 次，对每个角度取 max-pooling（不低估峰值需求）
- **溢出情况**（LCM > 50s）：用时间-比例映射，`angle_uni → time → angle_orig`

```python
if overflowed:
    t_us = a_uni * lcm // 360
    a_orig = (t_us * 360 // c.perimeter) % 360
    tile_bw[a_uni] = c.demand_at(a_orig)
else:
    r = lcm // c.perimeter
    a_start = (a_uni * r) % 360
    tile_bw[a_uni] = max(c.demand_at((a_start + i) % 360) for i in range(r))
```

**论文对应**：Section 3.2 末尾 "unified circle via LCM perimeter"。

---

### 2.3 `pair_compatibility.py` — 两两兼容性优化（论文 Section 3.3, Table 1）

**核心问题**：给定一条链路上的 N 个圆的带宽需求，找到每个圆的旋转角度，使总需求尽量不超过链路容量。

**兼容性分数公式**：

```python
def compute_score(circles, shifts_deg, link_capacity):
    for α in range(360):
        total_demand = Σ_j bw_j[(α - shift_j) % 360]
        excess = max(0, total_demand - link_capacity)
        total_excess += excess
    score = 1.0 - total_excess / (360 * link_capacity)
```

**论文对应**：Section 3.3 的 optimization formulation (Table 1)。

分数含义：
- **1.0** = 完美兼容（所有角度的总需求 ≤ 链路容量）
- **越低** = 拥塞越严重

**搜索策略**：

| 作业数 | 方法 | 代码函数 |
|--------|------|---------|
| 1 | 无需优化，分数=1.0 | 直接返回 |
| 2 | 穷举网格搜索：固定 job[0]=0°，搜索 job[1] 的全部角度 | `_optimize_two()` |
| 3+ | 迭代贪心：每次固定其他维度，对一个自由维度做网格搜索，收敛或 10 轮停止 | `_optimize_multi()` |

搜索角度由 `step_deg` 决定（默认 5° → 72 个候选角度）。

**关键设计**：0.02 阈值——如果旋转带来的 score 提升 < 0.02，则放弃该 shift（延迟对 makespan 的伤害 > 争用缓解的收益）。

**论文对应**：Section 3.3 的 grid-search optimization 和 Section 5.7 的 discretization precision 分析。

---

### 2.4 `affinity_graph.py` — 亲和图遍历（论文 Section 4.1-4.3, Algorithm 1）

**核心数据结构**：

```python
@dataclass
class AffinityGraph:
    job_links: dict[int, set[tuple[int, int]]]       # job → 经过的链路集合
    link_jobs: dict[tuple[int, int], set[int]]        # 链路 → 竞争的作业集合（仅 ≥2）
    link_capacities: dict[tuple[int, int], float]     # 链路容量 (Gbps)
    patterns: dict[int, CommunicationPattern]         # 每个 job 的通信模式
```

注意 `link_jobs` 只保留有 2+ 作业的**争用链路**（contended links），无竞争的链路不参与时间偏移计算。

**论文对应**：Section 4.1 "bipartite graph G = (J, L, E)"。

**入口函数** — `compute_cluster_time_shifts()`：

```python
def compute_cluster_time_shifts(graph, step_deg=5) -> dict[int, int]:
    components = _find_connected_components(graph)
    for comp_jobs in components:
        _bfs_traverse_component(graph, comp_jobs, fixed_shifts_us, step_deg)
```

**论文对应**：Algorithm 1 (BFS traversal of affinity graph)。

**连通分量分解** — `_find_connected_components()`：

通过共享的争用链路判断哪些作业属于同一个"竞争域"。两个作业如果在同一个连通分量中，它们的时间偏移需要一起优化；如果不在同一分量（使用完全不重叠的链路），则互不干扰。

**论文对应**：Section 4.1 "connected components of the affinity graph"。

**BFS 遍历** — `_bfs_traverse_component()`：

```
1. 选择 root: 通信总需求最大的 job（_job_total_demand）
2. root 的 time_shift = 0（固定参考点）
3. BFS 循环:
   a. 出队一个已固定的 job
   b. 收集它的未访问争用链路，按争用权重排序
   c. 对每条候选链路:
      - 构建所有 job 的圆
      - 将已固定的 job 的 shift 转为度数，传给 optimize_link_compatibility
      - 新作业被赋予优化后的 shift，入队
```

**争用权重** — `_link_contention_weight()`：

```python
weight = n_jobs × avg_bandwidth_demand_on_link
```

作业多且带宽需求大的链路优先处理，确保最"拥堵"的链路最先确定时间偏移关系。

**论文对应**：Section 4.2 Step 2: "rank links by contention weight"。

**一条链路的优化** — `_process_link()`：

```python
def _process_link(graph, link_id, all_jobs_on_link, fixed_on_link, unfixed_on_link, ...):
    circles, circle_to_job = _build_link_circles(graph, link_id, all_jobs_on_link)
    fixed_deg = {job→circle_idx: shift_us→deg for 已固定的job}
    result = optimize_link_compatibility(circles, capacity, step_deg,
                                         fixed_shifts_deg=fixed_deg)
```

这里 `_build_link_circles()` 内部调用 `CircleAbstraction.build_unified()` 处理不同迭代时间的 LCM 统一。

**论文对应**：Section 4.2 Step 3: "optimize compatibility on this link, fix new jobs, propagate to adjacent links"。

**Acyclic BFS Tree** — 论文强调图可能包含环。代码通过 `visited_jobs` 和 `visited_links` 将 BFS 限制为一棵树。

**论文对应**：Section 4.2 "acyclic BFS tree"。

---

## 3. CassiniAnalyzer — 组装全流程（论文 Algorithm 2）

**文件**: `src/static_analysis/strategies/cassini_strategy.py`

`CassiniAnalyzer` 是上述模块的编排器，将步骤串成一条管线：

```python
def analyze(self, workload, route_table=None) -> CassiniAnalysisResult:
    # Step 1: 路由 — 默认 BFS（可外部传入覆盖）
    if route_table is None:
        route_table = BfsStrategy().compute_routes(workload, self.topology)

    # Step 2: CPM 关键路径分析
    critical_path = analyze_critical_path(workload, route_table, self.topology)

    # Step 3: 通信模式提取
    patterns = extract_communication_patterns(workload, critical_path, route_table, self.topology)

    # Step 4: 亲和图遍历 → time_shifts
    time_shifts = self._compute_time_shifts(patterns, route_table, workload)

    # Step 5: C++ 参考计算序列化
    execution_plan = CppReferenceSerializer().serialize(workload)

    return CassiniAnalysisResult(route_table, critical_path, patterns, time_shifts, execution_plan)
```

**论文对应**：Algorithm 2 的完整 Cassini Module。

---

## 4. CassiniSchedulingPolicy — 运行时时间偏移执行

**文件**: `src/executor/policies/cassini_policy.py`

CassiniSchedulingPolicy 是运行时策略，将分析阶段算出的 `time_shifts` 落实到离散事件模拟中。

**核心机制** — `emit_ready_tasks()`：

```python
def emit_ready_tasks(self, current_time, ready_tasks):
    emitted = []
    for task in sorted(ready_tasks, key=lambda t: t.task_id):
        if not self._job_cleared(task.job_id, current_time):
            continue              # ← 时间偏移未到，阻塞
        if task.is_flow():
            emitted.append(task.task_id)
        elif task.is_compute() and self._is_next_compute(task):
            emitted.append(task.task_id)
    return emitted
```

`_job_cleared()` 是一个**一次性门控** — 作业被"清除"后（`current_time >= time_shift`），后续所有迭代自然按 DAG 依赖周期性执行，不需要重复施加偏移。

**论文对应**：Section 4.2 "Applying time-shifts at runtime"。

---

## 5. 与 Default / Puppeteer 的关系

### 5.1 对比

| 组件 | Default | Puppeteer | Cassini |
|------|---------|-----------|---------|
| 路由 | BFS 最短路径 | Greedy k-shortest-path | 同基础规划器 |
| 带宽分配 | FairShare（均分） | TTE-weighted（按优先级加权） | 同基础规划器 |
| 作业准入 | 无控制，DAG ready 即放行 | 可选的 co-start 协调 | **time_shift 门控** |
| 核心贡献 | — | TTE + 贪心路由 | 周期性通信模式时间偏移 |

### 5.2 Cassini 对基础规划器的依赖

Cassini 只需要基础规划器提供两样东西：
1. **路由表** (`RouteTable`) — 每个 flow task 走哪些链路
2. **带宽分配器** (`BandwidthAllocator`) — 决定运行时每条 flow 分多少带宽

无论是 Default（BFS + FairShare）还是 Puppeteer（Greedy + TTE），Cassini 的时间偏移逻辑都是正交的。

### 5.3 可插拔实现

- `CassiniAnalyzer.analyze(workload, route_table=None)` — 传入外部 route_table 跳过 BFS
- `CassiniSchedulingPolicy(analysis, allocator=None)` — 传入外部 allocator 替代 FairShare

这使实验脚本可以组合出四种配置：

| 配置 | route_table 来源 | allocator |
|------|-----------------|-----------|
| `cassini-default` | 内部 BFS | FairShareAllocator() |
| `cassini-puppeteer` | PuppeteerAnalyzer 的 greedy route_table | TteAwareAllocator(tte_info) |

---

## 6. 实验设计

### 6.1 实验脚本

**文件**: `scripts/run_cassini_e2e.py`

四种配置的 end-to-end 对比：
- `default` — BFS + FairShare（基线）
- `puppeteer` — Greedy + TTE-weighted
- `cassini-default` — Cassini on BFS + FairShare
- `cassini-puppeteer` — Cassini on Greedy + TTE-weighted

### 6.2 实验维度

| 对应论文实验 | 脚本参数 | 关键指标 |
|-------------|---------|---------|
| Experiment 1: 性能提升 | `--num-jobs 2/4/8` | makespan 加速比, per-job iteration time |
| Experiment 2: 拥塞减少 | 自动输出 | p99/avg flow time 比值 |
| Experiment 4: 部分兼容性 | 自动输出兼容性矩阵 | pairwise compatibility score |
| 扩展性 | `--num-jobs N` | 随作业数的扩展趋势 |

### 6.3 命令行示例

```bash
# 双作业完整对比（所有 4 种模式）
python scripts/run_cassini_e2e.py --num-jobs 2

# 仅对比 default vs cassini-default
python scripts/run_cassini_e2e.py --num-jobs 2 --modes default cassini-default

# 多 AICB 文件，每个文件是独立 workload entry
python scripts/run_cassini_e2e.py --aicb gpt.txt bert.txt --dp 2

# 使用 contention-spread placement 强制跨集群争用
python scripts/run_cassini_e2e.py --placement contention-spread --placement-clusters 2

# 使用 JSON 配置文件
python scripts/run_cassini_e2e.py --config experiments.json

# GPU placement 配置
#   contiguous（默认）— 每个 job 连续 GPU 范围，最小化跨服务器争用
#   contention-spread — 将 DP replica 分散到不同集群，强制跨 fabric 争用
```

### 6.4 输出说明

- `outputs/cassini_experiments/result_{mode}.json` — 每种模式的完整 `ExecutionResult`
- `outputs/cassini_experiments/comparison.json` — 原始 metrics 的 JSON 报告
- `outputs/cassini_experiments/comparison.txt` — 格式化对比表 + per-job 时间 + 加速比
- 终端输出：对比表格、per-job iteration time、speedup、flow time tail ratio

---

## 7. 关键文件索引

| 文件 | 作用 | 论文对应 |
|------|------|---------|
| `src/cassini/communication_pattern.py` | 通信模式提取：CPM时间 → 角度 → 带宽需求 | Section 3.1 |
| `src/cassini/circle_abstraction.py` | 几何圆：旋转、LCM 统一、平铺 | Section 3.2 |
| `src/cassini/pair_compatibility.py` | 两两兼容性：兼容性分数、网格搜索 | Section 3.3, Table 1 |
| `src/cassini/affinity_graph.py` | 亲和图：二分图 BFS、全局时间偏移求解 | Section 4.1-4.3, Algorithm 1 |
| `src/cassini/iteration_expansion.py` | GA 合并 + 多迭代复制 + 跨迭代 dep | —（实验工具） |
| `src/cassini/task_serializer_patch.py` | 跨迭代 DAG 执行序补丁 | —（兼容性） |
| `src/cassini/job_placement.py` | GPU placement 策略 | —（实验工具） |
| `src/cassini/diagnostics.py` | 指标提取 + 格式化报告 + 可视化保存 | —（实验工具） |
| `src/static_analysis/strategies/cassini_strategy.py` | 编排器：串起路由→CPM→模式→时间偏移 | Algorithm 2 |
| `src/executor/policies/cassini_policy.py` | 运行时：time_shift 门控执行 | Section 4.2 |
| `scripts/run_cassini_e2e.py` | 实验脚本：四种配置对比 | — |
| `tests/test_cassini_*.py` | 82 个单元/集成测试（全部通过） | — |

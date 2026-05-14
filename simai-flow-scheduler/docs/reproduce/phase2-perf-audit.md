# Phase 2 Performance Audit

> 对 Puppeteer 策略实现的性能审查，识别在 LLM 训练规模（十万级 task）
> 下无法线性扩展的瓶颈。

---

## 规模基准

- GPT-175B DP2 典型规模：107K tasks、88K flows
- 拓扑：154 节点（16 GPU + 138 Switch）
- O(n²) 在 88K 规模下意味着 ~7.7B 次操作

---

## 分类 1：关键瓶颈 (O(n²))

### 1.1 `compute_greedy_routes` — 热路径上的 link interval 线性扫描

**位置**: `src/static_analysis/passes/puppeteer_routing.py:146-152`

```python
for link in links:
    active_count = 0
    for (l_start, l_finish) in link_intervals.get(link, []):
        if start_us < l_finish and l_start < finish_us:
            active_count += 1
```

**问题**: `link_intervals[link]` 随已路由 flow 数线性增长。按 flow 排序后的顺序路由时，后处理的 flow 每条热路径都要遍历之前累积的全部 interval。在 Clos 拓扑下 spine link 作为共享瓶颈，interval 列表累积到 O(F) 规模。

**根因**: 用 list 存储 interval 并做线性扫描来判断重叠，而非用区间树（interval tree）或只在必要时检查重叠。

---

### 1.2 `k_shortest_paths` 按 per-flow 而非 per-(src,dst) 调用

**位置**: `src/static_analysis/passes/puppeteer_routing.py:132`（被 `compute_greedy_routes` 调用）

```python
candidates = k_shortest_paths(topology, task.src, task.dst, k)
```

**问题**: 对每个 flow task（88K）都独立调用一次 BFS k-shortest paths。LLM 训练中大量 flow 共享相同的 (src, dst) pair（如 TP allreduce 在同组 GPU pair 间重复数百次）。88K 次 BFS 探索完全相同的拓扑路径，无任何缓存。

此外 `k_shortest_paths` 内部的 BFS（`routing.py:48-57`）每次扩展都做 O(path_length) 的列表拷贝和成员检查，进一步放大了开销。

**根因**: 未识别 (src, dst) 重复模式，未做路径缓存。

---

## 分类 2：高影响

### 2.1 `compute_optimistic_timing` 的 compute-order 前驱查找

**位置**: `src/static_analysis/passes/puppeteer_tte.py:117-119`

```python
for pred, succs in compute_successors.items():
    if tid in succs and pred in earliest_finish:
        pred_finishes.append(earliest_finish[pred])
```

**问题**: 对每个 task，遍历所有 `compute_successors`（~19K compute tasks）判断当前 task 是否是后继。107K × 19K ≈ 20 亿次迭代。应构建 `compute_predecessors` 逆映射实现 O(1) 查找。

**根因**: 未预先构建反向索引。

---

### 2.2 k_shortest_paths BFS 路径操作开销

**位置**: `src/static_analysis/passes/puppeteer_routing.py:51-53`

```python
if neighbor in path:          # O(L) 线性查找
    continue
new_path = path + [neighbor]  # O(L) 完整拷贝
```

**问题**: 对 path（Python list）的 in 检查是 O(L)，path + [neighbor] 创建新 list 仍是 O(L)。在 154 节点拓扑中 L 约 4-8 跳，但 BFS 扩展可能是分支因子 × 深度的组合。88K 次调用积累大量分配。可用 `deque` + 回溯重构代替路径复制。

**根因**: 用不可变 list 做路径跟踪，而非回溯指针。

---

### 2.3 `TteAwareAllocator.strict_priority` 的 link 竞争重算

**位置**: `src/executor/allocators/tte_aware_allocator.py:164-171`

```python
count = sum(
    1 for f in group
    if any(
        (f.path[j], f.path[j + 1]) == link
        for j in range(len(f.path) - 1)
    )
)
```

**问题**: 对每个活跃 flow、每条 link，重新遍历整个 priority group 来统计该 link 上的 flow 计数。active_flows 在运行时可能达到数百甚至数千。正确的做法是在运行前一次性构建 `link → flows_in_tier` 的映射表。

**根因**: 每次 allocate 都实时推导拓扑关系而非预计算。

---

## 分类 3：中等影响

### 3.1 `compute_optimistic_timing` 重复执行

**位置**: `src/static_analysis/strategies/puppeteer_strategy.py:81,98`

`PuppeteerAnalyzer.analyze()` 在 step 3（最短路径 TTE）和 step 5（重算 TTE with 贪心路由）中各调用一次 `compute_tte`，每次内部都调用 `compute_optimistic_timing`。拓扑排序和 DAG 结构在两次调用间不变，可裁剪为一次。

---

### 3.2 child duration 未缓存

**位置**: `src/static_analysis/passes/puppeteer_tte.py:208-218`

在为 flow 的每个 child 计算 TTE 时，`_estimate_flow_duration_on_path` 被重复调用。同一个 child flow task 可能出现在多个父 flow 的 children 中。应缓存 `task_id → duration`。

---

### 3.3 link 列表重复计算

**位置**: `src/static_analysis/passes/puppeteer_routing.py:144 和 163`

同一 path 的 `[(path[i], path[i+1]) for i in range(...)]` 在 scoring 循环和路由后 update 中各算一次。可以提取到变量复用。

---

### 3.4 weight 线性查找

**位置**: `src/executor/allocators/tte_aware_allocator.py:103`

```python
my_w = next((w for fid, w in flows_on_link if fid == tid), 0.0)
```

`flows_on_link` 是 `list[tuple]`，每次 `next()` 扫描整个 list。应改为 `dict[int, float]` 在 `link_flows` 中直接存储 weight 映射表。

---

## 性能问题全景

| ID | 文件 | 复杂度 | 触发频次 | 瓶颈类型 |
|----|------|--------|----------|----------|
| 1.1 | routing.py | O(F × L × k × I) | 每次 analyst | 线性 interval 扫描 → 映射聚合 |
| 1.2 | routing.py | O(F × BFS) | 每次 analyst | 重复 BFS → (src,dst) 缓存 |
| 2.1 | tte.py | O(N × N_compute) | 每次 TTE 计算 | 缺失反向索引 |
| 2.2 | routing.py | O(L) per BFS step | 每次 analyst | 不可变路径 → 回溯指针 |
| 2.3 | tte_aware_allocator.py | O(A² × L) | 每轮调度 | 缺失 link 索引 |
| 3.1 | strategy.py | 2× 冗余 | 每次 analyst | 缓存不变结果 |
| 3.2 | tte.py | O(L × children) | 每次 TTE 计算 | 缺失缓存 |
| 3.3 | routing.py | 2× 计算 | 每次 analyst | 变量复用 |
| 3.4 | tte_aware_allocator.py | O(link_flows) per flow | 每轮调度 | 数据结构不当 |

> **Legend**: F=flow 数, N=总 task 数, N_compute=compute task 数, L=路径长度, k=候选路径数, I=每个 link 上累积的 interval 数, A=活跃 flow 数

---

## 核心修正思路

1. **按 (src, dst) 缓存路径** — 不再为每个 flow 独立 BFS
2. **预建反向索引替代线性扫描** — `compute_predecessors`、`link_to_flows`、`task_to_duration`
3. **路径跟踪改为回溯指针** — BFS 用 parent 指针重建路径而非复制
4. **区间树代替线性 interval 扫描** — 只在必要时检测重叠

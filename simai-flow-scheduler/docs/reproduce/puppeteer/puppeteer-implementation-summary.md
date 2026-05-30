# Puppeteer 策略复现总结

> 基于论文 *Puppeteer: A Network Planner for AI Training Workloads* (SIGCOMM '23)
> 与 `simai-flow-scheduler` 当前实现 (Phase 1-2) 的详细比较。

---

## 整体评估

| 维度 | 状态 | 说明 |
|------|------|------|
| Flow 级建模 | ✅ 完全实现 | Collective 展开为 P2P flow |
| 离线规划管线 | ✅ 完全实现 | BFS → TTE → Greedy Routing → Recompute TTE |
| 贪心路由 (Greedy Routing) | ✅ 已实现，有差异 | 离线规划 vs 论文的运行时决策 |
| TTE 分析 | ✅ 完全实现 | 公式、乐观 timing、优先级分类均对齐论文 |
| TTE 感知带宽分配 | ✅ 完全实现 | Weighted + Strict Priority 两种模式 |
| 资源依赖 / Handshake | ❌ 未实现 | `compute_resource_dependency` 返回空，Phase 3 规划中 |
| 分段级 (Segment) 速率控制 | ❌ 未实现 | Phase 3 规划中 |
| Clos 感知路由 | ❌ 未实现 | 使用通用 BFS k-shortest paths |
| Zero-queue 保证 | ⚠️ 部分近似 | 带宽分配不超容量，但缺少 handshake 防 jitter |

---

## 1. Flow 级建模

**论文描述：**
Puppeteer 不直接对 all-reduce 等 collective 原语做调度，而是先根据具体的 collective 算法（ring、tree 等）将其展开为一组 P2P flow。执行计划的最小单元是 flow。

**当前实现：**
`WorkloadBuilder.build_from_aicb()` 使用 `simcc` 将 AICB 格式的 collective 通信记录展开为点对点 flow。每个 flow 由 `Task(task_id, src, dst, size_bytes, deps)` 表示，compute 也作为 `Task` 节点存在 DAG 中。

**结论：** ✅ 完全实现。建模方式与论文一致。

---

## 2. 贪心路由 (Greedy Least-Active Path Selection)

这是当前实现与论文差异较大的地方，值得重点分析。

### 论文原始设计

| 属性 | 论文做法 |
|------|----------|
| 选路时机 | Flow 实际启动时（运行时决策），基于**当前** active flow 负载 |
| 路径范围 | Clos 分层按序选路：intra-node → intra-pod → inter-pod |
| 核心状态 | `link → active_flows` + `switch → bucketed outgoing links` |
| 决策粒度 | 每跳独立选择出链路 |

### 当前实现 (`GreedyStrategy` in `greedy.py`)

| 属性 | 当前做法 |
|------|----------|
| 选路时机 | 执行前离线预计算，基于**乐观 timing** 的规划区间 |
| 路径范围 | `k_shortest_paths`（通用 BFS，k=4），无 Clos 层级概念 |
| 核心状态 | `link → deque[finish_time]` 规划区间活动记录 |
| 决策粒度 | 整条路径评分后选择最优 |

### 实现细节

`GreedyStrategy.compute_routes()` 的流程：

1. 按 optimistic start time 排序所有 flow
2. 对每个 flow，用 `k_shortest_paths` 找 k 条候选路径
3. 对每条候选路径，计算 `score = (max_active, total_active, path_len, path)`
   - `max_active`：路径上所有 link 在 `[start, finish)` 区间内的最大并发 flow 数
   - `total_active`：所有 link 的活跃数之和（tie-breaker）
4. 选择 score 最低的路径
5. 将 `finish_us` 记录到路径上各 link 的 deque 中

### 差异分析

| 差异点 | 论文 | 当前实现 | 影响 |
|--------|------|----------|------|
| **选路时机** | 运行时 | 离线预计算 | 离线版本无法感知真实的运行时拥塞，但 AI workload 高度确定，差异不大 |
| **Clos 感知** | 有 | 无（通用 BFS） | 在 Clos 拓扑中，论文的选路天然具有负载均衡优势；当前实现依赖 k-shortest paths 的多样性 |
| **控制粒度** | 逐跳选择 | 整条路径评分 | 逐跳选择更灵活但复杂度更高；整条路径评分更稳定，但可能错过更好的单跳选择 |
| **状态维护** | `switch → bucket` | `link → deque` | 两者在效果上接近，当前实现更简单直接 |

**结论：** ✅ 已实现核心思想（least-active heuristic），但做了显著简化——从运行时决策变为离线规划，从 Clos 感知变为通用 BFS。对于 AI 训练这种高度确定性的 workload，这种简化在效果上损失有限。

---

## 3. 关键路径与乐观 Timing 分析

### 论文公式

```
TTE(flow) = min(Start(child) - Finish(flow))    over all children
```

### 当前实现 (`compute_optimistic_timing` + `compute_tte`)

**乐观 timing 算法：**

1. 构建 DAG 的拓扑排序（含 `compute_order` 隐式边）
2. Forward pass 计算每个 task 的 `earliest_finish`
   - compute: `duration = task.duration_us`
   - flow: 沿路径取瓶颈带宽，计算 `transmission_time + total_latency`
3. 对 flow 反向计算 TTE：
   ```
   TTE = min(child.earliest_start - flow.earliest_finish)
        = min((earliest_finish[child] - child_duration) - finish)
   ```

**优先级分类：**

| TTE 范围 | 优先级 | Priority Score |
|----------|--------|----------------|
| `≤ 0` | `critical` | 0.0 |
| `> 0` 且 `≤ threshold` | `elastic` | `1/max(tte, 1)` |
| `> threshold` 或 `inf` | `background` | `1/max(tte, 1)` |

**注意点：**

- **`compute_order` 隐式边已纳入**：每个 node 上的 compute 任务按 `execution_plan.compute_order` 串行化，`compute_predecessors/successors` 被加到乐观 timing 约束中。这是论文要求的关键细节，因为如果忽略 compute 顺序，TTE 计算会过于乐观。
- **论文的循环依赖处理**：TTE 需要 finish time，finish time 依赖带宽分配，带宽分配又依赖 TTE。当前实现采用论文相同的方式——先做一次 optimistic pass（假设 line rate），由此得出结构化近似，而非精确值。

**结论：** ✅ 完全实现。公式、乐观 timing pass、优先级分类均对齐论文。

---

## 4. TTE 感知带宽分配

### 论文动机

论文明确反对单纯的 max-min fairness，认为 AI 训练的目标是 iteration completion time，而不是 individual flow completion time。因此带宽应按 flow 在 DAG 中的"关键性"分配，而非公平均分。

### 当前实现 (`TteAwareAllocator`)

**Weighted 模式：**

```
weight(flow) = 1.0 / max(tte_us, epsilon)
alloc(flow, link) = link.bandwidth * weight(flow) / sum(weight of all flows on link)
final_alloc(flow) = min(alloc(flow, link) over all links in path)
```

**Strict Priority 模式：**

```
Tier 1: critical — 均分剩余带宽
Tier 2: elastic  — 均分 critical 用完后剩余的带宽
Tier 3: background — 均分 elastic 用完后剩余的带宽
```

两种模式都正确处理了多跳路径的瓶颈链路问题（`min over path links`）。

### 与论文对比

| 维度 | 论文 | 当前实现 | 评估 |
|------|------|----------|------|
| 核心思想 | 按关键性分配 | Weighted / Strict Priority | ✅ 一致 |
| 多跳瓶颈 | 隐式处理 | 显式 `min over links` | ✅ 更严谨 |
| 时间变化 | 带宽随时间变化（rate state） | 当前为静态分配（每次 realloc 重新计算） | ⚠️ 合理近似 |
| Starvation 防护 | 未明确提及 | 已移除 `min_background_share`（用户要求） | 用户决策 |

**结论：** ✅ 完全实现。两种模式都正确实现了 TTE 驱动的非公平带宽分配，且在瓶颈链路处理上比论文描述得更明确。

---

## 5. 资源依赖 / Handshake Barrier

### 论文设计

Puppeteer 认为仅靠离线时间表不够——运行时 compute jitter 会破坏 zero-queue 假设。解决方案是找出那些"在 DAG 中没有直接依赖但会在网络资源上相遇"的 flow，在它们之间插入 handshake barrier：

> 发送方先发一个轻量同步消息，对端确认也到达了计划中的状态，双方都 ready 后再开始真正数据传输。

### 当前实现

**分析 pass** (`puppeteer_coordination.py`):

```python
def compute_resource_dependency(...) -> ResourceDependencyTable:
    return ResourceDependencyTable()  # 空实现
```

**Policy 侧** (`puppeteer_policy.py`):

```python
def _can_emit_flow(self, task, ready_ids):
    if tid not in self._task_to_group:
        return True          # 没有 coordination 约束，直接准入
    gid = self._task_to_group[tid]
    members = self.resource_dependency.groups.get(gid, set())
    return all(m in ready_ids for m in members)  # 等待所有组成员 ready
```

### 差距分析

| 维度 | 论文 | 当前实现 | 差距 |
|------|------|----------|------|
| 依赖检测 | 分析 link 共享 + 时间窗重叠 | `return empty` | ❌ 未实现 |
| Co-start barrier | 分组同步后再发送 | Policy 逻辑已 ready，但无组可创建 | ⚠️ 半完成 |
| Handshake 协议 | NIC/transport 级握手 | 未建模 | 合理简化 |
| 插入 synthetic dep | 将 handshake 表达为 DAG 边 | 未实现 | ❌ 未实现 |

**根因分析：** 实现 Phase 2 时，检测哪些 flow 共享 link + 时间窗重叠 这一分析逻辑被有意推迟到 Phase 3。Phase 3 的 plan 进一步提出要拆分为 segment 级别同步而非 flow 级别同步（因为 `A: [1,3]` overlap `B: [2,4]` 的场景，在 `[1,2)` 区间 A 是 solo 的，不应等待 B）。

**结论：** ❌ 未实现。Policy 层代码结构已准备就绪，但核心的分析逻辑（创建 coordination group）和 segment 拆分均未实现。

---

## 6. Segment 级速率控制 (Phase 3)

### 论文未明确、Phase 3 plan 提出的改进

原始 Puppeteer 论文的带宽分配隐含了"同一 flow 在不同时间段可能获得不同带宽"的含义（因为随着其他 flow 的启停，竞争关系变化）。当前的实现是每次重新分配时所有 active flow 重新计算带宽，但不会主动将一个 flow 拆分为多个 rate state。

**Phase 3 plan 的核心想法：**

- 将每条 flow 按 planned timing 拆分为多个 segment task
- 每个 segment 有自己的 `planned_rate_gbps`
- Segment 边界由 overlap peer 的启停时间和 rate 变化点决定
- 在 segment 级别做 coordination（比 flow 级更精确）

**当前状态：** 未开始实现。

**结论：** ❌ 未实现。详细规划见 `puppeteer-phase3-plan.md`。

---

## 7. Zero-Queue 近似

### 论文实现路径

Puppeteer 的 zero-queue 不是靠交换机拥塞控制实现的，而是靠端侧三件事：

1. **路由避开冲突** —— ✅ 贪心路由已实现
2. **带宽不超容量** —— ✅ `TteAwareAllocator` 的 per-link `min` 计算确保总和不超过带宽
3. **运行时 handshake 防抖动** —— ❌ 未实现

### 当前实现的能力

在当前实现中，任意时刻 active flow 在共享链路上分配的带宽总和不会超过链路容量（weighted 模式通过 `sum(weight)` 比例分配天然不超；strict priority 模式通过逐层减 `link_rem` 也保证不超）。这意味着在规划层面，链路不会过载。但由于没有 handshake 机制，运行时 compute jitter 仍可能导致 flow 的实际启动时间偏离规划，从而在短时间内造成临时拥塞。

**结论：** ⚠️ 部分近似。前两项已实现，第三项未实现。

---

## 8. 复现模式对照表

Run 脚本支持的 5 种模式与论文策略的对应关系：

| 模式 | 路由 | 带宽分配 | Coordination | 对应论文程度 |
|------|------|----------|-------------|-------------|
| `default` | BFS 最短路径 | Fair Share | 无 | 论文 baseline（非 Puppeteer） |
| `route-only` | Greedy Routing | Fair Share | 无 | 仅路由策略 |
| `tte-only` | BFS 最短路径 | TTE-aware | 无 | 仅带宽策略 |
| `route-tte` | Greedy Routing | TTE-aware | 无 | ⭐ **核心策略** |
| `full` | Greedy Routing | TTE-aware | Co-start (空) | 预设完整版（但 coordination 暂无效） |

实际使用中建议以 **`route-tte`** 作为当前 Puppeteer 策略的代表。

---

## 9. 差距汇总与优先级

| # | 未实现 / 近似项 | 优先级 | 预估工作量 | 依赖 |
|---|-----------------|--------|------------|------|
| 1 | Resource dependency 分析 (flow 级) | **高** | 小 | Phase 2 完成 |
| 2 | Segment 级 flow 拆分 + coordination | **高** | 大 | 依赖 #1 |
| 3 | Clos 感知的路由 (分层选路) | 中 | 中 | 顶层需要 Clos 拓扑分类 |
| 4 | 运行时 handshake 协议建模 | 低 | 小 | 依赖 #2 |
| 5 | 动态 replanning（运行时 jitter 后调整计划） | 低 | 大 | 依赖 #3, #4 |

### 推荐下一步

**短期（高价值、低成本）：** 实现 flow 级的 resource dependency 分析（当前 `compute_resource_dependency` 的填充）。这可以让 `full` 模式的 co-start coordination 真正生效。Policy 层的逻辑已就绪，只需要补充检测共享 link + 时间窗重叠 + 排除已有 DAG 依赖 的逻辑。

**中期：** 推进 Phase 3 的 segment 拆分和 segment 级 coordination，这能更精确地匹配论文的 handshake 语义。

---

## 10. 实现文件索引

| 文件 | 角色 | 对应论文部分 |
|------|------|-------------|
| `src/static_analysis/passes/routing/greedy.py` | 离线贪心路由 | §4.2 Greedy Least-Active Path |
| `src/static_analysis/passes/puppeteer_tte.py` | TTE 分析 + 乐观 timing | §4.3 TTE Calculation |
| `src/static_analysis/passes/puppeteer_coordination.py` | 资源依赖分析（空实现） | §4.4 Resource Dependency |
| `src/static_analysis/strategies/puppeteer_strategy.py` | 策略组装管线 | §3 System Design |
| `src/executor/policies/puppeteer_policy.py` | Puppeteer 调度策略 | §4 Scheduling Policy |
| `src/executor/bandwidth_allocators/tte_aware_allocator.py` | TTE 感知带宽分配 | §4.3 TTE-driven Rate Allocation |
| `src/executor/analytical.py` | 离散事件模拟器 | §3.2 Event-driven Planner |
| `scripts/run_puppeteer_reproduce.py` | 端到端比较脚本 | —（实验工具） |

---

## 一句话总结

当前实现完整覆盖了 Puppeteer 的核心调度思想——**flow 级贪心路由 + TTE 驱动的非公平带宽分配**，但缺少论文中的 **resource dependency / handshake barrier** 运行时协调机制。这个缺失在纯模拟环境中对最终指标（makespan、平均 flow 完成时间）影响有限，但对于复现论文的 zero-queue 叙事是不够的。

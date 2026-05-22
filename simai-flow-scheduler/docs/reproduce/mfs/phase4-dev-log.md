# Phase 4 开发记录：Whole-Batch Feasibility Safeguard

## 概述

Phase 4 在 Phase 3 的基础上引入了可行性保障（Feasibility Safeguard）。通过静态分析预估每个 task 的执行时间，计算每个 request 的 critical path 总时长（截止到 P2D flow 完成），运行时动态维护 `remaining_us`，并在分配带宽时检查 request 是否已不可能满足 TTFT deadline。不可行的 EARLY RLI=0 flow 被降级到 queue 0，避免浪费带宽。

**最终结果：**
- 全部 580 个测试通过（含 17 个 Phase 4 测试），无回归
- 静态分析新增 `mfs_feasibility.py`，计算 per-task 时长和 per-request critical path
- Policy 在 `on_task_completed` 中维护 `remaining_us`
- Allocator 在 RED quantile 分级前检查可行性，降级不可行 flow

---

## 开发顺序与具体内容

### Task 13: 静态分析 Feasibility Pass（`mfs_feasibility.py`）

**目标：** 为可行性检查提供静态数据：per-task 预估时长和 per-request critical path 总时长。

**数据结构：**

```python
@dataclass
class FeasibilityInfo:
    task_duration_us: dict[int, int]            # per-task 预估时长（us）
    critical_path_tasks: set[int]               # 所有 request 的 critical path tasks 并集
    request_critical_tasks: dict[int, set[int]] # rid -> 该 request 的 critical path task set
    request_path_total_us: dict[int, int]        # rid -> critical path 总时长（us）
```

**实现：**

`build_feasibility_info(workload, context, route_table, topology)` 函数：

1. **Per-task 时长**：复用 `critical_path.py` 的 `_estimate_duration()`。Compute task 使用 `task.duration_us`，flow task 使用 `size_bits / bottleneck_bandwidth + propagation_delay`。

2. **Per-request critical path（截止到 P2D flow 完成）**：
   - 对每个 request，取其 task 集合 `rid_tasks`
   - 识别 P2D tasks：`{tid for tid in rid_tasks if context.task_info[tid].mfs_stage == MfsStage.P2D}`
   - 无 P2D tasks 的 request 直接跳过（无 TTFT deadline 需要保障）
   - 构建 sub-DAG：只保留 request 内部的 task，依赖也只保留内部依赖
   - Kahn's 拓扑排序 + longest path
   - **Critical path 终点只在 P2D tasks 中选择**：`max(dist[tid] + task_duration_us[tid] for tid in p2d_tasks)`
   - 回溯标记：从 longest path P2D 终端回溯，标记所有满足 `dist[dep] + dur[dep] == dist[tid]` 的前驱

3. **合并**：`critical_path_tasks` = 所有 `request_critical_tasks[rid]` 的并集

**关键设计决策：**

- **Critical path 截止到 P2D**：TTFT deadline 只关心到第一个 token 输出，即 P2D flow 完成。Decode 阶段的 compute/communication 不纳入 critical path。Prefill compute (BACKGROUND) 和 EARLY 通信作为 P2D 的前置仍然纳入。
- **Per-request critical path task set**：使用 `request_critical_tasks[rid]` 而非全局 `critical_path_tasks` 来做 `on_task_completed` 中的扣减，避免共享 task 误扣非关联 request 的 `remaining_us`。
- **无 P2D tasks 的 request 跳过**：纯 prefill 或纯 decode request 不参与可行性分析。

**测试：** 8 个测试覆盖 compute/flow task 时长估算、linear chain、parallel branches、无 P2D 跳过、decode 排除、多 request 独立性、per-request critical set。

### Task 14: 运行时 Feasibility 检查与降级

**目标：** Policy 维护 `remaining_us`，Allocator 检查可行性并降级不可行 flow。

**Policy 变更（`mfs_policy.py`）：**

1. `initialize()` 中从 `FeasibilityInfo.request_path_total_us` 初始化 `allocator.remaining_us`（copy 一份，运行时会修改）

2. `on_task_completed()` 中扣减 `remaining_us`：
   - 检查完成的 task 是否在 `critical_path_tasks` 中
   - 对该 task 关联的每个 request，检查 task 是否在该 request 的 `request_critical_tasks[rid]` 中
   - 满足条件时：`remaining_us[rid] -= task_duration_us[task_id]`

**Allocator 变更（`mfs_allocator.py`）：**

1. 新增动态状态：`remaining_us: dict[int, int]`（由 policy 更新）

2. 新增 `_is_infeasible(request_ids, current_time)` 方法：
   - 对所有关联 request 检查 `current_time + remaining_us[rid] > start_time + ttft_slo_us`
   - 任一 request 没有 SLO、没有 start_time、或仍可行 → 返回 False
   - 全部不可行 → 返回 True

3. 集成到 `allocate()` Phase 2（RED quantile 分级前）：
   ```python
   for f in rli0_flows:
       info = self.context.task_info.get(f.task_id)
       if info and self._is_infeasible(info.request_ids, current_time):
           queue_flows[0].append(f)  # 降级到 queue 0
           continue
       # ... 后续 RED quantile 逻辑 ...
   ```

**Strategy 变更（`mfs_strategy.py`）：**

- `MfsAnalysisResult` 新增 `feasibility_info: FeasibilityInfo | None = None` 字段
- `MfsAnalyzer.analyze()` 中调用 `build_feasibility_info()` 填充

**测试：**
- Allocator 层 6 个测试：不可行降级、可行正常、部分可行不降级、全部不可行降级、无 SLO 不降级、P2D 不受影响
- Policy 层 3 个测试：remaining_us 初始化匹配、critical task 完成时扣减、per-request 独立扣减

---

## 文件清单

### 新建文件

| 文件 | 说明 |
|------|------|
| `simai-flow-scheduler/src/static_analysis/passes/mfs_feasibility.py` | Feasibility 静态分析 pass |
| `simai-flow-scheduler/tests/test_mfs_feasibility.py` | Feasibility 分析测试（8 tests） |

### 修改文件

| 文件 | 修改内容 |
|------|---------|
| `simai-flow-scheduler/src/static_analysis/strategies/mfs_strategy.py` | 添加 `feasibility_info` 字段，调用 `build_feasibility_info()` |
| `simai-flow-scheduler/src/executor/policies/mfs_policy.py` | `initialize()` 初始化 `remaining_us`，`on_task_completed()` 扣减 |
| `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py` | 新增 `remaining_us` 状态、`_is_infeasible()` 方法，Phase 2 可行性检查 |
| `simai-flow-scheduler/tests/test_mfs_allocator.py` | 新增 `TestFeasibilityDemotion`（6 tests） |
| `simai-flow-scheduler/tests/test_mfs_policy.py` | 新增 `TestRemainingUsTracking`（3 tests） |

---

## 设计决策

1. **Critical path 截止到 P2D flow 完成：** TTFT deadline 衡量的是第一个 token 输出的延迟，对应 P2D KV cache transfer 完成时刻。之后的 decode 阶段不纳入。Prefill compute 和 EARLY 通信作为 P2D 的前置依赖仍然在 critical path 上。

2. **Per-request critical path task set（`request_critical_tasks`）：** 一个 task 可能关联多个 request，但只在其所属 request 的 critical path 上时才扣减该 request 的 `remaining_us`。这避免了共享 task（如 prefill compute）误扣非关联 request 的时间。

3. **可行性检查只在 EARLY RLI=0 分级时应用：** P2D flow 已有独立的 MLU 提升机制，EARLY default (RLI > 0) 不需要可行性检查。只有进入 RED quantile 分级的 EARLY RLI=0 flow 才会检查。

4. **ALL 不可行才降级：** flow 关联多个 request 时，任一 request 仍可行就不降级。只有全部关联 request 都不可行时才降级到 queue 0。这避免了对仍有机会的 request 过早放弃。

5. **无 SLO 的 request 永不触发降级：** 没有 `ttft_slo_us` 的 request 无法判断可行性，不应触发降级。`_is_infeasible()` 遇到无 SLO 的 request 直接返回 False。

6. **复用 `_estimate_duration()` 而非重新实现：** `critical_path.py` 已有完整的 duration 估算逻辑（compute task 使用 `duration_us`，flow task 使用 path bottleneck bandwidth + latency）。同 package 内 import 即可，无需重复。

7. **`FeasibilityInfo` 作为 `MfsAnalysisResult` 的可选字段：** 使用 `| None = None` 使现有代码无需修改即可工作。未来如果需要跳过可行性分析，设为 None 即可。

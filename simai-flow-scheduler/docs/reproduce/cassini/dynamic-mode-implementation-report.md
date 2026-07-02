# Cassini + DynamicExecutor 集成实现总结

> 日期: 2026-07-01
> 对应计划: [`dynamic-mode-integration-plan.md`](dynamic-mode-integration-plan.md)
> 状态: 已实现并验证

---

## 1. 概述

将 Cassini 调度策略（`CassiniSchedulingPolicy`）与 Phase 7 的 DynamicExecutor 集成，实现**按需展开的训练 workload 模拟**：

- **分析阶段**: CassiniAnalyzer 对代表性单 iteration workload 做一次分析（routing、CPM、comm pattern、time-shift）
- **执行阶段**: TrainingJobSlicer 创建 DAG，DynamicExecutor 逐步展开每个 iteration，每次只展开一个 job 的 tasks

避免全量展开 workload 的内存开销（O(活跃 jobs) 而非 O(总 iterations)）。

---

## 2. 新增/修改的文件

### 2.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `src/static_analysis/strategies/default_strategy.py` | +20 | LightweightAnalyzer — CppReferenceSerializer 的 compute_order 构建器（新增于现有文件） |
| `scripts/run_cassini_dynamic_e2e.py` | 420 | Cassini 动态模式入口脚本，二阶段流程 |

### 2.2 修改文件

| 文件 | 改动 |
|------|------|
| `src/workload_format/compact_workload.py` | `JobExpansionInfo` 增加 `job_group_id` 字段；`to_dict()` 序列化该字段 |
| `src/workload_generator/job_slicer.py` | `TrainingJobSlicer.slice_trace()` 新增 `job_group_id` 参数；修复 `assigned_nodes` 校验（`len % world_size == 0`）；修复 dp 计算 |
| `src/executor/policies/cassini_policy.py` | 新增 `update_analysis()` 方法（增量合并 compute_order + route_table）；新增 `DefaultAnalysisResult` import |
| `src/executor/job_manager.py` | 新增 `try_expand_eligible()` 公共方法 |
| `src/executor/job_expander.py` | 修复 `_expand_training()` 中 `entry_task_ids`/`terminal_task_ids` 计算逻辑 |
| `src/executor/dynamic_executor.py` | 多个 bug 修复（见下文） |

---

## 3. 架构设计

### 3.1 二阶段流程

```
Phase 1: One-shot 分析
─────────────────────
  对每个逻辑 Job 展开一个代表 iteration
  → merge_ga_to_one_iteration
  → JobMerger.merge()
  → CassiniAnalyzer.analyze(representative_workload)
    → route_table (通用, 所有 iteration 复用)
    → time_shifts (按 job_group_id 索引)
    → communication_patterns (只用于 time-shift 计算)
    → execution_plan (compute_order, 仅用于 static 模式初始化)

Phase 2: 动态执行
─────────────────
  TrainingJobSlicer → CompactWorkload (全部 N 个 iteration)
  DynamicExecutor.execute_dynamic():
    ① JobManager.initialize() → 展开 root jobs
    ② _analyze_and_inject() → LightweightAnalyzer → policy.update_analysis()
    ③ _drain_ready_pool() → 发出 ready tasks
    ④ 事件循环 → 处理 compute_done/flow_completion 事件
    ⑤ on_tasks_completed() → 检测 job 完成 → 展开后继 jobs
    ⑥ _reclaim_completed() → 回收 timing 到 ExecutionResult
```

### 3.2 LightweightAnalyzer

避免在每个 iteration 展开时重跑 CPM + BFS 的轻量级分析器：

```python
class LightweightAnalyzer:
    def analyze(self, workload):
        topo_order = _topological_sort(workload.tasks)  # Kahn 算法
        for tid in topo_order:
            task = task_map[tid]
            if task.is_compute() and task.node is not None:
                compute_order.setdefault(task.node, []).append(tid)
        return DefaultAnalysisResult(
            route_table=BfsRouteTable(topology),   # 空表，update_routes 幂等
            execution_plan=ExecutionPlan(compute_order=compute_order),
        )
```

路由表在 Phase 1 已预计算完全，每个 iteration 展开后通过 `update_routes()` 注册新 task_id（对已存在的 `(src,dst)` 无操作）。compute_order 用拓扑排序而非创建顺序，避免 `_is_next_compute()` 死锁。

### 3.3 job_group_id 映射

- `TrainingJobSlicer` 为每个 iteration 创建的 Job 带上 `job_group_id`
- 同一逻辑 Job 的所有 iteration 共享同一个 `job_group_id`
- `time_shifts[job_group_id]` 作为 `ExpandedJob.delay_us` 注入 entry tasks
- 通过 `delayed_queue` 机制实现 time-shift 延迟

---

## 4. 修复的 Bug

### Bug 1: `batch_completed` 作用域错误

**现象**: 事件循环运行但始终没有 job 被标记为完成。

**根因**: `batch_completed` 声明在内部 `while True` 循环里，每次处理一个 batch 后重置累加器，最终只有最后一个 batch 的 task IDs 被传入 `on_tasks_completed()`。

```python
# ❌ 修复前: 每个 batch 重置
while True:
    batch_events = [...]
    batch_completed: set[int] = set()  # 重置！
    for event in batch_events:
        ...
        batch_completed.add(event.task_id)
# ↑ 最终 batch_completed 只有最后一个 batch 的数据

# ✓ 修复后: 声明在外部 while 循环里，累积所有 batch
batch_completed: set[int] = set()
while True:
    batch_events = [...]
    for event in batch_events:
        batch_completed.add(event.task_id)  # 累积
```

**影响**: 终端 task completion 永远不会被 `on_tasks_completed()` 检测到，job 永远不被标记完成。

### Bug 2: `_reclaim_completed` 门控错误

**现象**: 即使 event 正确处理且 job 被标记完成，`self._per_task` 始终为空。

**根因**: `_reclaim_completed()` 放在 `if new_batches:` 代码块内。最后一个 job 完成后没有后继 job（`on_tasks_completed` 返回空 list），回收逻辑永远不会执行。

```python
# ❌ 修复前: 回收在 if 块内
new_batches = self._job_manager.on_tasks_completed(...)
if new_batches:         # 最后一个 job 完成后这里为 False
    ...
    self._reclaim_completed(...)  # 永不执行!

# ✓ 修复后: 回收无条件执行
new_batches = self._job_manager.on_tasks_completed(...)
for ej in new_batches: ...
self._reclaim_completed(...)  # 总是执行
```

### Bug 3: 事件循环提前退出

**现象**: 事件队列空了但 job 尚未完成时，循环直接退出。

**根因**: `not event_queue` 分支中的 `else: break` 没有检查 `is_all_jobs_done()`。

```python
# 修复前
else:
    if delayed_queue: ...
    break  # 直接退出，不管还有没有未完成的 job

# 修复后
else:
    if delayed_queue: ...
    new_jobs = self._job_manager.try_expand_eligible(sim_state)
    if new_jobs:
        self._analyze_and_inject(...)
        continue
    break  # 真没有可展开的 job 了才退出
```

**影响**: 多 iteration 场景在第 1 个 iteration 完成后事件队列空时直接退出，后续 iteration 永远不展开。

### Bug 4: `_expand_training()` entry/terminal task 计算错误

**现象**: training 展开后 `entry_task_ids` 包含错误 tasks，`terminal_task_ids` 始终为空。

**根因**: 代码注释写"entry tasks (no deps)"，但实际计算的是 `all_ids - has_dependents`（没有其他 task 依赖的 task），相当于 termnal 的含义。`terminal_ids` 的计算找的是指向不存在的 task_id 的 deps，对自包含的 workload 永远是空集。

```python
# ❌ 修复前
entry_ids = sorted(all_ids - has_dependents)        # 错误：相当于 terminal
terminal_ids = deps指向不存在task_id的集合            # 永远为空

# ✓ 修复后
entry_ids = sorted(t for t in tasks if not t.deps)  # 正确：没有 deps 的 task
terminal_ids = sorted(all_ids - has_dependents)      # 正确：没有后继的 task
```

**影响**: `_remaining_terminal[job_id]` 被设为 0，job 完成检测立即通过，但 `_terminal_to_job` 索引为空，`drain_completed()` 返回空 list。

### Bug 5: `compute_order` 被代表 workload 污染

**现象**: 少量 tasks 完成后死锁。

**根因**: `CassiniSchedulingPolicy.__init__()` 从 `CassiniAnalysisResult.execution_plan.compute_order` 初始化 `compute_order`，其中包含代表 workload 的 task_ids。动态展开后 `update_analysis()` 追加新 task_ids，但原始 task_ids 排在前面占据 `_is_next_compute()` 的判断位置。

```python
# 代表 workload 的 compute_order: [0, 6, 12, 18, 28, 38, ...]
# 动态展开的 compute_order:      [0, 2, 4, 6, 8, 10, ...]
# 合并后: [0, 6, 12, 18, 28, 38, ...] + [0, 2, 4, 6, ...]
#         ↑ 前几个是代表 workload 的 task_id，不是实际展开后对应 node 的 task
```

**修复**: 入口脚本中 `policy.compute_order = {}` 清空，完全由 `update_analysis()` 填充。

### Bug 6: `TrainingJobSlicer` 校验过严

**现象**: dp > 1 时 `slice_trace()` 报错。

**根因**: 校验条件 `len(assigned_nodes) == required (all_gpus)` 在 dp > 1 时失败（job 实际需要的节点数 = all_gpus × dp）。

**修复**: 改为 `len(assigned_nodes) % world_size == 0`，dp 从 `len(assigned_nodes) // world_size` 计算。

---

## 5. 验证结果

### 单 Job 场景 (dp=1, 1 job, 2 iterations)

```
Tasks: 838 → 展开为 1676 完成
Makespan: 1244.59 ms
Execution time: 0.02s
```

### 多 Job 场景 (dp=2, 2 jobs, 2 iterations)

```
Tasks: 3368 → 展开为 6736 完成
Makespan: 1325.66 ms
Execution time: 0.10s
```

---

## 6. 不修改的文件

| 文件 | 原因 |
|------|------|
| `src/cassini/affinity_graph.py` | 纯数学，只用于分析阶段 |
| `src/cassini/circle_abstraction.py` | 同上 |
| `src/cassini/pair_compatibility.py` | 同上 |
| `src/cassini/communication_pattern.py` | 同上 |
| `src/executor/analytical.py` | 父类零改动（设计约束） |
| `src/executor/policies/base_policy.py` | 接口已包含 `update_analysis()` |
| `src/cassini/iteration_expansion.py` | `replicate_with_cross_iteration_deps()` 只用于旧静态路径 |
| `src/workload_generator/inference_trace_expander.py` | 只用于 inference |

---

## 7. 用法

```bash
# 默认配置: ws2 tp2 pp1 dp2, 2 jobs, 5 iterations
uv run python scripts/run_cassini_dynamic_e2e.py

# 自定义
uv run python scripts/run_cassini_dynamic_e2e.py \
    --topo <topology> \
    --aicb job_A.txt job_B.txt \
    --dp 2 1 \
    --num-iters 10 5 \
    --output outputs/cassini_dynamic
```

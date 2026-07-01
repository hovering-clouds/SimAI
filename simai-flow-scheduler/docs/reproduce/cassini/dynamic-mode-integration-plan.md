# Cassini + DynamicExecutor 集成重构计划

> 日期: 2026-07-01
> 背景: Phase 7 实现了 DynamicExecutor（按需展开 Task），但 Cassini 策略仍走旧有的全量静态展开路径。
> 目标: 让 Cassini 在 DynamicExecutor 模式下工作，避免全量展开 C10+ 的多 iteration training workload。

---

## 1. 核心洞察

### 1.1 Cassini 的分析本质上是 One-Shot

Cassini 的完整分析流水线（`CassiniAnalyzer.analyze()`）包含 5 步：

| 步骤                             | 输入                 | 是否依赖 iteration 数量                                | 结论       |
| -------------------------------- | -------------------- | ------------------------------------------------------ | ---------- |
| ① BFS Routing                   | 拓扑 + (src,dst)     | 否 — 只与拓扑和节点对有关                             | 预计算即可 |
| ② CPM Critical Path             | Task DAG             | 否 — 每个 iteration 的 DAG 结构一致                   | 一轮就够了 |
| ③ Communication Pattern         | CPM timing + 路由    | 否 — 每个 flow task 被`% iteration_time` 折叠到圆上 | 一轮就够了 |
| ④ Affinity Graph → Time-shifts | CommunicationPattern | 否 — 只比较带宽圆，不关心执行次数                     | 一轮就够了 |
| ⑤ Compute Order                 | Task DAG             | 否 — 每个 iteration 的 compute 顺序相同               | 一轮就够了 |

关键证据在 `communication_pattern.py` 的 `_add_task_to_pattern()` 中：

```python
window_start = task_window_starts.get(task.task_id, 0)
cpm_start = timing.earliest_start_us - window_start  # 归一化到 iteration 窗口
# ...
offset_start = cpm_start % iteration_time_us         # 折叠进一个周期
```

无论 workload 中有 1 个 iteration 还是 1000 个，每个 flow task 经过 `- window_start` + `% iteration_time_us` 两步变换后落到**完全相同的角度桶**上。

### 1.2 动态展开路径已具备的基础设施

| 组件                                   | 现状           | 与本计划的关系                                 |
| -------------------------------------- | -------------- | ---------------------------------------------- |
| `TrainingJobSlicer`                  | 已实现         | 为每个 iteration 创建一个 Job，以 DAG 方式串联 |
| `JobExpander._expand_training()`     | 已实现         | 展开单个 training Job                          |
| `DynamicExecutor`                    | 已实现         | 按需展开 + 事件循环                            |
| `_analyze_and_inject()`              | 已实现         | 展开后对 tasks 做分析并合并到 Policy           |
| `SchedulingPolicy.update_analysis()` | 基类定义了接口 | CassiniSchedulingPolicy 需要实现               |
| `ExpandedJob.delay_us`               | 已实现         | 可携带 time-shift 作为 entry tasks 启动延迟    |

---

## 2. 方案设计

### 2.1 总体流程

```
┌── 阶段一：预分析 (one-shot) ──────────────────────────────────┐
│                                                               │
│  对每个"逻辑 Job"展开一个代表 iteration                          │
│    AICB_0 → WorkloadBuilder  → single-iter workload_0         │
│    AICB_1 → WorkloadBuilder  → single-iter workload_1         │
│        ↓                                                      │
│    JobMerger.merge(workloads).merged_workload                  │
│    (含每个逻辑 Job 各一轮 iteration 的 tasks)                   │
│        ↓                                                      │
│    CassiniAnalyzer.analyze(representative_workload)            │
│      → route_table        (所有 (src,dst) 的最短路径)          │
│      → critical_path      (一轮 iteration 的 CPM timing)       │
│      → communication_patterns (每个逻辑 Job 的带宽圆)          │
│      → time_shifts        (按逻辑 Job ID 索引)                 │
│      → execution_plan     (一轮 iteration 的 compute order)    │
│                                                               │
├── 阶段二：动态执行 ───────────────────────────────────────────┤
│                                                               │
│  TrainingJobSlicer → CompactWorkload (全部 N 个 iteration)     │
│  DynamicExecutor.execute_dynamic():                             │
│    Iter 0 → JobExpander.expand_job()                           │
│           → _analyze_and_inject() → policy.update_analysis()   │
│           → delay_us = time_shift_map[job_group_id]            │
│    Iter 1 → 同上 (自动等待 Iter 0 完成后触发)                  │
│    ...                                                         │
│    Iter N → 全部完成 → ExecutionResult                         │
│                                                               │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 新增/修改的文件清单

| 文件                                        | 操作               | 改动内容                                                                   |
| ------------------------------------------- | ------------------ | -------------------------------------------------------------------------- |
| `src/executor/policies/cassini_policy.py` | **修改**     | 新增`update_analysis()`, `initialize()` 兼容动态模式                   |
| `src/workload_format/compact_workload.py` | **修改**     | `JobExpansionInfo` 增加 `job_group_id` 字段                            |
| `src/executor/dynamic_executor.py`        | **修改**     | `_analyze_and_inject()` 适配 CassiniPolicy                               |
| `src/executor/job_expander.py`            | **修改**     | `_expand_training()` 支持 `prev_exits` 参数（跨 iteration 跨节点接线） |
| `scripts/run_cassini_e2e.py`              | **修改**     | 新增`--mode dynamic` 分支                                                |
| `scripts/run_dynamic_e2e.py`              | **可选修改** | 或新建一个独立入口                                                         |

---

## 3. 详细变更

### 3.1 `CassiniSchedulingPolicy.update_analysis()` — 增量合并

**位置**: `src/executor/policies/cassini_policy.py`

这是最关键的改动。目前 `CassiniSchedulingPolicy` 继承基类的空实现，需要像 `DefaultSchedulingPolicy` 一样做增量合并：

```python
def update_analysis(self, workload, analysis_result):
    """增量合并新展开的 iteration 的分析结果。

    Args:
        workload: 新 Job 展开后的 P2PWorkload（含该 iteration 的 tasks）。
        analysis_result: DefaultAnalysisResult（含 route_table + execution_plan）。
    """
    # 1. 合并 compute_order — 新 task_ids 追加到对应 node 的列表末尾
    for node_id, task_ids in analysis_result.execution_plan.compute_order.items():
        self.compute_order.setdefault(node_id, []).extend(task_ids)

    # 2. 合并路由 — 新 task_id 注册到 route table
    self.route_table.update_routes(analysis_result.route_table)
```

`compute_cursor` 是 `defaultdict(int)`，cursor 的递增只和已完成 compute task 的数量有关，不需要做额外处理。

> **注意**: `compute_position`（task_id → 列表中顺序位置的索引）在 `DefaultSchedulingPolicy` 和 `CassiniSchedulingPolicy` 中都有定义但**从未被读取**，调度决策实际使用 `compute_order` + `compute_cursor` 就够了。所以 `update_analysis()` 不需要重建它。

### 3.2 `initialize()` 兼容动态模式

Cassini e2e 脚本中，`initialize()` 接收包含代表性 iteration 的 workload 即可：

```python
# 在 run_cassini_e2e.py 的动态分支中
policy = CassiniSchedulingPolicy(cassini_result)
policy.initialize(representative_workload, topology)
```

`_job_started` 字典中的 job_id 是逻辑 Job ID（0, 1, 2,...），与动态展开时 `TrainingJobSlicer` 生成的具体 iteration job_id 不重合，但不影响——`_job_cleared()` 只对不在 `_job_started` 中的 job_id 返回 `True`。

### 3.3 `JobExpansionInfo.job_group_id` — 逻辑 Job 分组

**位置**: `src/workload_format/compact_workload.py`

```python
@dataclass
class JobExpansionInfo:
    depends_on: list[int] = field(default_factory=list)
    job_type: str = "inference"
    trace_src: str = ""
    trace_job_index: int = 0
    job_group_id: int = 0       # ← 新增：逻辑 Job 组 ID
```

`job_group_id` 的映射关系：

- 同一个 AICB 文件 + 同一组节点分配 → 同一个 `job_group_id`
- 所有共享此 ID 的 Job 共享同一个 Cassini time-shift

**`TrainingJobSlicer` 中的维护逻辑**：

```python
# job_slicer.py
def slice_trace(self, trace_path, assigned_nodes, repeat=1, job_group_id=0):
    ...
    for i in range(repeat):
        info[i] = JobExpansionInfo(
            depends_on=[i - 1] if i > 0 else [],
            job_type="training",
            trace_src=trace_path,
            trace_job_index=i,
            job_group_id=job_group_id,  # ← 传入
        )
```

### 3.4 time-shift 映射表

**位置**: 在 `DynamicExecutor.execute_dynamic()` 或入口脚本中维护

```python
# 分析阶段产出：cassini_result.time_shifts = {logical_job_id: shift_us}

# 执行阶段：构建 job_group_id → shift_us 的映射
time_shift_map: dict[int, int] = {}
for logical_job_id, shift_us in cassini_result.time_shifts.items():
    time_shift_map[logical_job_id] = shift_us
```

在 `_analyze_and_inject()` 或 `JobManager` 展开时，从 `JobExpansionInfo.job_group_id` 查表，设 `ej.delay_us = time_shift_map[job_group_id]`。

### 3.5 轻量级 Analyzer — 避免重复分析

#### 问题

`_analyze_and_inject()` 每次展开 iteration 都调用 `self._analyzer.analyze(mini_wl)`。对 Cassini 模式，这会**重复执行** BFS 路由、CPM、execution_plan 序列化——而这些在 Phase 1 的 one-shot 分析中已经完成。

实际上每个 iteration 展开时**唯一需要做**的是把新 task_ids 注册到 `compute_order` 中。`BfsRouteTable` 按 `(src, dst)` 寻址（`bfs.py:19-20`），`update_routes()` 是幂等的，所以路由表在第一次 iteration 后就不再变化。

#### 方案：LightweightAnalyzer

不修改 `_analyze_and_inject`，而是在 Cassini 模式传入一个轻量版 analyzer，只算 `compute_order`，不做 CPM/BFS：

**位置**: `src/executor/dynamic_executor.py`（新增内部类）或 `src/static_analysis/strategies/` 下

```python
class LightweightAnalyzer:
    """轻量分析器 — 只构建 compute_order，不做 CPM/BFS。

    用于 Cassini 动态模式：路由和时间片已在 Phase 1 one-shot 分析中预计算，
    每个 iteration 展开只需注册新 task_ids 到 compute_order。
    """

    def __init__(self, topology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> DefaultAnalysisResult:
        plan = ExecutionPlan()
        for task in workload.tasks:
            if task.is_compute() and task.node is not None:
                plan.compute_order.setdefault(task.node, []).append(task.task_id)

        return DefaultAnalysisResult(
            route_table=BfsRouteTable(self.topology),  # 空表 — update_routes 幂等
            execution_plan=plan,
        )
```

`_analyze_and_inject()` 的代码保持不变，只要在创建 `DynamicExecutor` 时传 `LightweightAnalyzer` 即可：

```python
executor = DynamicExecutor(
    topology=topology,
    policy=CassiniSchedulingPolicy(cassini_result),
    analyzer=LightweightAnalyzer(topology),   # ← 替代 DefaultAnalyzer
)
```

这样 `update_analysis()` 正常工作：`compute_order` 正常合并，`update_routes()` 收到空表后是 no-op。

### 3.6 `JobExpander._expand_training()` — 跨 iteration 接线

**位置**: `src/executor/job_expander.py`

当前 `_expand_training()` 内部的 `WorkloadBuilder.build_from_aicb()` 只展开一个独立 iteration，没有跨 iteration 依赖。但 `TrainingJobSlicer` 已经在 `JobExpansionInfo.depends_on` 中声明了 iteration 间的依赖关系。

依赖链的传递方式：

```
CompactWorkload 层面:
  Job 0 (iter_0) ──→ Job 1 (iter_1) ──→ Job 2 (iter_2) ──→ ...
  (depends_on=[-1])   (depends_on=[0])    (depends_on=[1])

每个 Job 展开后:
  ExpandedJob.terminal_task_ids → 在完成检测时用于触发后继 Job
  ExpandedJob.entry_task_ids    → 依赖满足后 ready 的 tasks
```

`JobManager` 已经处理了这个逻辑：

1. Job 0 的 terminal tasks 完成 → `mark_completed(0)` → Job 1 变为 eligible
2. `_expand_selected([1])` → 展开 Job 1 → 注入 tasks
3. Job 1 的 entry tasks 被注入 ready_pool

所以 `_expand_training()` 本身不需要额外参数或修改——跨 iteration 的依赖在 JobDAG 层面管理，不在 task 层面。

---

## 4. 入口脚本骨架

修改 `run_cassini_e2e.py`，新增 `--mode dynamic` 参数。以下是动态分支的核心逻辑：

```python
def run_cassini_dynamic(config, topology):
    """Cassini 动态模式入口。"""
    # ── Phase 1: 构建代表性 workload（每个逻辑 Job 一个 iteration）──
    parser = AicbParser()
    rep_jobs_list = []
    logical_jobs = []  # (aicb_path, dp, num_iters, nodes)

    for i, entry in enumerate(config["workloads"]):
        # 解析 + placement
        header, items = parser.parse(entry["aicb"])
        tp, dp, pp, ep, _ = resolve_parallelism(header, entry["dp"])
        nodes = assign_nodes(i, ...)  # 复用现有 placement 逻辑

        # 展开一个 iteration
        job = Job(job_id=i, name=Path(entry["aicb"]).stem, 
                  assigned_nodes=nodes,
                  parallelism=ParallelismConfig(tp=tp, dp=dp, pp=pp, ep=ep))
        wl = WorkloadBuilder().build_from_aicb(header, items, job, comm_algo="ring")
        wl = merge_ga_to_one_iteration(wl, header.ga)
        rep_jobs_list.append(wl)
        logical_jobs.append((entry["aicb"], entry["dp"], entry["num_iters"], nodes, i))

    # 合并为代表 workload
    if len(rep_jobs_list) == 1:
        rep_workload = rep_jobs_list[0]
    else:
        rep_workload = JobMerger().merge(rep_jobs_list).merged_workload

    # ── Phase 2: Cassini 分析 (one-shot) ──
    cassini_result = CassiniAnalyzer(topology, step_deg=step_deg).analyze(rep_workload)
    patch_iteration_time_us(cassini_result, rep_workload)

    # ── Phase 3: 构建动态执行 DAG ──
    slicer = TrainingJobSlicer()
    all_sliced = []
    for aicb_path, dp, num_iters, nodes, group_id in logical_jobs:
        compact_wl = slicer.slice_trace(
            trace_path=aicb_path,
            assigned_nodes=nodes,
            repeat=num_iters,
            job_group_id=group_id,
        )
        all_sliced.append(compact_wl)

    # 合并多个逻辑 Job 的 CompactWorkload
    merged_compact = merge_compact_workloads(all_sliced)

    # ── Phase 4: 动态执行 ──
    policy = CassiniSchedulingPolicy(cassini_result)
    policy.initialize(rep_workload, topology)

    executor = DynamicExecutor(
        topology=topology,
        policy=policy,
        analyzer=DefaultAnalyzer(topology),
    )

    result = executor.execute_dynamic(
        job_dag=JobDAG.from_compact(merged_compact),
        job_expansion_info=merged_compact.job_expansion_info,
        job_policy=FifoJobPolicy(),
        job_expander=JobExpander(
            task_id_allocator=TaskIdAllocator(),
            profile_store=None,      # training 不需要 profile
        ),
    )

    return result
```

### 辅助函数：合并多个 CompactWorkload

```python
def merge_compact_workloads(workloads: list[CompactWorkload]) -> CompactWorkload:
    """合并多个 CompactWorkload（调整 job_id 偏移量）。"""
    all_jobs = []
    all_info = {}
    offset = 0

    for wl in workloads:
        for job in wl.jobs:
            job.job_id += offset
            all_jobs.append(job)
        for jid, info in wl.job_expansion_info.items():
            info.depends_on = [d + offset for d in info.depends_on]
            all_info[jid + offset] = info
        offset += len(wl.jobs)

    return CompactWorkload(
        version="1.0",
        meta=Meta(num_jobs=len(all_jobs), num_nodes=...),
        jobs=all_jobs,
        job_expansion_info=all_info,
    )
```

---

## 5. 不修改的文件

| 文件                                                   | 原因                                                                       |
| ------------------------------------------------------ | -------------------------------------------------------------------------- |
| `src/cassini/affinity_graph.py`                      | 纯数学，只用于分析阶段                                                     |
| `src/cassini/circle_abstraction.py`                  | 同上                                                                       |
| `src/cassini/pair_compatibility.py`                  | 同上                                                                       |
| `src/cassini/communication_pattern.py`               | 同上                                                                       |
| `src/executor/analytical.py`                         | 父类零改动（设计约束）                                                     |
| `src/executor/policies/base_policy.py`               | 接口已包含`update_analysis()`                                            |
| `src/executor/job_manager.py`                        | 生命周期管理逻辑不变                                                       |
| `src/cassini/iteration_expansion.py`                 | `replicate_with_cross_iteration_deps()` 只用于旧静态路径，动态模式不走它 |
| `src/workload_generator/inference_trace_expander.py` | 只用于 inference，training 用`WorkloadBuilder`                           |

---

## 6. 实施顺序

```
Step 1: JobExpansionInfo 加 job_group_id 字段 + TrainingJobSlicer 支持传入
         → 最小改动，不影响现有逻辑

Step 2: CassiniSchedulingPolicy.update_analysis() 实现
         → 核心增量合并逻辑，可单独测试

Step 3: DynamicExecutor._analyze_and_inject() 验证
         → 确保 DefaultAnalyzer 产出能正确合并到 CassiniSchedulingPolicy

Step 4: 入口脚本 —— 新建或改造 run_cassini_e2e.py
         → 串联完整流程，端到端验证

Step 5: 正确性验证
         → 小规模 trace: 动态路径 vs 静态路径 makespan 一致
```

---

## 7. 潜在风险

| 风险                                                                                           | 影响                       | 缓解                                                          |
| ---------------------------------------------------------------------------------------------- | -------------------------- | ------------------------------------------------------------- |
| `_expand_training()` 产出的 task 结构与 `replicate_with_cross_iteration_deps()` 不完全一致 | makespan 偏差              | 小 trace 对比验证                                             |
| Cassini 分析用的代表性 iteration 与真实展开的 iteration 的 CPM timing 有偏差                   | iteration_time_us 估算偏差 | 用`patch_iteration_time_us()` 从单 iteration CPM 中取中位数 |

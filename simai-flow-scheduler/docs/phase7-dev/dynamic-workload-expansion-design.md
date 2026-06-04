# Phase 7: Workload 动态展开重构设计

> 日期: 2026-06-03
> 状态: 初稿
> 作者: Claude + User

---

## 1. 问题背景

当前 simai-flow-scheduler 在执行模拟前会将整个 workload DAG 全量展开。对于大规模场景（如 16 个 request 的 inference trace），展开后的 `workload.json` 可达 1.6 GB，包含 43.6 万+ tasks。

**膨胀根源**: 一个 168 KB 的 inference trace 膨胀约 800 倍，因为每个 batch × 61 layers × 16 ranks × 集合通信展开会产生大量 task。

**影响**: 无法扩展到真实规模的 training（数千 iteration）和 inference（数百 request）场景。

## 2. 设计决策总览

| 维度 | 决策 |
|------|------|
| 兼容性 | 双模式共存 — 静态（现有）+ 动态（新增），配置切换 |
| Job 粒度 | 用户可配置（`iteration` / `batch` / `wave` / `custom`） |
| 调度策略 | 单一统一插件接口（`JobPolicy`），类似 `SchedulingPolicy` 模式 |
| 静态分析 | Per-Job 按需分析，通过现有 `on_task_emitted` 回调处理增量状态 |
| 跨 Job 依赖 | DAG 式，在 JobDAG 层面管理（task 级别无跨 Job 依赖） |
| 内存管理 | 已完成 Job 的 task details 立即回收；timing 数据持久化到磁盘 |
| Executor 设计 | `DynamicExecutor` 继承 `AnalyticalExecutor` — 父类不做任何修改 |

## 3. 整体架构

### 3.1 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    入口 (scripts/)                        │
│   config.mode == "static"  →  AnalyticalExecutor        │
│   config.mode == "dynamic" →  DynamicExecutor           │
└──────────────┬──────────────────────┬───────────────────┘
               │ 静态                 │ 动态
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  现有静态路径（不变）  │  │  原始 Trace 文件                │
│  WorkloadBuilder      │  │  (inference JSON / AICB txt)   │
│  → P2PWorkload JSON   │  └───────────┬───────────────────┘
│  → Static Analysis    │              │
│  → Executor           │              ▼
└──────────────────────┘  ┌───────────────────────────────┐
                          │  JobSlicer                     │
                          │  → JobDAG (轻量)               │
                          │    (每个节点 = 一个 JobSpec)     │
                          └───────────┬───────────────────┘
                                      │
                                      ▼
                         ┌───────────────────────────────┐
                         │  DynamicExecutor               │
                         │    继承 AnalyticalExecutor       │
                         │  ┌─────────────────────────┐   │
                         │  │    JobManager            │   │
                         │  │  ├── JobDAG 生命周期管理  │   │
                         │  │  ├── JobPolicy 调度决策   │   │
                         │  │  ├── JobExpander 展开     │   │
                         │  │  ├── Per-Job Analysis     │   │
                         │  │  └── Task 注入与回收      │   │
                         │  └─────────────────────────┘   │
                         │  ┌─────────────────────────┐   │
                         │  │  DES Event Loop           │   │
                         │  │  (继承自父类)              │   │
                         │  └─────────────────────────┘   │
                         └───────────────────────────────┘
```

### 3.2 关键洞察：不需要新的文件格式

原始 input trace（inference JSON 或 AICB text）本身已经是紧凑的。膨胀发生在 `WorkloadBuilder` / `InferenceTraceExpander` 的展开阶段。

动态模式直接读取原始 trace，按需展开单个 Job，彻底避免 1.6 GB 中间文件的产生。

## 4. 数据模型

### 4.1 JobSpec — Job 的轻量描述

```python
@dataclass
class JobSpec:
    job_id: int                       # 全局唯一 ID
    job_type: JobType                 # TRAINING_ITERATION | INFERENCE_BATCH | CUSTOM
    name: str                         # 可读名称，如 "iter_5"、"batch_12"

    # Trace 来源参数
    iteration: int                    # GA step index (training) 或 0 (inference)
    batch_params: dict                # batch/iteration 级别的参数

    # 依赖
    depends_on: list[int]             # 依赖的 JobSpec.job_id 列表

    # 展开配置
    assigned_nodes: list[int]         # 默认节点分配（可被 JobPolicy 覆盖）
    parallelism: ParallelismConfig    # tp, dp, pp, ep（复用现有 schema）

    # 展开状态（展开后填充）
    expanded: bool = False
    terminal_task_ids: list[int] = field(default_factory=list)
    entry_task_ids: list[int] = field(default_factory=list)
```

### 4.2 JobDAG — 粗粒度依赖图

```python
class JobDAG:
    jobs: dict[int, JobSpec]              # job_id → JobSpec
    dependents: dict[int, list[int]]      # job_id → 依赖它的 job_ids
    dep_count: dict[int, int]             # job_id → 剩余依赖数
    root_jobs: list[int]                  # 无依赖的起始 jobs

    def mark_completed(self, job_id: int) -> list[int]:
        """标记 Job 完成，返回新 eligible 的 job IDs。"""
        ...

    def get_eligible_jobs(self) -> list[int]:
        """返回所有依赖已满足的 job IDs。"""
        ...
```

### 4.3 ExpandedJob — 纯数据容器

不存储分析结果。分析结果通过现有 Policy 回调机制传递。

```python
@dataclass
class ExpandedJob:
    job_id: int
    tasks: list[Task]                # 展开产生的全部 tasks
    entry_task_ids: list[int]        # 此 Job 的入口 task IDs
    terminal_task_ids: list[int]     # 此 Job 的出口 task IDs
```

### 4.4 SimulationState — 模拟器状态快照

只读，传递给 JobPolicy 做决策。

```python
@dataclass
class SimulationState:
    current_time_us: int
    completed_job_ids: set[int]
    active_job_ids: set[int]
    pending_job_ids: set[int]
    node_busy_until: dict[int, int]              # node_id → 预计空闲时间
    active_flow_count_per_link: dict[int, int]   # link_id → 活跃 flow 数
    total_bytes_transferred: int
    total_compute_time_us: int
```

## 5. 核心组件

### 5.1 JobPolicy — 统一的 Job 级调度策略接口

```python
class JobPolicy(ABC):
    """Job 级调度策略的统一接口。
    类似现有 SchedulingPolicy 模式，用户通过实现此类自定义 Job 调度。
    """

    @abstractmethod
    def can_emit(self, job: JobSpec, sim_state: SimulationState) -> bool:
        """此 Job 是否可以展开并注入 executor。
        返回 False 表示暂缓，下一轮重新检查。
        """

    @abstractmethod
    def get_placement(self, job: JobSpec, sim_state: SimulationState) -> list[int]:
        """决定此 Job 分配到哪些物理节点。
        默认返回 job.assigned_nodes（使用 JobSpec 中的默认值）。
        """

    @abstractmethod
    def get_delay_us(self, job: JobSpec, sim_state: SimulationState) -> int:
        """为此 Job 的 entry tasks 施加额外启动延迟（微秒）。
        返回 0 表示无延迟。可用于在通信高峰期延缓 Job 启动。
        """

    @abstractmethod
    def order_expansion(self, eligible_jobs: list[JobSpec], sim_state: SimulationState) -> list[JobSpec]:
        """对多个 eligible jobs 排序，决定展开优先级。
        当 DAG 中有多个依赖已满足的 Job 时调用。
        """
```

**内置实现示例**:

- `FifoJobPolicy`: 按 trace 顺序展开，不做任何调整
- `ContentionAwareJobPolicy`: 在网络拥塞时自动延缓 Job 启动

### 5.2 JobExpander — 复用现有展开逻辑

```python
class JobExpander:
    """将单个 JobSpec 展开为 Task 列表。
    复用现有 WorkloadBuilder / InferenceTraceExpander 的展开逻辑。
    """

    def __init__(self, task_id_allocator: TaskIdAllocator, profile_store, topology):
        ...

    def expand_job(self, job: JobSpec, context: ExpansionContext) -> ExpandedJob:
        if job.job_type == JobType.INFERENCE_BATCH:
            return self._expand_inference_batch(job, context)
        elif job.job_type == JobType.TRAINING_ITERATION:
            return self._expand_training_iteration(job, context)
```

**Inference**: 调用 `InferenceTraceExpander.expand_single_batch()` — 新方法，从现有 `expand()` 循环体中提取。

**Training**: 调用 `WorkloadBuilder.build_from_aicb()`，只传入当前 iteration 的 items。`WorkloadBuilder` 已天然支持单 Job 处理，无需修改。

### 5.3 TaskIdAllocator — 全局 Task ID 分配器

```python
class TaskIdAllocator:
    def __init__(self, start: int = 0):
        self._next_id = start

    def allocate(self, count: int) -> tuple[int, int]:
        """分配 count 个连续 ID，返回 (start_id, end_id)。"""
        ...
```

### 5.4 JobManager — 生命周期管理

`DynamicExecutor` 的内部组件，协调展开、分析、注入和回收。

```python
class JobManager:
    def __init__(self, job_dag, job_expander, job_policy, analyzer, topology):
        ...
        self.completed_jobs: set[int] = set()
        self.job_terminal_tasks: dict[int, list[int]] = {}    # 用于完成检测
        self.active_job_tasks: dict[int, set[int]] = {}        # 用于回收
        self.timing_writer: TimingWriter

    def initialize(self) -> list[Task]:
        """展开所有 root jobs，返回初始 tasks。"""
        ...

    def on_tasks_completed(self, completed_task_ids: list[int]) -> list[Task] | None:
        """Executor 调用：报告 tasks 完成。
        检查是否有 Job 完成 → 回收 → 展开后继 → 返回新 tasks。
        """

    def try_expand_eligible(self, sim_state: SimulationState) -> list[Task] | None:
        """尝试展开 eligible jobs。用于事件队列空但还有 Job 未完成的情况。"""
        ...

    def is_all_jobs_done(self) -> bool:
        ...
```

### 5.5 JobSlicer — 从原始 trace 构建 JobDAG

```python
class JobSlicer(ABC):
    @abstractmethod
    def slice_trace(self, trace_path: str, config: SlicerConfig) -> JobDAG:
        ...

class InferenceJobSlicer(JobSlicer):
    """读取 inference trace JSON → 按 batch 构建 JobDAG。
    依赖关系直接来自 trace 中 batch 的 depends_on 字段。
    """

class TrainingJobSlicer(JobSlicer):
    """读取 AICB text → 按 GA iteration 构建 JobDAG。
    每个 iteration 依赖前一个 iteration（顺序链）。
    """
```

## 6. Executor 集成

### 6.1 DynamicExecutor — 继承，不修改父类

`AnalyticalExecutor` 不做任何修改。

父类的辅助方法（`_drain_ready_pool`、`_handle_compute_done`、`_handle_flow_completion`、`_release_dependents`、`_reallocate_bandwidth` 等）全部通过参数传递状态，而非依赖实例变量。子类在 `execute_dynamic()` 中创建自己的局部状态变量，传递给继承的方法。

```python
class DynamicExecutor(AnalyticalExecutor):
    def __init__(self, topology, policy):
        super().__init__(topology, policy)
        self._job_manager: JobManager | None = None

    def execute_dynamic(self, compact_trace, job_policy, analyzer) -> ExecutionResult:
        # 1. 构建动态组件
        self._job_manager = self._build_job_manager(...)

        # 2. 局部状态（与父类 execute() 结构一致）
        task_map, dep_count, dependents, start_times, end_times = ...
        event_queue, active_flows, ready_pool = ...

        # 3. Policy 初始化
        self.policy.initialize(None, self.topology)

        # 4. 展开 root jobs，注入 tasks
        initial_tasks = self._job_manager.initialize()
        self._inject_tasks(initial_tasks, task_map, dep_count, dependents, ready_pool)

        # 5. 初次 drain
        self._drain_ready_pool(0, ready_pool, task_map, start_times, active_flows, push_event)

        # 6. 事件循环 — 复用父类全部 _handle_xxx / _release_xxx / _reallocate_xxx
        while event_queue or ready_pool or not self._job_manager.is_all_jobs_done():

            if not event_queue and not ready_pool:
                new_tasks = self._job_manager.try_expand_eligible(sim_state)
                if new_tasks:
                    self._inject_tasks(new_tasks, ...)
                    continue
                else:
                    break

            # 处理事件（与父类逻辑相同）
            for event in batch:
                completed_ids = self._process_event(event, ...)
                # ★ 动态注入触发点 ★
                if completed_ids:
                    new_tasks = self._job_manager.on_tasks_completed(completed_ids)
                    if new_tasks:
                        self._inject_tasks(new_tasks, ...)

            # 带宽重分配（继承自父类）
            self._reallocate_bandwidth(current_time, active_flows, push_event)

        return self._build_result(task_map, start_times, end_times)

    def _inject_tasks(self, new_tasks, task_map, dep_count, dependents, ready_pool):
        """将展开后的 tasks 注入 executor 状态。"""
        for task in new_tasks:
            tid = task.task_id
            task_map[tid] = task
            dep_count[tid] = len(task.deps)
            for dep in task.deps:
                dependents[dep].append(tid)
            if dep_count[tid] == 0:
                ready_pool.add(tid)

    def _process_event(self, event, ...) -> list[int]:
        """分发事件到父类处理器，返回完成的 task_id 列表。"""
        ...

    def _get_sim_state(self, ...) -> SimulationState:
        """从当前 executor 状态构建只读快照。"""
        ...
```

### 6.2 子类新增代码量

| 新增方法 | 行数 | 说明 |
|---------|------|------|
| `execute_dynamic()` | ~80 行 | 动态模式入口，事件循环主体 |
| `_inject_tasks()` | ~10 行 | 将新 tasks 注入状态 |
| `_process_event()` | ~10 行 | 分发事件到父类处理器 |
| `_get_sim_state()` | ~10 行 | 构建状态快照 |

**父类 AnalyticalExecutor 改动量**: 0 行。

### 6.3 内存回收机制

当一个 Job 的 terminal tasks 全部完成时：

```
1. 持久化 timing 数据 → 追加写入 task_timing.csv
2. 从 task_map 删除 task details
3. 从 dep_count / dependents 中清理对应条目
4. 保留 terminal_task_ids（仅用于完成判断）
5. 释放 active_job_tasks[job_id] 引用
```

顺序很重要：先写 timing 再删 details。

### 6.4 Timing 数据持久化

```python
class TimingWriter:
    """将每个 Job 的 timing 数据追加写入 CSV，不在内存中累积。"""

    def write_job_timing(self, job_id: int, task_ids: set[int], executor):
        # 在 task details 回收前调用
        # 追加写入: job_id, task_id, type, start_us, end_us, duration_us
```

### 6.5 静态分析在动态模式下的工作方式

不新增 Policy 接口方法。策略如下：

1. Per-Job 分析通过各 Analyzer 新增的 `analyze_job(tasks)` 方法运行（同样的分析 passes，作用于 task 子集）
2. 分析结果**不存储在 ExpandedJob 中**（不同策略结果结构不同）
3. Policy 通过现有的 `on_task_emitted` 回调感知新 tasks，在回调中自行更新内部状态
4. BFS 路由仅依赖拓扑，可预计算所有 `(src, dst)` 路径

## 7. 组件交互序列

```
DynamicExecutor                     JobManager
   │                                  │
   │── initialize() ────────────────> │
   │                                  │── job_dag.get_eligible_jobs() → roots
   │                                  │── job_policy.order_expansion(roots)
   │                                  │── job_policy.can_emit(job) ✓
   │                                  │── job_policy.get_placement(job)
   │                                  │── job_expander.expand_job(job)
   │                                  │── analyzer.analyze_job(tasks)
   │<── initial_tasks ────────────── │
   │                                  │
   │  ... DES event loop ...          │
   │                                  │
   │── on_tasks_completed(ids) ─────>│
   │                                  │── 检查 terminal_task_ids 完成情况
   │                                  │── 回收已完成 Job
   │                                  │   ├── 写 timing CSV
   │                                  │   └── 释放 task details
   │                                  │── job_dag.mark_completed(job_id)
   │                                  │── job_dag.get_eligible_jobs()
   │                                  │── job_policy.order_expansion(eligible)
   │                                  │── 逐个: can_emit → get_placement → expand → analyze
   │<── new_tasks (或 None) ─────────│
```

## 8. 双模式切换

```python
def run_simulation(config):
    topology = load_topology(config.topology_file)

    if config.mode == "static":
        # 现有路径 — 完全不变
        workload = load_workload(config.workload_file)
        analysis = analyzer.analyze(workload)
        policy = create_policy(config.policy_name, analysis=analysis)
        executor = AnalyticalExecutor(topology=topology, policy=policy)
        result = executor.execute(workload)

    elif config.mode == "dynamic":
        # 新的动态路径
        job_dag = create_slicer(config.trace_type).slice_trace(
            config.trace_file, config.slicer_config
        )
        job_policy = create_job_policy(config.job_policy_name)
        analyzer = create_analyzer(config.policy_name, topology)
        policy = create_policy(config.policy_name)

        executor = DynamicExecutor(topology=topology, policy=policy)
        result = executor.execute_dynamic(config, job_dag, job_policy, analyzer)
```

配置示例（YAML 或 CLI）:
```yaml
mode: dynamic                    # "static" | "dynamic"
trace_file: inputs/traces/inference_trace_stage1_pp1_req16.json
job_granularity: batch           # "iteration" | "batch" | "wave" | "custom"
job_policy: fifo                 # "fifo" | "least_load" | 自定义类名
wave_size: 4                     # 仅 granularity="wave" 时生效
```

## 9. 现有代码改动清单

### 需要改动

| 文件 | 改动 | 说明 |
|------|------|------|
| `inference_trace_expander.py` | 提取 `expand_single_batch()` | 从 `expand()` 循环体中提取单 batch 展开逻辑为独立方法 |
| `inference_trace_expander.py` | 添加 `task_id_offset` 参数 | 允许外部指定 task_id 起始偏移 |

### 不需要改动

| 文件 | 说明 |
|------|------|
| `analytical.py` | 父类完全不变 |
| `base_policy.py` | 接口不变 |
| `schema.py` | 复用现有 Task, Job, ParallelismConfig |
| `workload_builder.py` | 已天然支持单 Job，无需修改 |
| `collective_expander.py` | 无状态算法，直接复用 |
| `rank_grouper.py` | 无状态，直接复用 |

### 新增文件

| 文件 | 内容 |
|------|------|
| `src/executor/dynamic_executor.py` | `DynamicExecutor` 类 |
| `src/executor/job_manager.py` | `JobManager`、`ExpandedJob`、`SimulationState` |
| `src/executor/job_policy.py` | `JobPolicy` ABC + `FifoJobPolicy` 默认实现 |
| `src/executor/job_expander.py` | `JobExpander`、`TaskIdAllocator`、`ExpansionContext` |
| `src/executor/timing_writer.py` | `TimingWriter` 流式 CSV 输出 |
| `src/workload_generator/job_slicer.py` | `JobSlicer`、`InferenceJobSlicer`、`TrainingJobSlicer` |
| `src/workload_format/job_dag.py` | `JobDAG`、`JobSpec`、`JobType`、`SlicerConfig` |
| `scripts/run_dynamic_e2e.py` | 动态模式入口脚本 |

## 10. 内存占用对比

| 场景 | 静态模式 | 动态模式 |
|------|---------|---------|
| 16 req inference (512 batches) | ~1.6 GB（全部 436K tasks） | ~3 MB（1 batch × ~850 tasks） |
| 1000 iter training (64 GPUs) | 数十 GB | 几 MB / 活跃 iteration |
| 任意时刻内存中 tasks | 全部 tasks | 仅活跃 Jobs 的 tasks |

内存上界：`O(活跃 Jobs 数 × 单 Job task 数)` 而非 `O(总 task 数)`。

## 11. 待讨论 / 后续工作

1. **跨 Job 流依赖**: 当前 Jobs 在 task 层面完全隔离。如果后续调度策略需要细粒度流水线（Job B 的早期 layer 在 Job A 完全完成前启动），架构可通过在 `JobExpander` 中增加 `predecessor_terminals` 接线来支持。

2. **多 Replica 推理**: 多个 replica 服务不同 request 时，不同 replica 的 Jobs 相互独立可并行。JobDAG 通过 `depends_on` 自然表达。

3. **Cassini time-shifts 在动态模式下**: CassiniAnalyzer 基于全局通信模式计算 per-job time-shifts。动态模式下未来 Job 的 time-shifts 无法预计算。选项：(a) 仅对当前 Job 计算，(b) 用滑动窗口估算，(c) 退化为零 time-shift。需要实验验证。

4. **JobPolicy 内置策略**: 初始实现至少包含 `FifoJobPolicy`（按 trace 顺序展开）和 `ContentionAwareJobPolicy`（网络拥塞时自动延缓）。

5. **Checkpoint/Resume**: timing 数据流式写入磁盘，动态模式天然支持模拟断点续跑 — JobDAG 状态 + executor 状态可序列化保存。

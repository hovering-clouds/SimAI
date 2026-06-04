# Phase 7 实施计划: Workload 动态展开重构

> 日期: 2026-06-04
> 说明: 维持全量静态展开不变，新增一套动态展开路径。共拆分为 5 个 Task。

---

## 任务依赖关系

```
Task 1 (基础层) ──────────┐
  ├─ CompactWorkload      │
  │   JobExpansionInfo    │
  │   JobDAG / ExpandedJob│
  │   SimulationState     │
  │   TaskIdAllocator     │
  └─ JobPolicy 接口       │
                          ▼
Task 2 (JobSlicer) ───────┤
  ├─ Inference trace      │
  └─ Training trace       │
                          ▼
Task 3 (展开器) ───────────┤
  ├─ expand_single_batch  │
  └─ JobExpander          │
                          ▼
Task 4 (执行器) ───────────┤
  ├─ JobManager           │
  └─ DynamicExecutor      │
                          ▼
Task 5 (入口 + 验证) ──────┤
  ├─ run_dynamic_e2e.py   │
  └─ 正确性与性能验证     │
```

---

## Task 1 — 基础层（数据模型 + 接口 + 工具）

**目标**: 创建动态展开所需的所有数据结构和基础接口。

**新增文件**:

| 文件 | 内容 |
|------|------|
| `src/workload_format/compact_workload.py` | CompactWorkload, JobExpansionInfo, JobDAG, ExpandedJob, SimulationState, SlicerConfig, TaskIdAllocator |
| `src/executor/job_policy.py` | JobPolicy ABC + FifoJobPolicy 默认实现 |

**关键设计**:

```python
# ── CompactWorkload（取代 P2PWorkload 用于动态模式）──

@dataclass
class JobExpansionInfo:
    """单个 Job 的展开信息（不含现有 Job 类已有字段）。
    
    CompactWorkload 中按 job_id 索引，与现有 Job 类配合使用。
    """
    depends_on: list[int] = field(default_factory=list)
    job_type: str = "inference"              # "inference" | "training"
    trace_src: str = ""                      # trace 源文件路径
    trace_job_index: int = 0                 # 在 trace 中的序号（batch_idx / iter_idx）


@dataclass
class CompactWorkload:
    """紧凑 Workload — 镜像 P2PWorkload 结构，用 Job DAG 替代全量 tasks。
    
    对比 P2PWorkload:
      version, meta, network, jobs: list[Job]   ← 完全相同
      tasks: list[Task]                          ← 删掉
      job_expansion_info                         ← 新增（展开元数据 + 依赖关系）
    """
    version: str
    meta: Meta
    network: Network | None = None
    jobs: list[Job] = field(default_factory=list)

    # 按 job_id → JobExpansionInfo（替代 P2PWorkload 的全量 tasks）
    job_expansion_info: dict[int, JobExpansionInfo] = field(default_factory=dict)

    # ── JSON 序列化（实现直接，不在此展开）──
    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "CompactWorkload": ...
    def to_json(self, path: str): ...
    @classmethod
    def from_json(cls, path: str) -> "CompactWorkload": ...

    def to_json(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CompactWorkload":
        import json
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ── JobDAG：运行时工具类（不入序列化格式）──

class JobDAG:
    """从 CompactWorkload 构建的 DAG 索引。"""
    @classmethod
    def from_compact(cls, wl: CompactWorkload) -> "JobDAG": ...

    def mark_completed(self, job_id) -> None:
        """将依赖此 Job 的 dep_count 减 1。"""
    def get_eligible_jobs(self) -> list[int]:
        """返回所有 dep_count == 0 的 job_ids。"""


@dataclass
class ExpandedJob:
    """单个 Job 展开结果 — JobManager 提取后即可丢弃。"""
    job_id: int
    tasks: list[Task]
    entry_task_ids: list[int]
    terminal_task_ids: list[int]


@dataclass
class SimulationState:
    """传递给 JobPolicy 的状态快照。"""
    current_time_us: int
    completed_job_ids: set[int] = field(default_factory=set)


class TaskIdAllocator:
    def allocate(self, count) -> tuple[int, int]: ...


# ── JobPolicy 接口（面向 Job，不关心 CompactJob 等子类）──

class JobPolicy(ABC):
    @abstractmethod
    def can_emit(self, job: Job, sim_state: SimulationState) -> bool: ...
    @abstractmethod
    def get_placement(self, job: Job, sim_state: SimulationState) -> list[int]: ...
    @abstractmethod
    def get_delay_us(self, job: Job, sim_state: SimulationState) -> int: ...
    @abstractmethod
    def order_expansion(self, eligible_jobs: list[Job],
                        sim_state: SimulationState) -> list[Job]: ...

class FifoJobPolicy(JobPolicy):
    def can_emit(self, job, sim_state): return True
    def get_placement(self, job, sim_state): return job.assigned_nodes
    def get_delay_us(self, job, sim_state): return 0
    def order_expansion(self, eligible, sim_state): return sorted(eligible, key=lambda j: j.job_id)
```

**验收标准**:
- JobDAG 的 mark_completed/get_eligible_jobs 逻辑正确
- TaskIdAllocator 分配 ID 连续不重复

---

## Task 2 — JobSlicer（从原始 trace 构建 JobDAG）

**目标**: 将原始 trace 文件（inference JSON / AICB text）直接转换为 JobDAG，无需展开 task。

**关键洞察**: 原始 trace 本身就是紧凑格式——不需要新的文件格式。

**新增文件**: `src/workload_generator/job_slicer.py`

**关键设计**:

```python
class JobSlicer(ABC):
    @abstractmethod
    def slice_trace(self, trace_path: str, config: SlicerConfig) -> CompactWorkload: ...

class InferenceJobSlicer(JobSlicer):
    # 读 inference trace JSON
    # 每个 batch → 一个 Job（复用现有类） + 一个 JobExpansionInfo
    # batch.depends_on → JobExpansionInfo.depends_on
    # batch 数据存入 JobExpansionInfo.batch

class TrainingJobSlicer(JobSlicer):
    # 读 AICB text
    # 每个 GA iteration → 一个 Job + 一个 JobExpansionInfo
    # 依赖关系为顺序链 iter_N → iter_{N+1}
```

**关键点**: 
- 不对 trace 做任何展开，只提取 Job 级别的元数据和依赖关系
- `granularity: "wave"` 时合并多个连续 batch/iteration 为一个 Job
- 生成的 JobDAG 总大小 ≈ trace 文件大小（KB 级别），不是展开后的 GB 级别

**验收标准**:
- `inference_trace_stage1_pp1_req16.json` → 512 个 Job，依赖关系与 trace 一致
- AICB text → 正确的迭代数 + 线性依赖链

---

## Task 3 — 展开器重构（InferenceTraceExpander + JobExpander）

**目标**: 从 `InferenceTraceExpander.expand()` 中提取单 batch 展开方法，并创建 `JobExpander` 将单个 Job 展开为 ExpandedJob。

**改动文件 + 新增文件**:

| 操作 | 文件 | 说明 |
|------|------|------|
| 修改 | `src/workload_generator/inference_trace_expander.py` | 提取 `expand_single_batch()` 方法 |
| 新增 | `src/executor/job_expander.py` | JobExpander |

**InferenceTraceExpander 重构**:

`expand()` 的循环体（lines 144-288）中，跨 batch 的任务（PP inter-stage 通信、KV transfer、同阶段依赖）**是在处理依赖方 batch 时一起生成的**。例如 decode batch_B 展开时，内部调用 `_expand_kv_transfer()` 生成 KV transfer flows，这些 KV flows 和 batch_B 本身的 compute/flows 一起返回。

全量模式下：`expand()` 循环累积 `batch_exits` 字典，当前 batch 的 `prev_exits` = 前驱 batch 的 exit tasks。
动态模式下：`prev_exits` 由 **JobExpander 提供**（从已完成 Job 的 `terminal_task_ids` 构建）。

提取后的结构：

```python
class InferenceTraceExpander:
    # 保留原有 expand() — 行为不变，内部改为循环调用 expand_single_batch()
    def expand(self, trace, job_id=0, storage_node_ids=None): ...

    # 新增 — 只展开一个 batch，prev_exits 由调用者从已完成 Job 的 terminal_task_ids 构建
    def expand_single_batch(
        self,
        batch: dict,
        batch_lookup: dict,           # 查前驱 batch 元数据（PP stage、KV size 等）
        prev_exits: dict[int, list[int]],  # dep batch_id → [task_ids]，替代 batch_exits
        job_id: int,
        task_id_start: int,
        total_layers: int,
        profiles,
        ...
    ) -> tuple[list[FlowTask], dict, BatchTaskInfo, int]:
        # 逻辑与 expand() 循环体完全一致
        # 内部能正确处理 KV transfer / PP comm — 这些依赖 prev_exits 而非 batch_exits
        ...
```

**JobExpander**:

```python
class JobExpander:
    def __init__(self, task_id_allocator, profile_store, topology):
        ...
        self._trace_cache: dict[str, dict] = {}  # trace_src → parsed trace
    
    def _load_trace(self, trace_src: str) -> dict:
        """按需加载 trace 文件并缓存。"""
        if trace_src not in self._trace_cache:
            with open(trace_src) as f:
                self._trace_cache[trace_src] = json.load(f)
        return self._trace_cache[trace_src]
    
    def expand_job(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        if info.job_type == "training":
            return self._expand_training(job, info)
        else:
            return self._expand_inference(job, info)
    
    def _expand_inference(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        trace = self._load_trace(info.trace_src)
        batch = trace["batches"][info.trace_job_index]
        # prev_exits 由 caller（JobManager）从已完成 Job 的 terminal_task_ids 构建
        ...
    
    def _expand_training(self, job: Job, info: JobExpansionInfo) -> ExpandedJob:
        trace = self._load_trace(info.trace_src)
        # trace 中按 trace_job_index 取出对应 iteration 的 aicb_items
        ...
```

**验收标准**:
- 重构后 `expand()` 的输出与重构前完全一致
- `expand_single_batch()` 可以独立调用并返回正确结果
- JobExpander 展开单个 inference batch 返回 ExpandedJob 包含正确数量的 tasks

---

## Task 4 — 执行器（JobManager + DynamicExecutor）

**目标**: 创建 JobManager 管理 Job 生命周期 + DynamicExecutor 继承 AnalyticalExecutor 实现动态模式。

**新增文件**:

| 文件 | 内容 |
|------|------|
| `src/executor/job_manager.py` | JobManager（展开、回收、持久化） |
| `src/executor/dynamic_executor.py` | DynamicExecutor（继承 AnalyticalExecutor，父类零修改） |

**JobManager**:

```python
class JobManager:
    def __init__(self, job_dag, job_policy, job_expander, analyzer, topology): ...

    def initialize(self) -> list[Task]:
        """展开 root jobs，返回初始 tasks。"""

    def on_tasks_completed(self, completed_task_ids: set[int]) -> list[Task] | None:
        """检查完成 → 回收 → 展开新 eligible jobs → 返回新 tasks。"""

    def try_expand_eligible(self, sim_state) -> list[Task] | None:
        """事件队列为空时尝试展开。"""

    def is_all_jobs_done(self) -> bool: ...
```

**DynamicExecutor**:

```python
class DynamicExecutor(AnalyticalExecutor):
    """继承 AnalyticalExecutor，复用事件处理辅助方法。"""

    def execute_dynamic(self, job_dag, job_policy, analyzer) -> ExecutionResult:
        # 1. 创建 JobManager
        # 2. 初始化 executor 状态（task_map, dep_count, dependents, etc.）
        # 3. Policy.initialize(None, topology)
        # 4. 展开 root jobs → 注入 → drain
        # 5. 事件循环（使用父类的 drain_ready_pool / handle_xxx / reallocate_bandwidth）
        #    + 在事件处理后通过 on_tasks_completed 触发动态注入
        # 6. 返回结果
    
    def _inject_tasks(self, new_tasks, task_map, dep_count, dependents, ready_pool):
        """将展开后的 tasks 注入 executor 状态。"""
```

**关键设计点**:
- 父类 `AnalyticalExecutor` 不做任何修改
- 所有父类辅助方法（`_drain_ready_pool`, `_handle_compute_done` 等）通过参数传递状态，子类创建自己的局部变量直接调用
- 事件循环条件增加 `not self._job_manager.is_all_jobs_done()` 
- Job 完成时回收 task details（释放 task_map / dep_count / dependents 条目）
- Timing 不单独逐 Job 写入——最终统一通过 `ExecutionResult`（复用现有 `result.py`）输出

**验收标准**:
- 在小 trace 上执行后得到有效的 ExecutionResult（makespan > 0）
- 所有 Job 完成后正确退出
- 内存中不同时持有全部 tasks（验证：大 trace 峰值内存远小于全量展开）

---

## Task 5 — 入口脚本 + 验证

**目标**: 创建端到端入口脚本，验证动态模式正确性。

**新增文件**:

| 文件 | 内容 |
|------|------|
| `scripts/run_dynamic_e2e.py` | 动态模式入口脚本 |

**入口脚本**:

```python
# 步骤：
# 1. 加载配置（trace 路径、拓扑、profile_store、slicer_config）
# 2. JobSlicer 构建 JobDAG
# 3. 加载 topology + 创建 analyzer
# 4. 创建 DynamicExecutor
# 5. 执行 execute_dynamic()
# 6. 输出 timing CSV + QoS 报告
```

**验证内容**:

1. **功能正确性验证**:
   - 对同一个小 trace（如 1 batch），静态模式与动态模式 makespan 一致
   - JobDAG 依赖解析正确，不遗漏任何 batch
   - ExecutionResult 包含所有 task 的时间

2. **内存对比验证**:
   - 静态模式: 16 requests trace → 测量内存峰值
   - 动态模式: 同 trace → 测量内存峰值
   - 预期: 动态模式显著（100x+）更低

3. **压力测试**:
   - 512 batches 全流程跑通
   - 验证模拟完成后所有 Job 被标记为完成

**验收标准**:
- 迷你测试上动态与静态模式结果一致
- 大规模 trace 上动态模式内存占用远低于静态模式
- 无死锁（事件循环正确终止）

---

## 实施顺序

按 Task 序号顺序实施，每个 Task 完成且验证后再进入下一个：

```
Task 1 (基础层) ──→ Task 2 (JobSlicer) ──→ Task 3 (展开器重构) ──→ Task 4 (执行器) ──→ Task 5 (入口+验证)
```

### 整体文件变更清单

| 操作 | 文件 | 所属 Task |
|------|------|-----------|
| 新增 | `src/workload_format/compact_workload.py` | Task 1（CompactWorkload, JobExpansionInfo, JobDAG, ExpandedJob, SimulationState, TaskIdAllocator） |
| 新增 | `src/executor/job_policy.py` | Task 1（JobPolicy ABC, FifoJobPolicy） |
| 新增 | `src/workload_generator/job_slicer.py` | Task 2 |
| 修改 | `src/workload_generator/inference_trace_expander.py` | Task 3 |
| 新增 | `src/executor/job_expander.py` | Task 3 |
| 新增 | `src/executor/job_manager.py` | Task 4 |
| 新增 | `src/executor/dynamic_executor.py` | Task 4 |
| 新增 | `scripts/run_dynamic_e2e.py` | Task 5 |
| 不变 | `src/executor/analytical.py` | — |
| 不变 | `src/executor/policies/base_policy.py` | — |
| 不变 | `src/workload_format/schema.py` | — |

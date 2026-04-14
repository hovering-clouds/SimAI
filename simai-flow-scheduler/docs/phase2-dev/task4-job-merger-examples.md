# Phase 2 - Task 4 & Task 5 开发记录：多任务合并器与端到端示例

## 1. 目标与范围

Task 4 的目标是实现 `JobMerger` 类，将多个 P2P Workload 合并为一个多任务 workload，使多个训练任务可以在同一拓扑下共享网络资源并产生流量竞争。每个输入 workload 可以包含一个或多个 job，合并时自动处理 job_id 和 task_id 的冲突。Task 5 实现端到端示例（单任务和多任务），验证完整管线可运行。

**包含**：

- `JobMerger` 类实现：job_id + task_id 全局重映射、依赖引用更新、元数据合并
- `MergeResult` 数据结构：返回合并结果与映射关系
- 支持单任务和多任务 workload 灵活组合合并
- 单任务端到端示例（`examples/single_job/`）
- 多任务端到端示例（`examples/multi_job/`）
- JobMerger 单元测试（含 job_id 冲突场景）

**不包含**：

- 调度策略实现（Phase 3）
- 执行器实现（Phase 4）
- PP stage 间通信展开

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/workload_generator/
│   ├── job_merger.py            # 新增：JobMerger + MergeResult
│   └── __init__.py              # 更新：导出 JobMerger, MergeResult
├── tests/
│   └── test_job_merger.py       # 新增：27 个测试
└── examples/
    ├── single_job/
    │   ├── generate_workload.py     # 新增：单任务端到端示例脚本
    │   ├── sample_workload.txt      # 新增：示例 AICB workload 文件
    │   ├── single_job_workload.json # 生成：P2P Workload 输出
    │   └── README.md                # 新增：使用说明
    └── multi_job/
        ├── generate_multi_job.py    # 新增：多任务端到端示例脚本
        ├── job1_workload.txt        # 新增：Job 1 AICB workload
        ├── job2_workload.txt        # 新增：Job 2 AICB workload
        ├── multi_job_workload.json  # 生成：合并后 P2P Workload 输出
        └── README.md                # 新增：使用说明
```

### 2.2 核心 API

```python
@dataclass
class MergeResult:
    """合并结果。"""
    merged_workload: P2PWorkload
    task_id_mapping: dict[int, dict[int, int]]  # {workload_idx: {old_task_id: new_task_id}}
    job_mapping: dict[int, dict[int, int]]  # {workload_idx: {old_job_id: new_job_id}}

class JobMerger:
    """合并多个 P2P Workload（每个可包含一个或多个 job）。"""

    def merge(
        self,
        workloads: list[P2PWorkload],
        topology_file: str = "",
    ) -> MergeResult:
        """合并多个 workload。

        规则:
        1. job_id 全局重新编号（避免冲突）
        2. task_id 全局重新编号（避免冲突）
        3. tasks 中的 job_id 引用和 deps 引用用新 ID 替换
        4. meta.num_jobs = sum of all workloads' num_jobs
        5. meta.num_nodes = max num_nodes across workloads
        6. network topology 合并
        7. jobs 列表合并
        8. tasks 列表合并（含 job_id + task_id 重映射）
        """
        ...
```

---

## 3. 设计决策

### 3.1 Task ID 重映射策略

合并时采用**连续分配**策略：每个 workload 的 task_id 按顺序重新编号，形成一个连续的全局 task_id 空间。

```
Workload 0: task 0→0, task 1→1, ..., task N-1→N-1
Workload 1: task 0→N, task 1→N+1, ..., task M-1→N+M-1
```

**选择连续分配而非偏移量的原因**：
- 每个 workload 的内部 task_id 不一定从 0 开始（理论上可能有间隔）
- 连续分配保证合并后 task_id 紧凑、无间隔
- 映射关系清晰，便于调试和追踪

### 3.2 Job ID 重映射策略

合并时同时对 **job_id** 和 **task_id** 进行全局连续分配：

```
Job IDs:
  Workload 0: job 0→0, job 1→1 (if multi-job)
  Workload 1: next available → N
  ...

Task IDs:
  Workload 0: task 0→0, task 1→1, ...
  Workload 1: next available → M
  ...
```

**选择全局连续分配的原因**：
- 支持灵活的用户构建方式：用户可以先分别构建多个多-job workload，再合并
- 自动处理 job_id 冲突：即使两个 workload 都有 job_id=0，合并后会自动分配为 0, 1, 2...
- 保持 DAG 内部依赖正确：每个 job 内部的 task 依赖链保持不变
- 映射关系可追溯：`job_mapping[workload_idx][old_job_id]` 可追溯任何 job 的来源

### 3.3 Task 内的 job_id 引用更新

合并时需要同时更新 task 的两个引用字段：
1. `task.job_id` — 指向所属 job 的 ID
2. `task.deps` — 依赖的其他 task

```python
new_task = Task(
    job_id=job_old_to_new[task.job_id],  # 更新 job_id 引用
    task_id=task_old_to_new[task.task_id],
    deps=[task_old_to_new[dep] for dep in task.deps if dep in task_old_to_new],
    ...
)
```

### 3.3 Network 合并策略

```
1. 如果调用者提供了 topology_file 参数 → 使用它
2. 否则，使用第一个非 None 的 network
3. 如果所有 workload 都没有 network → 返回 None（实际上被 P2PWorkload.__post_init__ 转为空 Network）
```

**设计理由**：多任务场景下，通常有一个统一的拓扑文件。调用者可以通过 `topology_file` 参数指定，也可以让 merger 自动使用第一个 workload 的拓扑。

### 3.4 支持多任务 workload 输入

原设计要求每个输入 workload 只包含一个 job。新设计放开了这个约束：

- 每个输入 workload 可以包含任意数量的 job
- 所有 job_id 统一重新编号（避免冲突）
- 所有 task_id 统一重新编号
- 每个 task 的 `job_id` 字段会自动更新为重映射后的新 job_id
- 跨 workload 的 job_id 冲突会被自动解决

例如，合并以下两个 workload：
- W1: job_ids = [0, 1]（多 job）
- W2: job_ids = [0]（单 job）

结果：W1 的 jobs 映射到 [0, 1]，W2 的 job 映射到 [2]

### 3.5 MergeResult 结构

返回值不仅包含合并后的 workload，还包含映射关系：
- `task_id_mapping`：可用于外部系统将合并后的 task_id 映射回原始 workload
- `job_mapping`：追踪每个 workload 对应的 job_id

这对后续的调度分析器或结果分析工具很有用。

---

## 4. 关键实现细节

### 4.1 合并流程

```python
def merge(self, workloads, topology_file=""):
    # Step 1: Remap job_ids and task_ids (核心步骤)
    remapped_workloads, task_id_mapping, job_mapping = self._remap_ids(workloads)

    # Step 2: Merge meta (num_jobs=sum, num_nodes=max)
    merged_meta = self._merge_meta(remapped_workloads)

    # Step 3: Merge network (拓扑引用)
    merged_network = self._merge_network(remapped_workloads, topology_file)

    # Step 4: Merge jobs + tasks
    merged_jobs = self._merge_jobs(remapped_workloads)
    merged_tasks = self._merge_tasks(remapped_workloads)

    # Step 5: Build and return
    return MergeResult(merged_workload, task_id_mapping, job_mapping)
```

### 4.2 _remap_ids 实现

合并的核心逻辑，同时处理 job_id 和 task_id 的重映射：

```python
next_task_id = 0
next_job_id = 0
for idx, workload in enumerate(workloads):
    # 1. Remap job_ids
    job_old_to_new = {}
    new_jobs = []
    for job in workload.jobs:
        job_old_to_new[job.job_id] = next_job_id
        new_jobs.append(Job(job_id=next_job_id, ...))
        next_job_id += 1

    # 2. Remap task_ids and update job_id references
    task_old_to_new = {}
    new_tasks = []
    for task in workload.tasks:
        task_old_to_new[task.task_id] = next_task_id
        new_tasks.append(Task(
            task_id=task_old_to_new[task.task_id],
            job_id=job_old_to_new[task.job_id],  # 更新 job_id 引用
            deps=[task_old_to_new[dep] for dep in task.deps if dep in task_old_to_new],
            ...
        ))
        next_task_id += 1
```

---

## 5. 端到端示例

### 5.1 单任务示例

**输入**：`sample_workload.txt`（2 层，TP=2，DP=2，4 GPUs）

```
attention_column  -1 1000  ALLGATHER       134217728  2000  REDUCESCATTER  134217728  500  NONE  0  100
attention_row     -1 1500  REDUCESCATTER   134217728  2500  ALLGATHER      134217728  600  NONE  0  100
```

**运行**：
```bash
cd simai-flow-scheduler
python examples/single_job/generate_workload.py
```

**输出**：
- 40 tasks（24 compute + 16 flow）
- 4 个 rank 各自的 compute tasks
- Ring AllGather 和 ReduceScatter 的 P2P flow 展开
- Forward → Backward 依赖链

**管线步骤**：Parse → Define Job → Build P2P Workload → Validate → Write JSON

### 5.2 多任务示例

**输入**：两个独立 AICB workload 文件

| Job | 文件 | 模型 | TP | DP | GPUs | Nodes |
|-----|------|------|----|----|------|-------|
| 0 | job1_workload.txt | attention | 2 | 2 | 4 | [0,1,2,3] |
| 1 | job2_workload.txt | embedding | 4 | 1 | 4 | [4,5,6,7] |

**运行**：
```bash
cd simai-flow-scheduler
python examples/multi_job/generate_multi_job.py
```

**输出**：
- 160 tasks（40 from Job 0 + 120 from Job 1）
- task_id 重映射：Job 0 [0..39], Job 1 [40..159]
- 两个 Job 共享同一网络拓扑
- 每个 Job 内部依赖链独立，无跨 Job 依赖

**管线步骤**：Parse Job 1 → Build W1 → Parse Job 2 → Build W2 → Merge → Validate → Write JSON

### 5.3 示例中发现的问题

#### 问题：sample_workload.txt 与 header 参数不匹配

**现象**：使用 `example/microAllReduce.txt` 运行时断言失败：

```
AssertionError: Expected vpp=8, got 2
```

**原因**：`microAllReduce.txt` 的 header 声明 `vpp=8`，但实际只有 2 个 items。这是因为 micro benchmark 文件不是完整的 training workload，不满足 `num_layer_items == vpp * ga` 的约束。

**解决**：为示例创建专用的 `sample_workload.txt`，确保 header 参数（`vpp=2, ga=1`）与实际 item 数量一致。

#### 问题：assigned_nodes 与 parallelism 不匹配

**现象**：
```
AssertionError: Expected 2 nodes, got 4
```

**原因**：示例脚本中直接使用 `header.all_gpus` 作为 `assigned_nodes`，但 `ParallelismConfig` 仅设置 `tp=header.tp`，没有计算正确的 `dp` 值。`RankGrouper` 要求 `len(assigned_nodes) == tp * dp * ep * pp`。

**解决**：从 header 参数推导 DP size：
```python
dp_size = header.all_gpus // (header.tp * header.pp * header.ep)
```

---

## 6. 测试覆盖

### 6.1 JobMerger 测试

`tests/test_job_merger.py`（27 个测试）覆盖：

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestJobMergerBasic` | 4 | 单 workload 合并、双 workload 合并、空列表报错、多 job workload 成功 |
| `TestTaskIdRemapping` | 3 | 连续 ID 分配、映射关系正确性、deps 更新 |
| `TestMetadataMerging` | 2 | num_jobs 求和、num_nodes 取最大值 |
| `TestNetworkMerging` | 3 | topology_file 覆盖、自动选取首个、全 None 处理 |
| `TestJobMerging` | 2 | Job 信息保留、job_mapping 追踪 |
| `TestMergeResult` | 2 | 返回结构完整性、合并后 validate 通过 |
| `TestEdgeCases` | 3 | 空 tasks 合并、复杂依赖链无跨 Job 依赖、三方合并 |
| `TestJobIdRemapping` | 8 | 跨 workload job_id 冲突、同 workload 重复 job_id、混合多/单 job、task 中 job_id 更新、多 job 内依赖保留、合并后 validate、无跨 job 依赖、job_id 连续性 |

### 6.2 关键测试点

**无跨 Job 依赖验证**（`test_merge_workloads_with_complex_deps`）：
```python
for task in result.merged_workload.tasks:
    for dep in task.deps:
        dep_task = next(t for t in result.merged_workload.tasks if t.task_id == dep)
        assert dep_task.job_id == task.job_id, "Cross-job dependency detected!"
```

**合并后验证通过**（`test_result_is_valid_workload`）：
```python
errors = result.merged_workload.validate()
assert not errors, f"Merged workload validation failed: {errors}"
```

### 6.3 测试结果

```
tests/test_job_merger.py: 27 passed
Total: 152 passed (含 Phase 1、Task 1-3 全部测试)
```

---

## 7. 后续增强：放开多任务 workload 输入约束

### 7.1 增强背景

原设计要求每个输入 workload 只包含一个 job。但用户反馈：如果用户先构建一个包含多个 job 的 workload（例如用户手动组装），再与其他 workload 合并，原实现会报错。这限制了工作流的灵活性。

### 7.2 增强内容

- 移除单任务约束（`len(workload.jobs) != 1` 报错）
- 新增 `_remap_ids` 方法，同时处理 job_id 和 task_id 的全局重映射
- 更新 `MergeResult.job_mapping` 结构：从 `{workload_idx: job_id}` 改为 `{workload_idx: {old_job_id: new_job_id}}`
- 自动更新 task 中的 `job_id` 引用

### 7.3 增强后的测试

新增 8 个测试覆盖 job_id 重映射场景：

- `test_conflicting_job_ids_across_workloads`: 跨 workload job_id 冲突
- `test_conflicting_job_ids_in_same_workload`: 同一 workload 重复 job_id
- `test_multi_job_workload_plus_single_job_workload`: 混合合并
- `test_task_job_ids_updated_after_remap`: task 中 job_id 更新
- `test_multi_job_workload_preserves_per_job_deps`: 多 job 内依赖保留
- `test_multi_job_workload_validates`: 合并后 validate 通过
- `test_two_multi_job_workloads_no_cross_deps`: 无跨 job 依赖
- `test_job_id_contiguous_across_all_workloads`: job_id 连续分配

---

## 7. Phase 2 完整状态

### 7.1 测试汇总

| 模块 | 测试文件 | 测试数 |
|------|---------|--------|
| Workload Format | test_workload_format.py | 11 |
| Collective Expander | test_collective_expander.py | 46 |
| AICB Parser | test_aicb_parser.py | 30 |
| Rank Grouper | test_rank_grouper.py | 17 |
| Workload Builder | test_workload_builder.py | 21 |
| **Job Merger** | **test_job_merger.py** | **27** |
| **Total** | | **152** |

### 7.2 文件清单

```
simai-flow-scheduler/
├── src/
│   ├── workload_format/
│   │   ├── schema.py              # P2PWorkload 数据模型
│   │   ├── validator.py           # 格式验证
│   │   └── writer.py              # JSON 读写
│   └── workload_generator/
│       ├── aicb_parser.py         # Task 1: AICB 解析
│       ├── rank_grouper.py        # Task 2: Rank 分组
│       ├── collective_expander.py # Phase 1: Collective→P2P 展开
│       ├── workload_builder.py    # Task 3: Workload 构建
│       └── job_merger.py          # Task 4: 多任务合并（支持多 job 输入）
├── tests:                         # 152 tests
├── examples/
│   ├── single_job/                # Task 5a: 单任务端到端示例
│   └── multi_job/                 # Task 5b: 多任务端到端示例
└── docs/
    ├── specs/                     # 架构与格式文档
    └── phase2-dev/                # 开发记录
```

---

*开发时间：2026-04-13*
*测试状态：144 passed*
*Phase 2 全部完成，可进入 Phase 3（调度分析器）*

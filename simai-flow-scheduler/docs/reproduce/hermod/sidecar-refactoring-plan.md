# Hermod Sidecar 重构计划

> 日期：2026-07-17  
> 目标：将 Hermod 专属字段从泛型 `Task` schema 中剥离，改为 sidecar 模式  
> 参照：MFS 的 `MfsContext.task_info: dict[int, MfsTaskInfo]`、Puppeteer 的 `dict[int, TTEInfo]`  
> 状态：**计划阶段，待实施**

---

## 一、背景：为什么要重构

### 问题现状

Hermod 是当前唯一将策略专属字段直接写入泛型 `Task` 数据类的策略：

| 字段 | 声明位置 | 使用范围 | 问题 |
|------|---------|---------|------|
| `coflow_id: Optional[str]` | `schema.py:166` | 仅 Hermod | 污染泛型 IR |
| `microbatch_id: Optional[int]` | `schema.py:167` | 仅 Hermod | 污染泛型 IR |
| `logical_layer_id: Optional[int]` | `schema.py:168` | 仅 Hermod | 污染泛型 IR |
| `hermod_lid_source_operation` | 动态 `setattr`, `hermod_metadata.py:155` | 仅 Hermod 审计 | 运行时 hack |
| `hermod_lid_mapping_rule` | 动态 `setattr`, `hermod_metadata.py:156` | 仅 Hermod 审计 | 运行时 hack |

### 已有策略的对照

其他所有策略都使用 sidecar 模式，不碰 `Task`：

| 策略 | 侧边数据结构 | 存放位置 | 访问方式 |
|------|-------------|---------|---------|
| **MFS** | `MfsContext.task_info: dict[int, MfsTaskInfo]` | `MfsAnalysisResult.mfs_context` | `self.context.task_info.get(task_id)` |
| **Puppeteer** | `tte_info: dict[int, TTEInfo]` | `PuppeteerAnalysisResult.tte_info` | `self.tte_info[task_id]` |
| **Cassini** | `time_shifts: dict[int, int]` 等 | `CassiniAnalysisResult.time_shifts` | `self.time_shifts[job_id]` |

### 文件位置不当

Hermod 的两个文件目前放在通用机制的 `workload_generator/` 目录下，与 `aicb_parser.py`、`workload_builder.py` 等混在一起。应将其移至 `static_analysis/passes/`，与其他策略专属 pass 并列。

| 文件 | 当前位置 | 目标位置 |
|------|---------|---------|
| `hermod_aicb_metadata.py` | `workload_generator/` | `static_analysis/passes/hermod_metadata.py` |
| `hermod_placement.py` | `workload_generator/` | `static_analysis/passes/hermod_placement.py` |

最终 Hermod 的文件布局：

```
static_analysis/passes/
├── hermod_metadata.py       ← 元数据适配（从 workload_generator 迁入）
├── hermod_placement.py      ← GPU 放置（从 workload_generator 迁入）
├── hermod_priority.py       ← 优先级分析（已有）

static_analysis/strategies/
├── hermod_strategy.py       ← 策略编排（已有）

executor/policies/
├── hermod_policy.py         ← 调度策略（已有）

executor/bandwidth_allocators/
├── hermod_allocator.py      ← 带宽分配（已有）

executor/
├── hermod_training_expander.py  ← 动态展开（已有）
```

与 MFS 完全对称：

```
passes/mfs_context.py        vs  passes/hermod_metadata.py
passes/hermod_priority.py    vs  passes/mfs_feasibility.py  (各有专属 pass)
strategies/mfs_strategy.py   vs  strategies/hermod_strategy.py
policies/mfs_policy.py       vs  policies/hermod_policy.py
```

---

## 二、当前数据流（重构前）

```
workload_builder.py
  │  flow.coflow_id = "collective:j0:...:g0"    ← 含 subgroup_index
  ▼
collective_expander.py  FlowTask.to_task()
  │  将 coflow_id 传到 Task 对象
  ▼
hermod_metadata.py  HermodAicbMetadataAdapter.apply(workload)
  │  task.coflow_id         ← 读取已有字段
  │  task.microbatch_id = mid                   ← 🔴 写 Task schema
  │  task.logical_layer_id = lid                ← 🔴 写 Task schema
  │  task.hermod_lid_source_operation = ...     ← 🔴 动态 setattr
  │  task.hermod_lid_mapping_rule = ...         ← 🔴 动态 setattr
  │  return list[HermodMetadataRecord]          ← 已有 sidecar 雏形
  ▼
hermod_priority.py  HermodPriorityAnalysis.from_workload(workload)
  │  task.coflow_id / task.microbatch_id / task.logical_layer_id  ← 🔴 从 Task 读取
  ▼
schema.py / writer.py / validator.py / collective_expander.py
  │  Task 上 3 个 Optional 字段经序列化/反序列化传播  ← 🔴 IR 污染链
  ▼
dynamic_executor.py  _analyze_and_inject()
  │  getattr(t, "hermod_lid_*", None)  → 写入 _task_meta  ← 🔴 hack
```

---

## 三、目标数据流（重构后）

```
workload_builder.py  ← coflow_id 全部删除，不再生成 Hermod 标记
  │
  ▼
hermod_metadata.py  HermodAicbMetadataAdapter.apply(workload)
  │  从 Task 现有字段合成 coflow_id：
  │    coflow_id = f"j{jid}:i{iter}:{phase}:{ctype}"
  │  计算 microbatch_id / logical_layer_id
  │
  ├─ 构建 hermod_records: dict[int, HermodMetadataRecord]
  │     {task_id: HermodMetadataRecord(
  │         coflow_id, microbatch_id, logical_layer_id,
  │         coflow_type, provenance, source_operation, mapping_rule)}
  │
  │（不再在 Task 上设任何 Hermod 属性）
  │
  └─ return hermod_records                   ← 普通 dict
  ▼
hermod_priority.py  HermodPriorityAnalysis.from_workload(workload, hermod_records)
  │  hermod_records[task_id].coflow_id          ← ✅ 从 dict 读取
  │  hermod_records[task_id].microbatch_id
  │  hermod_records[task_id].logical_layer_id
  ▼
hermod_strategy.py  HermodAnalysisResult
  │  ├─ priority_analysis      ← 从 hermod_records 构建
  │  └─ hermod_records         ← dict[int, HermodMetadataRecord]
  ▼
hermod_policy.py  /  hermod_allocator.py
  │  → 不变（内部结构不受影响）
  ▼
dynamic_executor.py  _analyze_and_inject()
  │  _task_meta 只保留通用 7 个字段             ← ✅ 恢复原状
  │  Hermod 可视化数据从 hermod_records 另写文件
  ▼
【清理】schema.py / writer.py / validator.py / collective_expander.py
  │  删除 coflow_id / microbatch_id / logical_layer_id  ← ✅ 恢复到 merge 前
```

---

## 四、关键设计决策

### `subgroup_index` 不需要保留

`coflow_id` 在整个 Hermod 策略中作为**不透明字符串键**使用——没有任何代码对它做 `split()`、正则提取或子串解析。它仅在 `hermod_priority.py` 中按 `grouped.setdefault(task.coflow_id, [])` 做分组键，在 `hermod_allocator.py` 中按 `task_to_coflow.get(task_id)` 查表。

`subgroup_index`（`coflow_id` 中的 `:g{N}` 后缀）标明的是 workload builder 展开时的并行子组序号。同一 collective 调用的不同 subgroup 的 flows 在 Hermod 中有相同的 priority key `(job_id, MID, ctype_rank, LID)`，会被分到同一个优先级 tier 中，调度行为完全不受 subgroup 划分的影响。

因此 adapter 直接从 Task 现有字段合成 coflow_id，**不需要 subgroup_index，也不需要图分析**：

```python
def _compute_coflow_id(self, task: Task) -> str | None:
    ct = classify_coflow_type(task.comm_type)
    if ct is None:
        return None  # 非 Hermod 流量
    return f"j{task.job_id}:i{task.iteration}:{task.phase.value}:{ct.value}"
```

`coflow_id` 从此不再是提前存储在 Task 上的字段，而是 adapter 的一次性推导产物。

---

## 五、具体实施计划（7 个 Phase）

### Phase 1：移动文件 + 提取 `HermodMetadataRecord`

**操作：**

```
移动前                                          移动后
workload_generator/                             static_analysis/passes/
├── hermod_aicb_metadata.py    ──→              ├── hermod_metadata.py
│   ├── HermodMetadataRecord                    │   ├── HermodMetadataRecord        ← 保留
│   ├── HermodAicbMetadataAdapter               │   ├── HermodAicbMetadataAdapter   ← 保留
│   └── write_sidecar()                         │   └── write_sidecar()             ← 保留
│                                               │
├── hermod_placement.py        ──→              ├── hermod_placement.py
│   └── assigned_nodes_for()                    │   └── assigned_nodes_for()        ← 保留
│                                               │
│  (删除原文件)                                  │
```

**关键决策：**
- **不需要容器类。** `HermodMetadataRecord` 已经是 `frozen dataclass`，各策略类直接用 `dict[int, HermodMetadataRecord]` 传递和查询。与 Puppeteer 的 `dict[int, TTEInfo]` 模式一致。
- `write_sidecar()` 保留为独立方法。

**涉及文件：** 2 个移动 + 删除 2 个原位置文件
**依赖更新：** 所有 import `workload_generator.hermod_aicb_metadata` 或 `workload_generator.hermod_placement` 的地方改为 `static_analysis.passes.hermod_metadata` / `hermod_placement`。
**风险：** 低（纯移动 + 更新 import）

---

### Phase 2：修改 `hermod_metadata.py` 中的 `apply()`

**核心改动：`coflow_id` 由 adapter 自行合成，不再从 Task 读取。**

```python
class HermodAicbMetadataAdapter:
    @staticmethod
    def _compute_coflow_id(task: Task) -> str | None:
        ct = classify_coflow_type(task.comm_type)
        if ct is None:
            return None
        return f"j{task.job_id}:i{task.iteration}:{task.phase.value}:{ct.value}"

    def apply(self, workload) -> dict[int, HermodMetadataRecord]:
        records: dict[int, HermodMetadataRecord] = {}
        for task in workload.get_flow_tasks():
            coflow_id = self._compute_coflow_id(task)
            if coflow_id is None:
                continue
            # ... 计算 mid, lid ...
            records[task.task_id] = HermodMetadataRecord(
                task_id=task.task_id,
                coflow_id=coflow_id,
                microbatch_id=mid,
                logical_layer_id=lid,
                ...)
        return records
```

| 当前行为 | 重构后行为 |
|---------|-----------|
| `task.coflow_id` 从 workload_builder 读取 | **adapter 自行合成**，不依赖 workload_builder |
| `task.microbatch_id = mid` | **不设** |
| `task.logical_layer_id = lid` | **不设** |
| `task.hermod_lid_* = ...` | **不设** |
| 返回 `list[HermodMetadataRecord]` | 返回 `dict[int, HermodMetadataRecord]` |

**涉及文件：** 1 个
**风险：** 低

---

### Phase 3：修改 `hermod_priority.py`

```python
class HermodPriorityAnalysis:
    @classmethod
    def from_workload(
        cls,
        workload, variant, ep_mode,
        hermod_records: dict[int, HermodMetadataRecord] | None = None,
    ) -> "HermodPriorityAnalysis":
        for task in workload.get_flow_tasks():
            if hermod_records is not None:
                record = hermod_records.get(task.task_id)
                if record is None:
                    continue
                coflow_id = record.coflow_id
                microbatch_id = record.microbatch_id
                logical_layer_id = record.logical_layer_id
            else:
                # 回退路径（兼容旧 JSON 文件）
                coflow_id = task.coflow_id
                microbatch_id = task.microbatch_id
                logical_layer_id = task.logical_layer_id
            # ... 后续逻辑不变 ...
```

**涉及文件：** 1 个
**风险：** 低

---

### Phase 4：修改 `hermod_strategy.py`

```python
@dataclass
class HermodAnalysisResult:
    route_table: RouteTable
    execution_plan: ExecutionPlan
    priority_analysis: HermodPriorityAnalysis
    hermod_records: dict[int, HermodMetadataRecord] | None = None  # ← 新增


class HermodAnalyzer:
    def analyze(self, workload,
                hermod_records: dict[int, HermodMetadataRecord] | None = None):
        return HermodAnalysisResult(
            ...,
            priority_analysis=HermodPriorityAnalysis.from_workload(
                workload, ..., hermod_records=hermod_records),
            hermod_records=hermod_records,
        )
```

**涉及文件：** 1 个
**风险：** 低

---

### Phase 5：清理 `dynamic_executor.py`

```python
# 重构前 — 12 个 key
self._task_meta[t.task_id] = {
    k: getattr(t, k, None)
    for k in ("phase", "layer_id", "comm_type", "src", "dst", "node", "job_id",
              "coflow_id", "microbatch_id", "logical_layer_id",
              "hermod_lid_source_operation", "hermod_lid_mapping_rule")
}

# 重构后 — 恢复为 7 个 key
self._task_meta[t.task_id] = {
    k: getattr(t, k, None)
    for k in ("phase", "layer_id", "comm_type", "src", "dst", "node", "job_id")
}
```

**涉及文件：** 1 个
**风险：** 低

---

### Phase 6：清理污染链（6 个文件）

#### 6a. `workload_builder.py`

删除 3 处 `coflow_id` 赋值（collective 展开 + PP boundary flow）。

#### 6b. `collective_expander.py`

从 `FlowTask` 数据类中删除 `coflow_id` 字段，从 `to_task()` 中删除对应传参。

#### 6c. `schema.py`

删除 3 个字段定义、JSON schema 定义、docstring。

#### 6d. `validator.py`

删除 3 行读取。

#### 6e. `writer.py`

删除条件写入块和 `WorkloadReader` 中的对应读取。

**注意：** `item_id` 的序列化（`writer.py:124`）是独立修复，**保留**。

#### 6f. `job_merger.py`

删除 `coflow_id` 的透传和 remapping。

---

### Phase 7：修改 E2E 脚本

更新各脚本中的 import 路径，改为从 `static_analysis.passes.hermod_metadata` 导入；输出 `hermod_metadata.json` 时从 `hermod_records`（而非 `_task_meta`）读取。

**涉及文件：** `run_hermod_e2e.py`, `hermod_training_expander.py`, `run_hermod_dynamic_e2e.py`

---

## 六、总改动清单

| Phase | 文件 | 操作 | 行数 |
|-------|------|------|------|
| 1 | `hermod_metadata.py`, `hermod_placement.py` | 新建+移动；删除原文件 | ~210 |
| 2 | `hermod_metadata.py` | 改 `apply()`：合成 coflow_id + 返回 dict | ~±20 |
| 3 | `hermod_priority.py` | 改 `from_workload()` | ~+15 |
| 4 | `hermod_strategy.py` | 新增字段 + 参数 | ~+10 |
| 5 | `dynamic_executor.py` | 删 5 个 key | ~-5 |
| 6a | `workload_builder.py` | 删 3 处 coflow_id | ~-15 |
| 6b | `collective_expander.py` | 删字段 | ~-10 |
| 6c | `schema.py` | 删字段 + schema | ~-15 |
| 6d | `validator.py` | 删 3 行 | ~-3 |
| 6e | `writer.py` | 删写入 + 读取 | ~-15 |
| 6f | `job_merger.py` | 删 remapping | ~-5 |
| 7 | 3 个脚本 | 更新 import + 传参 | ~±10 |
| — | 各处 `__init__.py` | 更新导出路径 | ~+3 |
| **合计** | **~18 个文件** | | **~110 行净变更** |

---

## 七、测试计划

1. 全部现有测试：`uv run pytest tests/ -x --tb=short`
2. Hermod 专用测试：`uv run pytest tests/test_hermod_*.py -v`
3. E2E 结果对比：makespan 与重构前一致
4. 序列化 round-trip：JSON 不再含 Hermod 字段

---

## 八、实施顺序

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5
                                        → Phase 6
                                        → Phase 7
                                        → 全部测试验证
```

Phase 3 的 `hermod_records=None` 回退路径可在 Phase 6 最终删除时一并移除。

---

## 九、回退方案

1. **单文件回退：** `git checkout <path>`
2. **全量回退：** `git revert <commit>`
3. **补回 coflow_id 字段：** 在 `writer.py` 中保留条件写入（Phase 6e 跳过）即可向前兼容

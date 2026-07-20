# PR #5 合并审查报告

> 审查日期：2026-07-16
> 审查范围：`aa54e79` → `de4d886`（Merge pull request #5 from violetandevergarden/hxh/cassini）
> 审查目的：评估合并对现有功能的影响，标记所有侵入性改动

---

## 概览

本次合并引入了 Hermod §4.1 coflow 优先级调度策略的完整实现。共涉及 **35 个文件变更**（2,355 行新增，26 行删除），其中 **14 个已有文件被修改**，**21 个为新文件**。

所有 **749 个现有测试**全部通过，零失败。

---

## 改动分类体系

| 等级 | 标签 | 含义 |
|------|------|------|
| 🔴 | IR 污染 | 泛型数据模型被策略特定概念污染 |
| 🔴 | 动态 Hack | 利用 Python 动态特性张贴不属于对象的属性 |
| 🟡 | 行为变更 | 泛型代码逻辑改变，可能影响其他路径 |
| 🟢 | 元数据扩展 | 已有字典增加 key，完全向下兼容 |
| ✅ | 无侵入 | 纯新增结构，不影响已有逻辑 |

---

## 全部 14 个已有文件改动详解

### 🔴 IR 污染 — 泛型 Task 数据模型

#### 1. `src/workload_format/schema.py`

**新增 3 个字段到 `Task` dataclass（第 166-168 行）：**

```python
coflow_id: Optional[str] = None
microbatch_id: Optional[int] = None
logical_layer_id: Optional[int] = None
```

- `coflow_id` — 标识同一 collective 调用的所有 P2P 流，概念上通用但当前仅 Hermod 使用
- `microbatch_id` — Hermod §4.1 论文 MID（microbatch ID），从 AICB 的 GA step 映射而来
- `logical_layer_id` — Hermod §4.1 论文 LID（logical layer ID），从 AICB item 的 phase/顺序推导

**同时在 `P2P_WORKLOAD_JSON_SCHEMA` 中增加对应 JSON schema 定义（第 389-395 行）。**

**影响：** 每个 Task 实例现在多 3 个 Optional 字段。非 Hermod 路径下全部为 `None`，无运行时影响，但设计上不够干净。

---

#### 2. `src/workload_format/validator.py`

**在 `WorkloadValidator` 中添加字段读取（第 161-163 行）：**

```python
coflow_id=task_data.get("coflow_id"),
microbatch_id=task_data.get("microbatch_id"),
logical_layer_id=task_data.get("logical_layer_id"),
```

从 JSON 中反序列化 Hermod 元数据。对于不含这些字段的 JSON 文件，返回 `None`，安全。

---

#### 3. `src/workload_format/writer.py`

**在 `_task_to_dict()` 中添加条件写入（第 140-148 行）：**

```python
if task.coflow_id is not None:
    result["coflow_id"] = task.coflow_id
if task.microbatch_id is not None:
    result["microbatch_id"] = task.microbatch_id
if task.logical_layer_id is not None:
    result["logical_layer_id"] = task.logical_layer_id
```

采用 `if not None` 守卫，保证非 Hermod Task 写入时不会产生额外字段。

**附带改动：** `item_id` 字段被加入序列化（第 124 行）。此字段本就存在于 `Task` dataclass 中，以前未被写入 JSON，现在是"顺便修复"。

**双向一致性：** `WorkloadReader` 中对等添加读取（第 245-247 行）。

---

#### 4. `src/workload_generator/collective_expander.py`

**在 `FlowTask` 中间数据类中添加 3 个字段（第 70-72 行），并在 `to_task()` 中透传（第 83-87 行）：**

```python
coflow_id: Optional[str] = None
microbatch_id: Optional[int] = None
logical_layer_id: Optional[int] = None
```

`FlowTask` 是 `workload_builder.py` 展开 collective 时使用的中间结构，最终会转化为 `Task`。字段在此处添加是为了在展开阶段就能附着 Hermod 元数据。

---

#### 总结：污染传播链

```
流动方向                文件                         新字段
WorkloadBuilder ──→ collective_expander.FlowTask ──→ coflow_id, microbatch_id, logical_layer_id
                         ↓ to_task()
                   schema.Task ────────────────────→ 同上 3 个 Optional 字段
                         ↓ serialization
                   writer._task_to_dict() ─────────→ 同上（条件写入）
                         ↓ deserialization
                   validator / reader ──────────────→ 同上（get 读取）
```

---

### 🔴 动态 Hack — 运行时属性张贴

#### 5. `src/workload_generator/hermod_aicb_metadata.py`（第 155-156 行）

```python
task.hermod_lid_source_operation = operation
task.hermod_lid_mapping_rule = mapping_rule
```

**性质：** 直接在 Task 实例上 setattr 不存在的属性。

- `hermod_lid_source_operation` — AICB 操作名（`"attention_layer"`、`"mlp_layer"` 等），**不进 schema**
- `hermod_lid_mapping_rule` — LID 映射规则（`"attention_mlp_transformer_layer"` 等），**不进 schema**
- 这两个属性**仅用于审计/可视化**，不参与调度决策

**这是整个 merge 中最不优雅的做法。** 正确的做法是用 `HermodMetadataRecord` sidecar 传递。

---

#### 6. `src/executor/dynamic_executor.py`（第 254-258 行）

```python
self._task_meta[t.task_id] = {
    k: getattr(t, k, None)
    for k in ("phase", "layer_id", "comm_type", "src", "dst", "node", "job_id",
              "coflow_id", "microbatch_id", "logical_layer_id",
              "hermod_lid_source_operation", "hermod_lid_mapping_rule")
}
```

通过 `getattr` 反射读取 5 个新 key 写入 `_task_meta`。其中 2 个（`hermod_lid_*`）甚至不是 Task schema 字段，完全依赖 Python 的动态查找。

`_task_meta` key 数量从 **7 个 → 12 个**。所有消费者都用 `.get()` 或 `**` 展开，不会 KeyError。

---

### 🟡 行为变更 — 泛型代码逻辑改变

#### 7. `src/static_analysis/strategies/default_strategy.py`（第 89-120 行）

**`OneFOneBAnalyzer` 新增可选参数：**

```python
def __init__(self, topology, jobs_by_id: dict[int, object] | None = None):
```

**副作用：** `analyze()` 中当 `jobs_by_id` 非 `None` 时，会**就地修改 `workload.jobs`**：

```python
if self.jobs_by_id is not None:
    workload.jobs = [self.jobs_by_id[job_id] for job_id in sorted(job_ids)]
```

**风险评估：**
- ✅ `jobs_by_id` 有默认值 `None`，所有现有调用不受影响
- ⚠️ 修改入参 (`workload.jobs`) 是副作用，但 `DynamicExecutor` 的 mini-workload 本就是临时构造的，不影响已有路径

---

#### 8. `src/workload_generator/job_merger.py`（第 153-178 行）

**三项改动：**

**(a) 新增 `missing_deps` 校验（第 153-157 行）：**

```python
missing_deps = [dep for dep in task.deps if dep not in task_old_to_new]
if missing_deps:
    raise ValueError(f"Task {task.task_id} references missing dependencies: {missing_deps}")
```

之前缺失的 dep 被静默忽略，现在会直接报错。

**(b) 去掉 dep 过滤条件（第 178 行）：**

```python
# 之前：
deps=[task_old_to_new[dep] for dep in task.deps if dep in task_old_to_new]
# 之后：
deps=[task_old_to_new[dep] for dep in task.deps]
```

之前 `if dep in task_old_to_new` 过滤掉了不在映射中的依赖。现在去掉后，不存在的 dep 会导致 KeyError（或被 (a) 的校验提前拦截）。

**(c) 新增 3 个字段透传（第 163-167 行）：**

```python
coflow_id=...,
microbatch_id=...,
logical_layer_id=...,
```

**风险评估：**
- ✅ 现有 749 个测试全部通过，证明当前 workload 没有触发 dep 缺失的边界情况
- ⚠️ 如果将来出现 dep 引用不完整的 workload，原来静默运行现在会报错——这是 bug 修复而非回归

---

### 🟢 元数据扩展 — 仅增加字典 key

#### 9. `src/executor/dynamic_executor.py`（上述第 254-258 行）

已在上方（#6）覆盖。`_task_meta` 的 key 集从 7→12，属于纯扩展。

---

### ✅ 无侵入 — 纯新增结构

#### 10. 各 `__init__.py` 注册（3 个文件）

| 文件 | 新增导出 |
|------|---------|
| `executor/__init__.py` | `HermodSchedulingPolicy`, `HermodAllocator` |
| `executor/policies/__init__.py` | `HermodSchedulingPolicy` |
| `executor/bandwidth_allocators/__init__.py` | `HermodAllocator` |

纯新增引入，不影响已有类的导出。

---

#### 11. `.gitignore`

```
*.pdf
paper
```

无影响。

---

#### 12. `tests/test_job_merger.py`

新增测试 `test_missing_dependency_is_rejected_instead_of_dropped`（第 110-115 行），验证 #8(a) 的行为。

---

#### 13. `scripts/visualize_dynamic.py`

重构——将 trace 导出提取为 `export_trace()` 函数，并在事件中增加 `args` 字段透传 `task_id` 和 metadata。原有 `main()` 行为不变，新增函数可被其他脚本调用。

---

#### 14. `src/workload_generator/workload_builder.py`（第 530-535、1118、1137 行）

在 expander 生成的 collective flow 和 PP 边界 flow 上设置 `coflow_id` 字段。此字段仅在被 `hermod_aicb_metadata.py` 消费时有实际意义；非 Hermod 路径下 `coflow_id` 不会被任何代码引用。

---

## 新文件清单（21 个）

以下均为全新文件，不会影响已有功能：

| 分类 | 文件 | 行数 |
|------|------|------|
| 文档 | `docs/reproduce/hermod/hermod-4.1-e2e-minimal-plan.md` | 144 |
| 文档 | `docs/reproduce/hermod/hermod-4.1-implementation.md` | 228 |
| 文档 | `docs/reproduce/hermod/hermod-4.1-reproduction-plan.md` | 193 |
| 脚本 | `scripts/hermod_dynamic_e2e_config.json` | 18 |
| 脚本 | `scripts/hermod_e2e_config.json` | 14 |
| 脚本 | `scripts/run_hermod_dynamic_e2e.py` | 297 |
| 脚本 | `scripts/run_hermod_e2e.py` | 209 |
| 核心 | `src/executor/bandwidth_allocators/hermod_allocator.py` | 100 |
| 核心 | `src/executor/policies/hermod_policy.py` | 26 |
| 核心 | `src/executor/hermod_training_expander.py` | 54 |
| 核心 | `src/static_analysis/passes/hermod_priority.py` | 227 |
| 核心 | `src/static_analysis/strategies/hermod_strategy.py` | 75 |
| 数据 | `src/workload_generator/hermod_aicb_metadata.py` | 174 |
| 数据 | `src/workload_generator/hermod_placement.py` | 39 |
| 测试 | `tests/test_default_1f1b_dynamic.py` | 64 |
| 测试 | `tests/test_hermod_aicb_metadata.py` | 95 |
| 测试 | `tests/test_hermod_allocator.py` | 101 |
| 测试 | `tests/test_hermod_dynamic_placement.py` | 34 |
| 测试 | `tests/test_hermod_policy.py` | 35 |
| 测试 | `tests/test_hermod_priority.py` | 100 |
| 子模块 | `aicb`（指针变更 `46d60ac` → `c3542cc`） | - |

---

## 风险评估矩阵

| 优先级 | 问题 | 文件 | 状态 |
|--------|------|------|------|
| 🔴 | Hermod 字段污染泛型 Task schema | `schema.py` | 749 测试通过，运行时安全。设计上需后续重构 |
| 🔴 | 动态 setattr 属性不属于 Task | `hermod_aicb_metadata.py` | 纯审计用途，不参与调度。应改为 sidecar 模式 |
| 🟡 | JobMerger dep 校验行为变更 | `job_merger.py` | 经测试验证，现有 workload 未触发。如有旧 workload 含无效 dep 引用则需排查 |
| 🟡 | OneFOneBAnalyzer 副作用修改入参 | `default_strategy.py` | `jobs_by_id=None` 默认值保证现有路径不受影响 |
| 🟢 | `_task_meta` 字典 key 扩展 | `dynamic_executor.py` | 所有消费者安全（`get()` / `**` 展开） |
| ✅ | 各 `__init__.py` / `.gitignore` / 测试 | 多个 | 无影响 |

---

## 结论

**本次合并对已有功能没有产生破坏。** 所有 749 个测试通过。

主要的侵入性问题是设计层面的——Hermod 特定的概念（`microbatch_id`、`logical_layer_id`、`coflow_id`）被植入了泛型的 `Task` 数据模型，以及利用 Python 动态特性的 `setattr` hack。这些问题在运行时无害（默认值均为 `None`），但长期维护上建议将 Hermod 元数据抽离为独立的 sidecar 结构。

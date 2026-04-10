# Phase 2 - Task 3 开发记录：Workload Builder

## 1. 目标与范围

Task 3 的目标是实现 `WorkloadBuilder` 类，将 AICB training workload（通过 `AicbParser` 解析）转换为 P2P Workload IR（`P2PWorkload` 对象）。这是 Phase 2 的核心模块，依赖 Task 1 (`AicbParser`) 和 Task 2 (`RankGrouper`)。

**包含**：
- `WorkloadBuilder` 类实现，支持完整的 AICB → P2P Workload 转换
- GA iteration 自动分配机制
- 通信类型后缀到 RankGrouper 方法的映射
- 依赖链构建（forward → backward → DP phases，跨 item 串联）
- 12 个单元测试覆盖基本功能、GA 迭代、依赖链、rank 分组集成

**不包含**：
- PP 虚拟化建模（pp > 1 时简化处理，留作后续扩展）
- 多任务合并（Task 4: JobMerger）
- PP stage 间 flow 展开（当前仅建模为 compute task 依赖边）

---

## 2. 交付物

### 2.1 新增文件

```
simai-flow-scheduler/
├── src/workload_generator/
│   ├── workload_builder.py          # 新增：WorkloadBuilder 实现
│   └── __init__.py                  # 更新：导出 WorkloadBuilder
└── tests/
    └── test_workload_builder.py     # 新增：12 个测试
```

### 2.2 核心 API

```python
class WorkloadBuilder:
    """Convert AICB workload to P2P Workload."""

    def __init__(self):
        self.expanders = {
            "ALLREDUCE": AllReduceExpander(),
            "ALLGATHER": AllGatherExpander(),
            "REDUCESCATTER": ReduceScatterExpander(),
            "ALLTOALL": AlltoAllExpander(),
        }

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        """
        Convert AICB workload to P2P Workload.

        Key insight: AICB file already contains GA-expanded items.
        If ga=24 and vpp=80, there are 1920 layer items in the file.
        The builder does NOT need to loop over GA — it just assigns iteration IDs.
        """
        ...

    def _expand_comm(
        self,
        base_type: str,         # e.g. "ALLGATHER"
        context: str,           # e.g. "tp", "dp", "ep", "dp_ep"
        comm_size: int,
        grouper: RankGrouper,
        pp_idx: int, dp_idx: int, ep_idx: int, tp_idx: int,
        phase: Phase,
        job_id: int,
        task_id_start: int,
        algo: str = "ring",
    ) -> list[FlowTask]:
        """
        根据通信类型后缀选择正确的 rank group，然后调用对应的 expander。

        Context 映射:
        - "tp" → grouper.get_tp_group(pp_idx, dp_idx, ep_idx)
        - "dp" → grouper.get_dp_group(pp_idx, ep_idx, tp_idx)
        - "ep" → grouper.get_ep_group(pp_idx, dp_idx, tp_idx)
        - "dp_ep" → grouper.get_dp_ep_group(pp_idx, tp_idx)
        """
        ...
```

---

## 3. 设计原理

### 3.1 AICB 文件已经包含 GA 展开

**关键理解**：AICB workload 文件中的 items **已经包含了完整的 GA 展开**。如果 `ga=24` 且模型有 80 层（`vpp=80`），那么 workload 文件中会有 `80 * 24 = 1920` 个 layer items（不包括 pre/post items）。

**Builder 不需要自己做 GA 循环**，只需要按顺序处理所有 items，并根据 `ga` 和 `vpp` 计算每个 item 所属的 iteration：

```python
if item_idx < num_pre_items:
    iteration = 0
elif item_idx >= num_pre_items + num_layer_items:
    iteration = header.ga
else:
    iteration = (item_idx - num_pre_items) // header.vpp
```

### 3.2 通信类型后缀直接决定分组

使用 `AicbParser.parse_comm_type()` 解析后缀，直接映射到 `RankGrouper` 的对应方法：

| AICB comm | parse_comm_type 返回 | RankGrouper 调用 | 获取的 ranks |
|-----------|---------------------|------------------|-------------|
| `ALLREDUCE` | `("ALLREDUCE", "tp")` | `get_tp_group(pp_idx, dp_idx, ep_idx)` | TP group |
| `ALLGATHER_DP_EP` | `("ALLGATHER", "dp_ep")` | `get_dp_ep_group(pp_idx, tp_idx)` | DP×EP group |
| `ALLTOALL_EP` | `("ALLTOALL", "ep")` | `get_ep_group(pp_idx, dp_idx, tp_idx)` | EP group |

### 3.3 依赖关系规则

- **同一 item 内**：fwd_compute → fwd_flows → bwd_compute → bwd_flows → dp_compute → dp_flows（串行链）
- **跨 item**：前一 item 的最后 task → 当前 item 的第一个 task（按出现顺序串联）

### 3.4 PP 简化建模

当 `pp > 1` 时，当前版本不展开为 P2P flow，而是插入虚拟 compute task 作为依赖边。Phase 2 初期先只处理单 PP stage（`pp=1`）。

---

## 4. 算法流程

### 4.1 build_from_aicb 主流程

```python
def build_from_aicb(self, header, items, job, comm_algo="ring"):
    # Step 1: Create RankGrouper
    grouper = RankGrouper(job.assigned_nodes, job.parallelism)

    # Step 2: Validate structure
    num_pre_items = self._count_pre_items(items)
    num_post_items = self._count_post_items(items)
    num_layer_items = len(items) - num_pre_items - num_post_items
    assert num_layer_items % header.ga == 0
    assert num_layer_items // header.ga == header.vpp

    # Step 3: Initialize task collection
    all_tasks: list[FlowTask] = []
    task_id_counter = 0
    prev_last_task_id: Optional[int] = None

    # Step 4: Iterate through all items
    for item_idx, item in enumerate(items):
        # Calculate iteration
        if item_idx < num_pre_items:
            iteration = 0
        elif item_idx >= num_pre_items + num_layer_items:
            iteration = header.ga
        else:
            iteration = (item_idx - num_pre_items) // header.vpp

        # --- Forward phase ---
        compute_task = _create_compute_task(..., phase=FORWARD)
        all_tasks.append(compute_task)
        if item.forward_comm != "NONE":
            flows = _expand_comm(...)
            flows[0].deps.append(compute_task.task_id)
            all_tasks.extend(flows)

        # --- Backward phase ---
        # Similar pattern

        # --- DP phase ---
        # Similar pattern

        # Cross-item dependency
        if prev_last_task_id is not None:
            first_task_of_current_item.deps.append(prev_last_task_id)
        prev_last_task_id = all_tasks[-1].task_id

    # Step 5: Build P2PWorkload
    return P2PWorkload(version="1.0", meta=..., jobs=[job], tasks=[t.to_task() for t in all_tasks])
```

### 4.2 _expand_comm 方法

```python
def _expand_comm(self, base_type, context, comm_size, grouper, ...):
    # Select rank group based on context
    if context == "tp":
        ranks = grouper.get_tp_group(pp_idx, dp_idx, ep_idx)
    elif context == "dp":
        ranks = grouper.get_dp_group(pp_idx, ep_idx, tp_idx)
    elif context == "ep":
        ranks = grouper.get_ep_group(pp_idx, dp_idx, tp_idx)
    elif context == "dp_ep":
        ranks = grouper.get_dp_ep_group(pp_idx, tp_idx)

    # Call appropriate expander
    expander = self.expanders[base_type]
    if base_type == "ALLREDUCE":
        flows = expander.expand_allreduce(ranks, comm_size, ...)
    elif base_type == "ALLGATHER":
        flows = expander.expand_allgather(ranks, comm_size, ...)
    # etc.

    # Set phase on all flows
    for flow in flows:
        flow.phase = phase

    return flows
```

---

## 5. 开发过程（TDD）

### 5.1 方法

严格遵循 TDD 流程：先编写测试 → RED → GREEN → REFACTOR。

### 5.2 测试覆盖

`tests/test_workload_builder.py`（12 个测试）覆盖：

| 测试类 | 数量 | 覆盖内容 |
|--------|------|---------|
| `TestBasicWorkload` | 3 | 基本单层 workload、无通信场景、task_id 唯一性 |
| `TestGAIteration` | 2 | GA iteration 分配、pre/post items iteration |
| `TestDependencyChain` | 3 | forward→backward 依赖链、DAG 无环验证、跨 item 依赖 |
| `TestRankGroupIntegration` | 2 | TP context 用 TP group、DP_EP context 用 DP×EP group |
| `TestEdgeCases` | 2 | 单 GPU 无通信、多 phases 都有通信 |

### 5.3 测试结果

```
tests/test_workload_builder.py: 12 passed
Total: 116 passed (含 Phase 1、其他模块和之前 Task 测试)
```

---

## 6. 发现的问题及修复

### 问题 1：跨 item 依赖实现复杂度

**现象**：在初始设计中，跨 item 依赖需要找到当前 item 的第一个 task，但在流式构建过程中这个 task 可能还未创建。

**原因**：`first_task_of_item` 变量需要在 item 处理的早期设置，但第一个 task 可能在多个阶段（forward/backward/dp）中任意一个才真正创建。

**修复**：采用延迟绑定策略：
```python
first_task_of_item: Optional[int] = None
# 在每个阶段创建 compute_task 时检查并设置
if first_task_of_item is None:
    first_task_of_item = task_id_counter
# ... 在处理完所有 phases 后，再建立跨 item 依赖
if prev_last_task_id is not None and first_task_of_item is not None:
    for task in all_tasks:
        if task.task_id == first_task_of_item:
            task.deps.append(prev_last_task_id)
            break
```

**教训**：对于流式构建的场景，使用延迟绑定或事后修复比在构建过程中维护复杂状态更可靠。

### 问题 2：iteration 计算的边界条件

**现象**：pre-layer items 和 post-layer items 的 iteration 值应该如何设置？

**分析**：
- Pre-layer items（如 `grad_gather`）出现在 GA 循环之前，应该用 `iteration=0`
- Post-layer items（如 `embedding_norm`, `optimizer`）出现在所有 GA steps 之后，应该用 `iteration=ga`

**修复**：在 `_count_pre_items` 和 `_count_post_items` 辅助函数的帮助下，正确区分三类 items：
```python
if item_idx < num_pre_items:
    iteration = 0
elif item_idx >= num_pre_items + num_layer_items:
    iteration = header.ga
else:
    iteration = (item_idx - num_pre_items) // header.vpp
```

---

## 7. 使用示例

### 7.1 基本用法

```python
from src.workload_generator.workload_builder import WorkloadBuilder
from src.workload_generator.aicb_parser import AicbParser
from src.workload_format.schema import Job, ParallelismConfig

parser = AicbParser()
header, items = parser.parse("../../example/microAllReduce.txt")

job = Job(
    job_id=0,
    name="llama-7b",
    assigned_nodes=list(range(8)),
    parallelism=ParallelismConfig(tp=2, dp=2, pp=1, ep=1),
)

builder = WorkloadBuilder()
workload = builder.build_from_aicb(header, items, job)

print(f"Generated {len(workload.tasks)} tasks")
for task in workload.tasks:
    print(f"  Task {task.task_id}: {task.type}, phase={task.phase}, iteration={task.iteration}")
```

### 7.2 验证输出

```python
from src.workload_format.validator import WorkloadValidator

validator = WorkloadValidator()
is_valid, errors, _ = validator.validate(workload)
assert is_valid, f"Validation failed: {errors}"
```

---

## 8. 后续依赖

Task 3 的输出将被以下模块使用：

1. **Task 4（JobMerger）**：将多个单任务 P2P Workload 合并为一个多任务 workload
2. **Task 5（端到端示例）**：直接使用 WorkloadBuilder 完成从 AICB 文件到 P2P Workload 的转换
3. **Flow Scheduler**：消费生成的 P2P Workload 进行细粒度流量调度模拟

---

*开发时间：2026-04-10*
*测试状态：12 passed (total 116)*
*开发方法：TDD（RED → GREEN → REFACTOR）*

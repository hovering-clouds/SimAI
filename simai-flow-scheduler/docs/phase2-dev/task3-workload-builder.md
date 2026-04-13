# Phase 2 - Task 3 开发记录：Workload Builder

## 1. 目标与范围

Task 3 的目标是实现 `WorkloadBuilder` 类，将 AICB training workload（通过 `AicbParser` 解析）转换为 P2P Workload IR（`P2PWorkload` 对象）。这是 Phase 2 的核心模块，依赖 Task 1 (`AicbParser`) 和 Task 2 (`RankGrouper`)。

**包含**：

- `WorkloadBuilder` 类实现，支持完整的 AICB → P2P Workload 转换
- GA iteration 自动分配机制
- 通信类型后缀到 RankGrouper 方法的映射
- 依赖链构建（forward → backward → DP phases，跨 item 串联）
- 单元测试覆盖基本功能、GA 迭代、依赖链、rank 分组集成、DAG 完整性

**不包含**：

- PP 虚拟化建模（pp > 1 时简化处理，留作后续扩展）
- 多任务合并（Task 4: JobMerger）
- PP stage 间 flow 展开（当前仅建模为 compute task 依赖边）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/workload_format/
│   └── schema.py                          # 更新：Task/FlowTask 添加 item_id 字段和调度语义
├── src/workload_generator/
│   ├── workload_builder.py                # 新增：WorkloadBuilder 实现
│   ├── collective_expander.py             # 更新：FlowTask 添加 item_id 字段
│   └── __init__.py                        # 更新：导出 WorkloadBuilder
└── tests/
    └── test_workload_builder.py           # 新增：21 个测试
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
        Convert AICB workload to P2P Workload (Route C design).

        Key insight: AICB file already contains GA-expanded items.
        If ga=24 and vpp=80, there are 1920 layer items in the file.
        The builder does NOT need to loop over GA — it just assigns iteration IDs.

        Design principle: Only true data dependencies are encoded as hard edges.
        Cross-GA execution order is left to the scheduler using
        (iteration, layer_id, phase) hints for C++ order reproduction if needed.
        """
        ...
```

---

## 3. 设计原理

### 3.1 AICB 文件已经包含 GA 展开

**关键理解**：AICB workload 文件中的 items **已经包含了完整的 GA 展开**。如果 `ga=24` 且模型有 80 层（`vpp=80`），那么 workload 文件中会有 `80 * 24 = 1920` 个 layer items（不包括 pre/post items）。

证据来自 `aicb/workload_generator/SimAI_training_workload_generator.py:270-271`：

```python
for _ in range(self.ga_num):      # outer loop = GA
    for layer in layers:          # inner loop = layers
        # emit one work item
```

**Builder 不需要自己做 GA 循环**，只需要按顺序处理所有 items，并根据 `ga` 和 `vpp` 计算每个 item 所属的 iteration：

```python
if item_idx < num_pre_items:
    iteration = -1                        # pre-layer items
    layer_id = item_idx                   # sequential index
elif item_idx >= num_pre_items + num_layer_items:
    iteration = header.ga                 # post-layer items
    layer_id = item_idx - (num_pre_items + num_layer_items)
else:
    iteration = (item_idx - num_pre_items) // header.vpp   # GA step
    layer_id = (item_idx - num_pre_items) % header.vpp     # layer within GA
```

### 3.2 通信类型后缀直接决定分组

使用 `AicbParser.parse_comm_type()` 解析后缀，直接映射到 `RankGrouper` 的对应方法：

| AICB comm           | parse_comm_type 返回       | RankGrouper 调用                         | 获取的 ranks |
| ------------------- | -------------------------- | ---------------------------------------- | ------------ |
| `ALLREDUCE`       | `("ALLREDUCE", "tp")`    | `get_tp_group(pp_idx, dp_idx, ep_idx)` | TP group     |
| `ALLGATHER_DP_EP` | `("ALLGATHER", "dp_ep")` | `get_dp_ep_group(pp_idx, tp_idx)`      | DP×EP group |
| `ALLTOALL_EP`     | `("ALLTOALL", "ep")`     | `get_ep_group(pp_idx, dp_idx, tp_idx)` | EP group     |

### 3.3 路线 C：数据依赖 + 调度提示

**设计原则**：只在 workload 中建立真实的数据依赖边，让调度器可以自由选择执行顺序。同时提供 `(iteration, layer_id, phase)` 元组作为调度提示，允许调度器复现 C++ 参考实现的顺序。

**真实数据依赖**：

- Forward 链：`layer[i].fwd → layer[i+1].fwd`（激活值传递）
- Fwd→IG 桥接：最后一层 forward 完成后开始 backward
- IG reverse 链：`layer[i].ig → layer[i-1].ig`（input gradient 传递）
- 同层 IG→WG：`layer[i].ig → layer[i].wg`（gradient 完成后更新权重）

**移除的非数据依赖**：

- ~~GA bridge：`GA[k].wg[0] → GA[k+1].fwd[0]`~~ — 这只是 C++ 参考实现的执行顺序，不是真实数据依赖

**调度提示字段**：

```python
@dataclass
class Task:
    task_id: int
    job_id: int
    type: TaskType
    iteration: int      # GA step (pre: -1, layer: 0..ga-1, post: ga)
    phase: Phase        # FORWARD / BACKWARD_INPUT / BACKWARD_WEIGHT
    layer_id: int       # logical layer index within iteration
    item_id: int        # global AICB item index (0-based)
    deps: list[int]
    ...
```

调度器可以通过排序 `(iteration, layer_id, phase_order)` 来复现 C++ 顺序：

- Pre items: `iteration=-1`，按 layer_id 正序
- Layer items: `iteration=0..ga-1`，forward 正序 (0→N-1)，backward 逆序 (N-1→0)
- Post items: `iteration=ga`，按 layer_id 正序

### 3.4 Per-rank Compute Tasks

**C++ 参考行为**：astra-sim 中每个 rank 有独立的 `Workload` 实例和事件驱动模拟器，每个 rank 独立执行自己的 compute task。

**正确做法**：为参与该通信的所有 rank 各自创建 compute task。

示例：TP=2, DP=2 → 4 个 rank [0,1,2,3]

- forward ALLREDUCE (TP): TP groups [0,1] and [2,3]
  - 创建 4 个 compute task：rank 0, 1, 2, 3 各一个
  - 展开 2 组 ALLREDUCE flows（组 0: [0,1]，组 1: [2,3]）

### 3.5 Receiver-based 依赖

**设计决策**：下一步的 compute task 依赖于该 rank **接收到的所有 flows**（dst 指向该 rank），而不是发送的 flows。

**原因**：rank 只需等接收完其他 rank 发来的数据即可开始计算，不需要等自己完成发送。

以 2-rank Ring ALLREDUCE 为例：

```
Flow 0: 0→1, deps=[]        Flow 1: 1→0, deps=[]
Flow 2: 0→1, deps=[Flow 1]  Flow 3: 1→0, deps=[Flow 0]

receiver_index:
  rank 0: [Flow 1, Flow 3]   ← rank 0 作为 dst 的所有 flow
  rank 1: [Flow 0, Flow 2]   ← rank 1 作为 dst 的所有 flow
```

rank 0 的下一个 compute 必须依赖 Flow 1 和 Flow 3 都完成，缺一不可。

---

## 4. 算法流程

### 4.1 两阶段生成架构

```
Phase 1: 生成所有 tasks（无依赖）
  → 对每个 item，为所有 rank 创建 compute + flow tasks

Phase 2: 连接依赖关系
  → 按照 Forward/Backward/WG 顺序，per-node 地建立依赖边
```

### 4.2 数据结构

```python
@dataclass
class FlowGroupResult:
    """一次集合通信展开的结果，含增量追踪的 receiver 索引。"""
    flows: list[FlowTask]
    receiver_index: dict[int, list[int]]  # rank → 该 rank 作为 dst 的 task_id 列表

    @staticmethod
    def empty() -> "FlowGroupResult":
        return FlowGroupResult(flows=[], receiver_index={})

    def add_flow(self, flow: FlowTask):
        """添加一个 flow 并更新 receiver 索引。"""
        self.flows.append(flow)
        if flow.dst not in self.receiver_index:
            self.receiver_index[flow.dst] = []
        self.receiver_index[flow.dst].append(flow.task_id)


@dataclass
class ItemTasks:
    """一个 AICB work item 展开后的所有 tasks，按 phase 分组。"""
    fwd_computes: dict[int, FlowTask]   # rank → compute task
    ig_computes: dict[int, FlowTask]
    wg_computes: dict[int, FlowTask]
    fwd_result: FlowGroupResult
    ig_result: FlowGroupResult
    wg_result: FlowGroupResult
```

### 4.3 主流程伪代码

```python
def build_from_aicb(self, header, items, job, comm_algo="ring"):
    # Step 1: Create RankGrouper
    grouper = RankGrouper(job.assigned_nodes, job.parallelism)

    # Step 2: Validate structure
    num_pre_items = count_pre_items(items)
    num_post_items = count_post_items(items)
    num_layer_items = len(items) - num_pre_items - num_post_items
    assert num_layer_items % header.ga == 0
    assert num_layer_items // header.ga == header.vpp

    # Step 3: Phase 1 - Generate all tasks
    item_tasks_list = []
    for item_idx, item in enumerate(items):
        # Calculate scheduling fields
        if item_idx < num_pre_items:
            iteration = -1
            layer_id = item_idx
        elif item_idx >= num_pre_items + num_layer_items:
            iteration = header.ga
            layer_id = item_idx - (num_pre_items + num_layer_items)
        else:
            iteration = (item_idx - num_pre_items) // header.vpp
            layer_id = (item_idx - num_pre_items) % header.vpp

        item_id = item_idx
        item_tasks = ItemTasks()

        # Forward phase
        item_tasks.fwd_computes = _create_compute_tasks_for_phase(
            ranks, item.forward_compute_time, Phase.FORWARD,
            layer_id, iteration, item_id, ...)
        if item.forward_comm != "NONE":
            item_tasks.fwd_result = _expand_comm_all_groups(
                item.forward_comm, ..., layer_id, iteration, item_id, ...)
            _wire_compute_to_flows(item_tasks.fwd_computes, item_tasks.fwd_result)

        # Backward (IG) phase - similar pattern
        # DP (WG) phase - similar pattern

        item_tasks_list.append(item_tasks)

    # Step 4: Phase 2 - Wire dependencies
    _wire_dependencies(item_tasks_list, num_layer_items, num_pre_items, header)

    # Step 5: Build P2PWorkload
    return P2PWorkload(version="1.0", meta=..., jobs=[job],
                       tasks=[t.to_task() for t in all_flow_tasks])
```

### 4.4 依赖连接算法

```python
def _wire_dependencies(self, item_tasks_list, num_layer_items, num_pre_items, header):
    # Group layer items by GA step
    ga_groups = group_items_by_ga(item_tasks_list, num_pre_items, num_layer_items, vpp)

    for ga_group in ga_groups:
        num_layers = len(ga_group)

        # 1. Forward chain (正序 0 → N-1)
        for i in range(num_layers - 1):
            _wire_per_node_phase_transition(
                src_result=ga_group[i].fwd_result,
                dst_computes=ga_group[i + 1].fwd_computes)

        # 2. Fwd→IG bridge (最后一层)
        last_layer = ga_group[num_layers - 1]
        _wire_per_node_phase_transition(
            src_result=last_layer.fwd_result,
            dst_computes=last_layer.ig_computes)

        # 3. IG reverse chain (逆序 N-1 → 0) + WG same layer
        for i in range(num_layers - 1, -1, -1):
            # IG→WG same layer
            _wire_per_node_phase_transition(
                src_result=ga_group[i].ig_result,
                dst_computes=ga_group[i].wg_computes)
            # IG→previous layer IG (reverse)
            if i > 0:
                _wire_per_node_phase_transition(
                    src_result=ga_group[i].ig_result,
                    dst_computes=ga_group[i - 1].ig_computes)

        # 4. GA bridge removed - not a true data dependency
        # Scheduler can use (iteration, layer_id, phase) hints for ordering

    # Wire pre/post items linearly
```

---

## 5. 开发历史：问题发现与重构

### 5.1 初始实现问题

最初实现时存在三个根本性设计缺陷，通过分析 astra-sim C++ 参考实现（`Workload.cc`, `Layer.cc`）发现：

| 问题             | 初始行为                                     | C++ 参考行为                      | 修复方案                              |
| ---------------- | -------------------------------------------- | --------------------------------- | ------------------------------------- |
| Compute 任务粒度 | 每个 item 每个 phase 只创建一个 compute task | 每个 rank 都有独立的 compute task | 改为 per-rank 创建 compute tasks      |
| 依赖链顺序       | 线性顺序处理所有 item                        | Forward 正序 / Backward 逆序      | 实现正序 forward 链、逆序 backward 链 |
| 跨 item 依赖     | 单一全局依赖                                 | Per-node 依赖                     | 改为 per-node 的 receiver-based 依赖  |

### 5.2 重构历程

#### 第一轮：Per-rank Compute + 正确的 Forward/Backward 顺序

**触发原因**：阅读 C++ 参考代码发现每个 rank 独立执行 compute，且 forward/backward 有明确的正序/逆序要求。

**修改内容**：

- `_create_compute_task` → `_create_compute_tasks_for_phase`：为所有 ranks 创建 compute tasks
- 实现 Forward 正序链：`layer[i].fwd → layer[i+1].fwd`
- 实现 Backward 逆序链：`layer[i].ig → layer[i-1].ig`
- 实现同层 IG→WG 依赖

#### 第二轮：Receiver-based 依赖 + FlowGroupResult

**触发原因**：发现应该依赖 receiver flows 而非 sender flows，且需要追踪每个 rank 接收的所有 flows。

**修改内容**：

- 添加 `FlowGroupResult` 数据结构，增量维护 `receiver_index`
- `_expand_comm_all_groups` 返回 `FlowGroupResult` 而非简单 list
- `_wire_per_node_phase_transition` 基于 `receiver_index` 建立依赖

#### 第三轮：两阶段生成架构

**触发原因**：流式构建时跨 item 依赖难以处理，因为目标 task 可能尚未创建。

**修改内容**：

- Phase 1：生成所有 tasks（无依赖）
- Phase 2：统一连接依赖关系
- 引入 `ItemTasks` 数据结构组织 per-item 的 tasks

#### 第四轮：Route C - 移除 GA Bridge 依赖

**触发原因**：用户提出下面的分析，指出当前实现是"混合产物"——既没有完全模拟 C++ 顺序，又根据 GPU 资源约束建立了非数据依赖。

```txt
用户分析：

- 当前实现中，WG tasks不在关键路径上，下一个microbatch的fwd(0)只需要等待上一步的WG(0)结束后即可开始，
  但却不需要等前面的WG(k)，在实际调度上可能出现下一个microbatch都开始了，结果上一个microbatch的WG还没结束，
  不符合c++参考的实现；

- task3-refactor-plan.md中最后补充的“WG 通信与 Fwd 计算重叠”，试图让WG(0)的comm独立出去关键路径，因此让fwd(0)
  依赖于WG(0)的computation，但是fwd(0)也应该依赖于更前面的WG(k)的计算（与上一个问题相同）

- 之前在讨论phase2的开发计划的时候，agent就已经提醒过了c++参考实现中，WG和IG是交替进行的，问我要不要建立依赖关系
  来严格保证这种顺序，我当时没理解什么意思，只是根据我对两者的理解，觉得前一层的IG计算并不依赖后一层的WG，所以没有加依赖，
  让WG完全独立出去关键路径

- 感觉需要确定一下workload生成器中，到底是建立全量依赖关系来严格模拟c++参考实现的行为，还是依照真实的数据/通信依赖仅仅给
  任务之间加必要的边，把模拟c++参考的行为放在后续的调度层去实现。现在的实现完全是混合产物，一方面没有完全模拟c++参考实现
  的顺序，留了很多自由调度的余地（例如在关键路径外的WG任务），另一方面又根据“GPU 计算资源约束”把本来没有数据/通信依赖的跨
  GA任务建立依赖，其实如果允许显存到内存swap的话是可以混杂不同GA的任务调度顺序的，只不过这样性能很差就是了

- 严格模拟c++行为，建立依赖关系的问题：执行策略完全限制在原本的c++行为框架内，调度策略受限，例如无法允许WG通信与后续的计算重叠，傻等WG结束

- 仅建立必要依赖，自由调度的问题：现在的模拟框架不涉及GPU显存资源的建模，导致很多不违反必要依赖的调度方案实际上会导致
  超显存，例如跨GA任务混合编排、把各层的WG通信堆在一起（因为它们不在关键路径上，调度优先级可能降低）导致无法逐层释放占用的buffer等
```

**关键决策**：

- 只保留真实数据依赖（Forward 链、IG reverse 链、同层 IG→WG）
- 移除 GA bridge 依赖（`GA[k].wg[0] → GA[k+1].fwd[0]`）
- 通过 `(iteration, layer_id, phase)` 字段提供调度提示
- 允许调度器自由选择是否重叠 WG 通信和下一个 GA 的 forward 计算

**测试结果**：原 `test_ga_bridge_dependency` 失败，更新为 `test_ga_bridge_no_hard_dependency` 验证 GA bridge 确实被移除。

### 5.3 Schema 演变

| 字段          | 初始值          | Route C 最终值                    | 说明                 |
| ------------- | --------------- | --------------------------------- | -------------------- |
| `iteration` | 0（默认）       | pre: -1, layer: 0..ga-1, post: ga | 区分 item 类型       |
| `layer_id`  | 全局 layer 索引 | 迭代内的逻辑层索引                | 用于调度排序         |
| `item_id`   | 不存在          | 全局 AICB item 索引（0-based）    | 追溯到原始 AICB item |

---

## 6. 补充说明：fwd_in_bckwd 与 GA bridge

### 6.1 fwd_in_bckwd 是 Activation Checkpointing

C++ 参考中的 `iterate_hybrid_parallel_Transformer_fwd_in_bckwd()`（Workload.cc:791-929）实现的是 **activation checkpointing（激活重计算）**，与 GA 步骤间的重叠无关。

工作原理：forward 时只保存 checkpoint 层的激活值（节省显存），backward 时从最近的 checkpoint 层重新执行 forward 来恢复所需激活值。

### 6.2 GA bridge 的真实行为

C++ 参考在 `Forward_Pass` 状态开头显式等待 WG 通信完成（Workload.cc:797）：

```cpp
if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;  // ← 阻塞等待 layer 0 的 WG 通信完成
    }
}
```

但这是 C++ 实现的顺序约束，不是真实的数据依赖。Route C 设计中移除了这个硬依赖，允许调度器选择重叠策略。

### 6.3 后续阶段调度优化：WG 通信与 Fwd 计算重叠

> **注意**：以下优化在Route3重构方案后被推迟到调度器阶段，仅作为后续阶段参考。优化方案与 C++ 参考行为不一致，实施前需验证与实际训练框架（Megatron/DeepSpeed）的行为是否一致。

**分析**：fwd 不依赖 WG 的梯度数据，但受 GPU 计算资源约束——wg(0) compute 完成前 GPU 在忙，无法开始 fwd(0) compute。唯一可重叠的是 **wg(0) 的网络通信**与 **fwd(0→N-1) 的 GPU 计算**。

```
当前（与 C++ 一致，等通信完成）:
GPU:   ... → wg(0)compute → [等wg(0)_comm] → fwd(0) → fwd(1) → ... → fwd(N-1) → ig(N-1)

优化（不等通信，重叠 comm 和 fwd compute）:
GPU:   ... → wg(0)compute → fwd(0) → fwd(1) → ... → fwd(N-1) → ...
NIC:                          wg(0)_comm ──────────────────────────────┐
GPU:                                                                  ↓ ig(N-1)
```

---

## 7. 测试覆盖

`tests/test_workload_builder.py`（21 个测试）覆盖：

| 测试类                          | 数量 | 覆盖内容                                                     |
| ------------------------------- | ---- | ------------------------------------------------------------ |
| `TestPerRankCompute`          | 3    | 所有 ranks 都有 compute tasks、分配到正确 node、task_id 唯一 |
| `TestForwardBackwardOrdering` | 3    | Forward 正序链、Backward 逆序、同层 IG→WG                   |
| `TestDiamondDependency`       | 1    | IG 分叉到 WG 和下一层 IG                                     |
| `TestReceiverBasedDeps`       | 2    | Compute 依赖 receiver flows、无非必要 compute→compute 依赖  |
| `TestGAIteration`             | 2    | GA iteration 分配、GA bridge 无硬依赖                        |
| `TestRankGroupIntegration`    | 3    | TP/DP_EP context 使用正确的 rank groups                      |
| `TestDagIntegrity`            | 4    | DAG 无环、所有 deps 存在、无自依赖                           |
| `TestEdgeCases`               | 3    | 单 GPU 无通信、多 phases 都有通信、compute→all src flows    |

测试结果：

```
tests/test_workload_builder.py: 21 passed
Total: 125 passed (含 Phase 1、其他模块测试)
```

---

## 8. 使用示例

### 8.1 基本用法

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
for task in workload.tasks[:5]:
    print(f"  Task {task.task_id}: type={task.type}, phase={task.phase}, "
          f"iteration={task.iteration}, layer_id={task.layer_id}, item_id={task.item_id}")
```

### 8.2 验证输出

```python
from src.workload_format.schema import P2PWorkload

errors = workload.validate()
assert not errors, f"Validation failed: {errors}"
```

### 8.3 调度器使用 hints

```python
# 调度器可以基于 (iteration, layer_id, phase) 复现 C++ 顺序
phase_order = {
    Phase.FORWARD: 0,
    Phase.BACKWARD_INPUT: 1,
    Phase.BACKWARD_WEIGHT: 2,
}

sorted_tasks = sorted(
    workload.tasks,
    key=lambda t: (t.iteration, t.layer_id, phase_order.get(t.phase, 0))
)
```

---

## 9. 关键参考代码位置

| 位置                    | 说明                                                                                                         |
| ----------------------- | ------------------------------------------------------------------------------------------------------------ |
| `Workload.cc:700-790` | `iterate_hybrid_parallel_Transformer()` - 基础 Forward/IG/WG 状态机                                        |
| `Workload.cc:791-910` | `iterate_hybrid_parallel_Transformer_fwd_in_bckwd()` - 含 activation checkpointing                         |
| `Layer.cc:256-267`    | `get_fwd_pass_compute()` / `get_input_grad_compute()` / `get_weight_grad_compute()` - per-rank compute |
| `Layer.cc:1052-1231`  | `issue_forward_pass_comm()` - Blocking forward communication                                               |
| `Layer.cc:1232-1413`  | `issue_input_grad_comm()` - Blocking IG communication                                                      |
| `Layer.cc:1414-1594`  | `issue_weight_grad_comm()` - **Non-blocking** WG communication                                       |
| `Layer.hh:28-110`     | Layer 类定义 - fwd/ig/wg 三阶段字段                                                                          |

---

*开发时间：2026-04-10 ~ 2026-04-13*
*测试状态：21 passed (total 125)*
*设计路线：Route C（数据依赖 + 调度提示）*

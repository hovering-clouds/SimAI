# Task 3 重构计划：WorkloadBuilder 依赖模型修正

## 1. 背景

当前 `WorkloadBuilder` 的实现存在三个根本性设计缺陷，通过分析 astra-sim C++ 参考实现（`Workload.cc`, `Layer.cc`, `Layer.hh`）发现。本文档记录这些问题、正确的执行模型、以及重构方案。

---

## 2. 当前实现的三个核心问题

### 问题 1：Compute 任务应该是 per-rank，而非 per-group

**当前行为**：每个 item 的每个 phase（fwd/bwd/dp）只创建 **一个** compute task，分配给 `assigned_nodes[0]`。

**C++ 参考行为**：
- astra-sim 中每个 rank 有独立的 `Workload` 实例和 `Sys`（事件驱动模拟器）
- 每个 rank 独立执行自己的 compute task（`get_fwd_pass_compute()` 返回 AICB 文件中的 compute time）
- compute time 已经是 per-rank 的值（已按 TP 切分），不需要再除以 TP

**正确做法**：为参与该通信的所有 rank 各自创建 compute task。

示例：TP=2, DP=2 → 4 个 rank [0,1,2,3]
- forward ALLREDUCE (TP): TP group [0,1] 和 [2,3]
  - 创建 4 个 compute task：rank 0, 1, 2, 3 各一个
  - 展开 2 组 ALLREDUCE flows（组 0: [0,1]，组 1: [2,3]）

### 问题 2：依赖链应遵循 Forward 正序 / Backward 逆序

**当前行为**：按线性顺序处理所有 item，每个 item 内 fwd → bwd → dp 串行，item 之间也串行。

**C++ 参考行为**（`iterate_hybrid_parallel_Transformer_fwd_in_bckwd()` 状态机，Workload.cc 第 791-910 行）：

```
Forward_Pass (index++: layer 0 → N-1):
  layer[0].fwd_compute → layer[0].fwd_comm(blocking)
  layer[1].fwd_compute → layer[1].fwd_comm(blocking)
  ...
  layer[N-1].fwd_compute → layer[N-1].fwd_comm(blocking)
                                    ↓ (bridge)
Input_Gradient (从 layer[N-1] 开始):
  layer[N-1].ig_compute → layer[N-1].ig_comm(blocking)
      ↓
Weight_Gradient (同一层):
  layer[N-1].wg_compute → layer[N-1].wg_comm(non-blocking)
  wait for ig_comm_finished → index--
      ↓
Input_Gradient (继续逆序):
  layer[N-2].ig_compute → layer[N-2].ig_comm(blocking)
      ↓
  layer[N-2].wg_compute → layer[N-2].wg_comm(non-blocking)
  ...
  layer[0].ig → layer[0].wg → done, 进入下一个 GA step
```

**正确的依赖规则**（per-node 视角）：

| 规则 | 依赖边 | 说明 |
|------|--------|------|
| Forward 正序链 | `layer[i].fwd.last_task(R) → layer[i+1].fwd_compute(R)` | 每个 rank 正序前进 |
| Fwd→IG 桥接 | `layer[N-1].fwd.last_task(R) → layer[N-1].ig_compute(R)` | 最后一层完成后进入 backward |
| IG 逆序链 | `layer[i].ig.last_task(R) → layer[i-1].ig_compute(R)` | 反向传播逆序 |
| WG 同层依赖 | `layer[i].ig.last_task(R) → layer[i].wg_compute(R)` | IG 完成后做 WG |
| GA 步间桥接 | `layer[0].wg.last_task(R) → layer[0].fwd_compute(R)` (next GA) | 前一个 GA 完成后开始下一个 |

**关键**：IG→WG 和 IG→下一层 IG 形成分叉（diamond dependency），允许 WG 与下一层的 IG 并行，对应 C++ 中 WG comm 的 non-blocking 行为。

### 问题 3：跨 item 依赖应该是 per-node 的

**当前行为**：仅将下一个 item 的第一个 task 与上一个 item 的最后一个 task 建立单一全局依赖。

**正确行为**：每个 rank 在 item[i+1] 中的第一个 task，应该依赖于 **同一个 rank** 在 item[i] 中的最后一个 task。

原因：展开集合通信后，不同 rank 参与不同的 flow，完成时间不同。如果用单一全局依赖，快的 rank 被慢的 rank 阻塞，丢失了 per-node 的并行性。

---

## 3. 重构方案

### 3.1 整体架构：两阶段生成

```
Phase 1: 生成所有 tasks（无依赖）
  → 对每个 item，为所有 rank 创建 compute + flow tasks

Phase 2: 连接依赖关系
  → 按照 Forward/Backward/WG 顺序，per-node 地建立依赖边
```

### 3.2 Phase 1：Task 生成

#### 3.2.1 数据结构

```python
@dataclass
class FlowGroupResult:
    """一次集合通信展开的结果，含增量追踪的 receiver 索引。

    receiver_index 在生成 flow 的同时增量维护，避免后续扫描。
    key = rank (receiver/dst), value = 该 rank 作为 dst 接收到的所有 flow task_id 列表。

    设计理由：
    - 为什么只追踪 dst（receiver）而不追踪 src（sender）：
      一个 rank 的下一步计算只依赖它接收到的数据，不需要等自己完成发送。
      rank 收到了所有需要的数据就可以开始计算，自己的发送是为其他 rank 服务的。
    - 为什么是所有 receiver flows 而非仅最后一个：
      以 2-rank Ring ALLREDUCE 为例：
        Flow 0: 0→1, deps=[]        Flow 1: 1→0, deps=[]
        Flow 2: 0→1, deps=[Flow 1]  Flow 3: 1→0, deps=[Flow 0]
      rank 0 作为 receiver 收到 Flow 1 和 Flow 3。若仅依赖 Flow 3（最后一个），
      Flow 3 的依赖链是 Flow 3 ← Flow 0，不经过 Flow 1，rank 0 可能缺失 chunk 0 数据。
      因此必须依赖所有 receiver flows。
    - 效率：增量维护 O(1)/flow，无后续扫描。每个 rank 的 receiver flows 数为 O(n)（n=组大小）。
    """
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
```

```python
@dataclass
class ItemTasks:
    """一个 AICB work item 展开后的所有 tasks，按 phase 分组。"""
    # key = rank, value = FlowTask（每个 rank 的 compute task）
    fwd_computes: dict[int, FlowTask]
    ig_computes: dict[int, FlowTask]
    wg_computes: dict[int, FlowTask]
    # FlowGroupResult 含 flows + receiver_index
    fwd_result: FlowGroupResult
    ig_result: FlowGroupResult
    wg_result: FlowGroupResult
```

#### 3.2.2 Compute Task 生成逻辑

对于每个 item 的每个 phase，为 **所有 assigned_nodes** 创建 compute task：

```python
def _create_compute_tasks_for_phase(
    self,
    ranks: list[int],       # 所有 assigned_nodes
    duration_us: int,       # compute time (per-rank, from AICB)
    phase: Phase,
    layer_id: int,
    iteration: int,
    job_id: int,
    task_id_counter: int,   # 可变引用或返回新值
) -> dict[int, FlowTask]:
    """为每个 rank 创建一个 compute task。"""
    tasks = {}
    for rank in ranks:
        task = FlowTask(
            task_id=task_id_counter,
            job_id=job_id,
            type=TaskType.COMPUTE,
            node=rank,
            duration_us=duration_us,
            phase=phase,
            layer_id=layer_id,
            iteration=iteration,
        )
        tasks[rank] = task
        task_id_counter += 1
    return tasks, task_id_counter
```

#### 3.2.3 Flow Task 生成逻辑

对于每个通信操作，需要为 **每个并行子组** 分别展开。生成时同时维护 `FlowGroupResult`：

```python
def _expand_comm_all_groups(
    self,
    base_type: str,
    context: str,           # "tp", "dp", "ep", "dp_ep"
    comm_size: int,
    grouper: RankGrouper,
    phase: Phase,
    job_id: int,
    task_id_counter: int,
    algo: str = "ring",
) -> tuple[FlowGroupResult, int]:
    """为所有并行子组展开通信，同时增量构建 receiver_index。"""
    result = FlowGroupResult.empty()

    # 遍历所有并行子组
    for ranks in self._iter_subgroups(context, grouper):
        if len(ranks) >= 2:
            flows = self._call_expander(base_type, ranks, comm_size, ...)
            for flow in flows:
                flow.phase = phase
                result.add_flow(flow)  # 增量更新 receiver_index
            task_id_counter += len(flows)

    return result, task_id_counter

def _iter_subgroups(
    self, context: str, grouper: RankGrouper
) -> Iterator[list[int]]:
    """根据 context 遍历所有并行子组，yield 每个子组的 rank 列表。"""
    if context == "tp":
        for pp_idx in range(grouper.pp):
            for dp_idx in range(grouper.dp):
                for ep_idx in range(grouper.ep):
                    yield grouper.get_tp_group(pp_idx, dp_idx, ep_idx)
    elif context == "dp":
        for pp_idx in range(grouper.pp):
            for ep_idx in range(grouper.ep):
                for tp_idx in range(grouper.tp):
                    yield grouper.get_dp_group(pp_idx, ep_idx, tp_idx)
    elif context == "ep":
        for pp_idx in range(grouper.pp):
            for dp_idx in range(grouper.dp):
                for tp_idx in range(grouper.tp):
                    yield grouper.get_ep_group(pp_idx, dp_idx, tp_idx)
    elif context == "dp_ep":
        for pp_idx in range(grouper.pp):
            for tp_idx in range(grouper.tp):
                yield grouper.get_dp_ep_group(pp_idx, tp_idx)
```

#### 3.2.4 Compute → Flow 连接（同一 phase 内）

对于每个 rank 的 compute task → 该 rank 的 **所有** 发送流（src flow）。

rank 完成计算后才能开始发送数据，且一个 rank 可能同时发起多条独立的发送流（如 AlltoAll），
因此必须将 compute 连接到该 rank 的所有 src flow，而非仅第一个。

```python
def _wire_compute_to_flows(
    self,
    computes: dict[int, FlowTask],        # rank → compute task
    result: FlowGroupResult,               # 含 flows
):
    """将每个 rank 的 compute task 连接到该 rank 的所有发送流（src flow）。"""
    for flow in result.flows:
        if flow.src in computes:
            flow.deps.append(computes[flow.src].task_id)
```

**为什么不只连接第一个 src flow**：

以 AlltoAll 为例，rank 0 同时向 rank 1, 2, 3 发送 3 条独立 flow（无互相依赖）。
若仅连接 `compute(0) → Flow(0→1)`，则 Flow(0→2) 和 Flow(0→3) 缺少 compute 依赖，
理论上可在 compute 完成前开始发送。

对于 Ring 模式，后续 src flow 已通过 ring 依赖链间接依赖了前面的 flow，
多加一条 compute 边是冗余但无害的。统一使用"所有 src flow"简化了逻辑且对所有模式正确。

### 3.3 Phase 2：依赖关系连接

#### 3.3.1 核心算法

```python
def _wire_dependencies(
    self,
    item_tasks_list: list[ItemTasks],    # 所有 items 的 tasks
    num_layer_items: int,
    num_pre_items: int,
    header: AicbHeader,
):
    """
    按照 Forward/Backward/WG 的正确顺序连接依赖。

    对于一个 GA step 内的 layers [0, 1, ..., N-1]：
      Forward:  layer[0].fwd → layer[1].fwd → ... → layer[N-1].fwd
      Bridge:   layer[N-1].fwd → layer[N-1].ig
      IG reverse: layer[N-1].ig → layer[N-2].ig → ... → layer[0].ig
      WG same layer: layer[i].ig → layer[i].wg
      GA bridge: layer[0].wg(GA=k) → layer[0].fwd(GA=k+1)
    """
    # 将 items 按 GA step 分组
    ga_groups = self._group_items_by_ga(
        item_tasks_list, num_pre_items, num_layer_items, header.vpp
    )

    for ga_idx, ga_group in enumerate(ga_groups):
        num_layers = len(ga_group)

        # --- 1. Forward chain (正序 0 → N-1) ---
        for i in range(num_layers - 1):
            self._wire_per_node_phase_transition(
                src_result=ga_group[i].fwd_result,           # 当前层的 fwd flows
                src_computes=ga_group[i].fwd_computes,        # 当前层的 fwd computes（无通信时回退用）
                dst_computes=ga_group[i+1].fwd_computes,      # 下一层的 fwd computes
            )

        # --- 2. Fwd→IG bridge (最后一层) ---
        last_layer = ga_group[num_layers - 1]
        self._wire_per_node_phase_transition(
            src_result=last_layer.fwd_result,
            src_computes=last_layer.fwd_computes,
            dst_computes=last_layer.ig_computes,
        )

        # --- 3. IG reverse chain (逆序 N-1 → 0) + WG same layer ---
        for i in range(num_layers - 1, -1, -1):
            # IG→WG same layer
            self._wire_per_node_phase_transition(
                src_result=ga_group[i].ig_result,
                src_computes=ga_group[i].ig_computes,
                dst_computes=ga_group[i].wg_computes,
            )
            # IG→previous layer IG (reverse)
            if i > 0:
                self._wire_per_node_phase_transition(
                    src_result=ga_group[i].ig_result,
                    src_computes=ga_group[i].ig_computes,
                    dst_computes=ga_group[i-1].ig_computes,
                )

        # --- 4. GA bridge ---
        if ga_idx < len(ga_groups) - 1:
            next_ga_group = ga_groups[ga_idx + 1]
            self._wire_per_node_phase_transition(
                src_result=ga_group[0].wg_result,                # 当前 GA 第一层的 WG flows
                src_computes=ga_group[0].wg_computes,
                dst_computes=next_ga_group[0].fwd_computes,      # 下一个 GA 的第一层 fwd computes
            )
```

#### 3.3.2 Per-node Phase Transition

```python
def _wire_per_node_phase_transition(
    self,
    src_result: FlowGroupResult,             # 源 phase 的 flows + receiver_index
    src_computes: dict[int, FlowTask],        # 源 phase 的 computes (可能为空)
    dst_computes: dict[int, FlowTask],        # 目标 phase 的 computes (rank → task)
):
    """
    为每个 rank 建立跨 phase 依赖。

    规则：rank R 的 dst_compute 依赖 src_result 中 R 作为 receiver (dst) 的所有 flow。
    原因：rank 只需等接收完数据即可开始计算，不需要等自己完成发送。

    如果 src_result 为空（无通信），回退到 compute → compute 直连。
    """
    if not src_result.flows:
        # 无通信：直接 compute(R) → compute(R)
        for rank, dst_compute in dst_computes.items():
            if rank in src_computes:
                dst_compute.deps.append(src_computes[rank].task_id)
        return

    # 有通信：依赖该 rank 作为 receiver 的所有 flow
    for rank, dst_compute in dst_computes.items():
        received_ids = src_result.receiver_index.get(rank, [])
        dst_compute.deps.extend(received_ids)
```

**设计决策说明**：

1. **为什么依赖所有 receiver flows 而非仅最后一个**：

   以 2-rank Ring ALLREDUCE 为例（TP=2 常见场景）：
   ```
   Flow 0: 0→1, deps=[]        Flow 1: 1→0, deps=[]
   Flow 2: 0→1, deps=[Flow 1]  Flow 3: 1→0, deps=[Flow 0]
   ```
   rank 0 作为 receiver 收到 Flow 1（chunk 0）和 Flow 3（chunk 1）。
   若仅依赖 Flow 3（最后一个），其依赖链为 Flow 3 ← Flow 0，不经过 Flow 1。
   Flow 3 完成时 Flow 1 可能尚未完成，rank 0 缺失 chunk 0 数据。

   依赖所有 receiver flows 可以保证正确性。由于 ring 模式下每个 rank 的 receiver flows 数为 O(n)（n=组大小），依赖边数量可接受。

2. **为什么是 receiver (dst) 而非 sender (src)**：

   rank 的下一步计算只需要接收完其他 rank 发来的数据即可开始。
   rank 自己的发送是为其他 rank 服务的，不需要等自己发完。
   发送的完成由 ring 依赖链中的下一个 rank 的接收来保证。

### 3.4 Pre/Post layer items 的处理

Pre-layer items（`grad_gather`, `grad_param_comm` 等）和 Post-layer items（`embedding_norm`, `optimizer` 等）不属于 GA 循环内的层结构，它们的依赖处理：

- **Pre-layer items**：按线性顺序串行处理，最后一项的输出 → 第一个 GA step 的第一层 forward compute
- **Post-layer items**：按线性顺序串行处理，最后一个 GA step 的最后一层 WG 输出 → 第一个 post-item

### 3.5 无通信场景的处理

当某个 phase 的 comm 为 `NONE` 时：
- 仍然创建 compute tasks（所有 rank）
- `FlowGroupResult` 为空（无 flows，receiver_index 为空 dict）
- `_wire_per_node_phase_transition` 检测到 flows 为空后，自动回退到 compute(R) → compute(R) 直连

---

## 4. 依赖关系示例

### 4.1 示例：3 layers, TP=2, GA=1, 4 GPUs [0,1,2,3]

TP groups: [0,1], [2,3]

```
Forward chain (正序):

  Layer 0:
    rank 0: compute(fwd) ─┐
    rank 1: compute(fwd) ─┤
    rank 2: compute(fwd) ─┤
    rank 3: compute(fwd) ─┘
                           ↓
    TP group [0,1]: ALLREDUCE flows
    TP group [2,3]: ALLREDUCE flows
                           ↓
  Layer 1: (同上结构)
    rank 0: compute(fwd) ← layer[0] flows 中 rank 0 参与的最后一个
    rank 1: compute(fwd) ← layer[0] flows 中 rank 1 参与的最后一个
    rank 2: compute(fwd) ← layer[0] flows 中 rank 2 参与的最后一个
    rank 3: compute(fwd) ← layer[0] flows 中 rank 3 参与的最后一个
    ...
                           ↓
  Layer 2:
    rank 0-3: compute(fwd) → ALLREDUCE flows
                           ↓
Bridge (最后一层 fwd → 最后一层 ig):
  Layer 2:
    rank 0: compute(ig) ← layer[2] fwd flows 中 rank 0 的最后 flow
    rank 1: compute(ig) ← layer[2] fwd flows 中 rank 1 的最后 flow
    ...

IG reverse chain (逆序):

  Layer 2 → Layer 1:
    rank 0: compute(ig, layer 1) ← layer[2] ig flows 中 rank 0 的最后 flow
    rank 1: compute(ig, layer 1) ← layer[2] ig flows 中 rank 1 的最后 flow
    ...

  Layer 1 → Layer 0:
    rank 0: compute(ig, layer 0) ← layer[1] ig flows 中 rank 0 的最后 flow
    ...

WG same layer (IG → WG, 与 IG reverse 并行):

  Layer 2: rank 0: compute(wg) ← layer[2] ig flows 中 rank 0 的最后 flow
  Layer 1: rank 0: compute(wg) ← layer[1] ig flows 中 rank 0 的最后 flow
  Layer 0: rank 0: compute(wg) ← layer[0] ig flows 中 rank 0 的最后 flow
```

### 4.2 DAG 依赖图（单 rank 视角）

```
layer[0].fwd_compute(R) → layer[0].fwd_flows → layer[1].fwd_compute(R)
  → layer[1].fwd_flows → layer[2].fwd_compute(R) → layer[2].fwd_flows
  → layer[2].ig_compute(R) → layer[2].ig_flows ─┬→ layer[2].wg_compute(R)
                                                  └→ layer[1].ig_compute(R)
  → layer[1].ig_flows ─┬→ layer[1].wg_compute(R)
                        └→ layer[0].ig_compute(R)
  → layer[0].ig_flows ─┬→ layer[0].wg_compute(R)
                        └→ (GA bridge to next step, or end)
```

---

## 5. 实现步骤

### Step 1：重构 `_create_compute_task` → `_create_compute_tasks_for_phase`

- 输入：所有 assigned_nodes + duration + phase info
- 输出：`dict[int, FlowTask]`（rank → compute task）
- 为每个 rank 创建一个 compute task

### Step 2：重构 `_expand_comm` → `_expand_comm_all_groups`

- 遍历所有并行子组（所有 pp_idx × dp_idx × ep_idx × tp_idx 组合）
- 为每个子组展开通信
- 连接 compute → flow（每个 rank 的 compute → 该 rank 的第一个 src flow）

### Step 3：添加 `_wire_per_node_phase_transition`

- Per-node 依赖查找：找到 rank 在 flows 中最后参与的 flow
- 连接：last_flow_for_rank(R) → next_phase_compute(R)

### Step 4：重构 `build_from_aicb` 主流程

Phase 1（生成 tasks）：
```python
item_tasks_list = []
task_id_counter = 0

for item_idx, item in enumerate(aicb_items):
    item_tasks = ItemTasks()

    # Forward phase
    item_tasks.fwd_computes, task_id_counter = self._create_compute_tasks_for_phase(...)
    if item.forward_comm != "NONE":
        item_tasks.fwd_flows, task_id_counter = self._expand_comm_all_groups(...)
        self._wire_compute_to_flows(item_tasks.fwd_computes, item_tasks.fwd_flows)

    # Backward (IG) phase
    item_tasks.ig_computes, task_id_counter = self._create_compute_tasks_for_phase(...)
    if item.backward_comm != "NONE":
        item_tasks.ig_flows, task_id_counter = self._expand_comm_all_groups(...)
        self._wire_compute_to_flows(item_tasks.ig_computes, item_tasks.ig_flows)

    # DP (WG) phase
    item_tasks.wg_computes, task_id_counter = self._create_compute_tasks_for_phase(...)
    if item.dp_comm != "NONE":
        item_tasks.wg_flows, task_id_counter = self._expand_comm_all_groups(...)
        self._wire_compute_to_flows(item_tasks.wg_computes, item_tasks.wg_flows)

    item_tasks_list.append(item_tasks)
```

Phase 2（连接依赖）：
```python
self._wire_dependencies(item_tasks_list, num_layer_items, num_pre_items, header)
```

### Step 5：更新测试

需要更新的测试：
- `test_single_layer_allreduce`：验证所有 rank 都有 compute tasks
- `test_dependency_chain_forward_to_backward`：验证 per-node fwd → bwd 依赖
- `test_dag_no_cycles`：验证新 DAG 无环
- 新增：`test_forward_chain_per_node`：验证跨层 per-node 依赖
- 新增：`test_backward_reverse_order`：验证 backward 逆序依赖
- 新增：`test_wg_ig_parallel`：验证 WG 和下一层 IG 的分叉依赖

---

## 6. 关键参考代码位置

| 位置 | 说明 |
|------|------|
| `Workload.cc:700-790` | `iterate_hybrid_parallel_Transformer()` - 基础 Forward/IG/WG 状态机 |
| `Workload.cc:791-910` | `iterate_hybrid_parallel_Transformer_fwd_in_bckwd()` - 含 fwd_in_bckwd 的完整状态机 |
| `Layer.cc:256-267` | `get_fwd_pass_compute()` / `get_input_grad_compute()` / `get_weight_grad_compute()` - per-rank compute |
| `Layer.cc:1052-1231` | `issue_forward_pass_comm()` - Blocking forward communication |
| `Layer.cc:1232-1413` | `issue_input_grad_comm()` - Blocking IG communication |
| `Layer.cc:1414-1594` | `issue_weight_grad_comm()` - **Non-blocking** WG communication |
| `Layer.hh:28-110` | Layer 类定义 - fwd/ig/wg 三阶段字段 |

---

## 7. 风险与注意事项

| 风险 | 缓解 |
|------|------|
| Per-node 依赖查找效率 | 使用 `FlowGroupResult.receiver_index` 在生成 flow 时增量维护，O(1)/flow，无后续扫描 |
| 大规模 workload 任务数量膨胀 | 每个 rank 都有 compute task，TP=2 时 compute task 数翻倍。但这是正确模型，不可避免 |
| 无通信 phase 的依赖回退 | 当 flows 为空时，直接用 computes 作为 src 进行 per-node 连接 |
| Pre/Post items 不参与 Fwd/Bwd 逆序 | Pre items 按线性处理，输出馈入 GA 循环；Post items 接收 GA 循环输出 |
| GA 分组逻辑 | `(item_idx - num_pre_items) // vpp` 得到 ga_step，`% vpp` 得到 layer_idx_in_step |

---

*创建时间：2026-04-10*
*参考实现：astra-sim Workload.cc / Layer.cc*
*状态：待审查*

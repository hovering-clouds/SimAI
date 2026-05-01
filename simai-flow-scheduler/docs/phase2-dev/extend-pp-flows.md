# Phase 2 扩展：Pipeline Parallelism (PP) 通信流实现计划

## Context

当前 `WorkloadBuilder` 在 `pp > 1` 时未建模 PP stage 间的数据依赖。所有 PP stage 的 rank 独立执行全部 layer，没有跨 stage 的激活/梯度传输。

C++ 参考实现（astra-sim）对 PP 的处理方式是纯解析公式：`PP_time = 2 * vpp * GA * (pp_comm_size / bandwidth)`，`MockNcclGroup` 中 PP group 初始化为空。这意味着没有现成的 P2P PP 流实现可以参考，需要从 LLM 训练的实际数据流推导依赖关系。

**目标**：在 Phase 2 workload builder 中生成 PP stage 间的 P2P 流任务，并正确连接数据依赖。

---

## PP 数据依赖模型

### Forward PP（激活传递）

```
Stage 0: L0.fwd → L1.fwd → ... → L{vpp-1}.fwd ──PP_SEND──→ Stage 1: L0.fwd → ...
Stage 1: ...                            L{vpp-1}.fwd ──PP_SEND──→ Stage 2: L0.fwd → ...
```

- 每个 GA step 中，stage k 的 L{vpp-1}.fwd 完成后，将 activation 发送给 stage k+1
- Stage k+1 的 L0.fwd 依赖接收到的 PP flow
- PP flow 在 (dp_idx, ep_idx, tp_idx) 相同的 rank 间传输（TP rank 发送给对应的 TP rank）

### Backward PP（梯度传递）

```
Stage 2: L{vpp-1}.ig → ... → L0.ig ──PP_SEND──→ Stage 1: L{vpp-1}.ig → ... → L0.ig ──PP_SEND──→ Stage 0: L{vpp-1}.ig → ...
```

- Stage k+1 的 L0.ig（backward chain 的最后一个 compute）产生对 stage k activation 的梯度
- 梯度通过 PP flow 发送回 stage k
- Stage k 的 L{vpp-1}.ig（backward chain 的第一个 compute）接收该梯度
- Stage k 的 L{vpp-1}.ig 有**两个**依赖来源：(a) fwd→ig bridge（已由现有代码通过 post items bridge 隐式处理）+ (b) backward PP flow

### 关键约束

- vpp 已经是 per-stage 的层数（AICB 文件中 `vpp = model_total_layers / pp`）
- 所有 PP stage 共享相同的 AICB items，每个 stage 在自己的 ranks 上执行
- PP flow 在相同 (dp, ep, tp) 位置的 rank 间传输：`src = stage_k[dp,ep,tp]`, `dst = stage_{k+1}[dp,ep,tp]`
- PP flow 数量 = `ga_steps × (pp-1) × (dp × ep × tp) × 2`（forward + backward）

---

## 实现方案

### 修改文件

- `src/workload_generator/workload_builder.py` — 新增 PP 流生成和依赖连接方法
- `src/workload_generator/rank_grouper.py` — 新增 `get_pp_rank` helper 方法

### 新增数据结构

```python
@dataclass
class PPFlowResult:
    """PP flow generation result, indexed for wiring."""
    # (ga_idx, pp_boundary_idx) → sender_rank → flow
    forward_flows: dict[tuple[int, int], dict[int, FlowTask]]
    backward_flows: dict[tuple[int, int], dict[int, FlowTask]]
    all_flows: list[FlowTask]
```

### 修改 `build_from_aicb`（主流程）

在现有 Phase 1 和 Phase 2 之间插入 PP 流生成和连接：

```python
def build_from_aicb(self, aicb_header, aicb_items, job, comm_algo="ring"):
    # ... 现有 Phase 1（生成所有 item tasks）...

    # ===== Phase 1.5: 生成 PP flow tasks =====
    pp_result = None
    if grouper.pp > 1 and aicb_header.pp_comm_size > 0:
        ga_groups = self._group_items_by_ga(
            item_tasks_list, num_pre_items, num_layer_items, items_per_ga)
        pp_result, task_id_counter = self._generate_pp_flows(
            grouper, aicb_header, ga_groups, items_per_ga,
            job.job_id, task_id_counter)
        all_flow_tasks.extend(pp_result.all_flows)

    # ===== Phase 2: Wire dependencies =====
    self._wire_dependencies(
        item_tasks_list, num_layer_items, num_pre_items, items_per_ga)

    # ===== Phase 2.5: Wire PP dependencies =====
    if pp_result is not None:
        self._wire_pp_dependencies(
            pp_result, item_tasks_list, grouper,
            num_pre_items, num_layer_items, items_per_ga)

    # Build P2PWorkload（不变）
    ...
```

### 新增方法 `_generate_pp_flows`

为每个 GA step、每个 PP 相邻 stage 对、每个 (dp, ep, tp) 位置生成 forward 和 backward PP flow：

```python
def _generate_pp_flows(
    self, grouper, aicb_header, ga_groups, items_per_ga,
    job_id, task_id_counter,
) -> tuple[PPFlowResult, int]:
    forward_pp: dict[tuple[int, int], dict[int, FlowTask]] = {}
    backward_pp: dict[tuple[int, int], dict[int, FlowTask]] = {}
    all_flows: list[FlowTask] = []

    for ga_idx, ga_group in enumerate(ga_groups):
        for pp_boundary in range(grouper.pp - 1):
            fwd_flows = {}
            bwd_flows = {}

            for dp_idx in range(grouper.dp):
                for ep_idx in range(grouper.ep):
                    for tp_idx in range(grouper.tp):
                        src_rank = grouper.get_pp_rank(pp_boundary, dp_idx, ep_idx, tp_idx)
                        dst_rank = grouper.get_pp_rank(pp_boundary + 1, dp_idx, ep_idx, tp_idx)

                        # Forward PP flow: stage k → stage k+1
                        fwd_flow = FlowTask(
                            task_id=task_id_counter,
                            job_id=job_id,
                            type=TaskType.FLOW,
                            src=src_rank, dst=dst_rank,
                            size_bytes=aicb_header.pp_comm_size,
                            comm_type=CommType.PP_SEND,
                            phase=Phase.FORWARD,
                            layer_id=items_per_ga - 1,  # last layer boundary
                            iteration=ga_idx,
                        )
                        fwd_flows[src_rank] = fwd_flow
                        all_flows.append(fwd_flow)
                        task_id_counter += 1

                        # Backward PP flow: stage k+1 → stage k
                        bwd_flow = FlowTask(
                            task_id=task_id_counter,
                            job_id=job_id,
                            type=TaskType.FLOW,
                            src=dst_rank, dst=src_rank,  # reversed direction
                            size_bytes=aicb_header.pp_comm_size,
                            comm_type=CommType.PP_SEND,
                            phase=Phase.BACKWARD_INPUT,
                            layer_id=0,  # first layer boundary
                            iteration=ga_idx,
                        )
                        bwd_flows[dst_rank] = bwd_flow
                        all_flows.append(bwd_flow)
                        task_id_counter += 1

            forward_pp[(ga_idx, pp_boundary)] = fwd_flows
            backward_pp[(ga_idx, pp_boundary)] = bwd_flows

    return PPFlowResult(
        forward_flows=forward_pp,
        backward_flows=backward_pp,
        all_flows=all_flows,
    ), task_id_counter
```

注意：需要给 `RankGrouper` 新增一个 helper 方法 `get_pp_rank(pp_idx, dp_idx, ep_idx, tp_idx)` 来获取特定 PP stage 中特定 (dp, ep, tp) 位置的 rank：

```python
# rank_grouper.py 新增
def get_pp_rank(self, pp_idx, dp_idx, ep_idx, tp_idx) -> int:
    """Get the rank at a specific (pp, dp, ep, tp) position."""
    return self.nodes[
        pp_idx * (self.dp * self.ep * self.tp) +
        dp_idx * (self.ep * self.tp) +
        ep_idx * self.tp +
        tp_idx
    ]
```

### 新增方法 `_wire_pp_dependencies`

使用 receiver-based 模型连接 PP 流的依赖：

```python
def _wire_pp_dependencies(
    self, pp_result, item_tasks_list, grouper,
    num_pre_items, num_layer_items, items_per_ga,
):
    ga_groups = self._group_items_by_ga(
        item_tasks_list, num_pre_items, num_layer_items, items_per_ga)

    for ga_idx, ga_group in enumerate(ga_groups):
        last_item = ga_group[-1]   # L{vpp-1}
        first_item = ga_group[0]   # L[0]

        for pp_boundary in range(grouper.pp - 1):
            fwd_flows = pp_result.forward_flows[(ga_idx, pp_boundary)]
            bwd_flows = pp_result.backward_flows[(ga_idx, pp_boundary)]

            for dp_idx in range(grouper.dp):
                for ep_idx in range(grouper.ep):
                    for tp_idx in range(grouper.tp):
                        src_rank = grouper.get_pp_rank(pp_boundary, dp_idx, ep_idx, tp_idx)
                        dst_rank = grouper.get_pp_rank(pp_boundary + 1, dp_idx, ep_idx, tp_idx)

                        # ── Forward PP wiring ──
                        fwd_pp_flow = fwd_flows[src_rank]
                        # Sender dep: receiver-based — last layer's fwd output → PP flow
                        self._wire_to_flow_sender(
                            fwd_pp_flow, last_item.fwd_computes,
                            last_item.fwd_result, src_rank)
                        # Receiver dep: PP flow → first layer's fwd compute
                        first_item.fwd_computes[dst_rank].deps.append(fwd_pp_flow.task_id)

                        # ── Backward PP wiring ──
                        bwd_pp_flow = bwd_flows[dst_rank]  # sender is dst_rank (stage k+1)
                        # Sender dep: receiver-based — first layer's IG output → PP flow
                        self._wire_to_flow_sender(
                            bwd_pp_flow, first_item.ig_computes,
                            first_item.ig_result, dst_rank)
                        # Receiver dep: PP flow → last layer's IG compute
                        last_item.ig_computes[src_rank].deps.append(bwd_pp_flow.task_id)
```

### 新增辅助方法 `_wire_to_flow_sender`

将 receiver-based 模型封装为通用的 "source output → flow sender" 连接：

```python
def _wire_to_flow_sender(
    self, flow, src_computes, src_result, sender_rank,
):
    """Wire source phase output to a flow's sender side.

    Receiver-based: if source has comm flows, depend on flows where
    sender_rank is receiver (dst); otherwise depend on compute directly.
    """
    if src_result.flows:
        received_ids = src_result.receiver_index.get(sender_rank, [])
        flow.deps.extend(received_ids)
    else:
        if sender_rank in src_computes:
            flow.deps.append(src_computes[sender_rank].task_id)
```

---

## 依赖关系总结

以 pp=2, tp=2, dp=1, ep=1, vpp=2, ga=1 为例（ranks: stage0=[0,1], stage1=[2,3]）：

```
Forward:
  rank 0: L0.fwd → L1.fwd ──PP_SEND(0→2)──→ L0.fwd → L1.fwd
  rank 1: L0.fwd → L1.fwd ──PP_SEND(1→3)──→ L0.fwd → L1.fwd
  rank 2:                      ↑ wait        L0.fwd → L1.fwd
  rank 3:                      ↑ wait        L0.fwd → L1.fwd

Backward:
  rank 0: L1.ig ← L0.ig ←──PP_SEND(2→0)── L1.ig ← L0.ig
  rank 1: L1.ig ← L0.ig ←──PP_SEND(3→1)── L1.ig ← L0.ig
  rank 2:                      ↑ wait        L1.ig ← L0.ig
  rank 3:                      ↑ wait        L1.ig ← L0.ig
```

Key deps:
- Forward PP: `L1.fwd(0) → PP_SEND(0→2) → L0.fwd(2)` and `L1.fwd(1) → PP_SEND(1→3) → L0.fwd(3)`
- Backward PP: `L0.ig(2) → PP_SEND(2→0) → L1.ig(0)` and `L0.ig(3) → PP_SEND(3→1) → L1.ig(1)`

---

## 数据依赖 vs 资源约束

PP 场景下有两种不同性质的约束，分属不同阶段处理：

| 约束类型 | 性质 | 处理阶段 |
|---------|------|---------|
| 跨 stage 激活/梯度传递 | 数据依赖（stage k+1 的计算需要 stage k 的输出） | Phase 2（本文档） |
| 激活内存限制 | 资源约束（activation 占用显存，必须等 backward 消费完才能释放） | **Phase 3（待扩展）** |
| TP/DP/EP 集合通信 | 数据依赖 | Phase 2（已完成） |

### 激活内存约束详解（Phase 3 待扩展）

当 `GA > 1` 且 `pp > 1` 时，存在一个重要的**资源约束**：

- Stage k 完成 GA[0] 的 forward 后保存了 activation（用于后续 backward 计算 gradient）
- 该 activation 在 GA[0] 的 backward 完成前不能释放
- 如果 stage k 接着做 GA[1] 的 forward，需要存储新的 activation，但旧 activation 仍占用内存
- 因此 **GA[1].fwd 必须等 GA[0].bwd 完成（释放旧 activation）才能开始**

这不是数据依赖（GA[1] 不需要 GA[0].bwd 的任何计算结果），而是内存容量限制。如果内存无限大，所有 GA 的 forward 可以完全并行（GPipe 方式）。

1F1B pipeline schedule 本质上就是用这个资源约束来编排执行顺序：

```
1F1B Schedule (stage k, pp=4, GA=8):
  F0 → F1 → F2 → F3 → B0 → F4 → B1 → F5 → B2 → F6 → B3 → F7 → B4 → B5 → B6 → B7
  ↑ warmup (pp-1 forwards)  ↑ steady state (1F1B)              ↑ cooldown
```

**当前 Phase 3 的 `analytical.py` 使用 `compute_cursor` 做 per-node 线性序列化，尚未实现 pipeline schedule。** 后续需要在 Phase 3 中扩展：
- 增加 activation 内存模型（per-rank activation 存储计数）
- 实现 1F1B / interleaved / V-shape 等 pipeline 调度策略
- 将 forward/backward 的启动条件从纯数据依赖扩展为「数据就绪 AND 内存可用」

---

## 已知简化

1. **Pre/Post items**：所有 PP stage 执行相同的 pre/post items，无跨 stage PP 流。Pre/post items 在各 stage 内独立运行。这可能导致 stage 1+ 的 embedding 等操作被冗余计算，但作为初始实现可接受。

2. **PP flow size**：使用 `pp_comm_size` 作为每个 PP flow 的大小。`pp_comm_size` 由 AICB 生成器计算：`2 * micro_batch * seq_len * hidden_size`，默认不除以 TP（per-rank 完整大小）；仅开启 sequence parallel 时才除以 TP。workload 文件中读出的值可直接使用。

3. **Pipeline schedule**：当前建模为 strict pipeline（所有 stage 串行），未实现 1F1B 或 interleaved schedule。Phase 3 task serializer 可在此基础上编排更复杂的调度策略。

---

## 待研究：Pre/Post items 在 PP 场景下的语义（待验证）

基于对 `example/workload_analytical.txt`（pp=12, vpp=8, ga=24, tp=2, ep=16）的分析，pre/post items 的实际行为比当前简化假设更复杂。

### Pre-items（仅出现 1 次，在 workload 开头）

| 名称 | 实际操作 | 是否每个 stage 都执行？ |
|------|---------|---------------------|
| `grad_gather` | dp_comm=ALLGATHER (44GB)，梯度收集 | 待确认 |
| `grad_param_comm` | dp_comm=REDUCESCATTER (88GB)，梯度参数通信 | 待确认 |
| `grad_param_compute` | bwd_compute=29700224，梯度参数计算 | 待确认 |
| `embedding_grads` | bwd_comm=ALLREDUCE (100MB)，embedding 梯度 | 待确认 |
| `moe_grad_norm1` | dp_comm=ALLGATHER_DP_EP (19GB)，MoE 梯度归一化 | 待确认 |
| `moe_grad_norm2` | dp_comm=REDUCESCATTER_DP_EP (38GB)，MoE 梯度归一化 | 待确认 |

这些 pre-items 几乎全是 backward/dp 阶段的操作（fwd_compute≈0），与梯度同步和 MoE 梯度归一化有关。当前所有 PP stage 的所有 rank 都会为这些 item 创建任务，但实际中某些操作可能只涉及特定 stage。

### embedding_layer / final_column：嵌入在 layer items 中

这两个操作**不在** pre/post items 中，而是**以旋转模式嵌入 GA 循环**：

- `embedding_layer`：出现在**每个 GA step 的开头**（GA[0] pos=0, GA[1] pos=1, GA[2] pos=2, ...）——对应 stage 0 的 embedding 操作
- `final_column`：以类似旋转模式出现在 GA[1..23] 中（GA[1] pos=0, GA[2] pos=1, ...）——对应最后一个 PP stage 的输出层

```
GA[0]: [embedding_layer] [attn+mlp]×8
GA[1]: [final_column] [embedding_layer] [attn+mlp]×8
GA[2]: [mlp] [final_column] [embedding_layer] [attn+mlp]×8
GA[3]: [mlp] [mlp] [final_column] [embedding_layer] [attn+mlp]×8
...
```

这是 AICB 生成器对 **interleaved pipeline schedule** 的建模方式。embedding 和 final_column 被交错放置在不同 GA step 中，使不同 stage 的操作在时间上重叠。

**注意**：当前 `_count_pre_items` / `_count_post_items` 不会将 `embedding_layer` 和 `final_column` 识别为 pre/post items（它们不在 `PRE_LAYER_NAMES` / `POST_LAYER_NAMES` 中），因此它们已被正确归入 layer items，由 forward/backward chain 处理。但 `_count_post_items` 会将出现在 workload 末尾的 `final_column`（最后 1 个）识别为 post item，需要确认这是否符合预期。

### Post-items（仅出现 1 次，在 workload 末尾）

| 名称 | 操作 |
|------|------|
| `final_column` | fwd_comm=ALLGATHER, bwd_comm=REDUCESCATTER (TP 通信) |
| `cross_entropy`×3 | fwd_comm=ALLREDUCE (16KB) |
| `optimizer`×4 | fwd_comm=ALLREDUCE (4 bytes) |

cross_entropy 和 optimizer 的 ALLREDUCE 大小极小，是损失计算和优化器步骤，只在所有 GA 和 PP 完成后执行一次。所有 stage 参与 ALLREDUCE 是合理的。

### 需要后续验证的问题

1. Pre-items 中的 DP 通信操作（`grad_gather`, `grad_param_comm`）是否每个 PP stage 都参与？还是只在特定 stage？
2. Pre-items 中的 `embedding_grads` ALLREDUCE 是否只在 stage 0 执行？
3. `final_column` 出现在 workload 末尾（post-items 区域）的 1 个实例被 `_count_post_items` 识别为 post item，这与其他 GA step 中嵌入在 layer items 里的 `final_column` 是否应该有不同处理？
4. interleaved pipeline schedule 对 PP flow 生成的影响：当前 PP flow 只连接最后一个 layer → 第一个 layer，但 embedding_layer 和 final_column 的旋转位置暗示更复杂的 stage 边界？

---

## 测试计划

### 测试文件
`tests/test_workload_builder.py` — 新增 PP 相关测试

### 测试用例

1. **`test_pp_no_flows_when_pp1`**：pp=1 时不生成任何 PP flow
2. **`test_pp_forward_sender_dep`**：验证 forward PP flow 依赖 stage k 的 L{vpp-1}.fwd
3. **`test_pp_forward_receiver_dep`**：验证 stage k+1 的 L0.fwd 依赖 forward PP flow
4. **`test_pp_backward_sender_dep`**：验证 backward PP flow 依赖 stage k+1 的 L0.ig
5. **`test_pp_backward_receiver_dep`**：验证 stage k 的 L{vpp-1}.ig 依赖 backward PP flow
6. **`test_pp_multi_stage`**：pp=3 时验证所有相邻 stage 对之间都有 PP flow
7. **`test_pp_dag_no_cycles`**：验证完整依赖图无环
8. **`test_pp_with_layer_comm`**：layer 有 AllReduce 时，PP flow 依赖 receiver flows 而非 compute
9. **`test_pp_ga_iterations`**：ga=2 时验证每个 GA step 都有独立的 PP flow

---

## 验证步骤

1. `uv run pytest tests/test_workload_builder.py -v` — 所有 PP 测试通过
2. `uv run pytest tests/ -v` — 回归测试全部通过
3. 用 `example/workload_analytical.txt`（pp=12）端到端运行，检查生成结果中 PP flow 数量 = `ga × (pp-1) × (dp × ep × tp) × 2`
4. 检查 DAG 无环：对生成的任务做拓扑排序验证

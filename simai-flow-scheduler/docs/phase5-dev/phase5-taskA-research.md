# Phase 5 Task A: Vidur 调研报告

## 目标

深入调研 Vidur-Alibabacloud 的推理 batch 调度流程，确认可获取的 per-batch 信息，确定插桩点和可捕获的数据字段，为最终 trace JSON 格式提供设计依据。

---

## 1. Vidur 整体架构

### 1.1 入口和主循环

Vidur 使用事件驱动架构，核心入口：

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| 入口函数 | [`vidur/main.py`](../../vidur-alibabacloud/vidur/main.py#L6-L12) | 6-12 | `main()` → 创建 Simulator → `simulator.run()` |
| 事件循环 | [`vidur/simulator.py`](../../vidur-alibabacloud/vidur/simulator.py#L67-L108) | 67-108 | `Simulator.run()` — 优先队列 `(time, id, event_type)`，循环处理事件 |

主循环核心逻辑（`simulator.py:76-103`）：
```python
while self._event_queue and not self._terminate:
    event = heapq.heappop(self._event_queue)
    self._time = event.time
    new_events = event.handle_event()
    for new_event in new_events:
        heapq.heappush(self._event_queue, new_event)
```

### 1.2 完整事件链

```
RequestArrivalEvent
    ↓
GlobalScheduleEvent   ← 全局调度：分配 P/D 节点
    ↓
ReplicaScheduleEvent  ← 本地调度：组 batch
    ↓
BatchStageArrivalEvent (stage 0)
    ↓
ReplicaStageScheduleEvent  ← 获取执行时间，创建 BatchStage
    ↓
BatchStageEndEvent    ← 单个 PP stage 完成
    ↓ (非最后 stage)
BatchStageArrivalEvent (next stage)
    ↓ (最后 stage)
BatchEndEvent         ← batch 完成，触发下一 batch 调度
    ↓
ReplicaScheduleEvent  ← 调度下一个 batch
```

---

## 2. 请求生成

### 2.1 请求结构

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 唯一 ID（自增） |
| `arrived_at` | float | 到达时间 |
| `num_prefill_tokens` | int | 输入 token 数 |
| `num_decode_tokens` | int | 输出 token 数 |
| `num_processed_tokens` | int | 已处理 token 数（动态更新） |

**代码位置：** [`vidur/entities/request.py:39-95`](../../vidur-alibabacloud/vidur/entities/request.py#L39-L95)

### 2.2 请求生成器

- **SyntheticRequestGenerator**：[`vidur/request_generator/synthetic_request_generator.py:16-109`](../../vidur-alibabacloud/vidur/request_generator/synthetic_request_generator.py#L16-L109)
  - Poisson 到达间隔（`PoissonRequestIntervalGenerator`）
  - 多种长度分布：Fixed, Zipf, Uniform, Gamma

### 2.3 请求生命周期中的 PD 字段

| 字段 | 说明 |
|------|------|
| `request_type` | `RequestType.PREFILL` → `RequestType.DECODE` |
| `prefill_arrived_at` | Prefill 到达时间 |
| `decode_arrived_at` | Decode 到达时间（KV 传输完成后设置） |
| `prefill_replica_id` | 分配的 P-node ID |
| `decode_replica_id` | 分配的 D-node ID |
| `pd_p2p_comm_size` | KV cache 传输字节数 |
| `pd_p2p_comm_time` | KV cache 传输时间 |
| `pd_p2p_comm_bandwidth` | 传输带宽 |

**代码位置：** [`vidur/entities/request.py:39-95`](../../vidur-alibabacloud/vidur/entities/request.py#L39-L95)

---

## 3. 全局调度：PD 分离

### 3.1 SplitwiseGlobalScheduler

**代码位置：** [`vidur/scheduler/global_scheduler/splitwise_global_scheduler.py`](../../vidur-alibabacloud/vidur/scheduler/global_scheduler/splitwise_global_scheduler.py)

**初始化（L27-161）：**
- 根据 `pd_node_ratio` 分配 P-nodes 和 D-nodes
- P-nodes: replica_id ∈ `[0, num_prefill_nodes)`
- D-nodes: replica_id ∈ `[num_prefill_nodes, num_replicas)`
- 传输带宽：`transfer_bandwidth = 200 Gbps`（可配置）

**调度逻辑 `schedule()`（L271-417）：**
- Round-robin 分配 P-node 和 D-node
- 为每个 request 创建 DAG：
  ```
  prefill_task (PROMPT) → decode_task (TOKEN)
  ```
- 如果 P ≠ D，插入 KV cache transfer flow：
  ```
  prefill_task → kv_transfer_flow → decode_task
  ```

**KV cache 传输 `add_kv_cache_transfer()`（L233-266）：**
```python
flow_size = request.estimate_kv_cache_size(
    num_tokens=prefill_task.prompt_size,
    replica=src_replica
)
kv_transfer_flow = request.create_flow(
    FlowType.KVCacheTransfer,
    size=flow_size,
    src=src_replica,
    dest=dest_replica
)
# 重新连接 DAG
request.dag.remove_edge(prefill_task, decode_task)
request.dag.add_edge(prefill_task, kv_transfer_flow)
request.dag.add_edge(kv_transfer_flow, decode_task)
```

### 3.2 KV Cache 大小计算

**代码位置：** [`vidur/entities/request.py:452-498`](../../vidur-alibabacloud/vidur/entities/request.py#L452-L498)

```python
def estimate_kv_cache_size(self, num_tokens, replica):
    # bytes_per_token 根据 dtype（float16/bfloat16 = 2 bytes）
    return 2 * num_tokens * replica.mlp_hidden_dim * replica.num_layers * bytes_per_token
```

**示例（DeepSeek-671B）：**
- `mlp_hidden_dim = 18432`, `num_layers = 61`, `dtype = bfloat16`
- 128 tokens: `2 × 128 × 18432 × 61 × 2 ≈ 573 MB`

---

## 4. 本地调度：Batch 组成

### 4.1 SplitwiseReplicaScheduler

**代码位置：** [`vidur/scheduler/replica_scheduler/splitwise_replica_scheduler.py`](../../vidur-alibabacloud/vidur/scheduler/replica_scheduler/splitwise_replica_scheduler.py)

**`_get_next_batch()` 方法（L240-466）：**

#### Prefill batch 组成（L263-360）

```python
if self.replica.replica_type == ReplicaType.PREFILL:
    for request in self._request_queue:
        if request.request_type == RequestType.PREFILL and not request.is_prefill_complete:
            next_num_tokens = request.num_prefill_tokens  # 整个 prefill 一次处理

            # 约束检查：
            # 1. max_tokens_in_batch
            # 2. batch_size_cap
            # 3. max_micro_batch_size
            # 4. can_allocate()（内存）

            requests.append(request)
            num_tokens.append(next_num_tokens)
```

**关键点：**
- Prefill 一次处理所有 input tokens（`next_num_tokens == request.num_prefill_tokens`）
- 多个 request 的 prefill 可以合并为一个 batch（受 token 上限约束）
- 如果一个 request 的 prefill tokens 超过 `max_tokens_in_batch`，会被拆分为多个 micro-batch

#### Decode batch 组成（L361-457）

```python
elif self.replica.replica_type == ReplicaType.DECODE:
    for request in self._request_queue:
        if request.request_type == RequestType.DECODE:
            if request.is_prefill_complete:
                assert request.decode_arrived_at != float('inf')  # KV 传输已完成
                next_num_tokens = 1  # 每次只生成 1 个 token
                # 约束检查同上
                requests.append(request)
                num_tokens.append(1)
```

**关键点：**
- Decode **每次只生成 1 个 token**（`next_num_tokens == 1`）
- 多个 decode request 可以合并在同一个 batch 中（每个贡献 1 token）
- Decode request 必须满足：`is_prefill_complete == True` 且 `decode_arrived_at` 已设置

### 4.2 Batch 完成 `on_batch_end()`（L202-233）

**P-node 完成 prefill：**
- 若 request prefill 完成 → 释放内存，将 request 移到 D-node 队列
- 若被抢占 → 放回 P-node 抢占队列

**D-node 完成 decode：**
- 若 request 完成（所有 decode tokens）→ 释放内存
- 若被抢占 → 放回 D-node 抢占队列

### 4.3 Batch 数据结构

**代码位置：** [`vidur/entities/batch.py:29-153`](../../vidur-alibabacloud/vidur/entities/batch.py#L29-L153)

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | int | 唯一 batch ID |
| `replica_id` | int | 执行的 replica ID |
| `requests` | List[Request] | 批内 request 列表 |
| `num_tokens` | List[int] | 每个 request 的 token 数 |
| `total_num_tokens` | int | 总 token 数 |
| `total_num_tokens_rounded` | int | 向上取整到 8 的倍数（TP 对齐） |
| `num_prefill_tokens` | int | Prefill token 数 |
| `num_decode_tokens` | int | Decode token 数 |
| `scheduled_at` | float | 调度开始时间 |
| `completed_at` | float | 完成时间 |

---

## 5. Batch 完成事件（核心插桩点）

### 5.1 BatchEndEvent

**代码位置：** [`vidur/events/batch_end_event.py:27-142`](../../vidur-alibabacloud/vidur/events/batch_end_event.py#L27-L142)

这是最关键的插桩点。当 batch 完成时，所有信息都已确定：

```python
def handle_event(self):
    # 1. 更新 batch 完成时间
    self._batch.on_batch_end(self.time)

    # 2. 更新 replica scheduler
    replica_scheduler.on_batch_end(self._batch)

    # 3. 对每个 request（通过 batch.on_batch_end → request.on_batch_end）
    for request in batch.requests:
        # request.num_processed_tokens 已更新
        # request.is_prefill_complete 可能变为 True
        # request.completed 可能变为 True
```

### 5.2 KV Cache 传输时序（L87-106）

当 P-node 完成 prefill 时，计算 KV 传输：

```python
if request.is_prefill_complete and replica.replica_type == ReplicaType.PREFILL:
    # 计算 KV cache 大小
    request.pd_p2p_comm_size = request.estimate_kv_cache_size(
        request.num_processed_tokens, replica
    )
    # 获取传输带宽
    request.pd_p2p_comm_bandwidth = replica.pd_p2p_comm_bandwidth * 1024**3 / 8
    # 计算传输时间
    request.pd_p2p_comm_time = request.pd_p2p_comm_size / request.pd_p2p_comm_bandwidth
    # 设置 decode 到达时间
    request.decode_arrived_at = request.prefill_completed_at + request.pd_p2p_comm_time
    # 调度 D-node
    ReplicaScheduleEvent(request.decode_arrived_at, request.decode_replica_id)
```

### 5.3 Request 状态跟踪

**代码位置：** [`vidur/entities/request.py:259-338`](../../vidur-alibabacloud/vidur/entities/request.py#L259-L338) (`on_batch_end()`)

**关键状态转换：**

| 条件 | 动作 | 属性变化 |
|------|------|----------|
| `num_processed_tokens == num_prefill_tokens` | Prefill 完成 | `is_prefill_complete = True`，`num_processed_tokens += 1` |
| `num_processed_tokens > num_prefill_tokens` | 处理 decode token | 继续迭代 |
| `num_processed_tokens == total_tokens` | Request 完成 | `completed = True`，`completed_at = time` |

**重要细节（L282-289）：** Prefill 完成时 `num_processed_tokens += 1`，即 prefill 完成自动消耗第一个 decode token。

### 5.4 Batch 内 Request 可用信息汇总

在 `BatchEndEvent` 时，对 batch 内每个 request 可获取：

```python
{
    # 基础信息
    "request_id": request.id,
    "num_prefill_tokens": request.num_prefill_tokens,
    "num_decode_tokens": request.num_decode_tokens,
    "num_processed_tokens": request.num_processed_tokens,

    # PD 分离信息
    "prefill_replica_id": request.prefill_replica_id,
    "decode_replica_id": request.decode_replica_id,
    "request_type": request.request_type,  # PREFILL or DECODE

    # 状态
    "is_prefill_complete": request.is_prefill_complete,
    "completed": request.completed,

    # KV cache 传输
    "pd_p2p_comm_size": request.pd_p2p_comm_size,
    "pd_p2p_comm_time": request.pd_p2p_comm_time,

    # 时间戳（Vidur 理想条件下的值，trace 中不记录）
    # "prefill_arrived_at", "decode_arrived_at",
    # "prefill_completed_at", "completed_at"
}
```

---

## 6. 执行时间预测

### 6.1 AICB CSV Profiling 格式

Vidur 的 AICB backend 使用 tab-separated CSV 文件存储 per-layer profiling 数据。

**CSV 列：**

| 列名 | 类型 | 单位 | 说明 |
|------|------|------|------|
| `layer_id` | int | - | Transformer layer 编号（0, 1, 2, ...） |
| `layer_name` | str | - | 层类型：`attention` / `mlp` / `moe` |
| `comp_time` | float | ns | 计算时间（纳秒） |
| `comm_size` | float | bytes | 通信数据量（字节） |

**文件名模式：**
```
vidur-{MODEL}-world_size{WS}-tp{TP}-pp{PP}-ep{EP}-bs{BS}-seq{SEQ}-{PHASE}.csv
```
示例：`vidur-DeepSeek-671B-world_size32-tp1-pp1-ep32-bs4-seq4096-decode.csv`

**注意：** prefill 和 decode 使用**不同的 CSV 文件**（不同的 bs 和 seq 参数）。

**代码位置：**
- CSV 路径生成：[`vidur/entities/execution_time.py:213-223`](../../vidur-alibabacloud/vidur/entities/execution_time.py#L213-L223) (`_get_aicb_csv_path()`)
- CSV 加载：[`vidur/entities/execution_time.py:255-362`](../../vidur-alibabacloud/vidur/entities/execution_time.py#L255-L362) (`_load_aicb_data()`)

### 6.2 计算 vs 通信时间

**模型总计算时间（`model_time` 属性）：** [`vidur/entities/execution_time.py:493-542`](../../vidur-alibabacloud/vidur/entities/execution_time.py#L493-L542)

```python
@property
def model_time(self) -> float:
    for layer_id in range(start_layer, end_layer):
        att_time = self._get_attention_layer_execution_time_from_aicb(layer_id)
        mlp_time = self._get_mlp_layer_execution_time_from_aicb(layer_id)
        moe_time = self._get_moe_layer_execution_time_from_aicb(layer_id)
        total_block_time += att_time + mlp_time + moe_time
    return (total_block_time + pipeline_parallel_communication_time) * 1e-3
```

**各层时间获取：**

| 方法 | 代码位置 | 说明 |
|------|----------|------|
| `_get_attention_layer_execution_time_from_aicb()` | [`execution_time.py:103-121`](../../vidur-alibabacloud/vidur/entities/execution_time.py#L103-L121) | Attention 层计算时间 |
| `_get_mlp_layer_execution_time_from_aicb()` | [`execution_time.py:89-101`](../../vidur-alibabacloud/vidur/entities/execution_time.py#L89-L101) | MLP 层计算时间 |
| `_get_moe_layer_execution_time_from_aicb()` | [`execution_time.py:158-190`](../../vidur-alibabacloud/vidur/entities/execution_time.py#L158-L190) | MoE 层计算+通信时间 |

**MoE 层的特殊处理（L158-190）：**
```python
def _get_moe_layer_execution_time_from_aicb(self, layer_id):
    moe_comp_time = aicb_data['comp_time'] * 1e-9   # ns → s
    moe_comm_size = aicb_data['comm_size']            # bytes

    # prefill 用 RDMA 带宽，decode 用 NVLink 带宽
    if replica_stage == "prefill":
        cur_bw = rdma_bandwidth
    elif replica_stage == "decode":
        cur_bw = nvlink_bandwidth

    moe_comm_time = moe_comm_size / cur_bw
    return moe_comp_time + moe_comm_time
```

### 6.3 TP 通信量计算

**代码位置：** [`vidur/execution_time_predictor/communication_time_predictor.py:58-62`](../../vidur-alibabacloud/vidur/execution_time_predictor/communication_time_predictor.py#L58-L62)

```python
all_reduce_bytes = self.hidden_size * num_tokens_in_batch * self.tensor_size
```

其中：
- `hidden_size`：模型隐藏维度（如 DeepSeek-671B = 7168）
- `num_tokens_in_batch`：`batch.total_num_tokens_rounded`（向上取整到 8 的倍数）
- `tensor_size`：dtype 字节数（bfloat16 = 2）

---

## 7. AICB 推理 Workload 生成

### 7.1 生成脚本和参数

**脚本：** [`aicb/scripts/inference_workload_with_aiob.sh`](../../aicb/scripts/inference_workload_with_aiob.sh)

**生成器：** [`aicb/workload_generator/SimAI_inference_workload_generator.py`](../../aicb/workload_generator/SimAI_inference_workload_generator.py)

**关键参数：**
- `-m`：模型（deepseek-671B / qwen3-235B / qwen3-next-80B）
- `-p`：阶段（prefill / decode）
- `-s`：序列长度
- `-b`：micro batch size
- `-w`：world_size
- `-t`：tensor parallel size
- `-e`：expert parallel size

### 7.2 模型配置

| 模型 | 配置文件 | 层数 | hidden_size | 专家数 |
|------|----------|------|-------------|--------|
| DeepSeek-671B | [`inference_configs/deepseek_default.json`](../../aicb/scripts/inference_configs/deepseek_default.json) | 61 | 7168 | 288 |
| Qwen3-MoE-235B | [`inference_configs/qwen3_moe_default.json`](../../aicb/scripts/inference_configs/qwen3_moe_default.json) | 94 | 4096 | 128 |
| Qwen3-Next-80B | [`inference_configs/qwen3_next_default.json`](../../aicb/scripts/inference_configs/qwen3_next_default.json) | 48 | 2048 | 512 |

### 7.3 通信量公式

**TP AllReduce（attention / dense_mlp）：**
```
comm_size = 2 × batch_size × hidden_size
```

**EP AlltoAll（MoE dispatch / combine）：**
```
comm_size = 2 × batch_size × hidden_size × topk / TP × FP8_FACTOR
```
（FP8_FACTOR 仅 DeepSeek 适用：`(1 + 4/128) / 2`）

**与训练的区别：**
- 推理只有 forward pass（mode: 1），没有 backward
- 所有 backward_compute_time = 0，backward_comm = "NONE"
- Decode 阶段 `batch_size` 即为 micro_batch_size（通常很小），seq_length = 1
- Prefill 阶段 `batch_size` = seq_length

---

## 8. simai-flow-scheduler 现有架构参考

### 8.1 Schema

**代码位置：** [`simai-flow-scheduler/src/workload_format/schema.py`](../src/workload_format/schema.py)

| 类型 | 枚举值 | 说明 |
|------|--------|------|
| Phase | FORWARD, BACKWARD_INPUT, BACKWARD_WEIGHT, OPTIMIZER | 推理需扩展：PREFILL, DECODE |
| CommType | TP_ALLREDUCE_RING, TP_ALLGATHER_RING, ..., EP_ALLTOALL, PP_SEND, PP_RECV | 推理需扩展：KV_CACHE_TRANSFER |
| TaskType | COMPUTE, FLOW | 推理可直接复用 |

### 8.2 Workload Builder 构建模式

**代码位置：** [`simai-flow-scheduler/src/workload_generator/workload_builder.py`](../src/workload_generator/workload_builder.py)

两阶段构建：
1. **生成 tasks**：每个 rank × phase 创建 COMPUTE + 展开的 FLOW 任务
2. **连接依赖**：Forward 链式、Backward 反向链式、跨 phase receiver-based 依赖

推理可复用的模式：
- **Prefill batch** ≈ 训练的 Forward pass（COMPUTE + TP AllReduce FLOW）
- **Decode iteration** ≈ 训练的 Forward pass（但 seq_length=1，通信量更小）
- **KV cache transfer** = 新类型 P2P FLOW

### 8.3 Collective Expander

**代码位置：** [`simai-flow-scheduler/src/workload_generator/collective_expander.py`](../src/workload_generator/collective_expander.py)

已支持的展开算法：
- AllReduce Ring（n × 2(n-1) 个 flow）
- AllGather Ring（n × (n-1) 个 flow）
- ReduceScatter Ring（n × (n-1) 个 flow）
- AlltoAll（n × (n-1) 个独立 flow）

推理可直接复用这些展开器。

---

## 9. Trace 格式设计建议

基于以上调研，对 `phase5-plan.md` 中的 trace 格式提出以下调整建议：

### 9.1 节点标识问题

**原方案** 使用 `p_nodes: [0,1,2,...,7]` 列表。

**问题**：Vidur 中 `replica_id` 是节点组的逻辑 ID（一组 TP ranks），不是物理 GPU ID。simai-flow-scheduler 中节点是物理 rank。

**建议**：trace 中记录 `replica_id` + 并行度配置，由 `InferenceTraceExpander` 根据拓扑映射到物理 rank。

### 9.2 Decode Batch 结构

**原方案** 使用 `num_iterations` 表示 decode 步数。

**实际情况**：
- Vidur 中每个 decode step 是一个独立的 batch
- 同一 batch 中可包含多个 request 的各 1 个 token
- Prefill 完成时自动消耗第一个 decode token（`num_processed_tokens += 1`）

**建议**：每个 batch entry 就是一个 decode step，不使用 `num_iterations` 字段。decode 的 token 展开在 Expander 中处理（根据 `request.num_decode_tokens - 1`，因为第一个 token 被 prefill 消耗）。

### 9.3 Token 对齐

**实际情况**：Vidur 将 `total_num_tokens` 向上取整到 8 的倍数（`total_num_tokens_rounded`），TP 通信量基于这个值。

**建议**：trace 中记录原始 token 数，通信量计算时在 Expander 中做对齐。

### 9.4 修正后的 Trace 格式草案

```json
{
  "version": "1.0",
  "model": "deepseek-671b",
  "model_config": {
    "hidden_size": 7168,
    "num_layers": 61,
    "dense_layers": 3,
    "moe_topk": 8,
    "mlp_hidden_dim": 18432,
    "dtype": "bfloat16"
  },
  "parallelism": { "tp": 8, "pp": 1, "ep": 8 },
  "pd_config": {
    "pd_node_ratio": 0.5,
    "pd_p2p_comm_bandwidth_gbps": 200
  },
  "requests": {
    "0": { "num_prefill_tokens": 128, "num_decode_tokens": 64 },
    "1": { "num_prefill_tokens": 256, "num_decode_tokens": 32 }
  },
  "batches": [
    {
      "batch_id": "p0",
      "type": "prefill",
      "replica_id": 0,
      "request_ids": [0, 1],
      "num_tokens": [128, 256],
      "kv_cache_bytes": { "0": 123456, "1": 234567 },
      "depends_on": []
    },
    {
      "batch_id": "d0",
      "type": "decode",
      "replica_id": 4,
      "request_ids": [0, 1],
      "num_tokens": [1, 1],
      "kv_cache_bytes": null,
      "depends_on": ["p0"]
    },
    {
      "batch_id": "d1",
      "type": "decode",
      "replica_id": 4,
      "request_ids": [0, 1],
      "num_tokens": [1, 1],
      "kv_cache_bytes": null,
      "depends_on": ["d0"]
    }
  ]
}
```

### 9.5 与原方案的主要差异

| 差异点 | 原方案 | 修正建议 | 原因 |
|--------|--------|----------|------|
| 节点标识 | `p_nodes`/`d_nodes` 列表 | `replica_id` + parallelism | Vidur 中 replica_id 是逻辑 ID |
| Decode batch | `num_iterations` | 每个 step 一个 batch entry | Vidur 实际行为：per-token batch |
| KV cache | 单一 `kv_cache_bytes` | `per-request dict` | 同一 batch 中不同 request 的 KV 大小不同 |
| 模型配置 | 无 | 增加 `model_config` 字段 | Expander 需要隐藏维度等参数来计算通信量 |
| PD 配置 | 无 | 增加 `pd_config` 字段 | 需要带宽和 P/D 比例信息 |
| depends_on | 字符串 `"batch_0_prefill"` | batch_id 直接引用 | 更简洁，避免字符串解析 |

### 9.6 QoS 指标追踪方案

执行后需要追踪的 per-request 指标：

| 指标 | 计算方式 |
|------|----------|
| **TTFT** | 首 decode batch 完成时间（executor 结果） |
| **TBT** | 相邻 decode batch 完成时间差 |
| **E2E Latency** | 最后 decode batch 完成时间 |

**Trace 中不记录时间戳**：Vidur trace 是理想条件下的结果，simai-flow-scheduler replay 时网络拥塞完全不同，所有时间由 executor 重新计算。Trace 中不需要 `arrived_at` 等时间字段。

**Task → Request 回溯**：Expander 展开时额外输出一份 `batch_task_map`（batch_id → task_ids 映射），不修改现有 Task schema。执行后用这份映射按 request 聚合 task 时间戳：

```python
# Expander 输出
batch_task_map = {
    "p0": {"task_ids": [0,1,2,...], "request_ids": [0,1], "type": "prefill"},
    "d0": {"task_ids": [50,51,...], "request_ids": [0,1], "type": "decode"},
    ...
}

# 执行后：从 executor result 反推 QoS
for request_id in all_request_ids:
    decode_batches = [b for b in batches if request_id in b["request_ids"] and b["type"] == "decode"]
    decode_batches.sort(key=lambda b: b["end_time"])

    ttft = decode_batches[0]["end_time"]
    tbt_list = [decode_batches[i+1]["end_time"] - decode_batches[i]["end_time"]
                for i in range(len(decode_batches)-1)]
    e2e = decode_batches[-1]["end_time"]
```

**设计决策：**
- 不在 Task schema 中添加 metadata 字段，避免影响现有结构
- `batch_task_map` 作为 Expander 的附加输出，仅在需要 QoS 分析时使用
- 时间戳全部来自 executor 执行结果，trace 中不记录任何时间

---

## 10. 插桩方案

### 10.1 推荐插桩点：BatchEndEvent

**文件：** [`vidur/events/batch_end_event.py:27-142`](../../vidur-alibabacloud/vidur/events/batch_end_event.py#L27-L142)

在 `handle_event()` 末尾添加 trace 记录逻辑：

```python
def handle_event(self):
    # ... 现有逻辑 ...

    # === 新增：Trace 记录 ===
    self._trace_recorder.record_batch(
        batch_id=self._batch.id,
        batch_type="prefill" if replica.replica_type == ReplicaType.PREFILL else "decode",
        replica_id=self._batch.replica_id,
        requests=[{
            "request_id": r.id,
            "num_tokens": n_tokens,
            "is_prefill_complete": r.is_prefill_complete,
            "pd_p2p_comm_size": r.pd_p2p_comm_size if r.is_prefill_complete else None,
        } for r, n_tokens in zip(self._batch.requests, self._batch.num_tokens)],
        depends_on=...,  # 需要维护前驱 batch 映射
    )
```

### 10.2 需要额外维护的状态

- **前驱 batch 映射**：跟踪每个 replica 上最后一个 batch，用于确定 `depends_on`
- **Request → Prefill batch 映射**：跟踪每个 request 的 prefill batch，用于 decode batch 的依赖

### 10.3 不记录的信息

- **执行时间**：由 simai-flow-scheduler 的 executor 重新计算（网络拥塞效应不同）
- **Per-layer 细节**：Trace 只记录 batch 级别，layer 展开由 Expander 完成
- **调度开销**：Vidur 的 schedule_time、sampler_time 等（simai-flow-scheduler 不需要）

---

## 11. 风险和注意事项

1. **Prefill token 消耗**：Prefill 完成时 Vidur 自动 `num_processed_tokens += 1`，这意味着实际 decode 迭代次数 = `num_decode_tokens - 1`，Expander 需要注意这一点

2. **Token 对齐**：`total_num_tokens_rounded` 向上取整到 8 的倍数会影响 TP 通信量计算

3. **MoE 层带宽差异**：Vidur 中 prefill 用 RDMA 带宽，decode 用 NVLink 带宽来计算 MoE 通信时间。simai-flow-scheduler 中不做这种区分（P2P flow 自然反映路径带宽）

4. **CSV 文件可用性**：当前仓库中不存在实际的 AICB CSV profiling 文件，需要通过 AICB 在实际 GPU 上运行生成，或者构造模拟数据

5. **大 request 的 prefill 拆分**：如果一个 request 的 `num_prefill_tokens` 超过 `max_tokens_in_batch`，Vidur 会拆分为多个 micro-batch。Trace 需要正确反映这种拆分

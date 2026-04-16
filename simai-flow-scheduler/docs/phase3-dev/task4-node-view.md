# Phase 3 - Task 4 开发记录：Node-Centric Local Views（节点局部视图）

## 1. 目标与范围

Task 4 的目标是为每个节点预计算其局部调度视图，包括发送/接收的 flow 任务、compute 任务、估计调度时间（基于 ASAP 分析）和 idle ratio。

**包含**：
- `NodeLocalView` 数据类：单节点的 send_tasks、receive_tasks、compute_tasks、估计时间、idle ratio
- `build_node_views(workload, critical_path)` 函数：构建所有节点的局部视图

**不包含**：
- 动态调度决策（Phase 4 的职责）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/scheduler/
│   ├── __init__.py              # 更新：导出 NodeLocalView, build_node_views
│   └── node_view.py             # 新增：节点局部视图完整实现
├── tests/
│   └── test_node_view.py        # 新增：16 个测试
```

### 2.2 数据结构

```python
@dataclass
class NodeLocalView:
    node_id: int

    # Flows this node sends
    send_tasks: list[int]           # task_ids
    total_send_bytes: int

    # Flows this node receives
    receive_tasks: list[int]        # task_ids
    total_receive_bytes: int

    # Compute tasks on this node
    compute_tasks: list[int]        # task_ids
    total_compute_time_us: int

    # Estimated schedule (ASAP from critical path)
    estimated_send_times: list[tuple[int, int]]     # (start_time_us, task_id) — 发送方开始传输时刻
    estimated_receive_times: list[tuple[int, int]]  # (arrival_time_us, task_id) — 接收方数据到达时刻

    # Estimated idle ratio (communication time / total active time)
    estimated_idle_ratio: float
```

### 2.3 API

```python
# 主入口
def build_node_views(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
) -> dict[int, NodeLocalView]

# NodeLocalView 方法
view.add_send_flow(task_id, size_bytes, start_time)     # 添加发送 flow
view.add_receive_flow(task_id, size_bytes, arrival_time)  # 添加接收 flow
```

---

## 3. 设计原理

### 3.1 为什么需要节点局部视图

调度器在做决策时，经常需要从单个节点的角度了解：
- 这个节点有多少流量要发送/接收？
- 通信时间占总时间的比例是多少？
- 哪些时间点是发送/接收高峰？

这些信息可以帮助调度器：
- 识别通信热点节点（high idle_ratio）
- 避免在高峰时段安排更多流量
- 决定是否需要对某些节点做流量控制

### 3.2 Idle Ratio 计算

Idle ratio 表示通信开销占总活跃时间的比例。对于 GPU 节点来说，计算才是"本职工作"，通信是额外开销：

```
total_comm_time = sum(earliest_finish - earliest_start for flow tasks this node participates in)
total_time = total_compute_time + total_comm_time
idle_ratio = total_comm_time / total_time
```

**设计选择**：
- 使用 critical path 的 ASAP 时间差来估算 flow 持续时间（而非 size_bytes，量纲不对）
- 一个 flow 如果 node 同时是 src 和 dst（自环），只计算一次（使用 set union 去重）
- compute-only 节点 idle_ratio = 0.0，flow-only 节点 idle_ratio = 1.0

### 3.3 发送/接收时间语义

- **发送方**：使用 `earliest_start_us`（何时开始传输）
- **接收方**：使用 `earliest_finish_us`（何时数据到达），而非 `earliest_start_us`

发送方关心"我什么时候开始发"，接收方关心"我什么时候能拿到数据"。两者语义不同。

### 3.4 错误处理：缺失 critical path timing

`build_node_views` 要求 `critical_path.task_timings` 包含 workload 中所有 flow task 的时间信息。如果某个 task_id 查不到，说明 critical path 分析不完整，这是严重错误，直接 raise `ValueError` 而非静默 fallback 到 0。

---

## 4. 与设计文档的偏差

### 4.1 使用 `task.is_compute()` / `task.is_flow()` 替代 `task.type.value`

与 Task 1/2/3 保持一致。

### 4.2 修正 API 调用

计划中使用 `critical_path.earliest_start_us.get(task.task_id, 0)`，但 `CriticalPathInfo` 没有 `earliest_start_us` 属性。修正为直接访问 `critical_path.task_timings[task.task_id].earliest_start_us`。

### 4.3 修正 busy_ratio → idle_ratio

计划的 `busy_ratio = comm_time / total_time` 名称有误导性：GPU 的"忙碌"应该是计算，而通信代表"空闲"。重命名为 `idle_ratio`，定义不变（comm_time / total_time），但语义更准确。

### 4.4 修正接收方时间

计划中发送方和接收方都使用 `earliest_start_us`。修正为：接收方使用 `earliest_finish_us`（数据到达时刻）。

### 4.5 移除 busiest_outgoing_link / busiest_incoming_link

计划中包含这两个字段但未计算。分析后发现节点无法控制流的路由，这些信息对调度没有直接帮助。直接移除。

### 4.6 修正 idle_ratio 计算（量纲）

计划使用 `size_bytes` 累加计算 busy_ratio（量纲为字节），修正为使用 critical path timing 中的 `earliest_finish_us - earliest_start_us`（量纲为微秒）。

### 4.7 移除 fallback to 0

计划中当 task_id 不在 `critical_path.task_timings` 中时 fallback 到 0。修正为直接 raise `ValueError`，因为这种情况表明分析流程有严重错误。

---

## 5. 测试结果

```
tests/test_node_view.py: 16 passed
tests/ (完整回归): 304 passed (288 原有 + 16 node view)
```

### 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestNodeLocalView` | 4 | add_send_flow、add_receive_flow、多次添加、默认值 |
| `TestBuildNodeViews` | 12 | 空workload、单compute、单flow、多flow同sender、发送时间用earliest_start、接收时间用earliest_finish、idle_ratio（混合/flow-only/compute-only）、节点聚合、独立节点、缺失timing报错 |

---

## 6. 后续依赖

Task 4 的输出 (`NodeLocalView`) 将被以下模块使用：

1. **Task 7（WorkloadAnalyzer）**：统一入口，调用 `build_node_views` 作为分析流程的一步
2. **Phase 4（Scheduler）**：
   - `estimated_idle_ratio` — 判断节点通信负载
   - `estimated_send_times` / `estimated_receive_times` — 快速查询节点某时刻的流量

---

*开发时间：2026-04-16*
*测试状态：16 passed（node view）+ 288 passed（原有）= 304 total*
*关键设计：idle_ratio 量度通信开销；接收方用 earliest_finish_us；缺失 timing 直接报错*

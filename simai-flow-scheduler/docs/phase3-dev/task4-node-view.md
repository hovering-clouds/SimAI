# Phase 3 - Task 4 开发记录：Node-Centric Local Views（节点局部视图）

## 1. 目标与范围

Task 4 的目标是为每个节点预计算其局部调度视图，包括发送/接收的 flow 任务、compute 任务、估计调度时间（基于 ASAP 分析）和 busy ratio。

**包含**：
- `NodeLocalView` 数据类：单节点的 send_tasks、receive_tasks、compute_tasks、估计时间、busy ratio
- `build_node_views(workload, critical_path)` 函数：构建所有节点的局部视图

**不包含**：
- `busiest_outgoing_link` / `busiest_incoming_link` 的计算（需要路由信息，留给 Task 7 WorkloadAnalyzer）
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
│   └── test_node_view.py        # 新增：15 个测试
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
    estimated_send_times: list[tuple[int, int]]     # (start_time_us, task_id)
    estimated_receive_times: list[tuple[int, int]]

    # Bottleneck links (computed by WorkloadAnalyzer in Task 7)
    busiest_outgoing_link: tuple[int, int] | None
    busiest_incoming_link: tuple[int, int] | None

    # Estimated busy ratio (comm time / total time)
    estimated_busy_ratio: float
```

### 2.3 API

```python
# 主入口
def build_node_views(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
) -> dict[int, NodeLocalView]

# NodeLocalView 方法
view.add_send_flow(task_id, size_bytes, start_time)    # 添加发送 flow
view.add_receive_flow(task_id, size_bytes, start_time)  # 添加接收 flow
```

---

## 3. 设计原理

### 3.1 为什么需要节点局部视图

调度器在做决策时，经常需要从单个节点的角度了解：
- 这个节点有多少流量要发送/接收？
- 通信时间占总时间的比例是多少？
- 哪些时间点是发送/接收高峰？

这些信息可以帮助调度器：
- 识别通信热点节点（high busy_ratio）
- 避免在高峰时段安排更多流量
- 决定是否需要对某些节点做流量控制

### 3.2 Busy Ratio 计算

Busy ratio 表示通信时间占总活跃时间的比例：

```
total_comm_time = sum(earliest_finish - earliest_start for flow tasks this node participates in)
total_time = total_compute_time + total_comm_time
busy_ratio = total_comm_time / total_time
```

**设计选择**：
- 使用 critical path 的 ASAP 时间差来估算 flow 持续时间，而不是 size_bytes（量纲不对）
- 一个 flow 如果 node 同时是 src 和 dst（自环），只计算一次（使用 set union 去重）
- compute-only 节点 busy_ratio = 0.0，flow-only 节点 busy_ratio = 1.0

### 3.3 与计划文档的偏差修正

#### 3.3.1 使用 `task.is_compute()` / `task.is_flow()` 替代 `task.type.value`

与 Task 1/2/3 保持一致。

#### 3.3.2 修正 API 调用

计划中使用 `critical_path.earliest_start_us.get(task.task_id, 0)`，但 `CriticalPathInfo` 没有 `earliest_start_us` 属性。修正为：

```python
critical_path.task_timings[task.task_id].earliest_start_us
```

#### 3.3.3 修正 busy_ratio 计算

计划的 busy_ratio 使用 `size_bytes` 累加（量纲为字节），但 busy_ratio 应该是时间比例。修正为使用 critical path timing 中的 `earliest_finish_us - earliest_start_us` 作为 flow 持续时间。

---

## 4. 测试结果

```
tests/test_node_view.py: 15 passed
tests/ (完整回归): 303 passed (288 原有 + 15 node view)
```

### 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestNodeLocalView` | 4 | add_send_flow、add_receive_flow、多次添加、默认值 |
| `TestBuildNodeViews` | 11 | 空workload、单compute、单flow、多flow同sender、ASAP时间、busy ratio（混合/flow-only/compute-only）、节点聚合、独立节点 |

---

## 5. 与设计文档的偏差

### 5.1 使用 `task.is_compute()` / `task.is_flow()` 替代 `task.type.value == "compute"/"flow"`

与 Task 1/2/3 一致。

### 5.2 修正 API 调用

`critical_path.earliest_start_us.get()` → `critical_path.task_timings[].earliest_start_us`。

### 5.3 修正 busy_ratio 计算

使用 flow duration（from critical path timing）替代 size_bytes。

### 5.4 `busiest_outgoing_link` / `busiest_incoming_link` 未计算

这些字段需要路由信息（topology + routing_hints），而 `build_node_views` 只接收 `critical_path`。这些字段预留给 Task 7（WorkloadAnalyzer）填充，该模块可以访问所有分析结果。

---

## 6. 后续依赖

Task 4 的输出 (`NodeLocalView`) 将被以下模块使用：

1. **Task 7（WorkloadAnalyzer）**：统一入口，调用 `build_node_views` 作为分析流程的一步，并填充 `busiest_outgoing_link`/`busiest_incoming_link`
2. **Phase 4（Scheduler）**：
   - `estimated_busy_ratio` — 判断节点负载
   - `estimated_send_times` / `estimated_receive_times` — 快速查询节点某时刻的流量

---

*开发时间：2026-04-16*
*测试状态：15 passed（node view）+ 288 passed（原有）= 303 total*
*关键设计：使用 critical path timing 计算 busy ratio（而非 size_bytes），busiest link 预留给 Task 7 填充*

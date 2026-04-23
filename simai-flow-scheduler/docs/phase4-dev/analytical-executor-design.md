# Analytical Executor 设计文档

## 概述

Analytical Executor 是 Phase 4a 的核心组件，采用离散事件模拟方式，对 P2PWorkload + ExecutionPlan 进行时间推进模拟，输出每个 task 的 start/end 时间和各 job 的 iteration time。

---

## 文件结构

```
src/executor/
  __init__.py
  result.py          # TaskTiming + ExecutionResult 数据结构
  bandwidth.py       # BandwidthAllocator 基类 + FairShareAllocator 默认实现
  analytical.py      # AnalyticalExecutor（事件循环主体）
```

---

## 数据结构

### result.py

```python
@dataclass
class TaskTiming:
    task_id: int
    node: int          # compute 任务的节点；flow 任务用 src
    task_type: str     # "compute" | "flow"
    start_time_us: int
    end_time_us: int

@dataclass
class ExecutionResult:
    per_task: dict[int, TaskTiming]
    job_iteration_times: dict[int, int]   # job_id → 单次 iteration 时间
    total_time_us: int                    # 所有 task 的最晚 end_time
    makespan_us: int                      # max(end_time) - min(start_time)
```

`job_iteration_times[job_id]` = 该 job 所有 task 的 `max(end_time_us) - min(start_time_us)`。

### ActiveFlow（analytical.py 内部）

```python
@dataclass
class ActiveFlow:
    task_id: int
    src: int
    dst: int
    size_bytes: int
    remaining_bytes: int       # 剩余未传输字节数
    path: list[int]            # 网络路径（由 RoutingHints 查询）
    start_time: int            # 流开始时间（微秒）
    last_update_time: int      # 上次带宽重新分配的时间（用于计算 remaining_bytes）
    current_bw_gbps: float     # 当前分配带宽
    estimated_end_time: int    # 预计完成时间
    version: int               # 懒删除版本号，与 flow_completion 事件对应
```

### Event（analytical.py 内部）

```python
@dataclass(order=True)
class Event:
    time: int
    seq: int           # 单调递增，打破时间相同时的顺序
    kind: str          # "compute_ready" | "compute_done" | "flow_ready" | "flow_completion"
    task_id: int
    version: int = 0   # 仅 flow_completion 使用，用于懒删除
```

---

## 带宽分配（bandwidth.py）

### BandwidthAllocator 基类

```python
class BandwidthAllocator(ABC):
    @abstractmethod
    def allocate(
        self,
        active_flows: list[ActiveFlow],
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        current_time: int,
    ) -> dict[int, float]:
        """
        Returns:
            task_id → allocated_bw_gbps 的映射
        """
```

### FairShareAllocator（默认实现）

逐链路均分策略：

1. 统计每条链路上有多少 active flow 经过（`link_flow_count: dict[tuple[int,int], int]`）
2. 对每条 active flow，遍历其路径上的每条链路，计算该链路的 fair share = `link_bw / count`
3. 取路径上所有链路 fair share 的最小值作为该流的分配带宽（瓶颈链路决定）

---

## 事件循环（analytical.py）

### AnalyticalExecutor 接口

```python
class AnalyticalExecutor:
    def __init__(
        self,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        allocator: BandwidthAllocator | None = None,
    ):
        self.topology = topology
        self.routing_hints = routing_hints
        self.allocator = allocator or FairShareAllocator()

    def execute(
        self,
        workload: P2PWorkload,
        plan: ExecutionPlan,
    ) -> ExecutionResult:
        ...
```

### 初始化阶段

1. 构建 `task_map: dict[int, Task]`（task_id → Task）
2. 构建 `dep_count: dict[int, int]`（每个 task 还需要多少 dep 完成）
3. 构建 `dependents: dict[int, list[int]]`（task A 完成后，哪些 task 的 dep 计数减 1）
4. 对每个节点，取 `plan.compute_order[node][0]` 加入 `compute_ready` 事件（time=0）
5. 对 `dep_count[task_id] == 0` 的 flow task，加入 `flow_ready` 事件（time=0）

### 事件处理逻辑

**compute_ready(task_id)**:
- 记录 `start_time[task_id] = current_time`
- 安排 `compute_done(task_id, time = current_time + task.duration_us)`

**compute_done(task_id)**:
- 记录 `end_time[task_id] = current_time`
- 对 `dependents[task_id]` 中的每个 downstream task，`dep_count[downstream] -= 1`
  - 若归零且是 flow task → 安排 `flow_ready(downstream, time=current_time)`
  - 若归零且是 compute task → 检查是否是该节点 compute_order 中的下一个（见下）
- 检查该节点 compute_order 中的下一个 compute task，若其 `dep_count == 0` → 安排 `compute_ready`

> **注意**：compute_order 中相邻 compute 任务之间存在隐式顺序依赖（前一个完成后才能开始下一个），但这个依赖不在 workload.deps 中，而是由执行器根据 compute_order 在运行时强制执行。具体做法：每个节点维护一个 `compute_cursor[node]`，指向当前正在执行或等待执行的 compute 任务在 compute_order 中的位置。`compute_done` 时将 cursor 推进，并检查下一个 compute 的 dep_count 是否为 0。

**flow_ready(task_id)**:
- 通过 `routing_hints.get_path()` 查询路径
- 创建 `ActiveFlow`，加入 `active_flows: dict[int, ActiveFlow]`
- 记录 `start_time[task_id] = current_time`
- 调用 `_reallocate_bandwidth(current_time)`

**flow_completion(task_id, version)**:
- 懒删除检查：若 `active_flows[task_id].version != version` → 跳过（过期事件）
- 记录 `end_time[task_id] = current_time`
- 从 `active_flows` 移除
- 对 `dependents[task_id]` 减少 dep 计数，触发下游（同 compute_done 逻辑）
- 调用 `_reallocate_bandwidth(current_time)`

### _reallocate_bandwidth(current_time)

```
1. 更新所有 active flow 的 remaining_bytes：
   elapsed = current_time - flow.last_update_time
   transmitted_bytes = elapsed * flow.current_bw_gbps * 1e9 / 8
   flow.remaining_bytes = max(0, flow.remaining_bytes - int(transmitted_bytes))
   flow.last_update_time = current_time

2. 调用 allocator.allocate() 得到新的带宽分配 new_bw

3. 对每条 active flow：
   flow.current_bw_gbps = new_bw[flow.task_id]
   if flow.current_bw_gbps > 0:
       transmission_us = (flow.remaining_bytes * 8) / (flow.current_bw_gbps * 1e9)
       flow.estimated_end_time = current_time + int(transmission_us)
   else:
       flow.estimated_end_time = INF
   flow.version += 1
   安排新的 flow_completion(flow.task_id, flow.version, time=flow.estimated_end_time)
```

---

## 关键设计决策

### 1. 懒删除（Lazy Deletion）

带宽重新分配时，旧的 `flow_completion` 事件不从 heapq 中删除，而是在处理时通过 `version` 字段判断是否过期。这避免了 O(n) 的队列重建，是离散事件模拟的标准做法。

### 2. Compute 和 Flow 两类管道独立

- Compute 管道：每个节点严格串行，由 `compute_order` 控制，`compute_cursor` 追踪进度
- Flow 管道：只要 deps 满足就立即发出，多条流可同时在网络中传输
- 两类管道通过 `dep_count` 和 `dependents` 图相互触发，但各自独立推进

### 3. remaining_bytes 追踪

每次带宽重新分配时，先根据上次分配的带宽和经过的时间更新 `remaining_bytes`，再用新带宽计算新的 `estimated_end_time`。这确保了带宽变化时的正确性。

### 4. 路径传播延迟（简化模型）

当前版本在 `flow_ready` 时将传播延迟折算到完成时间中：

```python
propagation_delay = sum(
    topology.get_link(path[i], path[i+1]).latency_us
    for i in range(len(path) - 1)
)
transmission_us = (flow.size_bytes * 8) / (flow.current_bw_gbps * 1e9)
flow.estimated_end_time = current_time + propagation_delay + int(transmission_us)
```

**局限性**：传播期间流实际上还未开始占用链路带宽，但模型中已将其计入 active_flows 参与带宽竞争。在竞争激烈或多跳场景下会产生一定偏差。

**未来扩展：逐链路时间窗口模拟**

更精确的建模方式是让流依次经过路径上的每条链路：

1. 每条链路维护自己的 active_flow 队列和带宽分配
2. 流到达链路 i 时注册 `arrival_link` 事件
3. 链路 i 上的带宽分配 = link_bw / count_on_this_link
4. 流离开链路 i 的时间 = 到达时间 + size / allocated_bw
5. 流到达链路 i+1 的时间 = 离开链路 i 的时间 + link_latency(i, i+1)
6. 流的最终完成时间 = 离开最后一条链路的时间

这种方式能准确反映"传播延迟导致流在不同链路上处于不同阶段"的物理现实，但事件数量从 O(F) 增加到 O(F*H)（H = 平均跳数），实现复杂度显著提高。可作为后续精度优化方向。

---

## 验收标准

- 独占带宽模式下（无竞争），flow 延迟 = `size_bytes * 8 / (link_bw_gbps * 1e9)` 微秒
- 竞争模式下，两条流共享同一链路时各得一半带宽，完成时间约为独占时的 2 倍
- 简单 Ring AllReduce workload 的执行结果与手动计算一致
- 1000 条 flow 的 workload 在 10 秒内完成模拟

---

*文档创建日期：2026-04-22*

# Analytical Executor 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现 Analytical Executor，支持基于 P2PWorkload + ExecutionPlan 的离散事件模拟。

**Architecture:** 三个模块：result.py（数据结构）、bandwidth.py（带宽分配）、analytical.py（事件循环）。使用 heapq + 懒删除实现离散事件模拟。

**Tech Stack:** Python dataclasses, heapq, abc.ABC, src.workload_format.schema, src.static_analysis.*

---

## 文件映射

| 文件 | 职责 |
|------|------|
| `src/executor/result.py` | TaskTiming, ExecutionResult dataclass |
| `src/executor/bandwidth.py` | BandwidthAllocator ABC, FairShareAllocator |
| `src/executor/analytical.py` | AnalyticalExecutor, ActiveFlow, Event, 事件循环 |
| `src/executor/__init__.py` | 导出公共 API |
| `tests/test_analytical_executor.py` | 端到端测试 |

---

### Task 1: 创建 result.py

**Files:**
- Create: `src/executor/result.py`

- [ ] **Step 1: 定义 TaskTiming 和 ExecutionResult**

```python
"""Execution result data structures."""
from dataclasses import dataclass


@dataclass
class TaskTiming:
    """单个 task 的时间信息。"""
    task_id: int
    node: int          # compute 任务的节点；flow 任务用 src
    task_type: str     # "compute" | "flow"
    start_time_us: int
    end_time_us: int


@dataclass
class ExecutionResult:
    """执行结果。"""
    per_task: dict[int, TaskTiming]
    job_iteration_times: dict[int, int]   # job_id → 单次 iteration 时间
    total_time_us: int                    # 所有 task 的最晚 end_time
    makespan_us: int                      # max(end_time) - min(start_time)
```

- [ ] **Step 2: Commit**

```bash
git add src/executor/result.py
git commit -m "feat: add result data structures"
```

---

### Task 2: 创建 bandwidth.py

**Files:**
- Create: `src/executor/bandwidth.py`
- Test: `tests/test_bandwidth.py`

- [ ] **Step 1: 导入依赖**

```python
"""Bandwidth allocation strategies."""
from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..static_analysis.routing_hints import RoutingHints
from ..static_analysis.topology_loader import NetworkTopology


# Forward reference to avoid circular import
# ActiveFlow will be defined in analytical.py at runtime
```

- [ ] **Step 2: 定义 BandwidthAllocator 基类**

```python
class BandwidthAllocator(ABC):
    """带宽分配策略接口。"""

    @abstractmethod
    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        current_time: int,
    ) -> dict[int, float]:
        """
        根据当前 active flow 集合和拓扑信息，计算每条流的带宽分配。

        Args:
            active_flows: 当前正在传输的流列表 (list[ActiveFlow])
            topology: 网络拓扑
            routing_hints: 路由提示（用于路径查询）
            current_time: 当前全局时间

        Returns:
            task_id → allocated_bw_gbps 的映射
        """
        pass
```

- [ ] **Step 3: 实现 FairShareAllocator**

```python
class FairShareAllocator(BandwidthAllocator):
    """逐链路均分策略：每条链路上 n 条流各得 bw/n，取路径瓶颈。"""

    def allocate(
        self,
        active_flows: list,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        current_time: int,
    ) -> dict[int, float]:
        if not active_flows:
            return {}

        # Step 1: 统计每条链路上有多少 active flow 经过
        link_flow_count: dict[tuple[int, int], int] = {}
        flow_links: dict[int, list[tuple[int, int]]] = {}

        for flow in active_flows:
            path = flow.path
            links = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            flow_links[flow.task_id] = links
            for link in links:
                link_flow_count[link] = link_flow_count.get(link, 0) + 1

        # Step 2: 对每条 flow，计算其路径上每条链路的 fair share，取最小值
        result: dict[int, float] = {}
        for flow in active_flows:
            min_bw = float('inf')
            for link in flow_links[flow.task_id]:
                link_obj = topology.get_link(link[0], link[1])
                if link_obj is None:
                    continue
                fair_share = link_obj.bandwidth_gbps / link_flow_count[link]
                min_bw = min(min_bw, fair_share)
            result[flow.task_id] = min_bw if min_bw != float('inf') else 0.0

        return result
```

- [ ] **Step 4: Commit**

```bash
git add src/executor/bandwidth.py
git commit -m "feat: add bandwidth allocator with fair share"
```

---

### Task 3: 创建 analytical.py - 数据结构部分

**Files:**
- Create: `src/executor/analytical.py`

- [ ] **Step 1: 导入和 Event/ActiveFlow 定义**

```python
"""Analytical Executor - discrete event simulation for P2P workloads."""
import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from ..static_analysis.analyzer import WorkloadAnalysisResult
from ..static_analysis.routing_hints import RoutingHints
from ..static_analysis.topology_loader import NetworkTopology
from ..workload_format.schema import P2PWorkload, TaskType
from .bandwidth import BandwidthAllocator, FairShareAllocator
from .result import ExecutionResult, TaskTiming


@dataclass(order=True)
class Event:
    """事件：按时间排序，seq 打破平局。"""
    time: int
    seq: int = field(compare=True)
    kind: str = field(compare=False)  # "compute_ready" | "compute_done" | "flow_ready" | "flow_completion"
    task_id: int = field(compare=False)
    version: int = field(compare=False, default=0)  # 仅 flow_completion 使用


@dataclass
class ActiveFlow:
    """当前正在传输的流。"""
    task_id: int
    src: int
    dst: int
    size_bytes: int
    remaining_bytes: int       # 剩余未传输字节数
    path: list[int]            # 网络路径
    start_time: int            # 流开始时间（微秒）
    last_update_time: int      # 上次带宽重新分配的时间
    current_bw_gbps: float = 0.0
    estimated_end_time: int = 0
    version: int = 0           # 懒删除版本号
```

- [ ] **Step 2: AnalyticalExecutor 类骨架和 __init__**

```python
class AnalyticalExecutor:
    """离散事件模拟器：处理 P2PWorkload + ExecutionPlan。"""

    def __init__(
        self,
        topology: NetworkTopology,
        routing_hints: RoutingHints,
        allocator: Optional[BandwidthAllocator] = None,
    ):
        self.topology = topology
        self.routing_hints = routing_hints
        self.allocator = allocator or FairShareAllocator()

    def execute(
        self,
        workload: P2PWorkload,
        plan,  # ExecutionPlan, lazy import to avoid circular
    ) -> ExecutionResult:
        ...
```

- [ ] **Step 3: Commit（占位，后续步骤补充完整实现）**

```bash
git add src/executor/analytical.py
git commit -m "feat: add analytical executor skeleton"
```

---

### Task 4: 实现事件循环主体

在 `AnalyticalExecutor.execute()` 中实现完整的事件循环。

- [ ] **Step 1: 初始化数据结构**

```python
def execute(self, workload, plan) -> ExecutionResult:
    from .result import ExecutionResult, TaskTiming

    # 构建索引
    task_map = {task.task_id: task for task in workload.tasks}
    
    # dep_count: 每个 task 还需要多少 dep 完成
    dep_count: dict[int, int] = {}
    # dependents: task A 完成后，哪些 task 的 dep 计数减 1
    dependents: dict[int, list[int]] = defaultdict(list)
    
    for task in workload.tasks:
        dep_count[task.task_id] = len(task.deps)
        for dep in task.deps:
            dependents[dep].append(task.task_id)

    # compute_cursor: 每个节点当前执行到的 compute_order 位置
    compute_cursor: dict[int, int] = defaultdict(int)

    # 记录时间
    start_times: dict[int, int] = {}
    end_times: dict[int, int] = {}

    # 事件队列
    event_queue: list[Event] = []
    seq_counter = 0

    # active flows
    active_flows: dict[int, ActiveFlow] = {}

    # 初始化：将每个节点的第一个 compute 加入队列
    for node_id, compute_ids in plan.compute_order.items():
        if compute_ids:
            first_task_id = compute_ids[0]
            heapq.heappush(event_queue, Event(
                time=0, seq=seq_counter, kind="compute_ready", task_id=first_task_id
            ))
            seq_counter += 1

    # 初始化：将没有依赖的 flow 加入队列
    for task in workload.tasks:
        if task.is_flow() and dep_count[task.task_id] == 0:
            heapq.heappush(event_queue, Event(
                time=0, seq=seq_counter, kind="flow_ready", task_id=task.task_id
            ))
            seq_counter += 1
```

- [ ] **Step 2: 事件循环主逻辑**

```python
    # 事件循环
    while event_queue:
        event = heapq.heappop(event_queue)
        current_time = event.time

        if event.kind == "compute_ready":
            self._handle_compute_ready(event, task_map, start_times, event_queue, seq_counter)
            seq_counter += 1

        elif event.kind == "compute_done":
            self._handle_compute_done(
                event, task_map, end_times, dep_count, dependents,
                plan, compute_cursor, event_queue, seq_counter, active_flows
            )
            seq_counter += 1

        elif event.kind == "flow_ready":
            self._handle_flow_ready(
                event, task_map, start_times, active_flows,
                event_queue, seq_counter, current_time
            )
            seq_counter += 1

        elif event.kind == "flow_completion":
            self._handle_flow_completion(
                event, active_flows, end_times, dep_count, dependents,
                event_queue, seq_counter, current_time
            )
            seq_counter += 1

    # 构建结果
    per_task = {}
    for task_id, start in start_times.items():
        end = end_times.get(task_id, start)
        task = task_map[task_id]
        per_task[task_id] = TaskTiming(
            task_id=task_id,
            node=task.node if task.is_compute() else task.src,
            task_type="compute" if task.is_compute() else "flow",
            start_time_us=start,
            end_time_us=end,
        )

    # 计算 job_iteration_times
    job_times: dict[int, tuple[int, int]] = defaultdict(lambda: (float('inf'), 0))
    for task_id, timing in per_task.items():
        job_id = task_map[task_id].job_id
        s, e = job_times[job_id]
        job_times[job_id] = (min(s, timing.start_time_us), max(e, timing.end_time_us))

    job_iteration_times = {
        job_id: end - start for job_id, (start, end) in job_times.items()
    }

    all_starts = [t.start_time_us for t in per_task.values()]
    all_ends = [t.end_time_us for t in per_task.values()]
    total_time = max(all_ends) if all_ends else 0
    makespan = max(all_ends) - min(all_starts) if all_starts and all_ends else 0

    return ExecutionResult(
        per_task=per_task,
        job_iteration_times=job_iteration_times,
        total_time_us=total_time,
        makespan_us=makespan,
    )
```

- [ ] **Step 3: Commit**

```bash
git add src/executor/analytical.py
git commit -m "feat: add event loop main logic"
```

---

### Task 5: 实现事件处理器

- [ ] **Step 1: _handle_compute_ready**

```python
def _handle_compute_ready(
    self, event, task_map, start_times, event_queue, seq_counter
):
    task_id = event.task_id
    start_times[task_id] = event.time
    task = task_map[task_id]
    heapq.heappush(event_queue, Event(
        time=event.time + task.duration_us,
        seq=seq_counter,
        kind="compute_done",
        task_id=task_id,
    ))
```

- [ ] **Step 2: _handle_compute_done**

```python
def _handle_compute_done(
    self, event, task_map, end_times, dep_count, dependents,
    plan, compute_cursor, event_queue, seq_counter, active_flows
):
    task_id = event.task_id
    end_times[task_id] = event.time
    task = task_map[task_id]
    node_id = task.node

    # 触发下游依赖
    for downstream_id in dependents.get(task_id, []):
        dep_count[downstream_id] -= 1
        if dep_count[downstream_id] == 0:
            downstream_task = task_map[downstream_id]
            if downstream_task.is_flow():
                heapq.heappush(event_queue, Event(
                    time=event.time, seq=seq_counter,
                    kind="flow_ready", task_id=downstream_id,
                ))
                seq_counter += 1

    # 推进该节点的 compute_cursor
    compute_cursor[node_id] += 1
    cursor = compute_cursor[node_id]
    if node_id in plan.compute_order and cursor < len(plan.compute_order[node_id]):
        next_task_id = plan.compute_order[node_id][cursor]
        if dep_count[next_task_id] == 0:
            heapq.heappush(event_queue, Event(
                time=event.time, seq=seq_counter,
                kind="compute_ready", task_id=next_task_id,
            ))
            seq_counter += 1
```

- [ ] **Step 3: _handle_flow_ready**

```python
def _handle_flow_ready(
    self, event, task_map, start_times, active_flows,
    event_queue, seq_counter, current_time
):
    task_id = event.task_id
    task = task_map[task_id]
    start_times[task_id] = current_time

    # 查询路径
    path = self.routing_hints.get_path(self.topology, task.src, task.dst)

    # 计算传播延迟
    propagation_delay = 0
    for i in range(len(path) - 1):
        link = self.topology.get_link(path[i], path[i + 1])
        if link:
            propagation_delay += link.latency_us

    # 创建 ActiveFlow
    flow = ActiveFlow(
        task_id=task_id,
        src=task.src,
        dst=task.dst,
        size_bytes=task.size_bytes,
        remaining_bytes=task.size_bytes,
        path=path,
        start_time=current_time,
        last_update_time=current_time,
        version=0,
    )
    active_flows[task_id] = flow

    # 重新分配带宽
    self._reallocate_bandwidth(current_time, active_flows, event_queue, seq_counter)
```

- [ ] **Step 4: _handle_flow_completion**

```python
def _handle_flow_completion(
    self, event, active_flows, end_times, dep_count, dependents,
    event_queue, seq_counter, current_time
):
    task_id = event.task_id
    version = event.version

    # 懒删除检查
    if task_id not in active_flows or active_flows[task_id].version != version:
        return

    end_times[task_id] = current_time
    del active_flows[task_id]

    # 触发下游依赖
    for downstream_id in dependents.get(task_id, []):
        dep_count[downstream_id] -= 1
        if dep_count[downstream_id] == 0:
            # 这里简化处理：假设下游只可能是 flow（实际需根据 task 类型判断）
            # 需要访问 task_map 来确定类型
            pass  # 需要在参数中加入 task_map

    # 重新分配带宽
    self._reallocate_bandwidth(current_time, active_flows, event_queue, seq_counter)
```

- [ ] **Step 5: Commit**

```bash
git add src/executor/analytical.py
git commit -m "feat: add event handlers"
```

---

### Task 6: 实现 _reallocate_bandwidth

- [ ] **Step 1: 带宽重新分配方法**

```python
def _reallocate_bandwidth(
    self, current_time, active_flows, event_queue, seq_counter
):
    if not active_flows:
        return

    flows_list = list(active_flows.values())

    # Step 1: 更新所有 active flow 的 remaining_bytes
    for flow in flows_list:
        elapsed = current_time - flow.last_update_time
        if elapsed > 0 and flow.current_bw_gbps > 0:
            transmitted_bytes = int(elapsed * flow.current_bw_gbps * 1e9 / 8)
            flow.remaining_bytes = max(0, flow.remaining_bytes - transmitted_bytes)
        flow.last_update_time = current_time

    # Step 2: 调用 allocator 得到新的带宽分配
    new_bw = self.allocator.allocate(
        flows_list, self.topology, self.routing_hints, current_time
    )

    # Step 3: 更新每条 flow 的带宽和预计完成时间
    for flow in flows_list:
        flow.current_bw_gbps = new_bw.get(flow.task_id, 0.0)
        if flow.current_bw_gbps > 0 and flow.remaining_bytes > 0:
            transmission_us = int(flow.remaining_bytes * 8 / (flow.current_bw_gbps * 1e9))
            flow.estimated_end_time = current_time + transmission_us
        else:
            flow.estimated_end_time = current_time  # Already done or zero bandwidth
        flow.version += 1
        heapq.heappush(event_queue, Event(
            time=flow.estimated_end_time,
            seq=seq_counter,
            kind="flow_completion",
            task_id=flow.task_id,
            version=flow.version,
        ))
        seq_counter += 1

    return seq_counter
```

注意：`_handle_flow_completion` 中调用 `_reallocate_bandwidth` 时需要传入 `task_map` 来处理下游依赖。修正签名：

```python
def _reallocate_bandwidth(
    self, current_time, active_flows, event_queue, seq_counter
):
    ...
    return seq_counter
```

- [ ] **Step 2: Commit**

```bash
git add src/executor/analytical.py
git commit -m "feat: add bandwidth reallocation"
```

---

### Task 7: 创建 __init__.py

- [ ] **Step 1: 导出公共 API**

```python
"""SimAI Flow Scheduler Executor."""
from .analytical import AnalyticalExecutor
from .bandwidth import BandwidthAllocator, FairShareAllocator
from .result import ExecutionResult, TaskTiming

__all__ = [
    "AnalyticalExecutor",
    "BandwidthAllocator",
    "FairShareAllocator",
    "ExecutionResult",
    "TaskTiming",
]
```

- [ ] **Step 2: Commit**

```bash
git add src/executor/__init__.py
git commit -m "feat: add executor package exports"
```

---

### Task 8: 编写端到端测试

- [ ] **Step 1: 创建简单 Ring AllReduce 测试**

```python
"""Tests for AnalyticalExecutor."""
import pytest

from src.executor.analytical import AnalyticalExecutor
from src.executor.result import TaskTiming
from src.static_analysis.routing_hints import compute_routing_hints
from src.static_analysis.topology_loader import TopologyLoader
from src.workload_format.schema import (
    P2PWorkload, Meta, Network, Task, TaskType, CommType, Phase
)
from src.static_analysis.task_serializer import ExecutionPlan


def _make_simple_workload() -> P2PWorkload:
    """创建一个简单的 2 节点 workload：1 个 compute + 1 个 flow。"""
    tasks = [
        Task(
            task_id=0, job_id=0, type=TaskType.COMPUTE,
            node=0, duration_us=100, phase=Phase.FORWARD,
        ),
        Task(
            task_id=1, job_id=0, type=TaskType.FLOW,
            src=0, dst=1, size_bytes=1000, comm_type=CommType.TP_ALLREDUCE_RING,
            deps=[0], phase=Phase.FORWARD,
        ),
    ]
    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        network=Network(topology_file=""),
        tasks=tasks,
    )


def _make_simple_topology() -> NetworkTopology:
    """创建一个简单的 2 节点拓扑。"""
    from src.static_analysis.topology_loader import Link, NetworkTopology
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    topo.node_types = {0: "gpu", 1: "gpu"}

    link = Link(src=0, dst=1, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0.0)
    topo.add_link(link)
    rev_link = Link(src=1, dst=0, bandwidth_gbps=100.0, latency_us=1.0, error_rate=0.0)
    topo.add_link(rev_link)

    return topo


def test_simple_workload_execution():
    """测试简单 workload 的执行。"""
    workload = _make_simple_workload()
    topo = _make_simple_topology()
    routing_hints = compute_routing_hints(topo, workload)
    plan = ExecutionPlan(compute_order={0: [0]})  # Node 0 has compute task 0

    executor = AnalyticalExecutor(topo, routing_hints)
    result = executor.execute(workload, plan)

    # 验证 compute 任务
    assert 0 in result.per_task
    assert result.per_task[0].task_type == "compute"
    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100

    # 验证 flow 任务
    assert 1 in result.per_task
    assert result.per_task[1].task_type == "flow"
    # flow 应该在 compute 完成后开始（dep）
    assert result.per_task[1].start_time_us >= 100


def test_exclusive_bandwidth_latency():
    """测试独占带宽模式下的延迟计算。"""
    # size=1000 bytes, bw=100 Gbps => transmission = 1000*8 / 100e9 = 0.08 us
    # 加上 latency=1 us（单跳），总延迟应约为 1.08 us
    # 但由于我们用的是整数微秒，transmission_us = int(0.08) = 0
    # 所以主要看 latency
    workload = _make_simple_workload()
    topo = _make_simple_topology()
    routing_hints = compute_routing_hints(topo, workload)
    plan = ExecutionPlan(compute_order={0: [0]})

    executor = AnalyticalExecutor(topo, routing_hints)
    result = executor.execute(workload, plan)

    flow_timing = result.per_task[1]
    # start_time >= 100 (after compute), end_time = start + propagation_delay + transmission
    # propagation_delay = 1 us (single hop), transmission = int(1000*8 / 100e9) = 0
    assert flow_timing.end_time_us >= flow_timing.start_time_us + 1
```

- [ ] **Step 2: 运行测试**

```bash
cd simai-flow-scheduler
pytest tests/test_analytical_executor.py -v
```

Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_analytical_executor.py
git commit -m "test: add analytical executor end-to-end tests"
```

---

*Plan created: 2026-04-22*

# Phase 3 静态分析方向清单

本文档记录了 Phase 3（静态分析阶段）可以提取的 workload 特征信息。这些信息**必须通过分析 P2P Workload DAG 和网络拓扑才能得到**，不是用户配置的策略参数。

---

## 设计原则

### 为什么需要静态分析？

1. **全局视角 vs 局部视角**：实际集群中每个节点只有局部信息，Phase 3 提供全局预分析结果供动态调度器参考
2. **加速决策**：预计算复杂的全局分析（如关键路径、冲突组），避免 Phase 4 运行时重复计算
3. **辅助优化**：为动态调度器提供 workload 特征的"地图"，帮助其做出更智能的实时决策

### 静态分析的定位

- **不是**决定"用什么调度策略"（这是用户配置）
- **而是**提取 workload 的结构特征和统计信息
- 输出可以被 Phase 4 **可选地使用**（动态调度器应能独立工作）

---

## 分析方向清单

### 1. 关键路径分析（Critical Path Analysis）

**目标**：识别真正决定整体完成时间的任务链

#### 1.1 基于时间的关键路径

```python
@dataclass
class CriticalPathInfo:
    """Time-based critical path analysis."""
    
    # Earliest start time (ASAP - As Soon As Possible)
    earliest_start_us: dict[int, int]  # task_id → time
    
    # Earliest finish time
    earliest_finish_us: dict[int, int]
    
    # Latest start time (ALAP - As Late As Possible)
    latest_start_us: dict[int, int]
    
    # Latest finish time
    latest_finish_us: dict[int, int]
    
    # Slack time = latest_start - earliest_start
    slack_us: dict[int, int]
    
    # Tasks on the critical path (slack == 0)
    critical_path_tasks: list[int]
    
    # Total critical path duration
    critical_path_length_us: int
```

**计算方法**：
- Forward pass: 计算 `earliest_start` 和 `earliest_finish`
- Backward pass: 计算 `latest_start` 和 `latest_finish`
- `slack = latest_start - earliest_start`
- 关键路径 = slack == 0 的 tasks

**Flow 传输时间估算**：
```python
def estimate_flow_latency(flow: Task, topology: NetworkTopology) -> int:
    """使用最大可用带宽估算传输时间"""
    link = topology.get_link(flow.src, flow.dst)
    if link:
        # Optimistic estimate: assume exclusive link access
        return (flow.size_bytes * 8) / (link.bandwidth_gbps * 1e9)
    else:
        return _estimate_multihop_latency(flow, topology)
```

#### 1.2 每个 Task 的时间窗口

```python
@dataclass
class TaskTimeWindow:
    """Task 的可能执行时间窗口"""
    task_id: int
    earliest_start: int
    earliest_finish: int
    latest_start: int
    latest_finish: int
    slack: int
    is_flexible: bool  # slack > threshold
    is_critical: bool  # slack == 0
```

---

### 2. 链路竞争分析（Link Contention Analysis）

**目标**：识别哪些 flows 会竞争同一条物理链路

#### 2.1 潜在竞争组

```python
@dataclass
class LinkContentionGroup:
    """Flows sharing the same physical link."""
    link_id: tuple[int, int]  # (src_node, dst_node)
    competing_tasks: list[int]  # task_ids using this link
    total_data_bytes: int
    min_size_bytes: int
    max_size_bytes: int
    avg_size_bytes: float
    num_flows: int
    
    # Worst case: all flows send simultaneously
    worst_case_concurrency: int
    
    # Best case: considering dependencies, minimum possible concurrency
    best_case_concurrency: int
```

#### 2.2 并发可能性分析

```python
@dataclass
class ConcurrencyAnalysis:
    """Which flows can potentially execute concurrently?"""
    
    # Pairs that MAY execute concurrently (no dependency + overlapping time windows)
    potentially_concurrent_pairs: list[tuple[int, int]]
    
    # Pairs that CANNOT execute concurrently (have ancestor relationship)
    mutually_exclusive_pairs: list[tuple[int, int]]
    
    # Maximum possible concurrent flows per link
    max_concurrent_per_link: dict[tuple[int, int], int]
```

**判断并发的条件**：
```python
def may_concurrent(task_a: Task, task_b: Task, dag_info: DAGInfo) -> bool:
    # 1. No ancestor-descendant relationship
    if is_ancestor(a, b) or is_ancestor(b, a):
        return False
    
    # 2. Time windows may overlap
    if a.earliest_finish < b.earliest_start or b.earliest_finish < a.earliest_start:
        return False
    
    return True
```

---

### 3. 通信模式特征（Communication Pattern Profiling）

**目标**：统计 workload 的通信行为特征

#### 3.1 按执行阶段统计

```python
@dataclass
class PhaseCommunicationProfile:
    """Communication characteristics per execution phase."""
    phase: str  # forward, backward_input, backward_weight, optimizer
    
    total_flows: int
    total_bytes: int
    avg_flow_size: float
    comm_density: float  # bytes / layer
    
    # Communication type distribution
    comm_type_breakdown: dict[str, int]  # comm_type → count
```

#### 3.2 按 Layer 统计

```python
@dataclass
class LayerProfile:
    """Per-layer communication and computation profile."""
    layer_id: int
    
    compute_task_ids: list[int]
    total_compute_time_us: int
    
    comm_task_ids: list[int]
    total_comm_bytes: int
    comm_types: list[str]
    
    # Bottleneck analysis
    bottleneck: str  # "compute" or "communication"
    compute_comm_ratio: float  # compute_time / estimated_comm_time
```

#### 3.3 按 GA Iteration 统计

```python
@dataclass
class IterationProfile:
    """Per GA iteration profile."""
    iteration: int  # -1 (pre), 0 to ga-1 (layers), ga (post)
    
    num_layers: int
    total_comm_bytes: int
    total_compute_time_us: int
    comm_compute_ratio: float
```

---

### 4. 节点级别的局部视图（Node-Centric Local Views）

**目标**：为每个节点预计算它的"局部视角"，支持分布式调度决策

#### 4.1 单节点视图

```python
@dataclass
class NodeLocalView:
    """Local scheduling view for a single node."""
    
    node_id: int
    
    # Flows this node sends
    send_tasks: list[int]  # task_ids
    total_send_bytes: int
    
    # Flows this node receives
    receive_tasks: list[int]  # task_ids
    total_receive_bytes: int
    
    # Collective operations this node participates in
    collective_participations: list[int]  # collective_group_ids
    
    # Estimated schedule (based on ASAP)
    estimated_send_times: list[tuple[int, int]]  # (start_time_us, task_id)
    estimated_receive_times: list[tuple[int, int]]
    
    # Bottleneck links for this node
    busiest_outgoing_link: tuple[int, int]
    busiest_incoming_link: tuple[int, int]
    
    # Estimated busy ratio (time spent communicating / total time)
    estimated_busy_ratio: float  # 0.0 to 1.0
```

#### 4.2 流量矩阵

```python
@dataclass
class TrafficMatrix:
    """Node-to-node traffic volume matrix."""
    
    # traffic[(i, j)] = total bytes from node i to node j
    traffic: dict[tuple[int, int], int]
    
    # Top senders/receivers
    top_senders: list[tuple[int, int]]  # (node_id, total_sent_bytes)
    top_receivers: list[tuple[int, int]]
    
    # Bidirectional communication pairs
    bidirectional_pairs: list[tuple[int, int]]
```

---

### 5. 依赖链深度分析（Dependency Chain Analysis）

**目标**：理解任务间的依赖关系结构

#### 5.1 依赖链特征

```python
@dataclass
class DependencyChainProfile:
    """Profile of a dependency chain starting from a task."""
    start_task_id: int
    
    chain_length: int  # number of tasks
    chain_duration_us: int  # total time
    communication_fraction: float  # comm_time / total_time
    
    # Bottleneck task (longest duration in chain)
    bottleneck_task_id: int
    
    # Chains that can run in parallel with this one
    parallel_chain_ids: list[int]
```

#### 5.2 Fan-in/Fan-out 分析

```python
@dataclass
class FanAnalysis:
    """Fan-in and fan-out statistics."""
    
    fan_out: dict[int, int]  # task_id → number of dependent tasks
    fan_in: dict[int, int]   # task_id → number of dependencies
    
    # High fan-out tasks (potential synchronization points)
    high_fan_out_tasks: list[tuple[int, int]]  # (task_id, fan_out)
    
    # Barrier tasks (fan_out > threshold)
    barrier_tasks: list[int]
```

---

### 6. 集体通信模式分析（Collective Operation Analysis）

**目标**：识别属于同一个集体通信操作的 flow 集合

```python
@dataclass
class CollectiveGroup:
    """All flows belonging to one collective operation."""
    collective_id: int
    comm_type: str  # e.g., "tp_allreduce_ring"
    layer_id: int
    iteration: int
    phase: str
    
    participating_ranks: list[int]
    flow_task_ids: list[int]
    
    # First and last flow in the collective
    first_flow_task_id: int
    last_flow_task_id: int
    
    # Estimated completion time (all flows done)
    estimated_completion_time_us: int
    
    # Is this a synchronization point?
    is_synchronization_point: bool
```

---

### 7. 带宽需求分析（Bandwidth Demand Analysis）

**目标**：分析每条链路的带宽需求与容量关系

#### 7.1 每链路带宽需求

```python
@dataclass
class LinkBandwidthDemand:
    """Bandwidth demand analysis for a link."""
    link_id: tuple[int, int]
    
    # Total demand if all flows send simultaneously
    total_demand_gbps: float
    
    # Link capacity
    capacity_gbps: float
    
    # Overload ratio = demand / capacity
    overload_ratio: float  # > 1.0 means oversubscribed
    
    # Worst-case transmission time
    worst_case_transmission_time_us: int
```

#### 7.2 时间窗口化的带宽需求

```python
@dataclass
class TimeWindowedBandwidthDemand:
    """Bandwidth demand over time windows."""
    window_size_us: int
    
    # Demand per window
    demands: dict[int, LinkBandwidthDemand]  # window_start → demand
```

---

### 8. 拓扑感知路由信息（Topology-Aware Routing Hints）

**目标**：预计算路由信息，加速 Phase 4 的路径查找

```python
@dataclass
class RoutingHints:
    """Pre-computed routing information."""
    
    # Shortest paths (for multi-hop topologies)
    shortest_paths: dict[tuple[int, int], list[int]]  # (src, dst) → [hops...]
    
    # Hop counts
    hop_counts: dict[tuple[int, int], int]
    
    # Most frequently used links (bottleneck candidates)
    most_used_links: list[tuple[int, int]]
    
    # Bisection bandwidth
    bisection_bandwidth_gbps: float
```

---

### 9. 松弛时间和灵活性分析（Slack and Flexibility Analysis）

**目标**：识别哪些 tasks 有调度灵活性

```python
@dataclass
class FlexibilityInfo:
    """Scheduling flexibility for a task."""
    task_id: int
    
    # Slack time (can delay without affecting overall completion)
    slack_us: int
    
    # Relative flexibility = slack / duration
    flexibility_ratio: float
    
    # Can be deferred (has positive slack)
    can_defer: bool
    
    # Must schedule ASAP (on critical path)
    must_schedule_asap: bool
```

---

### 10. Workload 整体特征摘要（Workload Summary）

**目标**：workload 的全局统计信息

```python
@dataclass
class WorkloadSummary:
    """Overall workload characteristics."""
    
    # Basic statistics
    total_tasks: int
    total_compute_tasks: int
    total_flow_tasks: int
    total_communication_bytes: int
    total_compute_time_us: int
    
    # Communication/computation ratio
    comm_compute_ratio: float
    
    # Average DAG width (average concurrency)
    avg_dag_width: float
    
    # Critical path length
    critical_path_length_us: int
    
    # Communication fraction on critical path
    critical_path_comm_fraction: float
    
    # Parallelism factor = DAG width / critical path length
    parallelism_factor: float
    
    # Per-phase profiles
    phase_profiles: dict[str, PhaseCommunicationProfile]
    
    # Hottest links
    hot_links: list[tuple[int, int, int]]  # (link_id, bytes, num_flows)
```

---

## Phase 3 输出结构

综合以上分析，Phase 3 的输出数据结构：

```python
@dataclass
class WorkloadAnalysisResult:
    """Complete workload analysis result."""
    
    # 1. Critical path (time-based)
    critical_path: CriticalPathInfo
    
    # 2. Link contention groups
    contention_groups: dict[tuple[int, int], LinkContentionGroup]
    
    # 3. Concurrency analysis
    concurrency: ConcurrencyAnalysis
    
    # 4. Per-node local views
    node_views: dict[int, NodeLocalView]
    
    # 5. Traffic matrix
    traffic_matrix: TrafficMatrix
    
    # 6. Layer profiles
    layer_profiles: dict[int, LayerProfile]
    
    # 7. Iteration profiles
    iteration_profiles: dict[int, IterationProfile]
    
    # 8. Collective groups
    collective_groups: list[CollectiveGroup]
    
    # 9. Bandwidth demands
    bandwidth_demands: dict[tuple[int, int], LinkBandwidthDemand]
    
    # 10. Dependency chains
    dependency_chains: dict[int, DependencyChainProfile]
    
    # 11. Fan analysis
    fan_analysis: FanAnalysis
    
    # 12. Routing hints
    routing_hints: RoutingHints
    
    # 13. Flexibility info
    flexibility: dict[int, FlexibilityInfo]
    
    # 14. Overall summary
    summary: WorkloadSummary
```

---

## Phase 4 如何使用这些信息

### 示例 1: 基于关键路径的调度

```python
class CriticalPathScheduler:
    def __init__(self, analysis: WorkloadAnalysisResult):
        self.critical_tasks = set(analysis.critical_path.critical_path_tasks)
    
    def select_next_task(self, ready_queue: list[Task]) -> Task:
        # Prioritize critical path tasks
        for task in ready_queue:
            if task.task_id in self.critical_tasks:
                return task
        return ready_queue[0]  # Fallback
```

### 示例 2: 基于节点视图的带宽分配

```python
class NodeAwareAllocator:
    def __init__(self, analysis: WorkloadAnalysisResult):
        self.node_views = analysis.node_views
    
    def allocate_bandwidth(self, node_id: int, active_flows: list[FlowState]):
        view = self.node_views[node_id]
        if view.estimated_busy_ratio > 0.8:
            # Busy node: give more bandwidth
            ...
```

### 示例 3: 基于竞争组的拥塞控制

```python
class ContentionController:
    def __init__(self, analysis: WorkloadAnalysisResult):
        self.contention_groups = analysis.contention_groups
        self.current_load: dict[tuple[int, int], int] = defaultdict(int)
    
    def on_flow_start(self, flow: Task):
        link = (flow.src, flow.dst)
        self.current_load[link] += 1
        
        group = self.contention_groups.get(link)
        if group and self.current_load[link] > group.best_case_concurrency:
            flow.throttle = True  # Beyond expected concurrency
```

---

## 下一步

需要从以上分析方向中筛选出：
1. **最有价值**的分析（对动态调度决策帮助最大）
2. **计算成本可接受**的分析（不会使 Phase 3 过慢）
3. **Phase 4 实际会使用**的信息（避免无用计算）

建议优先实现：
- **1. 关键路径分析**（识别瓶颈）
- **2. 链路竞争组**（预知竞争）
- **4. 节点局部视图**（支持分布式调度）
- **10. Workload 摘要**（整体画像）

---

*创建日期：2026-04-14*
*基于 Phase 1 + Phase 2 已完成的基础设施*

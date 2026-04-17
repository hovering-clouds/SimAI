# Phase 3 开发计划：Workload 静态分析器

## 1. 目标

Phase 3 的目标是实现 Workload 静态分析器，从全局视角提取 P2P Workload 的结构特征和统计信息，为 Phase 4 的动态调度器提供决策依据。

**核心设计理念**：

- 实际集群中每个节点只有**局部信息**（自己的发送/接收表）
- Phase 3 提供**全局预分析**结果，弥补局部视角的不足
- Phase 3 **不决定调度策略**，只提取 workload 特征
- Phase 4 的动态调度器应能**独立工作**，但可以利用 Phase 3 的结果优化性能

**包含**：

1. 关键路径分析（基于时间的 ASAP/ALAP、slack 计算）
2. 链路竞争分析（哪些 flows 共享同一条物理链路）
3. 节点局部视图（每个节点的发送/接收表、瓶颈链路）
4. 流量矩阵（节点间的流量分布）
5. 路由信息（最短路径、bisection bandwidth）

**不包含**：

- 调度策略选择（这是用户配置或 Phase 4 的职责）
- 动态带宽分配（Phase 4 运行时决定）
- Collective 语义反推（保持 P2P Workload 的纯粹性）

---

## 2. 前置知识：已有基础设施

### 2.1 核心数据结构（schema.py）

```python
@dataclass
class Task:
    task_id: int
    job_id: int
    type: TaskType             # COMPUTE | FLOW
    iteration: int = 0
    phase: Phase               # FORWARD | BACKWARD_INPUT | BACKWARD_WEIGHT | OPTIMIZER
    layer_id: int = 0
    item_id: int = 0
    deps: list[int] = []
    # Compute fields
    node: Optional[int] = None
    duration_us: Optional[int] = None
    # Flow fields
    src: Optional[int] = None
    dst: Optional[int] = None
    size_bytes: Optional[int] = None
    comm_type: CommType = CommType.UNKNOWN
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None
```

### 2.2 运行测试

```bash
uv run pytest tests/ -v
```

当前状态：**153 tests passed**（Phase 1 + Phase 2）

---

## 3. Phase 3 输出结构

```python
@dataclass
class WorkloadAnalysisResult:
    """Complete workload analysis result."""
  
    # 1. Critical path (time-based)
    critical_path: CriticalPathInfo
  
    # 2. Link contention groups
    contention_groups: dict[tuple[int, int], LinkContentionGroup]
  
    # 3. Per-node local views
    node_views: dict[int, NodeLocalView]
  
    # 4. Traffic matrix
    traffic_matrix: TrafficMatrix
  
    # 5. Routing hints
    routing_hints: RoutingHints
  
    # 6. Overall summary
    summary: WorkloadSummary
```

---

## 4. 开发任务

### Task 0: TopologyLoader（拓扑加载器）

**文件**: `src/static_analysis/topology_loader.py`

**功能**：解析 astra-sim 格式的拓扑文件，构建内存中的网络拓扑图。

#### 4.1 拓扑文件格式

基于 `astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py` 生成的拓扑文件格式：

**示例** (`Spectrum-X_8g_8gps_400Gbps_H100`)：
```
18 8 1 9 24 H100              # 第1行: 节点统计
8 9 10 11 12 13 14 15 16 17   # 第2行: 交换机节点ID列表
0 8 2880Gbps 0.000025ms 0     # 第3行起: 链路定义 (src dst bandwidth latency error_rate)
0 9 400Gbps 0.0005ms 0
...
```

**第一行格式**：
- `total_nodes gpu_count nv_switch_count other_switch_count total_links type`

**第二行格式**：
- 所有交换机节点的 ID 列表（空格分隔）

**后续行格式**：
- `src_id dst_id bandwidth latency error_rate`
- **有向边**：`(src, dst)` 表示从 src 到 dst 的单向链路

#### 4.2 数据结构

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class NodeType(str, Enum):
    """Type of network node."""
    GPU = "gpu"
    NV_SWITCH = "nv_switch"       # NVLink switch
    ASW_SWITCH = "asw_switch"     # Aggregate switch
    PSW_SWITCH = "psw_switch"     # Pod switch
    DSW_SWITCH = "dsw_switch"     # Distribution switch


@dataclass
class Link:
    """Represents a unidirectional physical link in the topology.
    
    Link identification uses (src, dst) tuple as the key, matching
    the topology file format. This is sufficient for bandwidth
    contention analysis since we care about "traffic from A to B".
    """
    src: int
    dst: int
    bandwidth_gbps: float
    latency_us: float
    error_rate: float
    
    @property
    def link_id(self) -> tuple[int, int]:
        """Unique identifier for this link: (src, dst)."""
        return (self.src, self.dst)
    
    def __hash__(self):
        return hash((self.src, self.dst))
    
    def __eq__(self, other):
        if not isinstance(other, Link):
            return False
        return self.src == other.src and self.dst == other.dst


@dataclass
class NetworkTopology:
    """In-memory representation of the network topology.
    
    Provides graph traversal APIs for routing algorithms and
    link lookup for bandwidth/delay queries.
    """
    
    # All links indexed by (src, dst) tuple
    links: dict[tuple[int, int], Link] = field(default_factory=dict)
    
    # Adjacency list: node -> list of (neighbor, link)
    adjacency: dict[int, list[tuple[int, Link]]] = field(default_factory=dict)
    
    # Node type mapping: node_id -> type
    node_types: dict[int, NodeType] = field(default_factory=dict)
    
    # Metadata
    total_nodes: int = 0
    gpu_count: int = 0
    switch_count: int = 0
    
    # List of GPU node IDs for quick iteration
    gpu_nodes: list[int] = field(default_factory=list)
    
    # List of all switch node IDs
    switch_nodes: list[int] = field(default_factory=list)
    
    def add_link(self, link: Link):
        """Add a link and update adjacency list."""
        self.links[link.link_id] = link
        
        if link.src not in self.adjacency:
            self.adjacency[link.src] = []
        self.adjacency[link.src].append((link.dst, link))
        
        # Ensure dst node exists in adjacency (even if it has no outgoing edges yet)
        if link.dst not in self.adjacency:
            self.adjacency[link.dst] = []
    
    def get_link(self, src: int, dst: int) -> Optional[Link]:
        """Get link by (src, dst), or None if not found."""
        return self.links.get((src, dst))
    
    def get_neighbors(self, node: int) -> list[tuple[int, Link]]:
        """Get all outgoing neighbors of a node."""
        return self.adjacency.get(node, [])
    
    def get_gpu_nodes(self) -> list[int]:
        """Return list of all GPU node IDs."""
        return self.gpu_nodes
    
    def get_switch_nodes(self) -> list[int]:
        """Return list of all switch node IDs."""
        return self.switch_nodes
    
    def is_gpu_node(self, node: int) -> bool:
        """Check if a node is a GPU."""
        return node in self.gpu_nodes
    
    def is_switch_node(self, node: int) -> bool:
        """Check if a node is a switch."""
        return node in self.switch_nodes
```

#### 4.3 解析算法

```python
import re
from pathlib import Path


class TopologyLoader:
    """Load and parse astra-sim topology files."""
    
    def load(self, topo_file: str) -> NetworkTopology:
        """
        Parse topology file and build NetworkTopology.
        
        Args:
            topo_file: Path to topology file (astra-sim format)
            
        Returns:
            Populated NetworkTopology object
            
        Raises:
            FileNotFoundError: If topo_file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(topo_file)
        if not path.exists():
            raise FileNotFoundError(f"Topology file not found: {topo_file}")
        
        topo = NetworkTopology()
        
        with open(path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        if len(lines) < 2:
            raise ValueError("Invalid topology file: missing header lines")
        
        # Parse first line: metadata
        topo.total_nodes, topo.gpu_count, nv_switch_count, \
            other_switch_count, total_links, gpu_type = self._parse_header(lines[0])
        
        topo.switch_count = nv_switch_count + other_switch_count
        
        # Parse second line: switch node IDs
        switch_ids = list(map(int, lines[1].split()))
        
        # Classify nodes
        gpu_nodes_set = set(range(topo.gpu_count))
        switch_set = set(switch_ids)
        
        topo.gpu_nodes = sorted(gpu_nodes_set)
        topo.switch_nodes = sorted(switch_set)
        
        # Assign node types
        for node_id in topo.gpu_nodes:
            topo.node_types[node_id] = NodeType.GPU
        
        # Heuristic: classify switches based on ID ranges or position
        # For Spectrum-X: first few are NV switches, rest are ASW/PSW
        for i, sid in enumerate(switch_ids):
            if i < nv_switch_count:
                topo.node_types[sid] = NodeType.NV_SWITCH
            elif i < nv_switch_count + other_switch_count:
                # Could be ASW or PSW, need more context
                # For now, mark as generic switch
                topo.node_types[sid] = NodeType.ASW_SWITCH
        
        # Parse link definitions (lines 3+)
        for line in lines[2:]:
            link = self._parse_link_line(line)
            if link:
                topo.add_link(link)
        
        return topo
    
    def _parse_header(self, line: str) -> tuple[int, int, int, int, int, str]:
        """Parse first line of topology file."""
        parts = line.split()
        if len(parts) < 6:
            raise ValueError(f"Invalid header line: {line}")
        
        return (
            int(parts[0]),  # total_nodes
            int(parts[1]),  # gpu_count
            int(parts[2]),  # nv_switch_count
            int(parts[3]),  # other_switch_count
            int(parts[4]),  # total_links
            parts[5],       # gpu_type
        )
    
    def _parse_link_line(self, line: str) -> Optional[Link]:
        """Parse a link definition line."""
        parts = line.split()
        if len(parts) < 5:
            return None
        
        try:
            src = int(parts[0])
            dst = int(parts[1])
            bandwidth = self._parse_bandwidth(parts[2])
            latency = self._parse_latency(parts[3])
            error_rate = float(parts[4])
            
            return Link(
                src=src,
                dst=dst,
                bandwidth_gbps=bandwidth,
                latency_us=latency,
                error_rate=error_rate
            )
        except (ValueError, IndexError) as e:
            raise ValueError(f"Invalid link line: {line} ({e})")
    
    def _parse_bandwidth(self, bw_str: str) -> float:
        """Parse bandwidth string like '400Gbps' or '2880Gbps' to float (Gbps)."""
        match = re.match(r'([\d.]+)\s*(Gbps|Mbps|Tbps)', bw_str, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid bandwidth format: {bw_str}")
        
        value = float(match.group(1))
        unit = match.group(2).upper()
        
        if unit == 'MBPS':
            return value / 1000.0
        elif unit == 'TBPS':
            return value * 1000.0
        else:  # GBPS
            return value
    
    def _parse_latency(self, lat_str: str) -> float:
        """Parse latency string like '0.0005ms' or '25us' to float (microseconds)."""
        match = re.match(r'([\d.]+)\s*(ms|us|ns|s)', lat_str, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid latency format: {lat_str}")
        
        value = float(match.group(1))
        unit = match.group(2).lower()
        
        if unit == 's':
            return value * 1e6
        elif unit == 'ms':
            return value * 1e3
        elif unit == 'ns':
            return value / 1e3
        else:  # us
            return value
```

#### 4.4 使用示例

```python
# Load topology
loader = TopologyLoader()
topo = loader.load("./inputs/topo/Spectrum-X_8g_8gps_400Gbps_H100")

# Query links
link = topo.get_link(0, 9)  # GPU0 → ASW switch
if link:
    print(f"Bandwidth: {link.bandwidth_gbps} Gbps")
    print(f"Latency: {link.latency_us} us")

# Iterate GPU nodes
for gpu_id in topo.get_gpu_nodes():
    neighbors = topo.get_neighbors(gpu_id)
    print(f"GPU {gpu_id} has {len(neighbors)} outgoing links")
```

#### 4.5 测试

**文件**: `tests/test_topology_loader.py`

```python
def test_load_spectrum_x_topology():
    """Load Spectrum-X topology and verify basic properties."""
    loader = TopologyLoader()
    topo = loader.load("path/to/Spectrum-X_8g_8gps_400Gbps_H100")
    
    assert topo.gpu_count == 8
    assert topo.total_nodes == 18
    assert len(topo.gpu_nodes) == 8
    assert len(topo.switch_nodes) == 10

def test_get_direct_link():
    """Verify direct link lookup."""
    loader = TopologyLoader()
    topo = loader.load("path/to/topology")
    
    link = topo.get_link(0, 9)
    assert link is not None
    assert link.src == 0
    assert link.dst == 9
    assert link.bandwidth_gbps == 400.0

def test_adjacency_list():
    """Verify neighbor traversal."""
    loader = TopologyLoader()
    topo = loader.load("path/to/topology")
    
    neighbors = topo.get_neighbors(0)
    assert len(neighbors) > 0  # GPU should have outgoing links

def test_node_classification():
    """Verify GPU vs switch node classification."""
    loader = TopologyLoader()
    topo = loader.load("path/to/topology")
    
    assert topo.is_gpu_node(0)
    assert not topo.is_switch_node(0)
    assert topo.is_switch_node(8)

def test_invalid_file():
    """Verify error handling for missing file."""
    loader = TopologyLoader()
    with pytest.raises(FileNotFoundError):
        loader.load("nonexistent_file")
```

---

### Task 1: 路由信息（Routing Hints）

**文件**: `src/static_analysis/routing_hints.py`

**功能**：预计算拓扑的路由信息，为关键路径分析提供多跳路径信息。

**为什么 Task 1 需要做路由**：
- Task 2（关键路径分析）需要准确估算多跳 flow 的 duration
- 多跳 flow 的传输时间 = 各链路的 transmission delay + propagation latency
- 必须先知道完整路径才能找到瓶颈链路（最低带宽链路）

#### 4.1 数据结构

```python
from dataclasses import dataclass, field


@dataclass
class RoutingHints:
    """On-demand routing hints with path caching.
    
    Design: 
    - _cached_paths stores node-level paths [src, hop1, hop2, ..., dst]
    - When needed, convert path → links on demand
    - link_loads aggregates statistics for quick contention analysis
    
    Computed eagerly in Phase 3 Task 1 so that Task 2 (critical path)
    can use accurate multi-hop duration estimates.
    """

    # Shortest paths between actual (src, dst) pairs in the workload
    # Computed eagerly during initialization
    _cached_paths: dict[tuple[int, int], list[int]] = field(
        default_factory=dict, repr=False
    )

    # Link load statistics (aggregated across all flows during computation)
    # link_id → number of flows using this link
    link_loads: dict[tuple[int, int], int] = field(default_factory=dict)

    def get_path(self, topology: NetworkTopology, src: int, dst: int) -> list[int]:
        """Path lookup: compute via BFS on first access, cache for reuse."""
        key = (src, dst)
        if key not in self._cached_paths:
            self._cached_paths[key] = _bfs_shortest_path(topology, src, dst) or []
        return self._cached_paths[key]

    def get_flow_links(
        self, task: Task, topology: NetworkTopology
    ) -> list[tuple[int, int]]:
        """
        Convert cached path to physical links for a flow task.
        
        For multi-hop paths, converts [src, hop1, hop2, dst] → 
        [(src,hop1), (hop1,hop2), (hop2,dst)]
        """
        if task.src is None or task.dst is None:
            return []
        
        path = self.get_path(topology, task.src, task.dst)
        if not path:
            # Fallback: treat as direct link
            return [(task.src, task.dst)]
        
        return [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    def get_most_used_links(self, top_k: int = 20) -> list[tuple[tuple[int, int], int]]:
        """Compute top-K most used links on demand from link_loads."""
        return sorted(
            self.link_loads.items(), key=lambda x: x[1], reverse=True
        )[:top_k]
```

#### 4.2 算法

```python
from collections import deque


def compute_routing_hints(
    topology: NetworkTopology,
    workload: P2PWorkload,
) -> RoutingHints:
    """
    Compute routing hints by finding shortest paths for all flows.
    
    Steps:
    1. For each flow task, find shortest path via BFS
    2. Cache the path for later use by critical path analysis
    3. Aggregate link_loads for quick hotspot detection
    
    Memory: O(P * L) where P = unique (src,dst) pairs, L = avg path length
    Compute: O(F * (V+E)) - BFS for each flow task
    """
    hints = RoutingHints()

    # Process each flow task
    for task in workload.tasks:
        if task.type.value != "flow":
            continue
        if task.src is None or task.dst is None:
            continue
        
        # This triggers BFS and caches the path
        path = hints.get_path(topology, task.src, task.dst)
        
        # Aggregate link loads from this path
        if len(path) >= 2:
            links = [(path[i], path[i+1]) for i in range(len(path)-1)]
        else:
            links = [(task.src, task.dst)]  # fallback
        
        for link in links:
            hints.link_loads[link] = hints.link_loads.get(link, 0) + 1

    return hints


def _bfs_shortest_path(
    topology: NetworkTopology,
    src: int,
    dst: int,
) -> list[int] | None:
    """BFS to find shortest path (by hop count) from src to dst."""
    if src == dst:
        return [src]

    visited = {src}
    queue = deque([(src, [src])])

    while queue:
        current, path = queue.popleft()
        for neighbor, link in topology.get_neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == dst:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))

    return None
```

#### 4.3 测试

**文件**: `tests/test_routing_hints.py`

```python
def test_direct_link_path():
    """Direct link: path = [src, dst], hop_count = 1."""

def test_two_hop_path():
    """GPU0 → Switch → GPU1: path = [0, switch, 1], hop_count = 2."""

def test_flow_links_direct():
    """Flow on direct link maps to exactly one physical link."""

def test_flow_links_multihop():
    """Flow through switch maps to two physical links."""

def test_most_used_links_ordering():
    """Links used by more flows appear first."""

def test_no_path_fallback():
    """Flow with no path in topology falls back to direct link."""
```

---

### Task 2: 关键路径分析（Critical Path Analysis）

**文件**: `src/static_analysis/critical_path.py`

**功能**：基于时间的关键路径分析，计算每个 task 的时间窗口和 slack。

#### 4.1 统一输出格式

输出格式设计为**与分析算法无关**，三种算法（CPM / TTE / RCPSP）均填充相同的数据结构，Phase 4 只需读取 `slack_us` 和 `is_critical`，无需感知底层算法。

```python
from dataclasses import dataclass


@dataclass
class TaskTimingInfo:
    """
    Timing analysis result for a single task.

    Compatible with all analysis methods:
    - CPM:   all fields populated via forward/backward pass
    - TTE:   slack_us == TTE for flow tasks; compute tasks set to inf
    - RCPSP: earliest_* reflects resource-constrained schedule
    """
    task_id: int

    # Forward pass results (always present)
    earliest_start_us: int
    earliest_finish_us: int

    # Backward pass results (may be approximate depending on method)
    latest_start_us: int
    latest_finish_us: int

    # Slack = latest_start - earliest_start
    # For flow tasks under TTE method: equivalent to TTE value
    # float to allow inf (tasks with no downstream compute dependency)
    slack_us: float

    # Is this task on the critical path?
    is_critical: bool  # slack_us == 0


@dataclass
class CriticalPathInfo:
    """
    Critical path analysis result.

    analysis_method records which algorithm was used,
    allowing callers to interpret precision accordingly.
    """
    # Per-task timing info
    task_timings: dict[int, TaskTimingInfo]  # task_id → timing

    # Critical tasks (slack == 0)
    critical_tasks: list[int]

    # Total makespan (optimistic lower bound)
    makespan_us: int

    # Algorithm used: "cpm" | "tte" | "rcpsp"
    analysis_method: str

    def get_slack(self, task_id: int) -> float:
        return self.task_timings[task_id].slack_us

    def is_critical(self, task_id: int) -> bool:
        return task_id in self.critical_tasks

    def get_earliest_start(self, task_id: int) -> int:
        return self.task_timings[task_id].earliest_start_us
```

#### 4.2 初期实现：简单 CPM（含多跳延迟估算）

**Flow 传输时间估算**：需要区分两种延迟：

- **Transmission Delay（发送时延）** = `size / bandwidth`，由瓶颈链路（最低带宽）决定
- **Propagation Delay（传播时延）** = 路径上所有链路 latency 之和

**Total Duration = transmission_delay + propagation_delay**

```python
def analyze_critical_path(
    workload: P2PWorkload,
    topology: NetworkTopology,
    routing_hints: RoutingHints,  # ← From Task 1
) -> CriticalPathInfo:
    """
    Critical path analysis with multi-hop latency estimation.
    
    Flow duration = transmission_delay + propagation_delay
    - transmission_delay = size / bottleneck_bandwidth (min bw along path)
    - propagation_delay = sum of all link latencies along path
    
    Time complexity: O(V + E) where V = tasks, E = dependency edges.
    """
    tasks = workload.tasks
    
    # Step 1: Topological sort
    sorted_tasks = _topological_sort(tasks)
    
    # Step 2: Forward pass (ASAP) with accurate duration
    earliest_start: dict[int, int] = {}
    earliest_finish: dict[int, int] = {}
    
    for task in sorted_tasks:
        earliest_start[task.task_id] = (
            0 if not task.deps
            else max(earliest_finish[dep] for dep in task.deps)
        )
        earliest_finish[task.task_id] = (
            earliest_start[task.task_id] + 
            _estimate_duration(task, topology, routing_hints)
        )
    
    # Step 3: Backward pass (ALAP)
    makespan = max(earliest_finish.values()) if earliest_finish else 0
    latest_start: dict[int, int] = {}
    latest_finish: dict[int, int] = {}
    
    dependents: dict[int, list[int]] = {t.task_id: [] for t in tasks}
    for task in tasks:
        for dep in task.deps:
            dependents[dep].append(task.task_id)
    
    for task in reversed(sorted_tasks):
        latest_finish[task.task_id] = (
            makespan if not dependents[task.task_id]
            else min(latest_start[d] for d in dependents[task.task_id])
        )
        latest_start[task.task_id] = (
            latest_finish[task.task_id] - 
            _estimate_duration(task, topology, routing_hints)
        )
    
    # Step 4: Build output
    task_timings: dict[int, TaskTimingInfo] = {}
    for task in tasks:
        tid = task.task_id
        slack = float(latest_start[tid] - earliest_start[tid])
        task_timings[tid] = TaskTimingInfo(
            task_id=tid,
            earliest_start_us=earliest_start[tid],
            earliest_finish_us=earliest_finish[tid],
            latest_start_us=latest_start[tid],
            latest_finish_us=latest_finish[tid],
            slack_us=slack,
            is_critical=(slack == 0.0),
        )
    
    critical_tasks = [tid for tid, t in task_timings.items() if t.is_critical]
    
    return CriticalPathInfo(
        task_timings=task_timings,
        critical_tasks=critical_tasks,
        makespan_us=makespan,
        analysis_method="cpm",
    )


def _estimate_duration(
    task: Task,
    topology: NetworkTopology,
    routing_hints: RoutingHints,
) -> int:
    """
    Estimate flow duration for critical path analysis.
    
    Total duration = transmission_delay + propagation_delay
    - transmission_delay: size / bottleneck_bandwidth (lowest bw link in path)
    - propagation_delay: sum of all link latencies along the path
    
    Note: Per-link timing (which link the flow is on at which time) is NOT
    needed here — that's Task 3's job. Here we only need the total duration.
    """
    if task.type.value != "flow":
        return task.duration_us or 0
    
    if task.src is None or task.dst is None:
        return 0
    
    # Get full path through topology (from Task 1)
    path = routing_hints.get_path(topology, task.src, task.dst)
    if not path or len(path) < 2:
        # Fallback: assume direct link
        link = topology.get_link(task.src, task.dst)
        if link and link.bandwidth_gbps > 0:
            return int((task.size_bytes or 0) * 8 / (link.bandwidth_gbps * 1e9))
        return 0
    
    size_bits = (task.size_bytes or 0) * 8
    bottleneck_bw_gbps = float('inf')
    total_latency_us = 0.0
    
    for i in range(len(path) - 1):
        link = topology.get_link(path[i], path[i+1])
        if link:
            bottleneck_bw_gbps = min(bottleneck_bw_gbps, link.bandwidth_gbps)
            total_latency_us += link.latency_us
        else:
            return 0  # Link not found
    
    if bottleneck_bw_gbps <= 0 or bottleneck_bw_gbps == float('inf'):
        return 0
    
    # tx_time on bottleneck link (in microseconds)
    tx_time_us = size_bits / (bottleneck_bw_gbps * 1e9)
    
    return int(tx_time_us + total_latency_us)


def _topological_sort(tasks: list[Task]) -> list[Task]:
    """Topological sort using iterative DFS."""
    task_map = {t.task_id: t for t in tasks}
    visited: set[int] = set()
    result: list[Task] = []
    
    def visit(tid: int):
        if tid in visited:
            return
        visited.add(tid)
        for dep in task_map[tid].deps:
            visit(dep)
        result.append(task_map[tid])
    
    for task in tasks:
        visit(task.task_id)
    return result
```

#### 4.3 未来扩展方向

初期使用简单 CPM，接口固定，后续可替换内部实现而不影响 Phase 4。

**方案 B：TTE（Time-to-Exposed）**

来源：论文中针对 AI 训练 workload 的改进方案。

核心思路：

- 只做 forward pass（max bandwidth），不做 backward pass
- 对每条 flow `f`，找其 dependent 计算任务 `c`
- `TTE(f) = Start(c) - Finish(f)`，其中 `Start(c) = max(earliest_finish of all c's parents)`
- TTE = 0 表示 f 在关键路径上；TTE > 0 表示 f 可以被降速而不影响 JCT

与 CPM 的关系：对于 flow 任务，TTE 等价于 CPM 的 slack。TTE 方案的优势是不需要 backward pass，且语义更直接（直接量化"这条 flow 延迟多久会暴露计算等待"）。

```python
# 未来替换只需修改 analyze_critical_path 内部实现：
def analyze_critical_path_tte(...) -> CriticalPathInfo:
    # Forward pass only
    earliest_start, earliest_finish = _forward_pass(workload, topology)
    # Compute TTE per flow
    tte = _compute_tte(workload, earliest_start, earliest_finish)
    # Fill same TaskTimingInfo structure
    # analysis_method = "tte"
    ...
```

**方案 C：RCPSP（资源约束关键路径）**

适用场景：需要精确建模节点计算资源约束（每节点同时只能运行 1 个 compute task，可以多个flow task，但是共享有限的带宽）时。

核心思路：使用 Serial SGS（Serial Schedule Generation Scheme）启发式算法，在调度过程中同时考虑依赖约束和资源约束，得到资源可行的调度方案后再计算 slack。

注意：RCPSP 是 NP-hard 问题，对大规模 workload 只能求近似解，计算开销显著高于 CPM/TTE。

```python
# 未来替换：
def analyze_critical_path_rcpsp(...) -> CriticalPathInfo:
    # Serial SGS with node compute constraints
    schedule = _serial_sgs(workload, topology)
    # Fill same TaskTimingInfo structure
    # analysis_method = "rcpsp"
    ...
```

#### 4.4 测试

**文件**: `tests/test_critical_path.py`

```python
def test_linear_chain_all_critical():
    """A → B → C: all tasks have slack = 0."""

def test_parallel_branches_longer_is_critical():
    """Two branches of different length: only longer branch is critical."""

def test_flow_latency_from_bandwidth():
    """Flow duration = size * 8 / bandwidth."""

def test_slack_nonnegative():
    """All slack values must be >= 0."""

def test_critical_tasks_have_zero_slack():
    """Every task in critical_tasks must have slack_us == 0."""

def test_makespan_equals_max_finish():
    """makespan_us == max(earliest_finish_us) over all tasks."""
```

---

### Task 3: 链路竞争分析（Link Contention Analysis）

**文件**: `src/static_analysis/contention_analysis.py`

**功能**：识别哪些 flows 会竞争同一条物理链路，并结合时间窗口判断是否真正可能并发。依赖 Task 1（路由信息）提供完整路径，Task 2（关键路径）提供时间窗口。

**为什么需要时空结合**：两条 flow 即使共用某条链路，但如果它们的执行时间完全不重叠，就不会真正竞争。只有**时间上可能重叠且空间上共用链路**的 flows 才会产生实际竞争。例如：

```
Flow A: GPU0 → Switch1 → GPU3   (时间: 0-100us)
Flow B: GPU1 → Switch1 → GPU2   (时间: 50-150us)  ← 时间重叠，真正竞争
Flow C: GPU2 → Switch1 → GPU4   (时间: 200-300us) ← 时间不重叠，无竞争

虽然 A、B、C 都经过 Switch1，但只有 A∩B 需要带宽分配，C 可以独占链路。
```

#### 4.1 数据结构

```python
from dataclasses import dataclass, field


@dataclass
class LinkContentionGroup:
    """All flow tasks that traverse a specific physical link, with temporal analysis.
    
    Design philosophy (v1 - ASAP-based static analysis):
    - Uses ASAP (As Soon As Possible) timing from critical path analysis
    - time_windows represent idealized schedule, not guaranteed execution times
    - Serves as REFERENCE for Phase 4, not a guarantee
    - Actual timing depends on dynamic scheduler decisions in Phase 4
    
    Future improvement (v2 - iterative refinement):
    - After Phase 4 produces initial schedule, update time_windows with actual times
    - Iterate: analyze → schedule → measure → re-analyze
    - See "Future Improvements" section at the end of this document
    """
    
    link_id: tuple[int, int]  # (node_a, node_b)
    
    # All flows using this link (spatial contention only)
    all_flows: list[int] = field(default_factory=list)  # task_ids
    
    # Statistics
    total_data_bytes: int = 0
    min_size_bytes: int = 0
    max_size_bytes: int = 0
    avg_size_bytes: float = 0.0
    num_flows: int = 0
    
    # Concurrency estimates (based on ASAP timing)
    worst_case_concurrency: int = 0      # = num_flows (all flows simultaneous)
    best_case_concurrency: int = 0       # peak concurrent at any single time point
    
    # Time windows for each flow on THIS SPECIFIC LINK (per-link timing)
    # IMPORTANT: This is NOT the global flow start/end time!
    # entry_time = flow_start + cumulative_propagation_latency (from previous hops)
    # exit_time  = entry_time + transmission_delay (size / this_link_bandwidth)
    #
    # Example: Flow A→B→C→D, start=0
    #   Link A→B: entry=0,            exit=0+size/bw_AB
    #   Link B→C: entry=lat_AB,       exit=lat_AB + size/bw_BC
    #   Link C→D: entry=lat_AB+lat_BC, exit=lat_AB+lat_BC + size/bw_CD
    time_windows: dict[int, tuple[int, int]] = field(default_factory=dict)
    # task_id -> (entry_time_us, exit_time_us) on this specific link
    
    def add_flow(self, task_id: int, size_bytes: int,
                 entry_time: int, exit_time: int):
        """
        Add a flow task with its per-link timing information.
        
        Args:
            task_id: The flow task ID
            size_bytes: Data size to transfer
            entry_time: flow_start + cumulative_propagation_latency from previous hops
            exit_time: entry_time + transmission_delay (size / this_link_bandwidth)
        """
        self.all_flows.append(task_id)
        self.total_data_bytes += size_bytes
        self.num_flows += 1
        self.time_windows[task_id] = (entry_time, exit_time)
        
        if self.min_size_bytes == 0 or size_bytes < self.min_size_bytes:
            self.min_size_bytes = size_bytes
        if size_bytes > self.max_size_bytes:
            self.max_size_bytes = size_bytes
        self.avg_size_bytes = self.total_data_bytes / self.num_flows
        self.worst_case_concurrency = self.num_flows
    
    def get_concurrency_at_time(self, timestamp: int) -> int:
        """
        Query: how many flows are active on this link at a given timestamp?
        
        This is the primary interface for Phase 4 schedulers:
        - Node X wants to send flow F at time T
        - Check all links in F's path
        - For each link, query concurrency at time T
        - If concurrency is high, reduce rate or defer
        
        Args:
            timestamp: The time point to check (in microseconds)
            
        Returns:
            Number of flows active on this link during [entry, exit)
        """
        count = 0
        for fid in self.all_flows:
            entry, exit = self.time_windows[fid]
            if entry <= timestamp < exit:  # [entry, exit) interval
                count += 1
        return count
    
    def get_peak_concurrency_window(self) -> tuple[int, int, int]:
        """
        Find the time window with maximum concurrency on this link.
        
        Returns:
            (start_time, end_time, peak_concurrency)
            During [start_time, end_time], concurrency is at its peak.
            Useful for identifying hotspot periods to avoid.
        """
        if not self.time_windows:
            return (0, 0, 0)
        
        # Create events: (time, +1 for entry, -1 for exit)
        events = []
        for entry, exit in self.time_windows.values():
            events.append((entry, 1))
            events.append((exit, -1))
        
        # Sort by time, exits before entries at same timestamp
        events.sort(key=lambda x: (x[0], x[1]))
        
        # Sweep line algorithm to find peak
        max_concurrent = 0
        current = 0
        peak_start = 0
        peak_end = 0
        in_peak = False
        
        for time, delta in events:
            prev_current = current
            current += delta
            
            if delta == 1 and current > max_concurrent:
                max_concurrent = current
                peak_start = time
                in_peak = True
            elif delta == -1 and in_peak and prev_current == max_concurrent:
                peak_end = time
                in_peak = False
        
        return (peak_start, peak_end, max_concurrent)
    
    def analyze_temporal_contention(self):
        """
        Post-process: compute best_case_concurrency via sweep line algorithm.
        
        This identifies the peak concurrency considering actual time overlaps,
        not just worst-case assumption that all flows run simultaneously.
        
        Should be called after all flows are added to the group.
        """
        _, _, peak = self.get_peak_concurrency_window()
        self.best_case_concurrency = peak
    
    @property
    def has_temporal_contention(self) -> bool:
        """Check if this link has any potential temporal contention."""
        return self.best_case_concurrency > 1
    
    @property
    def contention_ratio(self) -> float:
        """
        Ratio of actual concurrency to worst case.
        
        High ratio (close to 1.0) = most flows overlap in time (high contention risk)
        Low ratio (close to 0.0) = flows are well-separated in time (low contention risk)
        
        Phase 4 can use this to decide scheduling strategy:
        - ratio > 0.7: enable aggressive flow control
        - ratio < 0.3: simple greedy scheduling is fine
        """
        if self.worst_case_concurrency < 2:
            return 0.0
        return self.best_case_concurrency / self.worst_case_concurrency
```

#### 4.2 算法

```python
def find_contention_groups(
    workload: P2PWorkload,
    topology: NetworkTopology,
    routing_hints: RoutingHints,  # From Task 1
    critical_path_info,           # From Task 2: CriticalPathInfo
) -> dict[tuple[int, int], LinkContentionGroup]:
    """
    Group flows by physical link they traverse, with accurate per-link temporal analysis.
    
    Key design decisions:
    1. Per-link timing (not global flow timing):
       - entry_time = flow_start + cumulative_propagation_latency_to_this_link
       - exit_time = entry_time + transmission_delay_on_this_link
       
       Example: Flow A→B→C→D, size=800Mb, start=0
         Links: A→B (400Gbps, 10us), B→C (200Gbps, 20us), C→D (400Gbps, 10us)
         
         Transmission delays:
           A→B: 800Mb/400Gbps = 2us
           B→C: 800Mb/200Gbps = 4us  ← bottleneck!
           C→D: 800Mb/400Gbps = 2us
         
         Cumulative latencies:
           To enter A→B: 0us
           To enter B→C: 10us (A→B latency)
           To enter C→D: 10+20 = 30us
         
         Per-link windows:
           Link A→B: [0, 0+2]     = [0, 2us]
           Link B→C: [10, 10+4]   = [10, 14us]
           Link C→D: [30, 30+2]   = [30, 32us]
       
       This is CORRECT because:
       - Flow doesn't occupy B→C until it propagates through A→B
       - Flow occupies each link for exactly transmission_delay duration
       - Different links may have different bandwidths (bottleneck analysis)
    
    2. ASAP-based static analysis (v1):
       - Uses earliest_start_us from critical path (ASAP scheduling)
       - LIMITATION: Actual Phase 4 scheduling may differ from ASAP
       - Serves as reference for Phase 4, not a guarantee
       - Future v2 will add iterative refinement (see "Future Improvements")
    
    Algorithm:
    For each flow task:
    1. Get global start time from critical path (Task 2)
    2. Get full path through topology (Task 1)
    3. Calculate per-link timing:
       - entry_time = flow_start + sum(previous link latencies)
       - exit_time = entry_time + (size / link_bandwidth)
    4. Add to contention group with accurate per-link timing
    5. Post-process to compute peak concurrency via sweep line
    
    Complexity: O(F * H) where F = num flows, H = avg hop count
    """
    groups: dict[tuple[int, int], LinkContentionGroup] = {}
    
    # Step 1: Collect all flows per link with accurate per-link timing
    for task in workload.tasks:
        if task.type.value != "flow":
            continue
        if task.src is None or task.dst is None:
            continue
        
        # Get global start time from Task 2's analysis (ASAP schedule)
        timing = critical_path_info.task_timings.get(task.task_id)
        if timing:
            flow_start_time = timing.earliest_start_us
        else:
            # Fallback: no timing info
            flow_start_time = 0
        
        # Get full path through topology (Task 1)
        path = routing_hints.get_path(topology, task.src, task.dst)
        if not path or len(path) < 2:
            # Direct link fallback
            path = [task.src, task.dst]
        
        size_bits = (task.size_bytes or 0) * 8
        cumulative_latency_us = 0.0
        
        # Calculate per-link timing with correct transmission/propagation separation
        for i in range(len(path) - 1):
            link_id = (path[i], path[i+1])
            link = topology.get_link(link_id[0], link_id[1])
            
            if link:
                # Transmission delay on THIS specific link
                transmission_delay_us = size_bits / (link.bandwidth_gbps * 1e9)
                
                # Per-link entry/exit times
                entry_time = flow_start_time + cumulative_latency_us
                exit_time = entry_time + transmission_delay_us
                
                if link_id not in groups:
                    groups[link_id] = LinkContentionGroup(link_id=link_id)
                
                groups[link_id].add_flow(
                    task.task_id, 
                    task.size_bytes or 0,
                    int(entry_time * 1e6),  # Convert to microseconds
                    int(exit_time * 1e6)
                )
                
                # Accumulate propagation latency for next hop
                cumulative_latency_us += link.latency_us
            else:
                # Link not found, skip this path
                break
    
    # Step 2: Post-process to analyze temporal contention
    for group in groups.values():
        group.analyze_temporal_contention()
    
    return groups
```

#### 4.3 Phase 4 使用示例

```python
# Example 1: Check concurrency at specific time before sending
def should_defer_transmission(
    flow_task: Task,
    proposed_start_time: int,
    contention_groups: dict[tuple[int, int], LinkContentionGroup],
    routing_hints: RoutingHints,
    topology: NetworkTopology,
) -> bool:
    """
    Should this flow defer its transmission to avoid contention?
    
    This is the primary use case for temporal contention analysis:
    - Node X wants to send flow F at time T
    - Check all links in F's path
    - For each link, query concurrency at the time when F would be on that link
    - If concurrency is too high, defer or reduce rate
    
    Returns:
        True if the flow should defer (too much contention)
    """
    # Get flow's path
    path = routing_hints.get_path(topology, flow_task.src, flow_task.dst)
    if not path or len(path) < 2:
        return False  # No path, can't evaluate
    
    # Estimate per-hop duration
    duration = _estimate_duration(flow_task, topology)
    num_hops = len(path) - 1
    per_hop = duration // num_hops if num_hops > 0 else duration
    
    # Check each link in the path
    CONGESTION_THRESHOLD = 3  # Configurable
    for i in range(num_hops):
        link_id = (path[i], path[i+1])
        group = contention_groups.get(link_id)
        
        if group:
            # When would this flow be on this specific link?
            entry_time = proposed_start_time + i * per_hop
            
            # Query: how many other flows are on this link at that time?
            concurrency = group.get_concurrency_at_time(entry_time)
            
            # Threshold: if too many flows, defer
            if concurrency > CONGESTION_THRESHOLD:
                return True
    
    return False


# Example 2: Find optimal start time to minimize contention
def find_optimal_start_time(
    flow_task: Task,
    contention_groups: dict[tuple[int, int], LinkContentionGroup],
    routing_hints: RoutingHints,
    topology: NetworkTopology,
    earliest_possible: int,
    latest_acceptable: int,
) -> int:
    """
    Find the best start time to minimize contention across all links.
    
    Simple approach: try different start times, pick the one with
    lowest maximum concurrency across all links in the path.
    
    Args:
        flow_task: The flow task to schedule
        contention_groups: Pre-computed contention groups
        routing_hints: Routing information
        topology: Network topology
        earliest_possible: Earliest acceptable start time
        latest_acceptable: Latest acceptable start time
        
    Returns:
        Optimal start time (in microseconds)
    """
    path = routing_hints.get_path(topology, flow_task.src, flow_task.dst)
    if not path or len(path) < 2:
        return earliest_possible
    
    duration = _estimate_duration(flow_task, topology)
    num_hops = len(path) - 1
    per_hop = duration // num_hops if num_hops > 0 else duration
    
    # Try different start times (granularity: 10us)
    best_time = earliest_possible
    best_max_concurrency = float('inf')
    
    GRANULARITY = 10  # microseconds
    for start_offset in range(0, latest_acceptable - earliest_possible, GRANULARITY):
        candidate_time = earliest_possible + start_offset
        max_concurrency = 0
        
        # Check all links in path
        for i in range(num_hops):
            link_id = (path[i], path[i+1])
            group = contention_groups.get(link_id)
            
            if group:
                entry_time = candidate_time + i * per_hop
                concurrency = group.get_concurrency_at_time(entry_time)
                max_concurrency = max(max_concurrency, concurrency)
        
        # Pick time with lowest max concurrency
        if max_concurrency < best_max_concurrency:
            best_max_concurrency = max_concurrency
            best_time = candidate_time
    
    return best_time


# Example 3: Adaptive bandwidth allocation based on contention
class ContentionAwareBandwidthAllocator:
    """Allocate bandwidth based on temporal contention analysis."""
    
    def __init__(self, contention_groups: dict[tuple[int, int], LinkContentionGroup]):
        self.groups = contention_groups
    
    def allocate_bandwidth(self, active_flows: list[Task], 
                          link_id: tuple[int, int]) -> dict[int, float]:
        """
        Allocate bandwidth for flows on a specific link.
        
        Strategy depends on contention ratio:
        - Low contention (< 0.3): greedy, give full bandwidth to first flow
        - Medium contention (0.3-0.7): fair share among concurrent flows
        - High contention (> 0.7): aggressive flow control, prioritize critical flows
        
        Args:
            active_flows: Flows currently ready to transmit
            link_id: The link to allocate bandwidth for
            
        Returns:
            Dictionary mapping task_id → bandwidth fraction (0.0 to 1.0)
        """
        group = self.groups.get(link_id)
        if not group:
            # No contention data, use equal sharing
            return {f.task_id: 1.0 / len(active_flows) for f in active_flows}
        
        allocation = {}
        
        if group.contention_ratio < 0.3:
            # Low contention: greedy
            # First flow gets full bandwidth, others wait
            if active_flows:
                allocation[active_flows[0].task_id] = 1.0
                for f in active_flows[1:]:
                    allocation[f.task_id] = 0.0
        
        elif group.contention_ratio < 0.7:
            # Medium contention: fair share
            equal_share = 1.0 / len(active_flows)
            for f in active_flows:
                allocation[f.task_id] = equal_share
        
        else:
            # High contention: prioritize by flow size (smaller flows first)
            # This reduces average completion time
            total_size = sum(f.size_bytes or 0 for f in active_flows)
            for f in active_flows:
                if total_size > 0:
                    allocation[f.task_id] = (f.size_bytes or 0) / total_size
                else:
                    allocation[f.task_id] = 1.0 / len(active_flows)
        
        return allocation
```

#### 4.4 测试

**文件**: `tests/test_contention_analysis.py`

```python
def test_per_link_timing_direct_link():
    """Single-hop flow: per-link timing equals global timing."""
    # Flow A→B, global [0, 100us], path = [A, B]
    # Link A→B should have entry=0, exit=100

def test_per_link_timing_multihop():
    """Multi-hop flow: per-link timing includes propagation delay."""
    # Flow A→B→C, global [0, 200us], path = [A, B, C]
    # Link A→B: entry=0, exit=100
    # Link B→C: entry=100, exit=200

def test_get_concurrency_at_specific_time():
    """Query concurrency at different time points on a link."""
    group = LinkContentionGroup(link_id=(0, 1))
    group.add_flow(1, 100, 0, 100)     # Flow 1: [0, 100)
    group.add_flow(2, 100, 50, 150)    # Flow 2: [50, 150)
    group.add_flow(3, 100, 200, 300)   # Flow 3: [200, 300)
    
    assert group.get_concurrency_at_time(0) == 1    # Only flow 1
    assert group.get_concurrency_at_time(50) == 2   # Flows 1 & 2
    assert group.get_concurrency_at_time(100) == 1  # Only flow 2 (flow 1 finished)
    assert group.get_concurrency_at_time(200) == 1  # Only flow 3
    assert group.get_concurrency_at_time(400) == 0  # No flows

def test_peak_concurrency_window():
    """Find the time window with maximum concurrency."""
    group = LinkContentionGroup(link_id=(0, 1))
    group.add_flow(1, 100, 0, 100)
    group.add_flow(2, 100, 50, 150)
    group.add_flow(3, 100, 75, 125)
    
    start, end, peak = group.get_peak_concurrency_window()
    assert peak == 3
    assert start == 75   # When 3rd flow enters
    assert end == 100    # When 1st flow exits

def test_shared_switch_contention():
    """Two flows through the same switch share a contention group."""
    # Both flows use link through switch node

def test_no_temporal_contention():
    """Two flows on same link but non-overlapping times."""
    # Flow A: [0, 100us], Flow B: [150, 250us] on same link
    # best_case_concurrency should be 1 (not 2)
    group = LinkContentionGroup(link_id=(0, 1))
    group.add_flow(1, 100, 0, 100)
    group.add_flow(2, 100, 150, 250)
    group.analyze_temporal_contention()
    
    assert group.best_case_concurrency == 1
    assert group.contention_ratio == 0.5  # 1/2

def test_contention_ratio_calculation():
    """Verify contention_ratio = best_case / worst_case."""
    group = LinkContentionGroup(link_id=(0, 1))
    group.add_flow(1, 100, 0, 100)
    group.add_flow(2, 100, 50, 150)
    group.add_flow(3, 100, 200, 300)
    group.analyze_temporal_contention()
    
    # worst_case = 3, best_case = 2 (flows 1&2 overlap)
    assert group.worst_case_concurrency == 3
    assert group.best_case_concurrency == 2
    assert abs(group.contention_ratio - 2/3) < 0.01

def test_has_temporal_contention_property():
    """Verify has_temporal_contention reflects actual overlap."""
    group1 = LinkContentionGroup(link_id=(0, 1))
    group1.add_flow(1, 100, 0, 100)
    group1.analyze_temporal_contention()
    assert not group1.has_temporal_contention  # Only 1 flow
    
    group2 = LinkContentionGroup(link_id=(0, 1))
    group2.add_flow(1, 100, 0, 100)
    group2.add_flow(2, 100, 50, 150)
    group2.analyze_temporal_contention()
    assert group2.has_temporal_contention  # 2 flows overlap

def test_contention_stats():
    """Verify total_data_bytes, avg_size_bytes, min/max calculations."""
    group = LinkContentionGroup(link_id=(0, 1))
    group.add_flow(1, 100, 0, 100)
    group.add_flow(2, 200, 50, 150)
    group.add_flow(3, 300, 200, 300)
    
    assert group.total_data_bytes == 600
    assert group.min_size_bytes == 100
    assert group.max_size_bytes == 300
    assert abs(group.avg_size_bytes - 200.0) < 0.01
```

---

### Task 4: 节点局部视图（Node-Centric Local Views）

**文件**: `src/static_analysis/node_view.py`

**功能**：为每个节点预计算它的局部调度视图。

#### 4.1 数据结构

```python
from dataclasses import dataclass, field


@dataclass
class NodeLocalView:
    """Local scheduling view for a single node."""
  
    node_id: int
  
    # Flows this node sends
    send_tasks: list[int] = field(default_factory=list)  # task_ids
    total_send_bytes: int = 0
  
    # Flows this node receives
    receive_tasks: list[int] = field(default_factory=list)  # task_ids
    total_receive_bytes: int = 0
  
    # Compute tasks on this node
    compute_tasks: list[int] = field(default_factory=list)  # task_ids
    total_compute_time_us: int = 0
  
    # Estimated schedule (based on ASAP from critical path)
    estimated_send_times: list[tuple[int, int]] = field(default_factory=list)
    # (start_time_us, task_id) — 发送方开始传输时刻 (earliest_start_us)

    estimated_receive_times: list[tuple[int, int]] = field(default_factory=list)
    # (arrival_time_us, task_id) — 接收方数据到达时刻 (earliest_finish_us)

    # Estimated idle ratio (communication time / total active time)
    # High value = node spends more time waiting on communication
    estimated_idle_ratio: float = 0.0
  
    def add_send_flow(self, task_id: int, size_bytes: int, start_time: int):
        self.send_tasks.append(task_id)
        self.total_send_bytes += size_bytes
        self.estimated_send_times.append((start_time, task_id))
  
    def add_receive_flow(self, task_id: int, size_bytes: int, arrival_time: int):
        self.receive_tasks.append(task_id)
        self.total_receive_bytes += size_bytes
        self.estimated_receive_times.append((arrival_time, task_id))
```

#### 4.2 算法

```python
def build_node_views(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
) -> dict[int, NodeLocalView]:
    """
    Build per-node local views.
  
    For each node:
    1. Collect send flows (where node is src)
    2. Collect receive flows (where node is dst)
    3. Collect compute tasks (where node is node)
    4. Estimate schedule using ASAP times from critical path
    5. Compute idle ratio (comm time / total active time)
    """
    views: dict[int, NodeLocalView] = {}
  
    # Initialize views for all nodes
    all_nodes = set()
    for task in workload.tasks:
        if task.is_compute() and task.node is not None:
            all_nodes.add(task.node)
        if task.is_flow():
            if task.src is not None:
                all_nodes.add(task.src)
            if task.dst is not None:
                all_nodes.add(task.dst)
  
    for node_id in all_nodes:
        views[node_id] = NodeLocalView(node_id=node_id)
  
    # Populate views
    for task in workload.tasks:
        if task.is_compute() and task.node is not None:
            node_id = task.node
            views[node_id].compute_tasks.append(task.task_id)
            views[node_id].total_compute_time_us += task.duration_us or 0
      
        elif task.is_flow():
            # Raises ValueError if task not in critical_path.task_timings
            timing = critical_path.task_timings[task.task_id]
            size_bytes = task.size_bytes or 0

            if task.src is not None:
                views[task.src].add_send_flow(
                    task.task_id, size_bytes,
                    timing.earliest_start_us,   # 发送方：开始传输时刻
                )
            if task.dst is not None:
                views[task.dst].add_receive_flow(
                    task.task_id, size_bytes,
                    timing.earliest_finish_us,  # 接收方：数据到达时刻
                )
  
    # Compute idle ratios using flow durations from critical path
    for node_id, view in views.items():
        node_flow_ids = send_task_ids[node_id] | recv_task_ids[node_id]
        total_comm_time = sum(
            critical_path.task_timings[tid].earliest_finish_us
            - critical_path.task_timings[tid].earliest_start_us
            for tid in node_flow_ids
        )
        total_time = view.total_compute_time_us + total_comm_time
        if total_time > 0:
            view.estimated_idle_ratio = total_comm_time / total_time
  
    return views
```

#### 4.3 测试

**文件**: `tests/test_node_view.py`

```python
def test_send_time_is_earliest_start():
    """Sender's estimated_send_times uses earliest_start_us."""

def test_receive_time_is_earliest_finish():
    """Receiver's estimated_receive_times uses earliest_finish_us."""

def test_idle_ratio_mixed_node():
    """Verify idle ratio for node with both compute and flow tasks."""

def test_raises_on_missing_critical_path_timing():
    """If a task is not in critical_path.task_timings, raise ValueError."""
```

---

### Task 5: 流量矩阵（Traffic Matrix）

**文件**: `src/static_analysis/traffic_matrix.py`

**功能**：统计节点间的流量分布。

#### 4.1 数据结构

```python
from dataclasses import dataclass, field


@dataclass
class TrafficMatrix:
    """Node-to-node traffic volume matrix."""
  
    # traffic[(i, j)] = total bytes from node i to node j
    traffic: dict[tuple[int, int], int] = field(default_factory=dict)
  
    # Top senders/receivers
    top_senders: list[tuple[int, int]] = field(default_factory=list)
    # (node_id, total_sent_bytes)
  
    top_receivers: list[tuple[int, int]] = field(default_factory=list)
  
    def get_traffic(self, src: int, dst: int) -> int:
        return self.traffic.get((src, dst), 0)
  
    def get_total_traffic(self) -> int:
        return sum(self.traffic.values())
```

#### 4.2 算法

```python
def compute_traffic_matrix(workload: P2PWorkload) -> TrafficMatrix:
    """
    Compute node-to-node traffic matrix.
  
    For each flow task, accumulate traffic[(src, dst)] += size_bytes.
    Then compute top senders/receivers.
    """
    tm = TrafficMatrix()

    # Accumulate traffic
    for task in workload.tasks:
        if not task.is_flow():
            continue
        if task.src is None or task.dst is None:
            continue

        link_id = (task.src, task.dst)
        tm.traffic[link_id] = tm.traffic.get(link_id, 0) + (task.size_bytes or 0)

    # Compute top senders/receivers
    sender_bytes: dict[int, int] = {}
    receiver_bytes: dict[int, int] = {}

    for (src, dst), bytes_count in tm.traffic.items():
        sender_bytes[src] = sender_bytes.get(src, 0) + bytes_count
        receiver_bytes[dst] = receiver_bytes.get(dst, 0) + bytes_count

    tm.top_senders = sorted(sender_bytes.items(), key=lambda x: x[1], reverse=True)
    tm.top_receivers = sorted(receiver_bytes.items(), key=lambda x: x[1], reverse=True)

    return tm
```

#### 4.3 测试

**文件**: `tests/test_traffic_matrix.py`

```python
def test_traffic_accumulation():
    """Verify traffic[(src, dst)] accumulation."""

def test_top_senders_receivers():
    """Verify top senders/receivers ordering."""
```

---

### Task 6: Workload 摘要（Workload Summary）

**文件**: `src/static_analysis/workload_summary.py`

**功能**：生成 workload 的全局统计信息。

#### 4.1 数据结构

```python
from dataclasses import dataclass


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

    # Hottest links
    hot_links: list[tuple[int, int, int]] = field(default_factory=list)
    # (link_id, bytes, num_flows)
```

#### 4.2 算法

```python
def compute_workload_summary(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
    contention_groups: dict[tuple[int, int], LinkContentionGroup],
) -> WorkloadSummary:
    """Compute overall workload statistics."""
    compute_tasks = [t for t in workload.tasks if t.is_compute()]
    flow_tasks = [t for t in workload.tasks if t.is_flow()]

    total_comm_bytes = sum(t.size_bytes or 0 for t in flow_tasks)
    total_compute_time = sum(t.duration_us or 0 for t in compute_tasks)

    # Communication time from critical path analysis (time-based, not bytes-based)
    total_comm_time = sum(
        critical_path.task_timings[t.task_id].earliest_finish_us
        - critical_path.task_timings[t.task_id].earliest_start_us
        for t in flow_tasks
        if t.task_id in critical_path.task_timings
    )

    comm_compute_ratio = (
        total_comm_time / total_compute_time if total_compute_time > 0 else 0.0
    )

    # DAG width = average number of tasks at each depth level
    dag_width = _compute_avg_dag_width(workload)

    # Critical path stats (time-based)
    cp_comm_time = sum(
        critical_path.task_timings[t.task_id].earliest_finish_us
        - critical_path.task_timings[t.task_id].earliest_start_us
        for t in flow_tasks
        if t.task_id in critical_path.critical_tasks
    )
    cp_total_time = critical_path.makespan_us
    cp_comm_fraction = cp_comm_time / cp_total_time if cp_total_time > 0 else 0.0

    # Hot links (top 10 by total bytes)
    hot_links = sorted(
        [(gid, g.total_data_bytes, g.num_flows)
         for gid, g in contention_groups.items()],
        key=lambda x: x[1],
        reverse=True,
    )[:10]

    return WorkloadSummary(
        total_tasks=len(workload.tasks),
        total_compute_tasks=len(compute_tasks),
        total_flow_tasks=len(flow_tasks),
        total_communication_bytes=total_comm_bytes,
        total_compute_time_us=total_compute_time,
        comm_compute_ratio=comm_compute_ratio,
        avg_dag_width=dag_width,
        critical_path_length_us=critical_path.makespan_us,
        critical_path_comm_fraction=cp_comm_fraction,
        hot_links=hot_links,
    )


def _compute_avg_dag_width(workload: P2PWorkload) -> float:
    """Compute average DAG width (number of tasks at each depth level)."""
    # Depth computed via memoized DFS (not len(task.deps))
    task_map = {t.task_id: t for t in workload.tasks}
    depths: dict[int, int] = {}

    def get_depth(task_id: int) -> int:
        if task_id in depths:
            return depths[task_id]
        task = task_map[task_id]
        if not task.deps:
            depths[task_id] = 0
        else:
            depths[task_id] = max(get_depth(dep) for dep in task.deps) + 1
        return depths[task_id]

    for task in workload.tasks:
        get_depth(task.task_id)

    levels: dict[int, int] = {}
    for depth in depths.values():
        levels[depth] = levels.get(depth, 0) + 1

    if not levels:
        return 0.0

    return sum(levels.values()) / len(levels)
```

---

### Task 7: 统一分析接口（WorkloadAnalyzer）

**文件**: `src/static_analysis/analyzer.py`

**功能**：统一的 workload 分析入口，整合所有分析模块。

```python
from dataclasses import dataclass
from .critical_path import CriticalPathInfo, analyze_critical_path
from .routing_hints import RoutingHints, compute_routing_hints
from .contention_analysis import LinkContentionGroup, find_contention_groups
from .node_view import NodeLocalView, build_node_views
from .traffic_matrix import TrafficMatrix, compute_traffic_matrix
from .workload_summary import WorkloadSummary, compute_workload_summary
from ..workload_format.schema import P2PWorkload
from .topology_loader import NetworkTopology


@dataclass
class WorkloadAnalysisResult:
    """Complete workload analysis result."""

    critical_path: CriticalPathInfo
    routing_hints: RoutingHints
    contention_groups: dict[tuple[int, int], LinkContentionGroup]
    node_views: dict[int, NodeLocalView]
    traffic_matrix: TrafficMatrix
    summary: WorkloadSummary


class WorkloadAnalyzer:
    """Unified entry point for workload analysis."""

    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload) -> WorkloadAnalysisResult:
        """Run all analysis modules on the workload.

        Execution order matters (routing must come before critical path):
        1. routing_hints  — depends on topology only (BFS shortest paths)
        2. critical_path  — depends on routing_hints (multi-hop duration estimation)
        3. contention_groups — depends on routing_hints + critical_path (paths + timing)
        4. node_views     — depends on critical_path (uses ASAP times)
        5. traffic_matrix — no dependencies
        6. summary        — depends on critical_path + contention_groups
        """
        # 1. Routing hints (BFS shortest paths - must come first!)
        routing_hints = compute_routing_hints(self.topology, workload)

        # 2. Critical path (uses routing_hints for accurate multi-hop duration)
        critical_path = analyze_critical_path(workload, self.topology, routing_hints)

        # 3. Link contention (uses paths from routing_hints + timing from critical_path)
        contention_groups = find_contention_groups(
            workload, self.topology, routing_hints, critical_path
        )

        # 4. Node views (uses ASAP times from critical_path)
        node_views = build_node_views(workload, critical_path)

        # 5. Traffic matrix (independent)
        traffic_matrix = compute_traffic_matrix(workload)

        # 6. Summary (aggregates results from above)
        summary = compute_workload_summary(
            workload, critical_path, contention_groups
        )

        return WorkloadAnalysisResult(
            critical_path=critical_path,
            routing_hints=routing_hints,
            contention_groups=contention_groups,
            node_views=node_views,
            traffic_matrix=traffic_matrix,
            summary=summary,
        )
---

### Task 8: 端到端示例

**文件**: `examples/analysis/run_analysis.py`

```python
"""
Example: Run workload analysis and print results.

Usage:
    python examples/analysis/run_analysis.py \
        --workload ../../example/workload_analytical.txt \
        --topo ./topologies/spectrum-x-8g.json \
        --output ./output/analysis_result.json
"""

import argparse
import json
import sys
sys.path.insert(0, "../../src")

from workload_generator.aicb_parser import AicbParser
from workload_generator.workload_builder import WorkloadBuilder
from workload_format.schema import Job, ParallelismConfig
from scheduler.analyzer import WorkloadAnalyzer
from scheduler.topology_loader import TopologyLoader


def main():
    parser = argparse.ArgumentParser(description="Workload analysis example")
    parser.add_argument("--workload", required=True, help="Path to AICB workload file")
    parser.add_argument("--topo", required=True, help="Path to topology file")
    parser.add_argument("--output", default="./output/analysis_result.json")
    args = parser.parse_args()

    # Step 1: Parse AICB workload
    aicb_parser = AicbParser()
    header, items = aicb_parser.parse(args.workload)

    # Step 2: Build P2P Workload
    job = Job(
        job_id=0,
        name="test-job",
        assigned_nodes=list(range(8)),
        parallelism=ParallelismConfig(tp=2, dp=1, pp=1, ep=1),
    )
    builder = WorkloadBuilder()
    workload = builder.build_from_aicb(header, items, job)

    # Step 3: Load topology
    topo_loader = TopologyLoader()
    topology = topo_loader.load(args.topo)

    # Step 4: Run analysis
    analyzer = WorkloadAnalyzer(topology)
    result = analyzer.analyze(workload)

    # Step 5: Print summary
    print(f"Total tasks: {result.summary.total_tasks}")
    print(f"Critical path length: {result.summary.critical_path_length_us} us")
    print(f"Comm/Compute ratio: {result.summary.comm_compute_ratio:.2f}")
    print(f"Hot links: {len(result.summary.hot_links)}")
  
    # Step 6: Save detailed result
    with open(args.output, "w") as f:
        json.dump({
            "critical_path_tasks": result.critical_path.critical_path_tasks,
            "contention_groups": {
                str(k): v.__dict__ for k, v in result.contention_groups.items()
            },
            "summary": result.summary.__dict__,
        }, f, indent=2)
  
    print(f"Analysis result saved to {args.output}")


if __name__ == "__main__":
    main()
```

---

### Task 9: 测试

**文件**: `tests/test_analyzer.py`

```python
def test_full_analysis_pipeline():
    """Run complete analysis on simple workload."""
    # Create simple workload
    # Run analyzer
    # Verify all result fields are populated

def test_critical_path_with_comm():
    """Critical path includes communication tasks."""

def test_contention_detection():
    """Verify contention groups are correct."""
```

---

## 5. 文件交付清单

| 文件                                     | 类型 | 说明                                         |
| ---------------------------------------- | ---- | -------------------------------------------- |
| `src/static_analysis/__init__.py`            | 新建 | 模块初始化                                   |
| `src/static_analysis/topology_loader.py`     | 新建 | 拓扑加载器（Task 0）                         |
| `src/static_analysis/routing_hints.py`       | 新建 | 路由信息（Task 1，关键路径分析的前置依赖）   |
| `src/static_analysis/critical_path.py`       | 新建 | 关键路径分析（Task 2，依赖 Task 1 的路由）   |
| `src/static_analysis/contention_analysis.py` | 新建 | 链路竞争分析（Task 3，依赖 Task 1 + Task 2） |
| `src/static_analysis/node_view.py`           | 新建 | 节点局部视图（Task 4）                       |
| `src/static_analysis/traffic_matrix.py`      | 新建 | 流量矩阵（Task 5）                           |
| `src/static_analysis/workload_summary.py`    | 新建 | Workload 摘要（Task 6）                      |
| `src/static_analysis/analyzer.py`            | 新建 | 统一分析入口（Task 7）                       |
| `tests/test_topology_loader.py`        | 新建 | 拓扑加载器测试                               |
| `tests/test_routing_hints.py`          | 新建 | 路由信息测试                                 |
| `tests/test_critical_path.py`          | 新建 | 关键路径测试                                 |
| `tests/test_contention_analysis.py`    | 新建 | 竞争分析测试                                 |
| `tests/test_node_view.py`              | 新建 | 节点视图测试                                 |
| `tests/test_traffic_matrix.py`         | 新建 | 流量矩阵测试                                 |
| `tests/test_analyzer.py`               | 新建 | 综合分析测试                                 |
| `examples/analysis/run_analysis.py`    | 新建 | 端到端示例                                   |

---

## 6. 开发优先级

建议按以下顺序实施（注意 Task 之间的依赖关系）：

1. **TopologyLoader**（Task 0，基础依赖，所有模块都需要）
2. **路由信息**（Task 1，关键路径分析需要路由来估算多跳 duration）
3. **关键路径分析**（Task 2，依赖 Task 1 的路由信息做精确估算，节点视图依赖它）
4. **链路竞争分析**（Task 3，依赖 Task 1 的路径 + Task 2 的时间窗口）
5. **节点视图**（Task 4，依赖 Task 2 的 ASAP 时间）
6. **流量矩阵**（Task 5，独立，可与节点视图并行）
7. **Workload 摘要**（Task 6，汇总前面所有结果）
8. **WorkloadAnalyzer 统一接口**（Task 7，整合所有模块）
9. **测试**（贯穿开发过程）
10. **示例**（最后补充）

依赖关系图：
```
Task 0: TopologyLoader
    ↓
Task 1: RoutingHints (BFS 最短路径)
    ↓
Task 2: CriticalPath (多跳 duration = transmission + propagation)
    ↓           ↓
Task 3:         Task 4:
Contention      NodeView
(路径+时间)    (ASAP 时间)
    ↓
Task 5: TrafficMatrix (独立)
    ↓
Task 6: WorkloadSummary (汇总)
    ↓
Task 7: WorkloadAnalyzer (统一入口)
```

---

## 7. 关键设计决策总结

本次更新的核心设计改进：

### 7.1 Task 执行顺序调整：路由信息提前到关键路径之前

**原设计问题**：
- `_estimate_duration` 假设 src 和 dst 之间直连（`topology.get_link(src, dst)`）
- 对于多跳 flow（如 GPU→Switch→GPU），找不到直连链路就返回 0
- 导致关键路径分析的 duration 估算不准确

**修正方案**：
- Task 1（路由信息）提前到 Task 2（关键路径）之前
- 关键路径分析使用 `_estimate_duration(task, topology, routing_hints)`
- 利用完整路径找到瓶颈链路（最低带宽链路），估算 `tx_time + propagation_delay`

**依赖关系变化**：
```
原方案: TopologyLoader → CriticalPath → RoutingHints → Contention
新方案: TopologyLoader → RoutingHints → CriticalPath → Contention
```

### 7.2 多跳 Flow Duration 估算

**关键洞察**：Task 2（关键路径）和 Task 3（链路竞争）对延迟的需求不同：

**Task 2 只需要总 duration**：
```
total_duration = tx_time + propagation_delay
             = size / bottleneck_bw + sum(link latencies)
```
- `tx_time`：发送时延，由瓶颈链路（最低带宽）决定
- `propagation_delay`：传播时延，路径上所有链路 latency 之和

**Task 3 需要 per-link timing**：
```
entry_time[link_i] = flow_start + sum(latency of link_0..link_{i-1})
exit_time[link_i]  = entry_time[link_i] + size / link_i.bandwidth
```
- 需要逐跳计算每条链路上的时间窗口，用于判断哪些 flows 在时间上真正重叠

**示例**：
```
Flow A→B→C→D, size=800Mb, start=0
Links: A→B (400Gbps, 10us), B→C (200Gbps, 20us), C→D (400Gbps, 10us)

Task 2 (total duration only):
  tx_time = 800Mb / 200Gbps = 4us  (bottleneck = B→C)
  propagation = 10 + 20 + 10 = 40us
  total = 4 + 40 = 44us

Task 3 (per-link timing):
  Link A→B: entry=0,  exit=0+2=2    (800Mb/400Gbps=2us)
  Link B→C: entry=10, exit=10+4=14  (800Mb/200Gbps=4us, bottleneck!)
  Link C→D: entry=30, exit=30+2=32  (800Mb/400Gbps=2us)
```

### 7.3 从空间竞争到时序竞争

**原设计问题**：
- 只记录哪些 flows 共用链路（空间竞争）
- 没有考虑时间维度：两条 flows 即使共用链路，如果执行时间不重叠，也不会真正竞争

**新设计**：
- 增加 `time_windows` 字段：存储 flow 在这条链路上的具体时间窗口（per-link timing）
- 增加 `get_concurrency_at_time(timestamp)` 接口：按需查询某时刻的并发数
- 删除 `potentially_concurrent_pairs`：存储所有重叠 pairs 对调度帮助有限

### 7.4 Per-Link Timing 的正确计算

**错误做法**（原设计）：平均分配 duration 到各跳
```python
per_hop_duration = global_duration // num_hops  # ← 错误！
```

**正确做法**：分离 transmission delay 和 propagation delay
```python
entry_time = flow_start + cumulative_latency     # 累加传播时延
exit_time = entry_time + transmission_delay       # 本链路发送时延
```

**好处**：
- 更精确的竞争检测：不同带宽的链路上，flow 占用时间不同
- 瓶颈链路会被正确识别为高并发区域
- 支持 Phase 4 按时间点查询

### 7.5 实用接口设计

**Phase 4 主要使用场景**：
1. **拥塞检测**：`group.get_concurrency_at_time(entry_time) > threshold` → defer
2. **最优调度**：遍历不同 start times，选择最大并发数最小的时间点
3. **带宽分配**：根据 `contention_ratio` 选择策略（greedy / fair-share / priority）

**关键指标**：
- `worst_case_concurrency`: 理论上界（所有 flows 同时）
- `best_case_concurrency`: 实际峰值（基于 ASAP 时间窗口）
- `contention_ratio = best / worst`: 量化竞争风险

### 7.6 简化设计的权衡

**当前采用方案 1（ASAP-based 静态分析）**：
- ✅ 简单直接，利用 Task 2 已有的 ASAP/ALAP 信息
- ✅ 计算快速，不需要迭代
- ⚠️ 局限：Phase 4 的实际调度可能偏离 ASAP 假设

**为什么先这样做**：
- Phase 3 定位为"静态参考"，提供 workload 的结构特征
- Phase 4 的动态调度器应能独立工作，可以选择性使用 Phase 3 的信息
- 后续可通过方案 3（迭代细化）提高准确性

---

## 8. 未来改进方向

### 8.1 迭代细化（Iterative Refinement - 方案 3）

**问题**：当前使用 ASAP 假设，但 Phase 4 的实际调度会改变时间

**解决方案**：
```python
# Iterative approach
contention_groups = None
for iteration in range(3):  # 2-3 iterations
    # Step 1: Build contention groups based on current timing estimates
    contention_groups = build_contention_groups(
        workload, topology, routing_hints, critical_path
    )
    
    # Step 2: Phase 4 does a preliminary schedule
    schedule = phase4_scheduler.schedule(workload, contention_groups)
    
    # Step 3: Extract actual timing from schedule
    actual_timing = extract_actual_schedule_times(schedule)
    
    # Step 4: Rebuild contention groups with actual timing
    contention_groups = rebuild_with_actual_timing(workload, actual_timing)
    
    # Check convergence: if timing didn't change much, stop
    if has_converged(actual_timing, previous_timing):
        break
```

**优点**：
- 最准确：反映了 Phase 4 的实际调度决策
- 自洽：分析结果与执行计划一致

**挑战**：
- 实现复杂：需要 Phase 4 支持"预调度"模式（dry-run）
- 计算开销：多次迭代可能较慢
- 收敛性：不能保证一定收敛

**适用场景**：
- 对精度要求极高的场景
- workload 规模较小（< 1000 flows）
- 离线分析（非实时调度）

### 8.2 ALAP 时间窗口（Flexibility Window）

**概念**：
- 除了 ASAP 时间，还提供 ALAP（As Late As Possible）时间
- Flow 可以在 `[asap_entry, alap_entry]` 窗口内灵活调度
- 利用 Task 2（关键路径分析）已计算的 `latest_start/latest_finish`

**数据结构**：
```python
@dataclass
class FlowLinkTiming:
    task_id: int
    link_id: tuple[int, int]
    
    asap_entry: int   # Earliest possible
    asap_exit: int
    
    alap_entry: int   # Latest without delaying JCT
    alap_exit: int
    
    @property
    def flexibility_window(self) -> int:
        return self.alap_entry - self.asap_entry
    
    def must_overlap_with(self, other: 'FlowLinkTiming') -> bool:
        """Check if two flows MUST overlap (even with flexibility)."""
        # Even if one delays to ALAP, still overlaps
        ...
```

**Phase 4 使用**：
- 优先调度 `flexibility_window` 小的 flows（它们更"紧急"）
- 对 `flexibility_window` 大的 flows，可以推迟以避开高峰

### 8.3 ECMP 多路径支持

**当前局限**：
- `_bfs_shortest_path` 只返回单一路径
- 实际网络可能有 ECMP（Equal-Cost Multi-Path）路由

**扩展方案**：
```python
def get_all_paths(topology, src, dst) -> list[list[int]]:
    """Return all shortest paths (ECMP)."""
    # Use BFS to find all shortest paths
    ...

@dataclass
class RoutingHints:
    # Change: store list of paths per (src, dst)
    _cached_paths: dict[tuple[int, int], list[list[int]]]
    
    def get_ecmp_paths(self, topology, src, dst) -> list[list[int]]:
        """Get all equal-cost shortest paths."""
        ...
```

**链路竞争分析调整**：
- Flow 可能被负载分担到多条路径上
- 每条路径上的 contention 都需要分析
- 或者保守估计：取所有路径中 contention 最高的

### 8.4 动态拓扑感知

**当前局限**：
- 假设拓扑是静态的
- 实际中链路可能故障、带宽可能动态变化

**扩展方向**：
```python
@dataclass
class DynamicTopology:
    """Time-varying topology support."""
    
    # Different bandwidth at different time windows
    time_varying_links: dict[tuple[int, int], list[tuple[int, int, float]]]
    # link_id -> [(start_time, end_time, bandwidth), ...]
    
    def get_bandwidth_at_time(self, link_id, timestamp) -> float:
        """Query link bandwidth at a specific time."""
        ...
```

---

*创建日期：2026-04-14*
*更新日期：2026-04-15*
*- v1: 初始计划，Task 0-7 定义*
*- v2: 调整 Task 顺序（路由→关键路径→竞争分析），增加时序竞争分析*
*- v3: 修正多跳 flow duration 估算（分离 transmission/propagation delay），路由信息提前到关键路径之前*
*- 简化设计：删除 potentially_concurrent_pairs，改用 get_concurrency_at_time() 接口*
*- 新增：关键设计决策总结（Section 7）和未来改进方向（Section 8）*
*前置阶段：Phase 1（已完成）+ Phase 2（已完成，153 tests passed）*
*下一阶段：Phase 4（执行器，使用 Phase 3 的分析结果进行动态调度）*

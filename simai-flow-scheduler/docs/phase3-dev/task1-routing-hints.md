# Phase 3 - Task 1 开发记录：RoutingHints（路由信息）

## 1. 目标与范围

Task 1 的目标是为拓扑中的所有 flow (src, dst) 对预计算 BFS 最短路径，缓存结果供后续模块使用。这是 Phase 3 依赖链中的关键一环：Task 2（关键路径分析）需要完整路径来估算多跳 flow 的 duration，Task 3（链路竞争分析）需要路径信息来判断哪些 flow 共享物理链路。

**包含**：
- `RoutingHints` 数据类：路径缓存 + 链路负载统计 + 自定义路由策略支持
- `compute_routing_hints(topology, workload, routing_strategy)` 函数：遍历所有 flow task，计算最短路径并聚合 link_loads
- `bfs_shortest_path(topology, src, dst)`：BFS 最短路径算法（按跳数，默认策略）
- `RoutingStrategy` 类型别名：自定义路由策略函数签名
- `get_flow_links(task, topology)`：将 flow 的缓存路径转为物理链路列表
- `get_most_used_links(top_k)`：按负载排序返回热点链路

**不包含**：
- ECMP 多路径支持（可通过自定义路由策略实现）
- 动态拓扑感知（假设拓扑是静态的）
- 内置的带权最短路径（默认 BFS 按跳数，可通过自定义策略实现 Dijkstra）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/static_analysis/
│   ├── __init__.py              # 更新：导出 RoutingHints, compute_routing_hints
│   └── routing_hints.py         # 新增：路由信息完整实现
├── tests/
│   ├── test_routing_hints.py    # 新增：34 个测试
│   └── test_topology_loader.py  # 修改：import 路径从 scheduler.xxx 改为 src.static_analysis.xxx
```

### 2.2 数据结构

```python
@dataclass
class RoutingHints:
    """On-demand routing hints with path caching and custom routing support."""

    # 自定义路由策略函数（默认：BFS 最短路径）
    routing_strategy: RoutingStrategy

    # 缓存的最短路径: (src, dst) → [src, hop1, hop2, ..., dst]
    _cached_paths: dict[tuple[int, int], list[int]]

    # 链路负载统计: link_id → 使用该链路的 flow 数量
    link_loads: dict[tuple[int, int], int]

# 路由策略函数签名
RoutingStrategy = Callable[[NetworkTopology, int, int], list[int] | None]
```

### 2.3 API

```python
class RoutingHints:
    def get_path(self, topology: NetworkTopology, src: int, dst: int) -> list[int]
    def get_flow_links(self, task: Task, topology: NetworkTopology) -> list[tuple[int, int]]
    def get_most_used_links(self, top_k: int = 20) -> list[tuple[tuple[int, int], int]]

def compute_routing_hints(
    topology: NetworkTopology, 
    workload: P2PWorkload,
    routing_strategy: RoutingStrategy | None = None
) -> RoutingHints

def bfs_shortest_path(topology: NetworkTopology, src: int, dst: int) -> list[int] | None
```

---

## 3. 设计原理

### 3.1 为什么路由信息需要单独作为一个 Task

**问题**：Task 2（关键路径分析）需要估算 flow 的 duration。对于多跳 flow（如 GPU→ASW→PSW→ASW→GPU），如果只知道 src 和 dst，不知道中间经过哪些链路，就无法准确估算传输时间。

**解决**：Task 1 在 Task 2 之前运行，为所有 flow 预计算最短路径。这样：
- Task 2 使用路径上的瓶颈链路（最低带宽）估算 transmission delay
- Task 2 使用路径上所有链路的 latency 之和估算 propagation delay
- Task 3 使用路径信息判断哪些 flow 共享同一条物理链路

### 3.2 惰性缓存设计

```python
def get_path(self, topology, src, dst):
    key = (src, dst)
    if key not in self._cached_paths:
        self._cached_paths[key] = _bfs_shortest_path(topology, src, dst) or []
    return self._cached_paths[key]
```

**设计选择**：`compute_routing_hints` 会遍历所有 flow 并触发 BFS，主动填充缓存。但 `get_path` 也可以被 Task 2/3 按需调用（如果需要查询一个 workload 中未出现的 src-dst 对），此时自动触发 BFS 并缓存结果。

### 3.3 BFS 路径查找

```python
def _bfs_shortest_path(topology, src, dst):
    if src == dst:
        return [src]
    visited = {src}
    queue = deque([(src, [src])])
    while queue:
        current, path = queue.popleft()
        for neighbor, _link in topology.get_neighbors(current):
            if neighbor not in visited:
                new_path = path + [neighbor]
                if neighbor == dst:
                    return new_path
                visited.add(neighbor)
                queue.append((neighbor, new_path))
    return None
```

BFS 保证找到跳数最少的路径。`queue` 中存储完整路径（而非仅 parent 指针），虽然内存开销略高，但代码简洁且路径重建无需回溯。

### 3.4 自定义路由策略支持（2026-04-16 扩展）

**设计目标**：支持用户自定义路由算法（如带宽感知、延迟感知、ECMP 等），而不修改核心代码。

**实现方式**：策略模式 + 可调用对象

```python
# 类型别名定义路由策略签名
RoutingStrategy = Callable[[NetworkTopology, int, int], list[int] | None]

@dataclass
class RoutingHints:
    routing_strategy: RoutingStrategy = field(default=None)
    
    def __post_init__(self):
        if self.routing_strategy is None:
            self.routing_strategy = bfs_shortest_path

def compute_routing_hints(
    topology, workload, 
    routing_strategy: RoutingStrategy | None = None
) -> RoutingHints:
    if routing_strategy is None:
        routing_strategy = bfs_shortest_path
    hints = RoutingHints(routing_strategy=routing_strategy)
    # ...
```

**使用示例**：

```python
# 默认 BFS（向后兼容）
hints = compute_routing_hints(topology, workload)

# 自定义：基于带宽的路由
def bandwidth_aware_routing(topology, src, dst):
    # Dijkstra，权重 = 1/bandwidth
    # 返回最大带宽路径
    pass

hints = compute_routing_hints(topology, workload, bandwidth_aware_routing)

# 自定义：基于延迟的路由
def latency_aware_routing(topology, src, dst):
    # Dijkstra，权重 = latency
    # 返回最低延迟路径
    pass

hints = compute_routing_hints(topology, workload, latency_aware_routing)
```

**优点**：
- 零侵入：现有代码无需修改，默认行为不变
- 灵活：支持任意路由算法
- 可测试：可以注入 mock 策略进行测试
- 类型安全：`RoutingStrategy` 类型别名提供清晰的接口契约

### 3.5 link_loads 的聚合方式

```python
for task in workload.tasks:
    if not task.is_flow():
        continue
    path = hints.get_path(topology, task.src, task.dst)
    links = [(path[i], path[i+1]) for i in range(len(path)-1)]
    for link in links:
        hints.link_loads[link] = hints.link_loads.get(link, 0) + 1
```

每条 flow 贡献 +1 到其路径上的每一条物理链路。同一 (src, dst) 对的多个 flow（例如 Ring AllReduce 的多个 chunk）会分别计数。

---

## 4. 关键发现

### 4.1 拓扑链路的双向性（2026-04-16 修正）

**初始假设（错误）**：拓扑文件中每一行定义单向链路。

**实际情况**：经过 Task 0 的 C++ 源码验证（`common.h:793-826`），拓扑文件中每一行代表**双向链路**。NS-3 的 `qbb.Install(snode, dnode)` 创建双向通道，并在两个方向上填充 `nbr2if` 映射。

**修正**：Task 0 已修改 `TopologyLoader`，对每一行解析出的链路创建正向和反向两个 `Link` 对象。因此 BFS 路由可以在双向拓扑上正常工作，GPU 间路径（如 GPU 0 → PSW 17 → ASW 9 → GPU 3）可以被正确找到。

### 4.2 路由失败处理策略（2026-04-16 修正）

**初始实现（不合理）**：当 BFS 找不到路径时，返回空列表，`get_flow_links` 退化为 `[(src, dst)]` 直连作为 fallback。

**问题**：对于正常的全连通网络拓扑，找不到路径表明拓扑配置错误或 BFS 实现有 bug，不应该静默 fallback。

**修正**：移除 fallback 行为，当 `_bfs_shortest_path` 返回 `None` 时，`get_path` 直接抛出 `ValueError`，明确指出拓扑连通性问题。这样可以在开发阶段尽早发现拓扑配置错误。

### 4.3 Import 路径约定

项目中 `src/` 是一个 Python 包（有 `__init__.py`），测试中必须使用 `from src.static_analysis.xxx import ...` 而非 `from scheduler.xxx import ...`，否则 `scheduler` 包内的相对导入（`from ..workload_format.schema`）会因 "attempted relative import beyond top-level package" 而失败。

此约定与 Phase 1/2 的测试一致（如 `from src.workload_generator.workload_builder import ...`）。

---

## 5. 开发过程

### 5.1 发现的问题及修复

#### 问题 1：Import 路径错误导致相对导入失败

**现象**：
```
from ..workload_format.schema import P2PWorkload, Task, TaskType
ImportError: attempted relative import beyond top-level package
```

**原因**：测试中使用 `from scheduler.routing_hints import ...`（无 `src.` 前缀），导致 Python 将 `scheduler` 视为顶级包，`..` 超出顶级包范围。

**修复**：将测试 import 改为 `from src.static_analysis.routing_hints import ...`，确保 `scheduler` 被视为 `src` 的子包。同时修正 Task 0 的 `test_topology_loader.py` 保持一致（虽然 topology_loader 不涉及跨包相对导入，不影响正确性，但风格统一）。

#### 问题 2：link_loads 聚合计数错误

**现象**：断言 `hints.link_loads[(0, 9)] == 1` 失败，实际值为 2。

**原因**：Flow 0→9 使用路径 [0, 9]，Flow 0→17 使用路径 [0, 9, 17]。两条 flow 都经过链路 (0, 9)，因此 load = 2。

**修复**：修正测试断言，准确反映多条 flow 共享同一链路的计数逻辑。

---

## 6. 测试结果

### 6.1 最终结果

```
tests/test_routing_hints.py: 39 passed (34 原有 + 5 自定义路由策略)
tests/test_topology_loader.py: 52 passed
tests/ (完整回归): 244 passed (153 原有 + 52 topology + 39 routing)
```

无回归，所有原有测试通过。

### 6.2 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestBfsShortestPath` | 7 | 同节点返回 [src]、直连邻居、2 跳、多跳线性、无路径返回 None、星形拓扑、直连优先于多跳 |
| `TestRoutingHintsGetPath` | 4 | 路径返回、缓存命中（同一对象）、无路径抛出异常、同节点路径 |
| `TestRoutingHintsGetFlowLinks` | 6 | 直连 flow、2 跳 flow、多跳 flow、无路径抛出异常、compute task 返回空、None src 返回空 |
| `TestGetMostUsedLinks` | 4 | 排序正确性、top_k 限制、空 loads、默认 top_k |
| `TestComputeRoutingHints` | 8 | 单 flow 直连、单 flow 多跳、多 flow 共享链路、缓存复用、跳过 compute task、跳过 None src/dst、空 workload、most_used_links 集成 |
| `TestIntegrationSpectrumX` | 5 | GPU→ASW 直连、GPU→PSW 两跳、GPU→NVSwitch 直连、反向路径存在（双向拓扑）、link_loads 聚合 |
| `TestCustomRoutingStrategy` | 5 | 自定义策略调用、compute_routing_hints 接受自定义策略、默认策略、自定义策略返回 None 抛出异常、自定义策略结果缓存 |
| `TestComputeRoutingHints` | 8 | 单 flow 直连、单 flow 多跳、多 flow 共享链路、缓存复用、跳过 compute task、跳过 None src/dst、空 workload、most_used_links 集成 |
| `TestIntegrationSpectrumX` | 5 | GPU→ASW 直连、GPU→PSW 两跳、GPU→NVSwitch 直连、反向路径存在（双向拓扑）、link_loads 聚合 |

### 6.3 拓扑 fixture 复用

测试中使用三个可复用的拓扑构建 helper：

| Helper | 结构 | 用途 |
|--------|------|------|
| `_make_linear_topo(n)` | 0→1→2→...→n-1 | 多跳路径测试 |
| `_make_star_topo(center, leaves)` | 中心交换机 + 叶子节点（双向） | 共享链路竞争测试 |
| `_make_workload_with_flows(specs)` | 最小 P2PWorkload + 指定 flow tasks | 快速构建测试 workload |

---

## 7. 与设计文档的偏差

### 7.1 使用 `task.is_flow()` 替代 `task.type.value != "flow"`

计划中使用 `task.type.value != "flow"` 过滤 flow task。实际实现使用已有的 `task.is_flow()` 方法，语义更清晰且与 schema.py 中定义的 helper 一致。

### 7.2 其他与计划一致

数据结构设计、API 接口、BFS 算法、缓存策略、link_loads 聚合方式均与计划文档完全一致。

---

## 8. 后续依赖

Task 1 的输出 (`RoutingHints`) 将被以下模块使用：

1. **Task 2（CriticalPath）**：
   - `routing_hints.get_path(topology, src, dst)` → 获取完整路径
   - 遍历路径找到瓶颈链路（最低带宽），估算 transmission delay
   - 累加路径上所有链路 latency，估算 propagation delay

2. **Task 3（ContentionAnalysis）**：
   - `routing_hints.get_flow_links(task, topology)` → 获取 flow 经过的物理链路列表
   - 按链路分组 flow，构建 contention groups
   - `routing_hints.link_loads` → 快速识别热点链路

3. **Task 7（WorkloadAnalyzer）**：
   - 统一入口，调用 `compute_routing_hints(topology, workload)` 作为第一步

---

*开发时间：2026-04-16（初始实现 + 双向拓扑修正 + 自定义路由策略扩展）*
*测试状态：39 passed（routing hints）+ 52 passed（topology loader）+ 153 passed（原有）= 244 total*
*重要修正：*
1. *拓扑链路为双向而非单向（Task 0 已修复，TopologyLoader 自动创建反向链路）*
2. *移除 fallback 行为：找不到路径时抛出 ValueError，而非静默退化为直连*
3. *支持自定义路由策略：通过 RoutingStrategy 类型别名和策略模式，允许用户注入自定义路由算法*

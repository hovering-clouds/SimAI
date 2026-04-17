# Phase 3 - Task 0 开发记录：TopologyLoader（拓扑加载器）

## 1. 目标与范围

Task 0 是 Phase 3 的基础设施，目标是解析 astra-sim 格式的网络拓扑文件，构建内存中的网络拓扑图 (`NetworkTopology`)。Phase 3 后续所有模块（路由信息、关键路径分析、链路竞争分析等）都依赖拓扑信息。

**包含**：
- `NodeType` 枚举：节点类型分类（GPU / NV_SWITCH / ASW_SWITCH / PSW_SWITCH / DSW_SWITCH）
- `Link` 数据类：单向物理链路表示，以 `(src, dst)` 为标识
- `NetworkTopology` 数据类：内存拓扑图，提供链路查找、邻居遍历、节点分类等 API
- `TopologyLoader` 解析器：解析 astra-sim 格式拓扑文件（Spectrum-X、AlibabaHPN、DCN+）
- 支持多种带宽单位（Gbps / Mbps / Tbps）和延迟单位（ms / us / ns / s）

**不包含**：
- 路由算法（Task 1 实现 BFS 最短路径）
- 关键路径分析（Task 2）
- 链路竞争分析（Task 3）

---

## 2. 交付物

### 2.1 新增文件

```
simai-flow-scheduler/
├── src/static_analysis/
│   ├── __init__.py              # 新增：模块初始化，导出核心类型
│   └── topology_loader.py       # 新增：拓扑加载器完整实现
└── tests/
    └── test_topology_loader.py  # 新增：52 个测试
```

### 2.2 数据结构

```python
class NodeType(str, Enum):
    """Type of network node."""
    GPU = "gpu"
    NV_SWITCH = "nv_switch"      # NVLink switch（机内高速互联）
    ASW_SWITCH = "asw_switch"    # Aggregate switch
    PSW_SWITCH = "psw_switch"    # Pod switch
    DSW_SWITCH = "dsw_switch"    # Distribution switch

@dataclass
class Link:
    """Represents a unidirectional physical link."""
    src: int
    dst: int
    bandwidth_gbps: float
    latency_us: float
    error_rate: float

    @property
    def link_id(self) -> tuple[int, int]:  # (src, dst)

@dataclass
class NetworkTopology:
    """In-memory representation of the network topology."""
    links: dict[tuple[int, int], Link]       # (src, dst) → Link
    adjacency: dict[int, list[tuple[int, Link]]]  # node → [(neighbor, link)]
    node_types: dict[int, NodeType]          # node_id → type
    total_nodes: int
    gpu_count: int
    switch_count: int
    gpu_type: str
    gpu_nodes: list[int]
    switch_nodes: list[int]
```

### 2.3 API

```python
class TopologyLoader:
    def load(self, topo_file: str | Path) -> NetworkTopology

class NetworkTopology:
    def add_link(self, link: Link)
    def get_link(self, src: int, dst: int) -> Optional[Link]
    def get_neighbors(self, node: int) -> list[tuple[int, Link]]
    def get_gpu_nodes(self) -> list[int]
    def get_switch_nodes(self) -> list[int]
    def is_gpu_node(self, node: int) -> bool
    def is_switch_node(self, node: int) -> bool
```

---

## 3. 拓扑文件格式详解

### 3.1 文件结构

基于 `astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py` 生成的拓扑文件格式：

```
第 1 行: total_nodes gpus_per_server nv_switch_count other_switch_count total_links gpu_type
第 2 行: 所有交换机节点 ID（空格分隔）
第 3 行起: src dst bandwidth latency error_rate（单向链路定义）
```

### 3.2 Spectrum-X 示例

文件：`Spectrum-X_8g_8gps_400Gbps_H100`

```
18 8 1 9 24 H100                # 18 节点, 8 gpu/server, 1 NV switch, 9 其他交换机, 24 链路
8 9 10 11 12 13 14 15 16 17     # 交换机节点 ID 列表
0 8 2880Gbps 0.000025ms 0       # GPU 0 → NV Switch 8: NVLink 2880Gbps
0 9 400Gbps 0.0005ms 0          # GPU 0 → ASW 9: 网卡 400Gbps
...
9 17 400Gbps 0.0005ms 0         # ASW 9 → PSW 17: 400Gbps
```

拓扑结构（Rail-Optimized SingleToR）：

```
GPU 0 ── NV Switch (8) ── GPU 1 ── GPU 2 ── ... ── GPU 7
  │                          │                       │
  └── ASW 9 ────── ... ─────┴── ASW 16 ──── ... ────┘
                      │
                  PSW (17)
```

### 3.3 AlibabaHPN 示例

文件：`AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_H100`

```
38 8 2 20 80 H100               # 38 节点, 8 gpu/server, 2 NV switch, 20 其他交换机, 80 链路
16 17 18 19 20 21 ... 37        # 22 个交换机节点 ID
0 16 2880Gbps 0.000025ms 0      # GPU 0 → NV Switch 16
0 18 200Gbps 0.0005ms 0         # GPU 0 → ASW 18: NIC 200Gbps
0 26 200Gbps 0.0005ms 0         # GPU 0 → ASW 26: 第二平面 NIC
...
```

双平面（DualPlane）特征：每个 GPU 有两条 NIC 链路连接到不同平面的交换机。

### 3.4 Header 字段 2 的语义

**重要发现**：Header 第 2 个字段是 `gpus_per_server`（每台服务器的 GPU 数），而非总 GPU 数。两种典型情况：

| 拓扑 | Header | 字段 2 值 | 实际 GPU 总数 | 计算方式 |
|------|--------|-----------|--------------|----------|
| Spectrum-X (8 GPU) | `18 8 1 9 24 H100` | 8 | 8 | 18 - (1+9) = 8 |
| AlibabaHPN (16 GPU) | `38 8 2 20 80 H100` | 8 | 16 | 38 - (2+20) = 16 |

对于 Spectrum-X，因为只有一台服务器，`gpus_per_server` 恰好等于 GPU 总数。但对于多服务器拓扑（如 AlibabaHPN 的 2 台服务器），两者不同。

正确推断方式：**GPU 总数 = total_nodes - switch_count**。

---

## 4. 关键实现细节

### 4.1 GPU 节点推断

```python
# 正确方式：GPU nodes = 全部节点 - 交换机节点
switch_set = set(switch_ids)
gpu_nodes_set = set(range(total_nodes)) - switch_set
topo.gpu_count = total_nodes - switch_count
```

不依赖 Header 字段 2 的值，而是通过排除法推断 GPU 节点集合。

### 4.2 交换机分类启发式

```python
for i, sid in enumerate(switch_ids):
    if i < nv_switch_count:
        topo.node_types[sid] = NodeType.NV_SWITCH
    elif i < nv_switch_count + other_switch_count:
        topo.node_types[sid] = NodeType.ASW_SWITCH
```

交换机按在 ID 列表中的位置分类：
- 前 `nv_switch_count` 个 → NV_SWITCH
- 其余 → ASW_SWITCH

**局限**：无法精确区分 ASW / PSW / DSW，后续可根据连接模式优化。

### 4.3 邻接表维护

```python
def add_link(self, link: Link):
    self.links[link.link_id] = link

    if link.src not in self.adjacency:
        self.adjacency[link.src] = []
    self.adjacency[link.src].append((link.dst, link))

    # 确保目标节点也出现在邻接表中（即使没有出边）
    if link.dst not in self.adjacency:
        self.adjacency[link.dst] = []
```

关键设计：确保只有入边、没有出边的节点（如汇聚层交换机 PSW）也出现在 `adjacency` 字典中，值为空列表。这避免了后续路由算法（BFS）中的 KeyError。

### 4.4 单位转换

带宽解析支持 Gbps / Mbps / Tbps，延迟解析支持 s / ms / us / ns：

```python
# 带宽 → 统一为 Gbps
if unit == "MBPS": return value / 1000.0
if unit == "TBPS": return value * 1000.0

# 延迟 → 统一为微秒 (us)
if unit == "s":  return value * 1e6
if unit == "ms": return value * 1e3
if unit == "ns": return value / 1e3
```

---

## 5. 开发过程

### 5.1 方法

直接实现 + 测试驱动。计划文档（phase3-plan.md）提供了完整的设计和参考代码，实现过程主要关注：
1. 确保与真实拓扑文件格式兼容
2. 验证数据结构的正确性
3. 覆盖边界情况

### 5.2 发现的问题及修复

#### 问题 1：Header 字段 2 语义误解

**现象**：AlibabaHPN 拓扑的测试失败，`gpu_count` 期望 16 但得到 8。

**原因**：计划文档中声明 Header 字段 2 为 `gpu_count`（总 GPU 数），但实际该字段为 `gpus_per_server`（每服务器 GPU 数）。对于 Spectrum-X（单服务器），两者恰好相同；对于 AlibabaHPN（双服务器），两者不同。

**验证**：
- Spectrum-X: `18 8 1 9 24 H100` → 字段 2 = 8，实际 GPU = 8 ✓（巧合）
- AlibabaHPN: `38 8 2 20 80 H100` → 字段 2 = 8，实际 GPU = 16 ✗（不一致）

通过阅读实际文件确认：GPU 节点 0-15 共 16 个（来自链路定义），而 Header 字段 2 为 8（对应 `8gps`，即每服务器 8 GPU）。

**修复**：改为通过 `total_nodes - switch_count` 推断 GPU 总数，不依赖 Header 字段 2 的值：

```python
topo.gpu_count = total_nodes - topo.switch_count
gpu_nodes_set = set(range(total_nodes)) - switch_set
```

#### 问题 2：Mini 拓扑测试数据不一致

**现象**：测试用的 4-GPU mini 拓扑 `switch_count` 断言失败。

**原因**：Mini 拓扑的 Header 写为 `10 4 1 4 12 H100`（NV=1, other=4, 总 switch=5），但第 2 行有 6 个交换机 ID：`4 5 6 7 8 9`。

**修复**：将 Header 修正为 `10 4 1 5 12 H100`（NV=1, other=5, 总 switch=6），与实际交换机 ID 数量一致。

**教训**：手动构造测试数据时，Header 字段必须与后续行数据的实际数量一致。

#### 问题 3：链接应为双向而非单向

**现象**：初始实现将拓扑文件中的每一行视为单向链路，导致 BFS 路由无法找到 GPU 间的完整路径（例如 GPU 0 → PSW 17 → ASW 9 → GPU 3 中，PSW 和 ASW 没有反向链路）。

**原因**：计划文档中假设拓扑文件定义的是单向链路，但实际上 NS-3 C++ 代码（`astra-sim-alibabacloud/astra-sim/network_frontend/ns3/common.h:793-826`）将每一行解析为**双向链路**：

```cpp
// common.h:807-826
NetDeviceContainer d = qbb.Install(snode, dnode);  // 创建双向通道

// 双向填充 nbr2if 映射
nbr2if[snode][dnode].idx = DintToQpMap[snode][dnode];
nbr2if[snode][dnode].up = true;
nbr2if[snode][dnode].delay = delay;
nbr2if[snode][dnode].bw = bw;

nbr2if[dnode][snode].idx = DintToQpMap[dnode][snode];  // 反向
nbr2if[dnode][snode].up = true;
nbr2if[dnode][snode].delay = delay;
nbr2if[dnode][snode].bw = bw;
```

`qbb.Install(snode, dnode)` 创建的是双向通道，且代码同时填充了 `nbr2if[snode][dnode]` 和 `nbr2if[dnode][snode]`，证明每条链路在两个方向上都可用。

**修复**：在 `TopologyLoader.load()` 中，对每一行解析出的链路，同时创建正向和反向两个 `Link` 对象：

```python
for line in lines[2:]:
    link = self._parse_link_line(line)
    if link:
        topo.add_link(link)
        # 添加反向链路（相同属性，相反方向）
        reverse = Link(
            src=link.dst, dst=link.src,
            bandwidth_gbps=link.bandwidth_gbps,
            latency_us=link.latency_us,
            error_rate=link.error_rate,
        )
        topo.add_link(reverse)
```

**测试更新**：
- `test_no_reverse_links` → `test_bidirectional_links`：断言反向链路**确实存在**
- `test_total_link_count`：链路数量翻倍（24 行 × 2 方向 = 48 个 Link 对象）
- `test_adjacency_switch_outgoing`：ASW 交换机现在有反向链路回到 GPU
- `test_adjacency_complete`：PSW 节点现在有出边（反向链路回到 ASW）

**影响**：此修复使得 Task 1（RoutingHints）的 BFS 路由能够找到完整的 GPU 间路径，与 NS-3 模拟器的实际行为一致。

---

## 6. 测试结果

### 6.1 最终结果

```
tests/test_topology_loader.py: 52 passed
tests/ (完整回归): 205 passed (153 原有 + 52 新增)
```

无回归，所有原有测试通过。

### 6.2 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestSpectrumXTopology` | 13 | 真实 Spectrum-X 文件：Header 解析、GPU/交换机节点、链路带宽/延迟、邻接表、节点分类、双向链路 |
| `TestAlibabaHPNTopology` | 3 | 真实 AlibabaHPN 文件：多服务器拓扑、16 GPU 推断、链路解析 |
| `TestMiniTopology` | 8 | 4-GPU 自定义拓扑：Header、节点分类、链路带宽、邻接完整性、邻居遍历 |
| `TestLinkDataclass` | 4 | Link 的 link_id、hash、equality、inequality |
| `TestNetworkTopology` | 9 | add_link、adjacency 更新、get_link、get_neighbors、is_gpu/is_switch |
| `TestParsingEdgeCases` | 15 | 带宽单位（Gbps/Mbps/Tbps）、延迟单位（s/ms/us/ns）、大小写、错误格式、文件不存在、空文件、Header 不足 |

### 6.3 使用真实拓扑文件验证

| 文件 | 验证内容 |
|------|----------|
| `Spectrum-X_8g_8gps_400Gbps_H100` | 18 节点 / 8 GPU / 10 switches / 24 lines × 2 = 48 links，NVLink 2880Gbps / NIC 400Gbps，双向链路 |
| `AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_H100` | 38 节点 / 16 GPU / 22 switches / 80 lines × 2 = 160 links，双平面 200Gbps，双向链路 |

---

## 7. 与设计文档的偏差

### 7.1 Header 字段 2 的语义修正

**计划文档**：`gpu_count`（总 GPU 数）
**实际实现**：`gpus_per_server`（每服务器 GPU 数），通过 `total_nodes - switch_count` 推断真正的 GPU 总数。

这是一个对计划文档的**必要修正**，因为计划中的假设在多服务器拓扑（如 AlibabaHPN）下是错误的。

### 7.2 新增 `gpu_type` 字段

计划文档中 `_parse_header` 返回 6 个值但没有存储 `gpu_type`。实际实现将 `gpu_type` 存储在 `NetworkTopology` 中，供后续分析使用。

### 7.3 其他与计划一致

数据结构设计（`NodeType`、`Link`、`NetworkTopology`）、API 接口、解析算法与计划文档完全一致。

---

## 8. 后续依赖

Task 0 的输出 (`NetworkTopology`) 将被以下模块使用：

1. **Task 1（RoutingHints）**：在拓扑图上执行 BFS 查找最短路径
2. **Task 2（CriticalPath）**：使用 `get_link()` 查询链路带宽，估算 flow duration
3. **Task 3（ContentionAnalysis）**：使用链路信息和路径，分析带宽竞争
4. **Task 7（WorkloadAnalyzer）**：统一入口，创建 `TopologyLoader` 并加载拓扑

---

*开发时间：2026-04-15（初始实现）+ 2026-04-16（双向链接修复）*
*测试状态：52 passed（topology loader）+ 153 passed（原有）= 205 total*
*重要修正：*
1. *Header 字段 2 为 gpus_per_server 而非 total_gpu_count，通过排除法推断 GPU 节点*
2. *链接为双向而非单向，与 NS-3 C++ 实现一致（每行拓扑文件 = 2 个 Link 对象）*

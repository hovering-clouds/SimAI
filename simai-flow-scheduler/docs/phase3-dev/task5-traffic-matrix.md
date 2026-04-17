# Phase 3 - Task 5 开发记录：Traffic Matrix（流量矩阵）

## 1. 目标与范围

Task 5 的目标是统计节点间的流量分布，识别 top senders/receivers，检测双向通信对。

**包含**：
- `TrafficMatrix` 数据类：节点对流量统计、top senders/receivers、双向通信对
- `compute_traffic_matrix(workload)` 函数：计算流量矩阵

**不包含**：
- 时间维度的流量分析（Task 3 已提供 per-link temporal analysis）
- 动态调度决策（Phase 4 的职责）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/scheduler/
│   ├── __init__.py                # 更新：导出 TrafficMatrix, compute_traffic_matrix
│   └── traffic_matrix.py          # 新增：流量矩阵完整实现
├── tests/
│   └── test_traffic_matrix.py     # 新增：22 个测试
```

### 2.2 数据结构

```python
@dataclass
class TrafficMatrix:
    # traffic[(i, j)] = total bytes from node i to node j
    traffic: dict[tuple[int, int], int]

    # Top senders/receivers
    top_senders: list[tuple[int, int]]      # (node_id, total_sent_bytes)
    top_receivers: list[tuple[int, int]]    # (node_id, total_received_bytes)

    # Bidirectional communication pairs
    bidirectional_pairs: list[tuple[int, int]]  # (node_a, node_b) where node_a < node_b
```

### 2.3 API

```python
# 主入口
def compute_traffic_matrix(workload: P2PWorkload) -> TrafficMatrix

# TrafficMatrix 方法
tm.get_traffic(src, dst)       # 查询 src → dst 的流量
tm.get_total_traffic()          # 查询总流量
```

---

## 3. 设计原理

### 3.1 为什么需要流量矩阵

流量矩阵提供节点间通信的全局视图：
- **识别热点**：哪些节点对之间流量最大？
- **负载均衡**：top senders/receivers 是否均衡？
- **通信模式**：双向通信对表明紧密协作的节点

这些信息可以帮助：
- 调度器避免在热点链路上安排更多流量
- 拓扑设计优化（为高流量节点对提供更高带宽）
- 识别通信瓶颈

### 3.2 流量累加

对于每个 flow task，累加 `traffic[(src, dst)] += size_bytes`。多个 flow 在同一节点对之间的流量会自动聚合。

### 3.3 Top Senders/Receivers

- **Top senders**：按每个节点的总发送字节数排序（降序）
- **Top receivers**：按每个节点的总接收字节数排序（降序）

一个节点可能同时是 top sender 和 top receiver（例如 all-to-all 通信中的中心节点）。

### 3.4 双向通信对检测

如果 `traffic[(i, j)]` 和 `traffic[(j, i)]` 都存在，则 `(i, j)` 是双向通信对。使用规范化顺序 `i < j` 避免重复。

---

## 4. 测试结果

```
tests/test_traffic_matrix.py: 22 passed
tests/ (完整回归): 326 passed (304 原有 + 22 traffic matrix)
```

### 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestTrafficMatrix` | 5 | get_traffic、get_total_traffic、默认值 |
| `TestComputeTrafficMatrix` | 17 | 空workload、单flow、流量累加、不同节点对、top senders/receivers排序、聚合多目的地/源、双向对检测、规范化顺序、单向不在双向对、跳过compute、跳过None src/dst、零大小flow、混合workload |

---

## 5. 与设计文档的偏差

### 5.1 使用 `task.is_flow()` 替代 `task.type.value != "flow"`

与 Task 1/2/3/4 保持一致。

### 5.2 其他与计划一致

数据结构设计、API 接口、流量累加算法、top senders/receivers 排序、双向对检测均与计划文档一致。

---

## 6. 后续依赖

Task 5 的输出 (`TrafficMatrix`) 将被以下模块使用：

1. **Task 7（WorkloadAnalyzer）**：统一入口，调用 `compute_traffic_matrix` 作为分析流程的一步
2. **Phase 4（Scheduler）**：
   - `top_senders` / `top_receivers` — 识别通信热点节点
   - `bidirectional_pairs` — 识别紧密协作的节点对
   - `get_traffic(src, dst)` — 查询特定节点对的流量

---

*开发时间：2026-04-16*
*测试状态：22 passed（traffic matrix）+ 304 passed（原有）= 326 total*
*关键设计：节点对流量聚合，top senders/receivers 排序，双向通信对检测*

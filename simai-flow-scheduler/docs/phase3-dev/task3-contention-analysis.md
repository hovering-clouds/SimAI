# Phase 3 - Task 3 开发记录：Link Contention Analysis（链路竞争分析）

## 1. 目标与范围

Task 3 的目标是识别哪些 flows 会竞争同一条物理链路，并结合时间窗口判断是否真正可能并发。只有**时间上可能重叠且空间上共用链路**的 flows 才会产生实际竞争。

**包含**：
- `LinkContentionGroup` 数据类：单条链路上的 flow 聚合、统计、时间窗口、并发查询
- `find_contention_groups(workload, topology, routing_hints, critical_path_info)` 函数：时空竞争分析
- 扫描线算法（sweep line）计算峰值并发

**不包含**：
- 动态调度决策（Phase 4 的职责）
- 迭代精化（v2：基于实际调度结果更新时间窗口）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/static_analysis/
│   ├── __init__.py                  # 更新：导出 LinkContentionGroup, find_contention_groups
│   └── contention_analysis.py       # 新增：链路竞争分析完整实现
├── tests/
│   └── test_contention_analysis.py  # 新增：19 个测试
```

### 2.2 数据结构

```python
@dataclass
class LinkContentionGroup:
    link_id: tuple[int, int]           # 物理链路 (node_a, node_b)
    all_flows: list[int]               # 使用该链路的所有 flow task_id
    num_flows: int                     # flow 数量
    total_data_bytes: int              # 总数据量
    time_windows: dict[int, tuple[int, int]]  # task_id → (entry_us, exit_us)
    worst_case_concurrency: int        # = num_flows
    best_case_concurrency: int         # 扫描线峰值并发
```

### 2.3 API

```python
# 主入口
def find_contention_groups(
    workload: P2PWorkload,
    topology: NetworkTopology,
    routing_hints: RoutingHints,       # From Task 1
    critical_path_info: CriticalPathInfo,  # From Task 2
) -> dict[tuple[int, int], LinkContentionGroup]

# LinkContentionGroup 方法
group.get_concurrency_at_time(timestamp)     # 查询某时刻的并发数
group.get_peak_concurrency_window()           # 峰值并发时间窗口
group.has_temporal_contention                 # 是否存在时间竞争
group.contention_ratio                        # 实际/最坏并发比
```

---

## 3. 设计原理

### 3.1 为什么需要时空结合

两条 flow 即使共用某条链路，如果执行时间完全不重叠，就不会真正竞争：

```
Flow A: GPU0 → Switch → GPU3  (0-100us)  ← 与 B 重叠，真正竞争
Flow B: GPU1 → Switch → GPU2  (50-150us)
Flow C: GPU2 → Switch → GPU4  (200-300us) ← 不重叠，无竞争
```

### 3.2 Per-link 时间窗口

**关键设计**：时间窗口是 per-link 的，不是全局 flow 时间。

```
Flow A→B→C→D, size=800Mb, start=0
Links: A→B (400Gbps, 10us), B→C (200Gbps, 20us), C→D (400Gbps, 10us)

Per-link windows:
  Link A→B: entry=0,            exit=0+2us      (tx=800Mb/400Gbps)
  Link B→C: entry=10us,         exit=10+4us     (tx=800Mb/200Gbps, bottleneck!)
  Link C→D: entry=30us(10+20),  exit=30+2us
```

Flow 不可能在到达之前就占用下游链路，每个链路上的传输时间由该链路带宽决定。

### 3.3 扫描线算法

计算峰值并发使用经典扫描线：

1. 为每个 flow 的 entry/exit 生成事件 (+1/-1)
2. 按时间排序，同时刻 exit 优先于 entry
3. 扫描事件，追踪当前并发数，记录峰值

时间复杂度：O(N log N) 其中 N = flow 数量。

### 3.4 与计划文档的单位修正

计划中 `transmission_delay_us = size_bits / (bandwidth_gbps * 1e9)` 给出的是秒，后续又乘 `1e6` 转微秒。但 `flow_start_time` 和 `cumulative_latency_us` 本身已经是微秒，存在单位不一致。

实际实现修正为：
```python
tx_delay_us = (size_bits / (link.bandwidth_gbps * 1e9)) * 1e6  # 秒 → 微秒
entry_time = flow_start_time + int(cumulative_latency_us)       # 全部微秒
```

---

## 4. 测试结果

```
tests/test_contention_analysis.py: 19 passed
tests/ (完整回归): 288 passed (269 原有 + 19 contention)
```

### 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestLinkContentionGroup` | 11 | 添加flow、统计、并发查询、边界条件、峰值窗口、无竞争、contention_ratio、空组 |
| `TestFindContentionGroups` | 8 | 空workload、单flow、共享链路、时间分离flow、per-link时序正确性、跳过compute、跳过零大小、依赖链时间 |

---

## 5. 与设计文档的偏差

### 5.1 使用 `task.is_flow()` 替代 `task.type.value != "flow"`

与 Task 1/2 一致，使用 `task.is_flow()` 方法。

### 5.2 单位修正

如 3.4 节所述，修正了计划中 transmission delay 的单位问题。

### 5.3 其他与计划一致

`LinkContentionGroup` 数据结构、API 接口、扫描线算法、per-link 时间窗口计算均与计划文档一致。

---

## 6. 后续依赖

Task 3 的输出 (`LinkContentionGroup`) 将被以下模块使用：

1. **Task 7（WorkloadAnalyzer）**：统一入口，调用 `find_contention_groups` 作为分析流程的一步
2. **Phase 4（Scheduler）**：
   - `get_concurrency_at_time(t)` — 查询某时刻的链路并发数
   - `has_temporal_contention` — 快速判断是否需要流控
   - `contention_ratio` — 决定调度策略（激进 vs 贪心）

---

*开发时间：2026-04-16*
*测试状态：19 passed（contention）+ 269 passed（原有）= 288 total*
*关键设计：per-link 时间窗口 + 扫描线峰值并发，为 Phase 4 调度器提供精确的竞争参考*

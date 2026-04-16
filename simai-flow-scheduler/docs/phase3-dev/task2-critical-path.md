# Phase 3 - Task 2 开发记录：CriticalPath（关键路径分析）

## 1. 目标与范围

Task 2 的目标是实现基于 CPM（Critical Path Method）的关键路径分析，计算每个 task 的时间窗口（earliest/latest start/finish）和 slack，识别关键路径上的任务。

**包含**：
- `TaskTimingInfo` 数据类：单个任务的时间分析结果
- `CriticalPathInfo` 数据类：整体分析结果 + 访问方法
- `analyze_critical_path(workload, topology, routing_hints)` 函数：CPM 前向/后向传播
- `_estimate_duration(task, topology, routing_hints)`：多跳 flow 持续时间估算
- `_topological_sort(tasks)`：拓扑排序

**不包含**：
- TTE（Time-to-Exposed）分析（未来扩展，接口已预留）
- RCPSP（资源约束关键路径）（未来扩展）
- Per-link 时序分析（Task 3 的职责）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/scheduler/
│   ├── __init__.py              # 更新：导出 TaskTimingInfo, CriticalPathInfo, analyze_critical_path
│   └── critical_path.py         # 新增：关键路径分析完整实现
├── tests/
│   └── test_critical_path.py    # 新增：22 个测试
```

### 2.2 数据结构

```python
@dataclass
class TaskTimingInfo:
    """单个任务的时间分析结果。"""
    task_id: int
    earliest_start_us: int    # 前向传播：最早开始时间
    earliest_finish_us: int   # 前向传播：最早完成时间
    latest_start_us: int      # 后向传播：最迟开始时间
    latest_finish_us: int     # 后向传播：最迟完成时间
    slack_us: float           # slack = latest_start - earliest_start
    is_critical: bool         # slack == 0

@dataclass
class CriticalPathInfo:
    """关键路径分析结果。"""
    task_timings: dict[int, TaskTimingInfo]
    critical_tasks: list[int]
    makespan_us: int
    analysis_method: str      # "cpm" | "tte" | "rcpsp"
```

### 2.3 API

```python
def analyze_critical_path(
    workload: P2PWorkload,
    topology: NetworkTopology,
    routing_hints: RoutingHints,
) -> CriticalPathInfo
```

---

## 3. 设计原理

### 3.1 为什么需要关键路径分析

**问题**：Phase 4 的调度器需要知道哪些 flow 任务是"关键的"（延迟会直接增加 JCT），哪些有"缓冲空间"（可以降速而不影响总时间）。

**解决**：CPM 通过前向传播 + 后向传播计算每个任务的 slack：
- slack == 0：任务在关键路径上，不能延迟
- slack > 0：任务有缓冲，可以适当降速或延后

### 3.2 CPM 算法流程

```
Step 1: 拓扑排序 → 保证处理顺序正确
Step 2: 前向传播（ASAP）→ earliest_start, earliest_finish
Step 3: 后向传播（ALAP）→ latest_start, latest_finish
Step 4: 计算 slack → slack = latest_start - earliest_start
```

### 3.3 Flow 持续时间估算

```
Total Duration = transmission_delay + propagation_delay

transmission_delay = size_bits / bottleneck_bandwidth
  - bottleneck_bandwidth = min(bandwidth along all hops)

propagation_delay = sum(link.latency_us for all hops)
```

**设计选择**：per-link 时序（哪个时刻在哪个链路上）不在 Task 2 范围内，那是 Task 3（链路竞争分析）的职责。Task 2 只需要总持续时间。

### 3.4 输出格式的可扩展性

`TaskTimingInfo` 和 `CriticalPathInfo` 设计为与分析算法无关：
- `analysis_method` 字段标识使用的算法
- 三种算法（CPM / TTE / RCPSP）填充相同的数据结构
- Phase 4 只需读取 `slack_us` 和 `is_critical`，无需感知底层算法

---

## 4. 测试结果

### 4.1 最终结果

```
tests/test_critical_path.py: 22 passed
tests/ (完整回归): 266 passed (244 原有 + 22 critical path)
```

无回归，所有原有测试通过。

### 4.2 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestEstimateDuration` | 6 | compute 任务持续时间、flow 直连链路、flow 多跳、零大小、None src |
| `TestTopologicalSort` | 4 | 空列表、单任务、线性链、菱形依赖 |
| `TestAnalyzeCriticalPath` | 12 | 空workload、单任务、线性链全关键、并行分支、独立任务、makespan、slack 非负、flow 在链中、菱形+flow、analysis_method、访问方法 |

---

## 5. 与设计文档的偏差

### 5.1 使用 `task.is_flow()` 替代 `task.type.value != "flow"`

计划中使用 `task.type.value != "flow"` 过滤 flow task。实际实现使用已有的 `task.is_flow()` 方法。

### 5.2 Flow duration 计算公式

计划中 `tx_time_us = size_bits / (bottleneck_bw_gbps * 1e9)`，这给出的是秒为单位。实际实现添加了 `* 1e6` 转换为微秒：

```python
tx_time_us = size_bits / (bottleneck_bw_gbps * 1e9) * 1e6
```

### 5.3 其他与计划一致

数据结构设计、API 接口、CPM 算法流程、拓扑排序、slack 计算均与计划文档一致。

---

## 6. 后续依赖

Task 2 的输出 (`CriticalPathInfo`) 将被以下模块使用：

1. **Task 3（ContentionAnalysis）**：结合 slack 信息和链路竞争，判断哪些 flow 可以安全降速
2. **Task 7（WorkloadAnalyzer）**：统一入口，调用 `analyze_critical_path` 作为分析流程的一步
3. **Phase 4（Scheduler）**：使用 `slack_us` 和 `is_critical` 做调度决策

---

*开发时间：2026-04-16*
*测试状态：22 passed（critical path）+ 244 passed（原有）= 266 total*
*关键设计：输出格式与分析算法解耦，支持 CPM → TTE → RCPSP 渐进升级*

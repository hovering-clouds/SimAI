# Phase 3 - Task 6 开发记录：Workload Summary（工作负载摘要）

## 1. 目标与范围

Task 6 的目标是生成 workload 的全局统计信息，包括基本统计、通信/计算比例、DAG 宽度、关键路径统计、热点链路。

**包含**：
- `WorkloadSummary` 数据类：全局统计指标
- `compute_workload_summary(workload, critical_path, contention_groups)` 函数：计算摘要

**不包含**：
- 动态调度决策（Phase 4 的职责）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/static_analysis/
│   ├── __init__.py                  # 更新：导出 WorkloadSummary, compute_workload_summary
│   └── workload_summary.py          # 新增：工作负载摘要完整实现
├── tests/
│   └── test_workload_summary.py     # 新增：17 个测试
```

### 2.2 数据结构

```python
@dataclass
class WorkloadSummary:
    # Basic statistics
    total_tasks: int
    total_compute_tasks: int
    total_flow_tasks: int
    total_communication_bytes: int
    total_compute_time_us: int

    # Communication/computation ratio (time-based)
    comm_compute_ratio: float  # total_comm_time / total_compute_time

    # Average DAG width (average concurrency level)
    avg_dag_width: float

    # Critical path length
    critical_path_length_us: int

    # Communication fraction on critical path
    critical_path_comm_fraction: float

    # Hottest links
    hot_links: list[tuple[tuple[int, int], int, int]]  # (link_id, bytes, num_flows)
```

### 2.3 API

```python
# 主入口
def compute_workload_summary(
    workload: P2PWorkload,
    critical_path: CriticalPathInfo,
    contention_groups: dict[tuple[int, int], LinkContentionGroup],
) -> WorkloadSummary
```

---

## 3. 设计原理

### 3.1 为什么需要 Workload Summary

全局摘要提供 workload 的鸟瞰视图：
- **通信/计算比例**：workload 是计算密集型还是通信密集型？
- **DAG 宽度**：平均并发度如何？
- **关键路径统计**：关键路径上通信占比多少？
- **热点链路**：哪些链路流量最大？

这些信息可以帮助：
- 调度器选择合适的策略（通信密集型 workload 需要更激进的流控）
- 拓扑设计优化（为热点链路提供更高带宽）
- Workload 对比分析

### 3.2 通信/计算比例（时间维度）

`comm_compute_ratio = total_comm_time / total_compute_time`

**设计选择**：使用时间而非字节数。字节数 / 时间的量纲不对（bytes/us），无法反映真实的通信开销。使用 critical path 的 timing 信息计算 flow 持续时间。

### 3.3 DAG 宽度计算

DAG 宽度 = 平均每个深度层级的任务数。

**深度计算**：使用记忆化 DFS，深度 = max(dep 的深度) + 1。根节点（无依赖）深度为 0。

**平均宽度**：统计每个深度层级的任务数，取平均值。

### 3.4 关键路径通信占比

`critical_path_comm_fraction = cp_comm_time / cp_total_time`

其中 `cp_comm_time` 是关键路径上所有 flow task 的持续时间之和，`cp_total_time` 是 makespan。

---

## 4. 与设计文档的偏差

### 4.1 使用 `task.is_compute()` / `task.is_flow()` 替代 `task.type.value`

与 Task 1/2/3/4/5 保持一致。

### 4.2 修正 API 不一致

- `critical_path.critical_path_tasks` → `critical_path.critical_tasks`
- `critical_path.critical_path_length_us` → `critical_path.makespan_us`

### 4.3 修正 `comm_compute_ratio` 量纲

计划使用 `total_comm_bytes / total_compute_time_us`（bytes/us），量纲不对。修正为 `total_comm_time / total_compute_time`（us/us，无量纲）。

### 4.4 修正 DAG 宽度计算

计划使用 `depth = len(task.deps)`，这不是正确的深度计算（只考虑直接依赖数量，不考虑依赖链长度）。修正为记忆化 DFS 计算真实深度。

### 4.5 修正 `parallelism_factor` 量纲

计划使用 `dag_width / critical_path_length_us`（tasks/us），量纲不对。修正为直接使用 `avg_dag_width`（无量纲，表示平均并发度）。后因与 `avg_dag_width` 完全重复，移除 `parallelism_factor` 字段。

---

## 5. 测试结果

```
tests/test_workload_summary.py: 16 passed
tests/ (完整回归): 350 passed
```

### 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestComputeWorkloadSummary` | 16 | 空workload、单compute、单flow、混合、comm_compute_ratio时间维度、DAG宽度（单任务/线性链/并行/菱形）、关键路径长度、关键路径通信占比（无/全部/混合）、热点链路排序/top10/结构 |

---

## 6. 后续依赖

Task 6 的输出 (`WorkloadSummary`) 将被以下模块使用：

1. **Task 7（WorkloadAnalyzer）**：统一入口，调用 `compute_workload_summary` 作为分析流程的最后一步
2. **Phase 4（Scheduler）**：
   - `comm_compute_ratio` — 判断 workload 类型，选择调度策略
   - `avg_dag_width` / `parallelism_factor` — 评估并行度
   - `hot_links` — 识别流量热点

---

*开发时间：2026-04-16*
*测试状态：17 passed（workload summary）+ 326 passed（原有）= 343 total*
*关键设计：时间维度的 comm_compute_ratio，记忆化 DFS 计算 DAG 深度*

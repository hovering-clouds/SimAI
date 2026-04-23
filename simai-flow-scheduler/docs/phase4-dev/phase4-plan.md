# Phase 4 开发规划

## 概述

Phase 4 的核心目标是实现执行器和调度策略，分为两个阶段：

- **Phase 4a**：执行器（Analytical + NS3）
- **Phase 4b**：调度策略（可插拔的带宽竞争计算器和排序策略）

前置条件：Phase 3 扩展的 Task Serializer 已完成，输出 `ExecutionPlan` 文件。详见 [phase3-extend.md](../phase3-dev/phase3-extend.md)。

---

## Phase 4a: 执行器

### 目标

实现 Analytical 和 NS3 两个执行后端。两者都读取原始 P2PWorkload + ExecutionPlan 文件进行时间推进模拟。

### Analytical Executor

#### 核心设计

采用**离散事件模拟**方式，维护全局时间线和 active flow 集合。

#### 输入

- `P2PWorkload`（原始 workload 文件）：所有 task 定义 + deps
- `ExecutionPlan`（Serializer 输出）：每个节点上 compute task 的执行顺序
- `NetworkTopology`（拓扑文件）：用于路由查询和带宽查询

#### 事件类型

| 事件类型 | 触发条件 | 处理逻辑 |
|---------|---------|---------|
| `compute_ready` | 某节点 compute_order 中的下一个 compute 的所有 deps 满足 | 开始执行 compute，end_time = current + duration |
| `compute_done` | 某 compute 任务执行完成 | 释放下游依赖：检查哪些 flow task 现在 deps 满足，将它们全部发出 |
| `flow_ready` | 某 flow task 的所有 deps 满足 | 创建 ActiveFlow，加入 active flow 集合 |
| `flow_completion` | 某 active flow 传输完成 | 从 active flow 集合移除，释放下游依赖 |

#### 事件循环

```
1. 初始化：各节点 compute_order 的第一个 compute 设为 compute_ready（时间=0）
2. 从事件优先队列取出最早事件
3. 推进全局时间到该事件时刻
4. 处理事件：
   a. compute_ready:
      - 安排 compute_done 事件：end_time = current_time + duration_us
   b. compute_done:
      - 检查依赖该 compute 的 flow task：哪些 flow 的 deps 现在全部满足？
      - 对每个满足 deps 的 flow task，创建 ActiveFlow，加入集合 → flow_ready
      - 检查 compute_order 中的下一个 compute，其 deps 是否满足 → 安排 compute_ready
   c. flow_ready:
      - 创建 ActiveFlow（通过 RoutingHints 查询路径），加入集合
      - 调用竞争带宽计算器重新分配带宽
      - 计算所有 active flow 的预计完成时间，更新 flow_completion 事件
   d. flow_completion:
      - 从 active flow 集合移除该 flow
      - 释放下游依赖（可能触发其他节点的 compute_ready 或 flow_ready）
      - 调用竞争带宽计算器重新分配剩余 flow 的带宽
5. 重复步骤 2-4，直到队列为空
```

**关键特性**：
- Compute 和 flow 是独立管道：compute 执行不阻塞 flow 发出（只要 deps 满足）
- 多条 flow 可同时在网：每条 flow_ready 都创建独立的 ActiveFlow
- 带宽竞争在 flow_ready 和 flow_completion 时动态计算

#### ActiveFlow 数据结构

```python
@dataclass
class ActiveFlow:
    """当前正在传输的流。"""
    task_id: int                  # 对应 P2PWorkload 中的 flow task
    src: int
    dst: int
    size_bytes: int               # 总大小
    remaining_bytes: int          # 剩余字节数
    path: list[int]               # 网络路径（由 RoutingHints 查询）
    start_time: int               # 开始时间（微秒）
    current_bw_gbps: float        # 当前分配带宽
    estimated_end_time: int       # 预计完成时间
```

#### 竞争带宽计算器接口

```python
class BandwidthAllocator(ABC):
    """带宽分配策略接口。"""

    @abstractmethod
    def allocate(
        self,
        active_flows: list[ActiveFlow],
        topology: NetworkTopology,
        current_time: int,
    ) -> dict[int, float]:
        """
        根据当前 active flow 集合和拓扑信息，计算每条流的带宽分配。

        Args:
            active_flows: 当前正在传输的流列表
            topology: 网络拓扑
            current_time: 当前全局时间

        Returns:
            task_id → allocated_bw_gbps 的映射
        """
        pass
```

**默认实现**：链路带宽均分。n 条流共享同一链路，各得 bandwidth / n。

#### ExecutionResult 输出结构

```python
@dataclass
class TaskTiming:
    """单个 task 的时间信息。"""
    task_id: int
    node: int
    task_type: str                # "compute" | "flow"
    start_time_us: int
    end_time_us: int

@dataclass
class ExecutionResult:
    """执行结果。"""
    per_task: dict[int, TaskTiming]      # task_id → 时间信息
    job_iteration_times: dict[int, int]  # job_id → 单次 iteration 时间
    total_time_us: int                   # 总执行时间
    makespan_us: int                     # 从最早开始到最晚结束
```

#### 性能优化

- 只在 active flow 集合发生变化时重新计算带宽分配
- 链路竞争分析可利用 Phase 3 的 `ContentionAnalysis` 预计算信息加速
- 大规模 workload（>10k flows）可考虑时间窗口批量处理

### NS3 Executor（C++ 实现）

#### 设计思路

NS3 是 C++ 项目，因此 NS3 执行器**必须用 C++ 实现**，作为 `astra-sim-alibabacloud` 项目的扩展。它直接读取原始 P2PWorkload + ExecutionPlan JSON 文件，利用 NS3 的端侧流注册和回调机制以及网络侧的流量控制（优先级、队列分配等）进行高保真模拟。

#### 开发步骤

1. 实现 P2PWorkload + ExecutionPlan JSON 解析器（C++ 端，可使用 nlohmann/json 等库）
2. 根据每个节点的 compute_order，确定计算任务的执行时序
3. 根据 workload 中的 flow task 和 deps，向 NS3 注册流任务及回调
4. 配置网络侧的流量控制（队列、优先级等）
5. 运行 NS3 事件循环，通过回调收集各 task 的 start_time / end_time
6. 输出 ExecutionResult 格式的结果文件（JSON），供 Python 端对比分析

#### 与 Analytical Executor 的关系

- **不统一接口**：Analytical 用 Python，NS3 用 C++，各自独立实现
- **共享输入**：两者都读取同一份 P2PWorkload + ExecutionPlan JSON 文件
- **结果对比**：NS3 输出的 JSON 结果文件与 Analytical 的 `ExecutionResult` 结构对应，便于交叉验证

### 文件结构

```
src/
  executor/
    __init__.py
    analytical.py        # AnalyticalExecutor（事件循环 + 竞争带宽计算）
    bandwidth.py         # BandwidthAllocator 基类 + 默认均分实现
    result.py            # ExecutionResult / TaskTiming 数据结构

# NS3 Executor 位于 astra-sim-alibabacloud 项目中（C++ 实现）
# 不在 simai-flow-scheduler 的 Python 代码内
```

### 验收标准

- Analytical Executor 能正确运行简单的 Ring AllReduce workload，结果与手动计算一致
- 独占带宽模式下（无竞争），flow 延迟 = size * 8 / link_bw + sum(latency)
- 竞争模式下，多条流共享链路时带宽正确分配
- NS3 Executor 能成功转换格式并调用模拟器（如条件允许）
- 同一 workload 两种后端结果趋势一致

---

## Phase 4b: 调度策略

### 目标

在执行器已验证的基础上，实现可插拔的调度策略。

### 策略维度

| 维度 | 影响位置 | 默认行为 |
|------|---------|---------|
| 任务排序 | Task Serializer (Phase 3 扩展) | C++ 参考实现顺序 |
| 带宽分配 | Analytical Executor 的 BandwidthAllocator | 链路带宽均分 |
| NS3 流量控制 | NS3 Executor 的配置映射 | 默认队列行为 |

### 待实现策略

1. **优先级带宽分配**：按通信类型（TP > EP > DP）分配不同带宽比例
2. **加权公平队列（WFQ）**：按 job GPU 数量配置权重
3. **自定义排序策略**：如最短任务优先、关键路径优先等
4. **NS3 优先级映射**：将调度策略转换为 NS3 的队列/优先级配置

### 验收标准

- 同一 workload 应用不同策略，输出不同结果
- 策略可通过配置文件切换，无需修改代码
- 关键路径优先策略应比默认策略减少 makespan

---

## 整体文件结构变化

```
src/
  executor/             # Phase 4a: 新增
    __init__.py
    analytical.py
    bandwidth.py
    result.py
  scheduler/            # Phase 4b: 新增
    __init__.py
    policies.py

# TaskSerializer 位于 static_analysis/task_serializer.py，详见 phase3-dev/phase3-extend.md
# NS3 Executor 位于 astra-sim-alibabacloud 项目中（C++ 实现）
```

---

## 相关文档

- [phase3-extend.md](../phase3-dev/phase3-extend.md) — Phase 3 扩展: Task Serializer
- [flow-scheduler-design.md](../specs/flow-scheduler-design.md) — 整体架构设计
- [cpp-execution-order.md](../specs/cpp-execution-order.md) — C++ 参考实现执行顺序分析

---

*文档创建日期：2026-04-21*

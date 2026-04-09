# AI 训练集群流量调度模拟器 - 设计文档

## 1. 项目概述

### 1.1 背景与目标

当前 SimAI 项目提供了基于集合通信（Collective Communication）的训练模拟能力，但其调度粒度仅到 TP/DP/EP 组级别，无法支持针对单条点对点流（Point-to-Point Flow）的带宽分配和优先级调度研究。

本项目旨在扩展 SimAI，支持：
1. **细粒度 workload 描述**：将集合通信展开为点对点流，包含节点、流量大小、依赖关系等
2. **多任务仿真**：支持多个独立训练任务同时运行，任务间共享网络拓扑并产生流量竞争
3. **多调度策略验证**：支持离线静态调度分析，可插拔的调度策略实现
4. **双模拟后端**：理论计算模式（快速迭代）+ ns3 模式（高保真验证）

### 1.2 核心价值

- **与现有 SimAI 兼容**：复用 AICB 的模型参数解析和 workload 生成逻辑
- **通用 workload 格式**：P2P Workload 文件与调度策略、执行后端均解耦
- **快速迭代**：调度策略用 Python 实现，支持快速开发和测试

---

## 2. 系统架构

### 2.1 整体分层

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: 输入描述层                                  │
│  网络拓扑 (JSON) + 训练任务描述 (JSON)                │
│  [模型参数、并行策略 TP/DP/PP/EP、节点放置]           │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Workload 生成层 (Python)                   │
│                                                     │
│  ┌─────────────────┐    ┌──────────────────────┐   │
│  │ AICB 适配器      │    │ Collective→P2P 展开器 │   │
│  │ (复用现有逻辑)   │ →  │ Ring/Tree/AlltoAll    │   │
│  └─────────────────┘    └──────────────────────┘   │
│                                  ↓                  │
│                    ┌─────────────────────────┐      │
│                    │ 多任务合并器             │      │
│                    │ (合并多个 Job 的 flow)   │      │
│                    └─────────────────────────┘      │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: P2P Workload 文件 (JSON)                   │
│  计算任务 + 通信流 + 依赖关系 DAG                     │
│  与调度策略无关，与模拟后端无关                        │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Layer 4: 执行层 (Python)                            │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ 调度分析器 (可插拔策略)                        │   │
│  │ 静态分析冲突 → 输出带宽分配 / 优先级           │   │
│  └──────────────────────────────────────────────┘   │
│                       ↓                             │
│  ┌─────────────────┐    ┌──────────────────────┐   │
│  │ 理论计算执行器   │    │ ns3 执行器            │   │
│  │ DAG 拓扑排序     │    │ 生成 ns3 输入文件     │   │
│  │ + 时延公式       │    │ + 调用现有 ns3 后端   │   │
│  └─────────────────┘    └──────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### 2.2 目录结构

```
simai-flow-scheduler/
├── inputs/                      # Layer 1: 输入描述
│   ├── topologies/              # 网络拓扑 JSON
│   └── jobs/                    # 训练任务描述 JSON
├── src/
│   ├── workload_generator/      # Layer 2: Workload 生成
│   │   ├── __init__.py
│   │   ├── aICB_adapter.py      # AICB 输出适配器
│   │   ├── collective_expander.py  # Collective→P2P 展开（基类 + 所有算法实现）
│   │   └── job_merger.py        # 多任务合并
│   ├── workload_format/         # Layer 3: P2P Workload 格式
│   │   ├── __init__.py
│   │   ├── schema.py            # JSON Schema 定义
│   │   ├── validator.py         # 格式验证器
│   │   └── writer.py            # 文件写入
│   ├── scheduler/               # Layer 4: 调度分析器
│   │   ├── __init__.py
│   │   ├── base.py              # 基类定义
│   │   ├── static_analysis.py   # 静态冲突分析
│   │   └── policies/            # 具体调度策略
│   │       ├── __init__.py
│   │       ├── priority.py      # 优先级调度
│   │       ├── bandwidth.py     # 带宽分配
│   │       └── wfq.py           # 加权公平队列
│   └── executor/                # Layer 4: 执行器
│       ├── __init__.py
│       ├── analytical.py        # 理论计算执行器
│       └── ns3.py               # ns3 执行器
├── tests/
│   ├── test_collective_expander.py
│   ├── test_scheduler.py
│   └── test_executor.py
├── examples/
│   ├── single_job/              # 单任务示例
│   └── multi_job/               # 多任务示例
├── docs/
│   └── WORKLOAD_FORMAT.md       # Workload 格式文档
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 3. P2P Workload 格式设计

### 3.1 文件结构

```json
{
  "version": "1.0",
  "meta": {
    "num_jobs": 2,
    "num_nodes": 16,
    "generated_at": "2026-04-08T10:30:00Z"
  },
  "network": {
    "topology_file": "topologies/spectrum-x-16g.json"
  },
  "jobs": [
    {
      "job_id": 0,
      "name": "llama-70b",
      "model": "llama-70B",
      "assigned_nodes": [0, 1, 2, 3, 4, 5, 6, 7],
      "parallelism": {
        "tp": 8, "dp": 1, "pp": 1, "ep": 1
      }
    }
  ],
  "tasks": [
    {
      "task_id": 0,
      "job_id": 0,
      "iteration": 0,
      "phase": "forward",
      "layer_id": 0,
      "type": "compute",
      "node": 2,
      "duration_us": 1500,
      "deps": []
    },
    {
      "task_id": 1,
      "job_id": 0,
      "iteration": 0,
      "phase": "forward",
      "layer_id": 0,
      "type": "flow",
      "src": 2,
      "dst": 3,
      "size_bytes": 134217728,
      "comm_type": "TP_ALLREDUCE_RING",
      "chunk_id": 0,
      "num_chunks": 8,
      "deps": [0]
    },
    {
      "task_id": 2,
      "job_id": 0,
      "iteration": 0,
      "phase": "forward",
      "layer_id": 0,
      "type": "flow",
      "src": 3,
      "dst": 4,
      "size_bytes": 134217728,
      "comm_type": "TP_ALLREDUCE_RING",
      "chunk_id": 1,
      "num_chunks": 8,
      "deps": [1]
    }
  ]
}
```

### 3.2 字段说明

| 字段 | 类型 | 描述 |
|------|------|------|
| `task_id` | int | 全局唯一任务 ID |
| `job_id` | int | 所属任务 ID |
| `iteration` | int | 训练迭代序号 |
| `phase` | string | 阶段：`forward`, `backward_input`, `backward_weight`, `optimizer` |
| `layer_id` | int | 模型层序号（-1 表示全局操作如 embedding/optimizer） |
| `type` | string | 任务类型：`compute`（计算）或 `flow`（通信流） |
| `node` | int | 计算任务所在节点 ID（仅 compute 类型） |
| `src` | int | 流的源节点 ID（仅 flow 类型） |
| `dst` | int | 流的目标节点 ID（仅 flow 类型） |
| `size_bytes` | int | 流的数据量（仅 flow 类型） |
| `comm_type` | string | 通信类型语义标签，用于调度策略参考 |
| `chunk_id` | int | Ring/Tree 展开后的 chunk 序号 |
| `num_chunks` | int | 总 chunk 数量 |
| `duration_us` | int | 计算任务预估耗时（微秒） |
| `deps` | list[int] | 依赖任务 ID 列表，DAG 边 |

### 3.3 约束规则

1. **DAG 完整性**：`deps` 中的所有 task_id 必须在同一文件中存在，且不能形成环
2. **节点范围**：`node`, `src`, `dst` 的值必须在 `[0, num_nodes-1]` 范围内
3. **Chunk 依赖**：同一 collective 展开的多个 chunk，chunk i 依赖于 chunk i-1（ring 的环依赖）
4. **Phase 顺序**：同一 iteration 内，`forward` → `backward_input` → `backward_weight` → `optimizer` 顺序不能乱

---

## 4. 模块设计

### 4.1 Collective→P2P 展开器

#### 4.1.1 基类接口

```python
class CollectiveExpander(ABC):
    """Collective 通信到 P2P 流的展开器基类"""

    @abstractmethod
    def expand_allreduce(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",  # "ring", "tree", "nvls"
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """将 AllReduce 展开为多个 P2P 流任务"""
        pass

    @abstractmethod
    def expand_allgather(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        pass

    @abstractmethod
    def expand_reducescatter(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        pass

    @abstractmethod
    def expand_alltoall(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        pass
```

每个子类负责一种集合通信操作，内部按 `algo` 参数分派到具体算法实现：

```python
class AllReduceExpander(CollectiveExpander):
    def expand_allreduce(self, ranks, data_size, algo="ring", ...):
        if algo == "ring":
            return self._expand_ring(...)
        elif algo == "tree":
            return self._expand_tree(...)
        raise ValueError(f"unsupported algo '{algo}'")

    def _expand_ring(self, ranks, data_size, job_id, task_id_start):
        # Ring AllReduce 具体实现
        ...
```

#### 4.1.2 Ring AllReduce 展开算法

Ring AllReduce 将 N 个 rank 的数据分成 N 个 chunk，分三阶段：

**Phase 1: 初始 chunk**（1 步，RS 第 1 步）：
- 每个 rank 发送自己的数据到 next rank，无依赖

**Phase 2: Reduce-Scatter 迭代**（n-2 步，RS 第 2 到 n-1 步）：
- 每个 flow 依赖 ring 上前一个 rank 在上一步的 flow（对角线依赖）

**Phase 3: AllGather 迭代**（n-1 步，AG 第 1 到 n-1 步）：
- RS 完成后才开始，与 Phase 1 无关
- 同样的对角线依赖模式

总 flow 数: `n + n*(n-2) + n*(n-1) = n * 2*(n-1)`

```python
def _expand_ring(self, ranks, data_size, job_id, task_id_start):
    n = len(ranks)
    chunk_size = data_size // n
    chunk_count = 2 * (n - 1)
    ring = _build_ring_topology(ranks)

    # Phase 1: 初始 chunk (chunk_id=0)
    task_list = {}
    for rank in ranks:
        tasks.append(FlowTask(..., chunk_id=0, deps=[]))
        task_list[rank] = task_id; task_id += 1

    # Phase 2: RS 迭代 (n-2 步, chunk_id=1..n-2)
    for step in range(n - 2):
        for rank in ranks:
            deps = [task_list[ring[rank]["prev"]]]  # 对角线依赖
            tasks.append(FlowTask(..., chunk_id=1+step, deps=deps))
            ...

    # Phase 3: AG 迭代 (n-1 步, chunk_id=n-1..2n-3)
    for step in range(n - 1):
        for rank in ranks:
            deps = [task_list[ring[rank]["prev"]]]  # 对角线依赖
            tasks.append(FlowTask(..., chunk_id=(n-1)+step, deps=deps))
            ...
```

#### 4.1.3 与 MockNcclGroup 的对应关系

| MockNcclGroup 函数 | 本项目对应函数 | 说明 |
|-------------------|---------------|------|
| `genAllReduceFlowModels` | `AllReduceExpander.expand_allreduce(algo="ring")` | Ring 算法 |
| `genAllGatherFlowModels` | `AllGatherExpander.expand_allgather(algo="ring")` | Ring 算法 |
| `genReduceScatterFlowModels` | `ReduceScatterExpander.expand_reducescatter(algo="ring")` | Ring 算法 |
| `genAlltoAllFlowModels` | `AlltoAllExpander.expand_alltoall()` | 全连接 |

**验证策略**：使用相同的输入参数，对比本项目展开结果与 MockNcclGroup.cc 生成的 flow 列表，确保 `src/dst/deps/chunk_id` 完全一致。

### 4.2 调度分析器

#### 4.2.1 基类接口

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class SchedulingDecision:
    """调度决策结果"""
    task_id: int
    priority: int          # 优先级（越大越高）
    allocated_bw_gbps: float  # 分配带宽 (Gbps)
    start_time_us: int     # 允许开始时间（相对 iteration 开始）

class Scheduler(ABC):
    """调度策略基类"""

    @abstractmethod
    def analyze(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
    ) -> dict[int, SchedulingDecision]:
        """
        静态分析 workload 和 topology，输出调度决策
        """
        pass
```

#### 4.2.2 冲突分析模块

```python
class ConflictAnalyzer:
    """分析哪些流会竞争同一条链路"""

    def find_conflicts(
        self,
        tasks: list[FlowTask],
        topology: NetworkTopology,
    ) -> list[ConflictGroup]:
        """
        识别冲突组：同一时刻使用同一链路的流集合
        """
        pass

    def compute_link_usage(
        self,
        tasks: list[FlowTask],
        topology: NetworkTopology,
        start_time: int,
        end_time: int,
    ) -> dict[int, float]:
        """
        计算指定时间窗口内每条链路的带宽使用率
        link_id -> usage_ratio [0.0, 1.0]
        """
        pass
```

#### 4.2.3 预置调度策略

| 策略类 | 文件 | 描述 |
|--------|------|------|
| `StaticPriorityScheduler` | `policies/priority.py` | 严格优先级调度：TP > PP > DP |
| `BandwidthAllocationScheduler` | `policies/bandwidth.py` | 按任务 GPU 数量比例分配带宽 |
| `WFQScheduler` | `policies/wfq.py` | 加权公平队列，权重可配置 |
| `CoflowAwareScheduler` | `policies/coflow.py` | 识别同一 collective 的所有 flow，统一调度 |

### 4.3 执行器

#### 4.3.1 理论计算执行器

```python
class AnalyticalExecutor:
    """基于理论公式的计算执行器"""

    def __init__(self, scheduling_decisions: dict[int, SchedulingDecision]):
        self.decisions = scheduling_decisions
        self.link_bw = {...}  # 从 topology 加载

    def execute(self, workload: P2PWorkload) -> ExecutionResult:
        # 1. DAG 拓扑排序，获得任务执行顺序
        # 2. 按序调度任务：
        #    - 计算任务：直接累加 duration
        #    - 流任务：
        #      a. 等待所有 deps 完成
        #      b. 检查链路冲突，累加等待时间
        #      c. 计算传输时延 = size / allocated_bw
        # 3. 汇总每个 job 的 iteration time
        pass
```

**关键公式**：

```python
def compute_flow_latency(
    flow: FlowTask,
    allocated_bw_gbps: float,
    link_contention_delay: int,
) -> int:
    """计算单条流的传输时延（微秒）"""
    transmission_us = (flow.size_bytes * 8) / (allocated_bw_gbps * 1e9)
    return int(transmission_us + link_contention_delay)
```

#### 4.3.2 ns3 执行器

```python
class NS3Executor:
    """调用现有 ns3 后端执行"""

    def __init__(self, simai_dir: str):
        self.simai_dir = simai_dir
        self.ns3_binary = f"{simai_dir}/bin/SimAI_simulator"

    def execute(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
        scheduling_decisions: dict[int, SchedulingDecision],
    ) -> ExecutionResult:
        # 1. 将 P2P Workload 转换为 ns3 兼容格式
        # 2. 将调度决策转换为 ns3 流量控制配置
        # 3. 调用 bin/SimAI_simulator
        # 4. 解析输出，提取各任务的 iteration time
        pass
```

---

## 5. 开发阶段划分

### Phase 1: 基础设施（预计 2-3 周）

**目标**：建立项目框架，实现基础的 P2P Workload 生成和解析

| 周 | 任务 | 交付物 |
|----|------|--------|
| 1 | 项目初始化：目录结构、pyproject.toml、依赖管理 | `simai-flow-scheduler/` 骨架代码 |
| 1 | 定义 P2P Workload JSON Schema | `workload_format/schema.py` |
| 1 | 实现 JSON 读写和验证器 | `workload_format/writer.py`, `validator.py` |
| 2 | 实现 Ring AllReduce 展开器（对照 MockNcclGroup 验证） | `collective_expander/ring_allreduce.py` |
| 2 | 实现 AllGather / ReduceScatter / AlltoAll 展开器 | `collective_expander/*.py` |
| 2-3 | 实现 AllToAll 展开器 | `collective_expander/alltoall.py` |
| 3 | 单元测试：展开器正确性验证 | `tests/test_collective_expander.py` |

**验收标准**：
- 相同输入参数下，本项目的 Ring AllReduce 展开结果与 MockNcclGroup.cc 逐条 flow 对比完全一致
- P2P Workload JSON 文件格式验证通过

### Phase 2: 多任务支持（预计 1-2 周）

**目标**：支持多个训练任务同时存在于同一 workload 中

| 周 | 任务 | 交付物 |
|----|------|--------|
| 4 | 实现 AICB 输出适配器，复用现有 workload 生成逻辑 | `workload_generator/aicb_adapter.py` |
| 4 | 实现多任务合并器：将单任务 workload 合并为多任务 | `workload_generator/job_merger.py` |
| 5 | 示例：单任务（LLaMA-70B）+ 拓扑（Spectrum-X）生成完整 P2P Workload | `examples/single_job/` |
| 5 | 示例：多任务（Job1 + Job2）共享同一拓扑 | `examples/multi_job/` |

**验收标准**：
- 用 AICB 生成的现有格式 workload 能成功转换为 P2P Workload
- 多任务场景下，每个任务的 flow 独立生成，节点分配正确

### Phase 3: 调度分析器（预计 2-3 周）

**目标**：实现可插拔的调度策略框架和基础策略

| 周 | 任务 | 交付物 |
|----|------|--------|
| 6 | 实现调度分析器基类和静态冲突分析 | `scheduler/base.py`, `static_analysis.py` |
| 6 | 实现拓扑加载器：解析网络拓扑 JSON，识别链路 | `topology_loader.py` |
| 7 | 实现优先级调度策略 | `scheduler/policies/priority.py` |
| 7 | 实现带宽分配策略 | `scheduler/policies/bandwidth.py` |
| 8 | 实现 WFQ 调度策略 | `scheduler/policies/wfq.py` |

**验收标准**：
- 调度策略可通过配置文件切换
- 同一 workload 分别应用不同策略，输出不同的调度决策
- 调度决策可序列化保存

### Phase 4: 执行器（预计 2-3 周）

**目标**：实现理论计算执行器和 ns3 执行器

| 周 | 任务 | 交付物 |
|----|------|--------|
| 9 | 实现 DAG 拓扑排序和任务调度引擎 | `executor/analytical.py` 核心逻辑 |
| 9 | 实现链路带宽竞争模型 | `executor/bandwidth_model.py` |
| 10 | 实现理论计算执行器完整流程 | `executor/analytical.py` |
| 10 | 实现 ns3 执行器（workload 转换 + 调用现有二进制） | `executor/ns3.py` |
| 11 | 集成测试：用同一 workload 分别运行两个执行器，对比结果 | 集成测试报告 |

**验收标准**：
- 理论计算执行器能在 10 秒内完成 1000 条 flow 的调度模拟
- ns3 执行器输出与理论计算执行器的结果趋势一致（误差 < 20%）

### Phase 5: 验证与优化（预计 1-2 周）

**目标**：端到端验证系统正确性，优化性能

| 周 | 任务 | 交付物 |
|----|------|--------|
| 12 | 端到端测试：真实模型参数 → P2P Workload → 调度 → 执行 → 结果分析 | 完整测试报告 |
| 12 | 性能优化：并行化、缓存、减少内存占用 | 优化后的代码 |
| 13 | 文档完善：使用说明、API 文档、示例 | `README.md`, `docs/` |

---

## 6. 关键技术决策

### 6.1 为什么不用 Protobuf

- **理由**：JSON 格式更易于人类阅读和调试，且 SimAI 项目其他部分也使用 YAML/JSON
- **Trade-off**：序列化和反序列化性能略低于 Protobuf，但在这个场景下可接受

### 6.2 为什么调度策略用 Python 而非 C++

- **理由**：调度策略是快速迭代的部分，Python 的开发效率远高于 C++
- **Trade-off**：理论计算执行器用 Python 实现，对于大规模 workload 可能有性能瓶颈——可以通过 Cython 或多进程优化

### 6.3 为什么 ns3 执行器不直接修改 ns3 代码

- **理由**：保持与 SimAI 现有 ns3 后端的兼容性，避免重复维护
- **Trade-off**：如果 ns3 不支持某种流量控制机制，需要在"调度决策→ns3 配置转换"时做近似或降级处理

---

## 7. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Ring/Tree 展开算法与 MockNcclGroup 不一致 | 结果验证失败 | Phase 1 必须完成逐条 flow 对比验证 |
| 多任务场景下的资源竞争模型过于简化 | 理论计算结果与 ns3 偏差大 | Phase 4 用 ns3 结果校准理论模型参数 |
| 调度策略数量膨胀，难以维护 | 代码难以扩展 | 策略严格遵循基类接口，用注册模式管理 |

---

## 8. 后续扩展方向

1. **在线调度**：将静态调度升级为运行时调度，嵌入事件循环
2. **PP 通信支持**：当前仅支持 TP/DP/EP 展开，PP 的 Pipeline 并行需要扩展
3. **更多拓扑**：除 Spectrum-X 外，支持 AlibabaHPN、DCN+ 等
4. **可视化**：开发 Web 界面，展示 DAG、调度决策、时延分解

---

*文档版本：1.0*
*创建日期：2026-04-08*
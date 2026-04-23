# ExecutionResult 可视化设计文档

## 概述

将 `ExecutionResult` 导出为 Chrome Trace JSON 格式（`chrome://tracing`），提供交互式时间线视图，帮助用户定位 AI 训练模拟中的性能瓶颈。

---

## Chrome Trace 格式简介

Chrome Trace 是一种标准化的 JSON 时间线格式，被 PyTorch Profiler、TensorBoard 等工具广泛采用。用户在 Chrome 浏览器中打开 `chrome://tracing`，加载 JSON 文件即可查看。

核心数据结构是事件列表，每个事件：

```json
{
  "name": "事件名称",
  "cat": "分类",
  "ph": "X",          // X = Complete Event（有起止时间的条形）
  "ts": 1000,         // 开始时间（微秒）
  "dur": 500,         // 持续时间（微秒）
  "pid": 0,           // 进程 ID → 映射为行组（Group）
  "tid": 1,           // 线程 ID → 映射为行（Track）
  "args": {}          // 附加元数据（点击条形时显示）
}
```

**行组织方式**：
- `pid`（进程）= 行组，在 Trace Viewer 中显示为可折叠的分组
- `tid`（线程）= 行，每个 tid 是时间线上的一行

---

## 映射设计

### 行组织

每个节点拆分为 **Compute** 和 **Comm** 两行，直观展示计算与通信的 overlap：

```
Trace Viewer 显示：
┌─ Job 0 ────────────────────────────────────────────────┐
│  Node 0 (Compute)  [■ fwd L0 ■]                  [■ bwd_i L3 ■]  │
│  Node 0 (Comm)          [■ tp_ar 0→1 ■■]  [■ dp_ar 0→3 ■■■]      │
│  Node 1 (Compute)  [■ fwd L0 ■]         [■ bwd_i L3 ■]          │
│  Node 1 (Comm)          [■ tp_ar 1→2 ■]       [■ dp_ar 1→0 ■■■]  │
│  Node 2 (Compute)  [■ fwd L0 ■]    [■ bwd_i L3 ■]               │
│  Node 2 (Comm)          [■ tp_ar 2→0 ■]                          │
├─ Network ──────────────────────────────────────────────┤
│  Link 0→1  [==== flow ====]      [=== flow ===]                    │
│  Link 1→2       [= flow =]            [= flow =]                  │
└────────────────────────────────────────────────────────┘
```

**设计理由**：GPU 的计算单元（SM）和通信单元（NIC/NVLink）是独立硬件，计算与通信可以并行。拆成两行后：
- 两行同时有活动条形 → 正在 overlap
- 只有 Compute 行活跃 → 纯计算，通信空闲
- 只有 Comm 行活跃 → 纯通信，计算空闲
- 两行都空闲 → 资源浪费，潜在瓶颈

| Trace 字段 | 映射 | 说明 |
|-----------|------|------|
| `pid` | Job 分组 | 每个 Job 一个 pid，名为 "Job {job_id}" |
| `tid` | `(node_id, track_type)` | 每个节点两行：Compute 行和 Comm 行 |
| Network 的 `pid` | 独立 pid | 名为 "Network"，tid 为链路标识 |

**tid 编码**：`node_id * 2` 为 Compute 行，`node_id * 2 + 1` 为 Comm 行。

### 事件字段映射

**Compute Task**：

| Trace 字段 | 值 | 示例 |
|-----------|-----|------|
| `name` | `{phase} L{layer_id}` | `fwd L0`, `bwd_i L3`, `wg L0` |
| `cat` | `compute` | 固定值 |
| `ph` | `X` | Complete Event |
| `ts` | `start_time_us` | 直接取值 |
| `dur` | `end_time_us - start_time_us` | 计算得出 |
| `pid` | `job_id` | 分组 |
| `tid` | `node * 2` | Compute 行（偶数 tid） |
| `args` | 附加信息 | 见下 |

**Flow Task**：

| Trace 字段 | 值 | 示例 |
|-----------|-----|------|
| `name` | `{comm_type} {src}→{dst}` | `tp_ar 0→1`, `dp_allreduce 2→3` |
| `cat` | `flow` | 固定值 |
| `ph` | `X` | Complete Event |
| `ts` | `start_time_us` | 直接取值 |
| `dur` | `end_time_us - start_time_us` | 计算得出 |
| `pid` | `job_id` | 分组 |
| `tid` | `src * 2 + 1` | Comm 行（奇数 tid，取 src 节点） |
| `args` | 附加信息 | 见下 |

### args 字段

**Compute Task args**：
```json
{
  "task_id": 0,
  "phase": "forward",
  "iteration": 0,
  "layer_id": 0,
  "duration_us": 1500
}
```

**Flow Task args**：
```json
{
  "task_id": 1,
  "src": 0,
  "dst": 1,
  "size_bytes": 134217728,
  "comm_type": "tp_allreduce_ring",
  "phase": "forward",
  "iteration": 0
}
```

### 行名称元数据

为了让 Trace Viewer 显示有意义的行名称，需要插入 metadata 事件：

```json
{
  "name": "thread_name",
  "ph": "M",
  "pid": 0,
  "tid": 0,
  "args": {"name": "Node 0 (Compute)"}
}
{
  "name": "thread_name",
  "ph": "M",
  "pid": 0,
  "tid": 1,
  "args": {"name": "Node 0 (Comm)"}
}
{
  "name": "process_name",
  "ph": "M",
  "pid": 0,
  "args": {"name": "Job 0"}
}
```

### 事件名称缩写规则

| Phase | 缩写 |
|-------|------|
| `forward` | `fwd` |
| `backward_input` | `bwd_i` |
| `backward_weight` | `bwd_w` |
| `optimizer` | `opt` |

| CommType 关键词 | 缩写 |
|----------------|------|
| `tp_allreduce_ring` | `tp_ar` |
| `tp_allreduce_tree` | `tp_at` |
| `tp_allgather_ring` | `tp_ag` |
| `tp_reducescatter_ring` | `tp_rs` |
| `dp_allreduce` | `dp_ar` |
| `ep_alltoall` | `ep_a2a` |
| 其他 | 保留原样 |

---

## 两种显示模式

### Verbose 模式

展开所有 flow，每条 P2P flow 单独显示为一个条形。适合深入分析单条流的调度细节和带宽竞争。

```
Node 0 (Comm)  [■ tp_ar 0→1 ■][■ tp_ar 0→2 ■][■ tp_ag 0→1 ■]...
```

**优点**：信息完整，能看到每条流的传输时间和竞争关系
**缺点**：大规模 workload 下条形过多（8 节点 Ring AllReduce = 112 条），难以宏观把握

### Compact 模式

将同一次集合通信操作的所有 flow 合并为一个条形。适合宏观把握训练迭代的整体结构和计算/通信 overlap。

**合并规则**：按 `(job_id, iteration, phase, layer_id, comm_type, item_id)` 分组，每组中的所有 flow 合并为一个事件：

```python
merged_start = min(flow.start_time for flow in group)  # 该节点上最早的 flow 开始时间
merged_end = max(flow.end_time for flow in group)      # 该节点上最晚的 flow 结束时间
```

```
Node 0 (Comm)  [■■■■■■■ AllReduce(TP) L0 ■■■■■■■■]  [■■ RS(DP) L3 ■■]
```

**合并后事件的 args**：
```json
{
  "merged_from": [1, 2, 3, 4, 5],
  "num_flows": 5,
  "total_bytes": 536870912,
  "comm_type": "tp_allreduce_ring",
  "phase": "forward",
  "iteration": 0,
  "layer_id": 0
}
```

**优点**：时间线简洁，一次 AllReduce 从 112 个条形压缩到 8 个
**缺点**：丢失单条流的细节

### Flow Event 箭头（Compact 模式专属）

Compact 模式下使用 Chrome Trace 的 Flow Event 在事件之间画箭头，展示依赖/因果关系。

**实现方式**：每个箭头由一对 `ph: "s"`（发送端）和 `ph: "f"`（接收端）事件组成，共享同一个 `id`：

```json
// compute_0 完成后触发 AllReduce
{"name": "dep", "cat": "dep", "ph": "s", "id": "dep_1", "ts": 1500, "pid": 0, "tid": 0}
{"name": "dep", "cat": "dep", "ph": "f", "id": "dep_1", "ts": 1500, "pid": 0, "tid": 1}

// AllReduce 完成后触发下一个 compute
{"name": "dep", "cat": "dep", "ph": "s", "id": "dep_2", "ts": 8000, "pid": 0, "tid": 1}
{"name": "dep", "cat": "dep", "ph": "f", "id": "dep_2", "ts": 8000, "pid": 0, "tid": 0}
```

**箭头连接逻辑**：
1. 遍历每个合并事件的 deps 列表
2. 找到每个 dep 对应的合并事件（或 compute 事件）
3. 在 dep 源事件的 end_time 发出 `ph: "s"`，在目标事件的 start_time 发出 `ph: "f"`

**数量估计**：典型训练迭代 50 层 × 4 phase ≈ 200 个集合通信，每节点约 300 条箭头。Chrome Trace 处理这个量级没有问题。

### 依赖信息（两种模式通用）

所有事件的 args 中包含 `deps` 字段，点击条形时显示该任务依赖哪些任务：

```json
// compute_2 的 args
{
  "task_id": 2,
  "phase": "backward_input",
  "iteration": 0,
  "layer_id": 3,
  "duration_us": 1500,
  "deps": [15, 16],
  "dep_descriptions": ["flow tp_ar 0→1 (L0)", "flow tp_ar 1→2 (L0)"]
}
```

`dep_descriptions` 提供人类可读的描述，用户点击任务后可直接看到"这个计算在等哪条流"。

---

## Network 视图（可选，Phase 2 扩展）

在时间线底部添加一个独立的 "Network" 进程，显示每条链路的活跃状态：

```
┌─ Network ──────────────────────────────────────┐
│  Link 0→1  [■■■■■■■■■]    [■■■■■■]             │  ← 链路被占用的时间段
│  Link 1→2       [■■■■]        [■■■■■■]         │
│  Link 2→0  [■■■■■■■]              [■■■]        │
└────────────────────────────────────────────────┘
```

这需要从 `ActiveFlow` 的生命周期推导每条链路的占用时间窗口，不在当前 ExecutionResult 中直接提供。可作为后续扩展，在 `AnalyticalExecutor` 中记录链路级事件。

---

## 颜色规则

Chrome Trace 支持通过 `cname` 字段指定颜色，或通过 `cat` 自动着色。

| 类别 | 颜色建议 | 含义 |
|------|---------|------|
| `compute` | 蓝色系 | 计算任务 |
| `compute_idle` | 灰色 | 节点空闲（可选） |
| `flow` | 绿色系 | 通信任务 |
| `flow_blocked` | 红色 | 通信等待（deps 未满足） |

默认情况下 Trace Viewer 按 `cat` 自动着色，compute 和 flow 会自动获得不同颜色，无需额外配置。

---

## 输出接口设计

### 类继承结构

```
ChromeTraceVisualizer (基类)
├── ChromeTraceCompact       # 合并集合通信 + Flow Event 箭头
├── ChromeTraceVerbose       # 展开所有 P2P flow
├── ChromeTraceNetwork       # 链路级占用视图（未来扩展）
└── ...                      # 其他粒度模式
```

### ChromeTraceVisualizer 基类

```python
from abc import ABC, abstractmethod

class ChromeTraceVisualizer(ABC):
    """Chrome Trace 可视化基类。

    负责公共逻辑：JSON 文件写入、metadata 事件生成、行名称管理。
    子类通过实现 _build_events() 定义不同的展示粒度。
    """

    def __init__(self, workload: P2PWorkload):
        self.workload = workload

    def export(self, result: ExecutionResult, path: str) -> None:
        """导出为 Chrome Trace JSON 文件。"""
        events = self.to_events(result)
        with open(path, "w") as f:
            json.dump(events, f, indent=2)

    def to_events(self, result: ExecutionResult) -> list[dict]:
        """转换为 Chrome Trace 事件列表。"""
        events = []
        events.extend(self._build_metadata_events(result))
        events.extend(self._build_events(result))
        return events

    @abstractmethod
    def _build_events(self, result: ExecutionResult) -> list[dict]:
        """子类实现：生成具体的任务事件。"""
        pass

    def _build_metadata_events(self, result) -> list[dict]:
        """生成行名称和分组名称的 metadata 事件。"""
        ...

    def _tid_for_compute(self, node_id: int) -> int:
        """Compute 行 tid（偶数）。"""
        return node_id * 2

    def _tid_for_comm(self, node_id: int) -> int:
        """Comm 行 tid（奇数）。"""
        return node_id * 2 + 1
```

### Compact 模式子类

```python
class ChromeTraceCompact(ChromeTraceVisualizer):
    """紧凑模式：合并同一集合通信的所有 flow，附带 Flow Event 箭头。"""

    def _build_events(self, result: ExecutionResult) -> list[dict]:
        events = []
        # 1. Compute 任务直接输出
        events.extend(self._build_compute_events(result))
        # 2. Flow 任务按集合通信分组后合并
        merged = self._merge_collective_flows(result)
        events.extend(self._build_merged_flow_events(result, merged))
        # 3. 生成 Flow Event 箭头
        events.extend(self._build_flow_arrows(result, merged))
        return events

    def _merge_collective_flows(self, result) -> dict:
        """按 (job_id, iteration, phase, layer_id, comm_type) 分组 flow。"""
        ...

    def _build_flow_arrows(self, result, merged) -> list[dict]:
        """生成 compute ↔ 集合通信之间的 Flow Event 箭头。"""
        ...
```

### Verbose 模式子类

```python
class ChromeTraceVerbose(ChromeTraceVisualizer):
    """详细模式：每条 P2P flow 单独显示。"""

    def _build_events(self, result: ExecutionResult) -> list[dict]:
        events = []
        events.extend(self._build_compute_events(result))
        events.extend(self._build_flow_events(result))
        return events

    def _build_flow_events(self, result) -> list[dict]:
        """每条 flow 生成一个独立的事件。"""
        ...
```

### 文件结构

```
src/executor/
  visualizer.py              # ChromeTraceVisualizer 基类 + Compact/Verbose 子类
```

---

## 使用方式

```python
from src.executor import AnalyticalExecutor
from src.executor.visualizer import ChromeTraceCompact, ChromeTraceVerbose
from src.static_analysis.routing_hints import compute_routing_hints

# 运行模拟
executor = AnalyticalExecutor(topology, routing_hints)
result = executor.execute(workload, plan)

# 紧凑模式（合并集合通信 + 箭头）
viz = ChromeTraceCompact(workload)
viz.export(result, "output/trace_compact.json")

# 详细模式（展开所有流）
viz = ChromeTraceVerbose(workload)
viz.export(result, "output/trace_verbose.json")

# 然后在 Chrome 中打开 chrome://tracing，加载 JSON 文件
```

---

## 验收标准

- 生成的 JSON 文件能被 `chrome://tracing` 正确加载和显示
- Verbose 模式：每条 flow 单独显示
- Compact 模式：同一集合通信的 flow 合并为一个条形
- Compact 模式：事件之间显示 Flow Event 箭头表示依赖
- 两种模式下点击条形均能看到 deps 信息
- 行名称显示 "Node X (Compute)" 和 "Node X (Comm)"
- 多 Job 场景下分组正确
- `ChromeTraceVisualizer` 基类可扩展，新增粒度模式只需实现 `_build_events()`

---

*文档创建日期：2026-04-23*

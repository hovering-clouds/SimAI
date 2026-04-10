# Phase 2 开发计划：多任务支持

## 1. 目标

Phase 2 的目标是实现从 AICB 训练 workload 到 P2P Workload 的完整转换管线，并支持多个训练任务在同一拓扑下的合并。完成 Phase 2 后，用户可以：

1. 读入 AICB 格式的训练 workload 文件，自动转换为 P2P Workload JSON
2. 将多个单任务 workload 合并为一个多任务 P2P Workload
3. 通过命令行或 Python API 完成端到端的 workload 生成

---

## 2. 前置知识：已有基础设施（Phase 1）

### 2.1 项目结构

```
simai-flow-scheduler/
├── pyproject.toml
├── src/
│   ├── workload_format/
│   │   ├── schema.py          # P2PWorkload, Task, Job, Meta 等数据模型
│   │   ├── validator.py       # 结构 + 语义验证
│   │   └── writer.py          # WorkloadWriter + WorkloadReader
│   └── workload_generator/
│       └── collective_expander.py  # 4 个展开器
├── tests/
│   ├── test_workload_format.py     # 11 tests
│   └── test_collective_expander.py # 46 tests
```

### 2.2 核心数据模型（schema.py）

```python
@dataclass
class P2PWorkload:
    version: str               # "1.0"
    meta: Meta                 # num_jobs, num_nodes
    network: Optional[Network] # topology_file, bandwidth_gbps, latency_us
    jobs: list[Job]            # job_id, name, model, assigned_nodes, parallelism
    tasks: list[Task]          # task_id, job_id, type, iteration, phase, layer_id, deps, ...

@dataclass
class Task:
    task_id: int
    job_id: int
    type: TaskType             # COMPUTE | FLOW
    iteration: int = 0
    phase: Phase               # FORWARD | BACKWARD_INPUT | BACKWARD_WEIGHT | OPTIMIZER
    layer_id: int = 0
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

### 2.3 展开器接口

```python
class CollectiveExpander(ABC):
    def expand_allreduce(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0) -> list[FlowTask]
    def expand_allgather(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0) -> list[FlowTask]
    def expand_reducescatter(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0) -> list[FlowTask]
    def expand_alltoall(self, ranks, data_size, job_id=0, task_id_start=0) -> list[FlowTask]
```

已实现的 expander：`AllReduceExpander`, `AllGatherExpander`, `ReduceScatterExpander`, `AlltoAllExpander`。

`FlowTask` 通过 `.to_task()` 方法转换为 `Task` 对象。

### 2.4 运行测试

```bash
uv run pytest tests/ -v
```

---

## 3. AICB Workload 格式

### 3.1 文件结构

AICB 使用纯文本格式（`.txt`），Tab 分隔：

```
第 1 行: Header（并行策略配置）
第 2 行: Workload 条目数量
第 3~N 行: 各层/操作的详细描述
```

### 3.2 Header 行

```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: <tp> ep: <ep> pp: <pp> vpp: <num_layers> ga: <ga_num> all_gpus: <world_size> checkpoints: <cp> checkpoint_initiates: <ci> pp_comm <pp_comm_size>
```

关键参数：
- `model_parallel_NPU_group`: TP size
- `ep`: Expert Parallelism size
- `pp`: Pipeline Parallelism size
- `vpp`: Virtual Pipeline Parallelism（通常等于层数）
- `ga`: Gradient Accumulation steps
- `all_gpus`: 总 GPU 数量
- `pp_comm`: PP stage 间 activation 数据量（字节）

### 3.3 Workload 条目格式

每行 12 个 Tab 分隔字段：

| 字段序号 | 名称 | 类型 | 说明 |
|---------|------|------|------|
| 1 | name | str | 操作名（如 `embedding_layer`, `attention_column`） |
| 2 | placeholder | int | 占位符（通常 -1） |
| 3 | forward_compute_time | int | 前向计算耗时 |
| 4 | forward_comm | str | 前向通信类型：`NONE`, `ALLREDUCE`, `ALLGATHER`, `REDUCESCATTER`, `ALLTOALL` 等 |
| 5 | forward_comm_size | int | 前向通信数据量（字节） |
| 6 | backward_compute_time | int | 反向计算耗时 |
| 7 | backward_comm | str | 反向通信类型 |
| 8 | backward_comm_size | int | 反向通信数据量（字节） |
| 9 | dp_compute_time | int | DP 计算耗时 |
| 10 | dp_comm | str | DP 通信类型 |
| 11 | dp_comm_size | int | DP 通信数据量（字节） |
| 12 | process_time | int | 处理耗时（默认 100） |

### 3.4 Micro Benchmark 格式

```
MICRO
<count>
<name> <compute_time1> <comm_type> <comm_size> <count> <compute_time2> <comm_type2> <comm_size2> <count2>
```

示例：
```
MICRO
1
AllReduce 0 AllReduce 1073741824 1 0 AllReduce 1073741824 1
```

---

## 4. SimAI 拓扑格式

### 4.1 拓扑文件格式（文本）

```
第 1 行: <total_nodes> <gpus_per_server> <nv_switch_count> <switch_count> <link_count> <gpu_type>
第 2 行: <switch_node_ids...>
第 3+ 行: <src> <dst> <bandwidth> <latency> <error_rate>
```

示例（Spectrum-X 8 GPU）：
```
18 8 1 9 24 H100
8 9 10 11 12 13 14 15 16 17
0 8 2880Gbps 0.000025ms 0
0 9 400Gbps 0.0005ms 0
...
```

### 4.2 Busbw YAML 格式

```yaml
test-prefix
TP:
  allreduce,: 300      # 单位 GB/s
  allgather,: 280
  reducescatter,: 280
  alltoall,: 230
DP:
  allgather,: 380
  reducescatter,: 380
EP:
  allgather,: 45
  reducescatter,: 45
  alltoall,: 80
PP:
  busbw: 47.5
```

### 4.3 拓扑生成脚本

```bash
python3 ./astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py \
  -topo Spectrum-X -g 128 -gt A100 -bw 100Gbps -nvbw 2400Gbps
```

---

## 5. AICB Workload 语义详解

### 5.1 Item 结构：每层生成 1~5 个 item

AICB workload 文件不是"每层一个 item"，而是根据层类型和并行策略，**每层生成 1~5 个 item**：

```
Pre-layer items (GA 循环外):
  grad_gather          → DP ALLGATHER
  grad_param_comm      → DP REDUCESCATTER
  grad_param_compute   → DP 计算
  embedding_grads      → backward ALLREDUCE (TP)
  moe_grad_norm1       → ALLGATHER_DP_EP (仅 EP != DP 时)
  moe_grad_norm2       → REDUCESCATTER_DP_EP (仅 EP != DP 时)

GA 循环 (ga_num 次):
  每个 GA step 遍历所有层:
    embedding_layer     → 1 item (forward ALLREDUCE if TP>1)
    attention_column    → 1 item (forward ALLGATHER, backward REDUCESCATTER)
    attention_row       → 1 item (forward REDUCESCATTER, backward ALLGATHER)
    mlp_moelayer        → 5 items:
      1. EP ALLGATHER
      2. EP ALLTOALL dispatch
      3. TP ALLGATHER
      4. TP REDUCESCATTER
      5. EP ALLTOALL combine

Post-layer items:
  embedding_norm       → ALLREDUCE
  cross_entropy1~3     → ALLREDUCE
  optimizer1~4         → ALLREDUCE
```

**一个模型的所有 item 在一个文件中**，不是按 PP stage 分文件。

### 5.2 Compute Time 的含义

AICB 的 `forward_compute_time` 和 `backward_compute_time` 是**每个 rank 独立的计算时间**（已经按 TP 切分后的）。也就是说，如果 TP=2，每层的 compute_time 已经是切分后的值，不需要再除以 TP。

单位待确认（可能是 cycles、ns 或 us），初期先保持原值，在 executor 阶段再处理单位转换。

### 5.3 流水线并行（PP）的处理方式

**PP 不在 workload item 中体现**，关键事实：

- workload header 只有 `pp`（PP size）和 `pp_comm`（stage 间 activation 数据量）两个参数
- 所有 PP stage 读同一个 workload 文件，各自执行不同的层子集
- PP 的调度逻辑（如 1F1B）由 astra-sim 运行时处理
- `vpp`（virtual pipeline parallelism）= 模型总层数，仅用于 bubble time 计算

**Phase 2 的 PP 处理策略**：简化建模为计算依赖边：
- 将 PP stage 间通信建模为固定延迟的虚拟 compute task（不展开为 P2P flow）
- 依赖关系：当前 stage 的最后一层 output → 下一个 stage 的第一层 input
- 保留 `PP_SEND`/`PP_RECV` 的 CommType 枚举，后续可替换为真实 flow
- Phase 2 初期先只处理单 PP stage（pp=1），PP 扩展作为后续增强

### 5.4 通信类型的 TP/DP/EP 判断

AICB workload 中的通信类型**已经自带后缀标记**，直接解析即可：

| 位置 | 后缀 | 并行类型 | 说明 |
|------|------|---------|------|
| forward 字段（字段 4-5） | 无后缀 | TP | 前向通信通常是 TP |
| forward 字段 | `_EP` | EP | 带后缀的例外 |
| backward 字段（字段 7-8） | 无后缀 | TP | 反向通信通常是 TP |
| backward 字段 | `_EP` | EP | 带后缀的例外 |
| DP 字段（字段 10-11） | `_DP` | DP | DP 字段自然是 DP |
| DP 字段 | `_DP_EP` | DP×EP | DP 和 EP 的联合组 |
| 任何位置 | `ALLTOALL`（无后缀） | EP | AlltoAll 通常关联 EP |

判断方式：**直接使用 AICB 的通信类型后缀**，如 `ALLGATHER_DP_EP` → base=`ALLGATHER`, context=`dp_ep`。

---

## 6. Rank 分组设计

### 6.1 设计决策：通过 rank 列表顺序控制分组

用户传入 `assigned_nodes`（rank 列表），列表中元素的顺序隐式编码了多维并行网格的映射关系。

**约定规则**：rank 列表按 `[PP][DP][EP][TP]` 的嵌套顺序排列（TP 最内层，PP 最外层）。

**示例**：8 个 rank，tp=2, dp=2, pp=2

```python
ranks = [0, 1, 2, 3, 4, 5, 6, 7]

# 按 [PP=2][DP=2][TP=2] 排列:
#   PP stage 0: ranks[0:4] = [0,1,2,3]
#   PP stage 1: ranks[4:8] = [4,5,6,7]
#
#   在 PP stage 0 内, dp=2, tp=2:
#     DP group 0: [0,1]    DP group 1: [2,3]
#     TP group 0: [0,2]    TP group 1: [1,3]
```

**用户如何改变分组**：如果想不同的分组，调整传入的 rank 列表顺序。

```python
# 如果想让 [0,1] 和 [4,5] 成为同一个 TP group:
ranks = [0, 4, 1, 5, 2, 6, 3, 7]  # 重排顺序
```

### 6.2 Rank 分组推导算法

```python
class RankGrouper:
    """
    根据 assigned_nodes 列表顺序 + parallelism 配置推导并行分组。

    列表排列约定: [PP=pp][DP=dp][EP=ep][TP=tp]
    索引公式: global_idx = pp_idx*(dp*ep*tp) + dp_idx*(ep*tp) + ep_idx*tp + tp_idx
    """

    def __init__(self, assigned_nodes: list[int], parallelism: ParallelismConfig):
        self.nodes = assigned_nodes
        self.tp = parallelism.tp
        self.dp = parallelism.dp
        self.ep = parallelism.ep
        self.pp = parallelism.pp
        assert len(assigned_nodes) == self.tp * self.dp * self.ep * self.pp

    def get_tp_group(self, pp_idx: int, dp_idx: int, ep_idx: int) -> list[int]:
        """获取指定 (PP, DP, EP) 下的 TP group ranks。"""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for tp_idx in range(self.tp)
        ]

    def get_dp_group(self, pp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (PP, EP, TP) 下的 DP group ranks。"""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for dp_idx in range(self.dp)
        ]

    def get_ep_group(self, pp_idx: int, dp_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (PP, DP, TP) 下的 EP group ranks。"""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for ep_idx in range(self.ep)
        ]

    def get_dp_ep_group(self, pp_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (PP, TP) 下的 DP×EP 联合 group ranks。"""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for dp_idx in range(self.dp)
            for ep_idx in range(self.ep)
        ]

    def get_pp_group(self, dp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (DP, EP, TP) 下的 PP group ranks。"""
        return [
            self.nodes[
                pp_idx * (self.dp * self.ep * self.tp) +
                dp_idx * (self.ep * self.tp) +
                ep_idx * self.tp +
                tp_idx
            ]
            for pp_idx in range(self.pp)
        ]
```

### 6.3 与 AICB 通信类型的对应

| AICB comm 后缀 | 调用 | 获取的 ranks |
|----------------|------|-------------|
| 无后缀 (TP) | `get_tp_group(pp_idx, dp_idx, ep_idx)` | 同一 PP/DP/EP 内的所有 TP rank |
| `_DP` | `get_dp_group(pp_idx, ep_idx, tp_idx)` | 同一 PP/EP/TP 内的所有 DP rank |
| `_EP` | `get_ep_group(pp_idx, dp_idx, tp_idx)` | 同一 PP/DP/TP 内的所有 EP rank |
| `_DP_EP` | `get_dp_ep_group(pp_idx, tp_idx)` | 同一 PP/TP 内的所有 DP×EP rank |

---

## 7. 开发任务

### Task 1: AICB Workload 解析器

**文件**: `src/workload_generator/aicb_parser.py`

**功能**：解析 AICB 格式的 `.txt` 文件，提取结构化数据。

```python
@dataclass
class AicbHeader:
    tp: int
    ep: int
    pp: int
    vpp: int          # = 模型总层数
    ga: int           # gradient accumulation steps
    all_gpus: int     # world size
    pp_comm_size: int # PP stage 间 activation 大小

@dataclass
class AicbWorkItem:
    name: str
    forward_compute_time: int
    forward_comm: str          # "NONE", "ALLREDUCE", "ALLGATHER_DP_EP", etc.
    forward_comm_size: int
    backward_compute_time: int
    backward_comm: str
    backward_comm_size: int
    dp_compute_time: int
    dp_comm: str
    dp_comm_size: int

class AicbParser:
    def parse(self, file_path: str) -> tuple[AicbHeader, list[AicbWorkItem]]:
        """Parse AICB workload file into header + work items.
        Supports HYBRID_TRANSFORMER_FWD_IN_BCKWD format."""
        pass

    def parse_micro(self, file_path: str) -> list[AicbWorkItem]:
        """Parse MICRO format benchmark file."""
        pass

    @staticmethod
    def parse_comm_type(comm_str: str) -> tuple[str, str]:
        """Parse comm type string like 'ALLGATHER_DP_EP' into (base_type, context).
        Returns e.g. ('ALLGATHER', 'dp_ep'), ('ALLREDUCE', 'tp'), ('NONE', '')."""
        pass
```

---

### Task 2: Rank 分组器

**文件**: `src/workload_generator/rank_grouper.py`

**功能**：根据 rank 列表顺序和 parallelism 配置推导 TP/DP/EP/PP 分组。

详见第 6.2 节的 `RankGrouper` 类设计。独立模块，可与 Task 1 并行开发。

---

### Task 3: Workload 构建器

**文件**: `src/workload_generator/workload_builder.py`

**功能**：将解析后的 AICB 数据 + 展开器输出 + RankGrouper 分组组合成完整的 P2PWorkload。

这是 Phase 2 的核心模块，负责：
1. 使用 RankGrouper 推导每个通信操作的参与 rank 列表
2. 遍历 AICB 的每个 work item，为 forward/backward/dp 各阶段生成 compute + flow tasks
3. 构建 task 之间的依赖关系（DAG）
4. 管理 task_id 的全局递增
5. 处理 PP（初期简化为依赖边，不展开为 flow）

```python
class WorkloadBuilder:
    def __init__(self):
        self.expanders = {
            "ALLREDUCE": AllReduceExpander(),
            "ALLGATHER": AllGatherExpander(),
            "REDUCESCATTER": ReduceScatterExpander(),
            "ALLTOALL": AlltoAllExpander(),
        }

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        """
        Convert AICB workload to P2P Workload.

        Algorithm:
        1. 从 Job 的 assigned_nodes + parallelism 创建 RankGrouper
        2. 遍历 aicb_items:
           for each item:
             a. Forward: compute_task(fwd_compute_time)
                → if fwd_comm != NONE: expand_comm(fwd_comm, fwd_comm_size)
             b. Backward: compute_task(bwd_compute_time)
                → if bwd_comm != NONE: expand_comm(bwd_comm, bwd_comm_size)
             c. DP: compute_task(dp_compute_time)
                → if dp_comm != NONE: expand_comm(dp_comm, dp_comm_size)
           依赖链: fwd_compute → fwd_flows → bwd_compute → bwd_flows → dp_compute → dp_flows
           跨 item: 前一 item 最后 task → 当前 item 第一个 task
        3. PP 处理（pp > 1 时）：
           在 PP stage 间插入虚拟 compute task 作为依赖边
        4. 收集所有 tasks，构建 P2PWorkload
        """
        pass

    def _expand_comm(
        self,
        comm_type: str,         # e.g. "ALLGATHER_DP_EP"
        comm_size: int,
        grouper: RankGrouper,
        pp_idx: int, dp_idx: int, ep_idx: int, tp_idx: int,
        phase: Phase,
        job_id: int,
        task_id_start: int,
    ) -> list[FlowTask]:
        """
        根据通信类型后缀选择正确的 rank group，然后调用对应的 expander。

        comm_type 解析:
        - "ALLGATHER" → base="ALLGATHER", context="tp" → grouper.get_tp_group()
        - "ALLGATHER_DP_EP" → base="ALLGATHER", context="dp_ep" → grouper.get_dp_ep_group()
        - "ALLTOALL" → base="ALLTOALL", context="ep" → grouper.get_ep_group()
        - "REDUCESCATTER_DP" → base="REDUCESCATTER", context="dp" → grouper.get_dp_group()
        """
        pass
```

**关键设计决策**：

1. **通信类型后缀直接决定分组**：使用 `AicbParser.parse_comm_type()` 解析后缀，直接映射到 RankGrouper 的对应方法。
2. **PP 简化处理**：pp > 1 时，在 PP stage 间插入一个虚拟 compute task（duration = 估算传输延迟），不展开为 P2P flow。
3. **依赖关系**：同一 item 内 compute → flow 串行；跨 item 按出现顺序串联。

---

### Task 4: 多任务合并器

**文件**: `src/workload_generator/job_merger.py`

**功能**：将多个单任务 P2P Workload 合并为一个多任务 workload。

```python
class JobMerger:
    def merge(
        self,
        workloads: list[P2PWorkload],
        topology_file: str = "",
    ) -> P2PWorkload:
        """
        Merge multiple single-job workloads into one multi-job workload.

        Rules:
        1. task_id 全局重新编号（避免冲突）
        2. job_id 保持各自原有值
        3. deps 内部引用用新 task_id 替换
        4. meta.num_jobs = sum of all workloads' num_jobs
        5. meta.num_nodes = max num_nodes across workloads
        6. network topology 合并
        7. jobs 列表合并
        8. tasks 列表合并（含 task_id 重映射）
        """
        pass

    def _remap_task_ids(
        self,
        workloads: list[P2PWorkload],
    ) -> tuple[list[P2PWorkload], dict[int, dict[int, int]]]:
        """
        Remap task_ids across workloads to ensure global uniqueness.
        Returns remapped workloads and mapping: {workload_idx: {old_task_id: new_task_id}}
        """
        pass
```

---

### Task 5: 端到端示例

**目录**：`examples/single_job/` 和 `examples/multi_job/`

#### 单任务示例

```python
parser = AicbParser()
header, items = parser.parse("../../example/workload_analytical.txt")

job = Job(
    job_id=0,
    name="llama-7b",
    assigned_nodes=list(range(8)),
    parallelism=ParallelismConfig(tp=8, dp=1, pp=1, ep=1),
)

builder = WorkloadBuilder()
workload = builder.build_from_aicb(header, items, job)

writer = WorkloadWriter()
writer.write(workload, "output.json")

validator = WorkloadValidator()
is_valid, errors, _ = validator.validate_file("output.json")
assert is_valid, f"Validation failed: {errors}"
```

#### 多任务示例

```python
# Job 1: llama-7b, 8 GPUs, TP=8
job1 = Job(job_id=0, assigned_nodes=list(range(8)), parallelism=ParallelismConfig(tp=8))
w1 = builder.build_from_aicb(header1, items1, job1)

# Job 2: deepseek-moe, 8 GPUs, TP=2, EP=4
job2 = Job(job_id=1, assigned_nodes=list(range(8, 16)), parallelism=ParallelismConfig(tp=2, ep=4))
w2 = builder.build_from_aicb(header2, items2, job2)

merger = JobMerger()
merged = merger.merge([w1, w2], topology_file="topologies/spectrum-x-16g.json")
writer.write(merged, "multi_job_output.json")
```

---

### Task 6: 测试

**文件**: `tests/test_aicb_parser.py`, `tests/test_rank_grouper.py`, `tests/test_workload_builder.py`, `tests/test_job_merger.py`

测试策略：

1. **AicbParser 测试**：构造小型 AICB 文本（含 header + 2-3 个 item），验证解析结果
2. **RankGrouper 测试**：
   - tp=4, dp=1: 验证 TP group 正确
   - tp=2, dp=2: 验证 TP 和 DP group 各自正确
   - tp=2, dp=2, ep=2: 验证三维分组
   - 自定义 rank 顺序：验证改变顺序后分组结果变化
3. **WorkloadBuilder 测试**：
   - 单层 workload（1 个 ALLREDUCE）：验证 compute + flow tasks
   - 依赖链验证：compute → flow → compute
   - task_id 连续递增
   - phase 和 layer_id 正确
4. **JobMerger 测试**：
   - 合并 2 个简单 workload：task_id 重映射、deps 更新
   - 合并后 validate() 通过
5. **端到端测试**：用 `example/microAllReduce.txt` 完整走一遍

---

## 8. 文件交付清单

| 文件 | 类型 | 说明 |
|------|------|------|
| `src/workload_generator/aicb_parser.py` | 新建 | AICB 格式解析器 |
| `src/workload_generator/rank_grouper.py` | 新建 | Rank 分组推导器 |
| `src/workload_generator/workload_builder.py` | 新建 | Workload 构建器（AICB → P2P Workload） |
| `src/workload_generator/job_merger.py` | 新建 | 多任务合并器 |
| `tests/test_aicb_parser.py` | 新建 | 解析器测试 |
| `tests/test_rank_grouper.py` | 新建 | 分组器测试 |
| `tests/test_workload_builder.py` | 新建 | 构建器测试 |
| `tests/test_job_merger.py` | 新建 | 合并器测试 |
| `examples/single_job/` | 新建 | 单任务示例 |
| `examples/multi_job/` | 新建 | 多任务示例 |

---

## 9. 开发优先级

建议按以下顺序实施：

1. **AicbParser**（Task 1）— 基础依赖
2. **RankGrouper**（Task 2）— 可与 Task 1 并行开发
3. **WorkloadBuilder**（Task 3）— 核心逻辑，依赖 Task 1 和 Task 2
4. **JobMerger**（Task 4）— 相对独立
5. **测试**（Task 6）— 贯穿开发过程
6. **示例**（Task 5）— 最后补充

---

## 10. 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| AICB 格式变体多，解析容易遗漏 | 先覆盖最常见的 `HYBRID_TRANSFORMER` 和 `MICRO` 两种格式 |
| 依赖关系构建错误 | 用小型 workload 手动验证 DAG 正确性 |
| TP/DP/EP/PP 四维分组逻辑复杂 | RankGrouper 独立模块 + 充分的单元测试覆盖各种组合 |
| Compute time 单位不明确 | 保持原值，在文档中标注单位待确认 |
| PP 简化模型过于粗糙 | Phase 2 先只支持 pp=1，PP 扩展作为后续增强 |

---

## 附录 A：参考源码文件路径

### AICB Workload 生成

| 文件 | 说明 |
|------|------|
| `aicb/workload_generator/SimAI_training_workload_generator.py` | AICB 训练 workload 生成器，包含 `SIMAI_workload` 类和 `Work_Item` dataclass |
| `aicb/workload_generator/SimAI_inference_workload_generator.py` | AICB 推理 workload 生成器 |

### AICB 示例 Workload 文件

| 文件 | 说明 |
|------|------|
| `example/workload_analytical.txt` | 大规模 MoE 模型训练 workload（1789 items, 9216 GPUs, tp=2/ep=16/pp=12） |
| `example/microAllReduce.txt` | 微基准测试 workload（AllReduce, 8 GPUs） |
| `aicb/workload/simAI/micro_test/all_reduce.txt` | AllReduce 微基准 |
| `aicb/workload/simAI/micro_test/all_gather.txt` | AllGather 微基准 |
| `aicb/workload/simAI/micro_test/all_to_all.txt` | AlltoAll 微基准 |
| `aicb/workload/simAI/micro_test/muti_all_reduce.txt` | 多 AllReduce 微基准 |

### astra-sim Workload 解析（C++ 参考实现）

| 文件 | 说明 |
|------|------|
| `astra-sim-alibabacloud/astra-sim/workload/Workload.cc` | workload 解析主逻辑，`initialize_workload()` 函数（约 1134-1549 行）解析 header 和 items |
| `astra-sim-alibabacloud/astra-sim/workload/Workload.hh` | Workload 类定义，包含 `Layer` vector 和并行策略字段 |
| `astra-sim-alibabacloud/astra-sim/workload/Layer.cc` | Layer 执行逻辑，包含通信类型到 ComType 的映射和 PP bubble 计算 |
| `astra-sim-alibabacloud/astra-sim/workload/Layer.hh` | Layer 类定义，包含 forward/backward/DP 各阶段的 compute_time、comm_type、comm_size |

### 拓扑与配置

| 文件 | 说明 |
|------|------|
| `astra-sim-alibabacloud/inputs/topo/gen_Topo_Template.py` | 拓扑生成脚本（支持 Spectrum-X、AlibabaHPN、DCN+ 等模板） |
| `astra-sim-alibabacloud/inputs/config/SimAI.conf` | 模拟配置文件 |
| `example/busbw.yaml` | Busbw 带宽配置示例 |

### MockNcclGroup（Collective→P2P 参考实现）

| 文件 | 说明 |
|------|------|
| `astra-sim-alibabacloud/astra-sim/system/MockNcclGroup.cc` | Ring AllReduce/AllGather/ReduceScatter/AlltoAll 的 flow 生成逻辑 |
| `astra-sim-alibabacloud/astra-sim/system/MockNcclChannel.h` | `SingleFlow` 结构定义（flow_id, src, dst, flow_size, prev, chunk_id 等） |

> 注：以上路径相对于 SimAI 项目根目录 `/Users/hovering-clouds/Desktop/workspace/SimAI/`。

---

*创建日期：2026-04-09*
*更新日期：2026-04-09（融入 PP 简化建模 + Rank 分组设计决策）*
*前置阶段：Phase 1（已完成，57 tests passed）*

# Phase 2 - Task 1 开发记录：AICB Workload 解析器

## 1. 目标与范围

Task 1 的目标是实现 AICB 格式训练 workload 文件的解析器，将 AICB 的文本格式转换为结构化的 Python 数据对象。这是 Phase 2 整条管线的起点：后续的 RankGrouper（Task 2）和 WorkloadBuilder（Task 3）都依赖解析器输出的结构化数据。

**包含**：
- `HYBRID_TRANSFORMER_FWD_IN_BCKWD` 格式解析
- `MICRO` 格式解析
- 通信类型后缀解析（`ALLGATHER_DP_EP` → `("ALLGATHER", "dp_ep")`）

**不包含**：Rank 分组推导（Task 2）、Workload 构建（Task 3）、多任务合并（Task 4）。

---

## 2. 交付物

### 2.1 新增文件

```
simai-flow-scheduler/
├── src/workload_generator/
│   ├── __init__.py              # 更新：导出 AicbParser, AicbHeader, AicbWorkItem
│   └── aicb_parser.py           # 新增：解析器实现
└── tests/
    └── test_aicb_parser.py      # 新增：30 个测试
```

### 2.2 数据结构

```python
@dataclass
class AicbHeader:
    tp: int                 # Tensor Parallelism size
    ep: int                 # Expert Parallelism size
    pp: int                 # Pipeline Parallelism size
    vpp: int                # Virtual Pipeline Parallelism (= 模型总层数)
    ga: int                 # Gradient Accumulation steps
    all_gpus: int           # 总 GPU 数量 (world size)
    pp_comm_size: int       # PP stage 间 activation 大小 (0 如果不存在)

@dataclass
class AicbWorkItem:
    name: str
    forward_compute_time: int
    forward_comm: str       # "NONE", "ALLREDUCE", "ALLGATHER_DP_EP", etc.
    forward_comm_size: int
    backward_compute_time: int
    backward_comm: str
    backward_comm_size: int
    dp_compute_time: int
    dp_comm: str
    dp_comm_size: int
```

### 2.3 API

```python
class AicbParser:
    def parse(self, file_path: str) -> tuple[AicbHeader, list[AicbWorkItem]]
    def parse_micro(self, file_path: str) -> list[AicbWorkItem]
    def parse_comm_type(comm_str: str) -> tuple[str, str]  # static
```

---

## 3. AICB 文件格式详解

### 3.1 HYBRID_TRANSFORMER_FWD_IN_BCKWD 格式

```
第 1 行: Header（空格分隔的 key:value 对）
第 2 行: Item 数量
第 3~N 行: 每个 item 12 个空白分隔的字段
```

**Header 示例**：
```
HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 2 ep: 16 pp: 12 vpp: 8 ga: 24 all_gpus: 9216 checkpoints: 0 checkpoint_initiates: 0 pp_comm 50331648
```

注意 `pp_comm` 没有冒号，而其他字段（如 `ep:`）都有冒号。解析器需要同时处理两种情况。

**Item 字段映射**（以 `grad_gather` 为例）：

| 位置 | 字段 | 示例值 | 含义 |
|------|------|--------|------|
| 0 | name | `grad_gather` | 操作名 |
| 1 | placeholder | `-1` | 占位符（忽略） |
| 2 | forward_compute_time | `1` | 前向计算时间 |
| 3 | forward_comm | `NONE` | 前向通信类型 |
| 4 | forward_comm_size | `0` | 前向通信数据量 |
| 5 | backward_compute_time | `1` | 反向计算时间 |
| 6 | backward_comm | `NONE` | 反向通信类型 |
| 7 | backward_comm_size | `0` | 反向通信数据量 |
| 8 | dp_compute_time | `1` | DP 计算时间 |
| 9 | dp_comm | `ALLGATHER` | DP 通信类型 |
| 10 | dp_comm_size | `2807758848` | DP 通信数据量 |
| 11 | process_time | `100` | 处理耗时（忽略） |

### 3.2 MICRO 格式

```
第 1 行: "MICRO"
第 2 行: Item 数量
第 3~N 行: 与 HYBRID 格式相同的 12 字段结构
```

MICRO 格式没有并行策略 header，因此 `parse_micro()` 只返回 `list[AicbWorkItem]`。

### 3.3 与 C++ 参考实现的对应

C++ 中的字段映射（`Workload.cc` 的 `initialize_workload()` 函数）：

| C++ 变量 | 本项目字段 | Phase 语义 |
|----------|-----------|-----------|
| `fp_compute_time` | `forward_compute_time` | 前向 (Forward Pass) |
| `ig_compute_time` | `backward_compute_time` | 反向输入梯度 (Input Gradient) |
| `wg_compute_time` | `dp_compute_time` | 反向权重梯度 / DP |
| `wg_update_time` | (忽略) | 权重更新时间 |

在 `HYBRID_TRANSFORMER_FWD_IN_BCKWD` 模式下，C++ 的 ig（input gradient）对应本项目的"backward"，wg（weight gradient）对应本项目的"dp"。命名不同但语义一致。

---

## 4. 关键实现细节

### 4.1 Header 解析算法

Header 行通过 `split()` 分词后逐 token 遍历，使用 key-value 匹配模式：

```python
i = 1  # 跳过第 0 个 token（并行策略名称）
while i < len(tokens):
    if tok == "model_parallel_NPU_group:":
        tp = int(tokens[i + 1]); i += 2
    elif tok in ("pp_comm", "pp_comm:"):  # pp_comm 可能有也可能没有冒号
        pp_comm_size = int(tokens[i + 1]); i += 2
    elif tok in ("checkpoints:", "checkpoint_initiates:"):
        count = int(tokens[i + 1])
        i += 2 + count  # 跳过 count 值 + layer ID 列表
    else:
        i += 1
```

关键点：
- `pp_comm` 可能没有冒号（对比 C++ 中 `tokens[i] == "pp_comm" || tokens[i] == "pp_comm:"` 的处理）
- `checkpoints:` 后跟一个计数 + 若干 layer ID，需要跳过

### 4.2 Item 行解析

使用 `split()` 而非 `split('\t')`，与 C++ 的 `>>` 操作符行为一致（任意空白作为分隔符）。实际 AICB 文件中存在混合空白的情况（如 `example/microAllReduce.txt` 使用多空格分隔）。

### 4.3 通信类型后缀解析

```python
@staticmethod
def parse_comm_type(comm_str: str) -> tuple[str, str]:
```

后缀优先级（最长匹配优先）：
1. `_DP_EP` → context = "dp_ep"
2. `_DP` → context = "dp"
3. `_EP` → context = "ep"
4. 无后缀 → 默认 "tp"（包括 ALLTOALL 无后缀的情况）

**关键规则**：所有通信类型无后缀时都默认对应 **TP group**。这与 astra-sim 源码（Workload.cc 第 1329-1369 行）的行为完全一致。

---

## 5. 开发过程（TDD）

### 5.1 方法

严格遵循 TDD 流程：RED → GREEN → REFACTOR。

### 5.2 RED 阶段

先编写 `tests/test_aicb_parser.py`（30 个测试），覆盖：
- `TestParseCommType`（11 个）：各种后缀变体 + ALLTOALL 特殊规则
- `TestParseHeader`（3 个）：完整 header、无 pp_comm header、字段类型检查
- `TestParseItems`（7 个）：各种 comm 类型、空白分隔、字段不足跳过
- `TestParseMicro`（3 个）：MICRO 格式解析
- `TestRealFiles`（6 个）：用真实 AICB 文件做集成测试

运行测试，确认全部因 `ModuleNotFoundError` 失败：

```
E   ModuleNotFoundError: No module named 'src.workload_generator.aicb_parser'
```

### 5.3 GREEN 阶段

编写 `aicb_parser.py`，实现最小功能使所有测试通过。

### 5.4 发现的问题及修复

#### 问题 1：集成测试中 SIMAI_ROOT 路径错误

**现象**：6 个 `TestRealFiles` 测试被 skip，因为路径不存在。

**原因**：`Path(__file__).resolve().parents[3]` 多上了一级。

```
# 实际路径层级：
# parents[0] = simai-flow-scheduler/tests/
# parents[1] = simai-flow-scheduler/
# parents[2] = SimAI/           ← 正确的 root
# parents[3] = workspace/       ← 错误
```

**修复**：改为 `parents[2]`。修复后 6 个集成测试全部从 SKIP 变为 PASS。

#### 问题 2：workload_analytical.txt 的 item 索引错误

**现象**：`test_parse_workload_analytical_first_items` 失败：

```
E   AssertionError: assert 'embedding_grads' == 'embedding_layer'
```

**原因**：编写测试时基于直觉假设 item 顺序，没有对照实际文件内容。

实际文件前 8 个 item 顺序为：
```
0: grad_gather
1: grad_param_comm
2: grad_param_compute
3: embedding_grads        ← 我误以为是 embedding_layer
4: moe_grad_norm1
5: moe_grad_norm2
6: embedding_layer        ← 实际在第 7 个位置
7: attention_column
```

**修复**：根据实际文件内容修正测试中的索引和期望值。这个 bug 的教训是：**集成测试的期望值必须来自实际数据，不能凭记忆或直觉假设**。

#### 问题 3：ALLTOALL 无后缀时的默认分组错误

**现象**：初始实现中 `parse_comm_type("ALLTOALL")` 返回 `("ALLTOALL", "ep")`。

**原因**：基于直觉假设 ALLTOALL 通常与 EP 关联，因此无后缀时默认返回 `"ep"`。但查阅 astra-sim 源码 `Workload.cc` 第 1329-1337 行后发现：

```cpp
if (wg_comm_type_s.substr(0,8) == "ALLTOALL") {
  if(wg_comm_type_s == "ALLTOALL"){
    wg_group_type = MockNccl::GroupType::TP;      // 无后缀 → TP
  } else if(wg_comm_type_s == "ALLTOALL_EP"){
    wg_group_type = MockNccl::GroupType::EP;      // _EP → EP
  } else if(wg_comm_type_s == "ALLTOALL_DP_EP"){
    wg_group_type = MockNccl::GroupType::DP_EP;   // _DP_EP → DP×EP
  }
}
```

**正确规则**：所有通信类型（包括 ALLTOALL）无后缀时都默认对应 **TP group**，有后缀时才根据后缀判断。

**修复**：
1. 删除 `parse_comm_type()` 中针对 `ALLTOALL` 的特殊处理逻辑
2. 修改测试 `test_alltoall_no_suffix_is_tp` 的期望值为 `"tp"`
3. 更新 docstring 反映正确规则

**影响范围**：此修正确保了与 astra-sim 源码的行为完全一致，避免后续 WorkloadBuilder 在处理 ALLTOALL 通信时选择错误的 rank group。

---

## 6. 测试结果

### 6.1 最终结果

```
tests/test_aicb_parser.py: 30 passed
tests/test_collective_expander.py: 46 passed
tests/test_workload_format.py: 11 passed
──────────────────────────────────
Total: 87 passed, 0 failed
```

### 6.2 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestParseCommType` | 11 | NONE、空串、无后缀 TP、ALLTOALL→TP 规则、_DP、_EP、_DP_EP |
| `TestParseHeader` | 3 | 含 pp_comm、不含 pp_comm、字段类型 |
| `TestParseItems` | 7 | item 计数、字段映射、空白分隔、字段不足跳过 |
| `TestParseMicro` | 3 | MICRO 返回类型、字段解析、末尾 item |
| `TestRealFiles` | 6 | microAllReduce.txt、workload_analytical.txt、MICRO benchmark 文件 |

### 6.3 集成测试验证的真实文件

| 文件 | 验证内容 |
|------|----------|
| `example/microAllReduce.txt` | 2 个 embedding_layer，tp=8, ep=1 |
| `example/workload_analytical.txt` | 1789 个 item，tp=2, ep=16, pp=12, ga=24 |
| `aicb/workload/simAI/micro_test/all_reduce.txt` | MICRO 格式，19 个 ALLREDUCE |
| `aicb/workload/simAI/micro_test/all_gather.txt` | MICRO 格式，19 个 ALLGATHER |
| `aicb/workload/simAI/micro_test/all_to_all.txt` | MICRO 格式，19 个 ALLTOALL |

---

## 7. 与设计文档的偏差

无偏差。实现完全按照 [phase2-plan.md](phase2-plan.md) 第 5.5 节和 Task 1 的设计。

补充说明：
- 设计中 `AicbWorkItem` 不包含 `placeholder`（字段 1）和 `process_time`（字段 11），因为这两个字段在后续构建流程中不需要。解析时跳过它们。
- `parse_comm_type()` 的规则来自设计文档 5.5 节的通信类型判断表，与 astra-sim 源码（Workload.cc 第 1329-1369 行）完全一致。所有无后缀的通信类型都默认对应 **TP group**。

---

## 8. 后续依赖

Task 1 的输出将被以下模块使用：

1. **Task 3（WorkloadBuilder）**：使用 `AicbHeader` 的并行策略信息 + `AicbWorkItem` 的三层（forward/backward/dp）计算和通信数据，结合 RankGrouper 和展开器，构建完整的 P2PWorkload
2. **Task 5（端到端示例）**：直接调用 `AicbParser.parse()` 读取示例 workload 文件

---

*开发时间：2026-04-10*
*测试状态：30 passed（AICB parser）+ 46 passed（collective expander）+ 11 passed（workload format）= 87 total*
*开发方法：TDD（RED → GREEN → REFACTOR）*
*重要修正：ALLTOALL 无后缀时返回 "tp" 而非 "ep"，与 astra-sim 源码一致*

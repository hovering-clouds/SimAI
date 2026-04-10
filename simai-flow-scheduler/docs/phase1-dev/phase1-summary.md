# Phase 1 开发总结

## 1. 目标与范围

Phase 1 的目标是建立 simai-flow-scheduler 的项目框架，实现 P2P Workload 格式定义和 Collective→P2P 展开器的核心逻辑。这是整个系统的 Layer 2（Workload 生成层）和 Layer 3（P2P Workload IR）的基础。

**不包含**：多任务支持（Phase 2）、调度策略（Phase 3）、执行器（Phase 4）。

---

## 2. 交付物

### 2.1 项目结构

```
simai-flow-scheduler/
├── pyproject.toml                          # Python 3.13, jsonschema + pyyaml
├── src/
│   ├── __init__.py
│   ├── workload_format/
│   │   ├── __init__.py
│   │   ├── schema.py                       # 数据模型 + JSON Schema
│   │   ├── validator.py                    # 结构 + 语义验证
│   │   └── writer.py                       # JSON 序列化/反序列化
│   └── workload_generator/
│       ├── __init__.py
│       └── collective_expander.py          # 基类 + 4 个展开器实现
├── tests/
│   ├── __init__.py
│   ├── test_workload_format.py             # 11 个测试
│   └── test_collective_expander.py         # 46 个测试
└── docs/
    ├── specs/flow-scheduler-design.md      # 总体设计文档
    └── phase1-dev/
        ├── RING_ALLREDUCE_FIX_REPORT.md    # Ring AllReduce 调试记录
        └── phase1-summary.md              # 本文件
```

### 2.2 核心模块

| 模块 | 职责 |
|------|------|
| `schema.py` | 定义 `P2PWorkload`、`Task`、`Job`、`Meta`、`Network` 等 dataclass，以及 `P2P_WORKLOAD_JSON_SCHEMA`（Draft 7） |
| `validator.py` | `WorkloadValidator`：JSON Schema 校验 + 语义校验（DAG 无环、task_id 唯一、字段互斥） |
| `writer.py` | `WorkloadWriter`（序列化到 JSON 文件/字符串）+ `WorkloadReader`（反序列化） |
| `collective_expander.py` | 基类 `CollectiveExpander`(ABC) + 4 个具体实现类 |

### 2.3 展开器实现

| 类 | 支持的算法 | flow 数公式 |
|----|-----------|------------|
| `AllReduceExpander` | ring | n * 2*(n-1) |
| `AllGatherExpander` | ring | n * (n-1) |
| `ReduceScatterExpander` | ring | n * (n-1) |
| `AlltoAllExpander` | 全连接 | n * (n-1) |

### 2.4 测试覆盖

**57 个测试全部通过**（`uv run pytest`）：

- `test_workload_format.py`：11 个测试（Task 创建/验证、P2PWorkload 验证、读写一致性、文件校验）
- `test_collective_expander.py`：46 个测试
  - AllReduce：12 个（含与 MockNcclGroup.cc 的逐条对比）
  - AllGather：12 个（含与 C++ 逐条对比）
  - ReduceScatter：12 个
  - AlltoAll：12 个

---

## 3. 关键设计决策

### 3.1 Python-only 实现

选择纯 Python 实现，而非 C++ bridge 方案。原因：MockNcclGroup.cc 的构造函数假设所有 GPU 参与同一任务（`_TP_size * _DP_size * _PP_size == _ngpus`），GroupIndex 是全局的 `std::map<std::pair<int, GroupType>, int>`，无法支持多任务场景。重新实现更灵活。

### 3.2 Expander 类结构

每个 expander 类对应一种集合通信操作，内部按 `algo` 参数分派到不同算法实现：

```python
class AllReduceExpander(CollectiveExpander):
    def expand_allreduce(self, ranks, data_size, algo="ring", ...):
        if algo == "ring":
            return self._expand_ring(...)
        raise ValueError(f"unsupported algo '{algo}'")
```

这样设计的好处：
- 新增算法（如 tree、nvls）只需在对应 expander 内添加 `_expand_tree()` 方法
- 基类的 `expand_*` 方法签名统一，便于上层代码多态调用

### 3.3 P2P Workload 格式

采用 JSON 格式作为 IR（中间表示），特点：
- 与调度策略无关
- 与模拟后端无关
- 支持 `compute` 和 `flow` 两种 task 类型
- 通过 `deps` 字段编码 DAG 依赖关系
- `comm_type` 语义标签（如 `tp_allreduce_ring`）供调度策略参考

### 3.4 FlowTask 中间结构

展开过程使用 `FlowTask` dataclass（轻量级），最终通过 `.to_task()` 转换为完整的 `Task` 对象。这简化了展开逻辑，避免在构建过程中处理所有 Task 字段。

---

## 4. Ring 展开算法要点

Ring AllReduce 是最复杂的展开算法，分为三个阶段：

1. **Phase 1**（初始 chunk，1 步）：n 个 flow，无依赖
2. **Phase 2**（RS 迭代，n-2 步）：每步 n 个 flow，对角线依赖
3. **Phase 3**（AG 迭代，n-1 步）：每步 n 个 flow，对角线依赖

关键点：
- Phase 1 只服务 RS，AG 有自己独立的 n-1 步
- 依赖关系使用 `task_list[prev_rank]` 查表，不是简单的 `task_id - n` 线性偏移
- `num_chunks = 2*(n-1)`（总 step 数），`chunk_id` 顺序递增

AllGather 和 ReduceScatter 的 ring 结构相同，都是 n*(n-1) 个 flow，只是语义不同。AlltoAll 最简单：n*(n-1) 个独立 flow，无依赖。

详细调试过程见 [RING_ALLREDUCE_FIX_REPORT.md](RING_ALLREDUCE_FIX_REPORT.md)。

---

## 5. 与 MockNcclGroup.cc 的对应关系

| C++ 函数 | Python 对应 |
|----------|------------|
| `genAllReduceFlowModels` | `AllReduceExpander.expand_allreduce(algo="ring")` |
| `genAllGatherFlowModels` | `AllGatherExpander.expand_allgather(algo="ring")` |
| `genReduceScatterFlowModels` | `ReduceScatterExpander.expand_reducescatter(algo="ring")` |
| `genAlltoAllFlowModels` | `AlltoAllExpander.expand_alltoall()` |

验证策略：4 ranks 场景下，Python 展开的每个 flow 的 `src/dst/chunk_id/num_chunks/deps` 与 C++ 逐条对比，完全一致。

---

## 6. 已知限制与后续方向

1. **仅 Ring 算法**：AllReduce/AllGather/ReduceScatter 目前只实现了 ring，tree 和 nvls 待后续补充
2. **单任务**：展开器每次只处理一个集合通信操作，多任务合并是 Phase 2 的工作
3. **无 AICB 集成**：目前需要手动构造参数调用展开器，Phase 2 将实现 AICB 适配器
4. **无计算任务生成**：当前只生成 flow task，compute task 需要上层（AICB 适配器）提供

---

*Phase 1 完成时间：2026-04-09*
*测试状态：57 passed*

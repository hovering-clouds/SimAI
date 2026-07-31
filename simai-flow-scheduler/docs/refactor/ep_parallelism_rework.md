# EP 并行维度模型修正

## 背景：现象与问题

### 问题陈述

前期开发中将 Expert Parallelism (EP) 建模为与 DP、TP、PP 完全独立的第四维：

```
total_gpus = tp × dp × ep × pp
排名布局：[PP][DP][EP][TP]
```

但后续调研发现，实际系统中 **EP 并不增加总的 GPU 数量**。EP 作用的对象是 MoE 层的专家参数，而 TP 作用的是 attention 层参数，二者在一个 PP stage 内"正交"共享同一组 GPU。EP 本质上是将 DP 维度做了进一步划分：

```
实际模型：total_gpus = tp × dp × pp，其中 dp = ep × dp_ep
         dp_ep = 每个专家组内的数据并行副本数
```

### 更严重的问题：通信组语义颠倒

对 `astra-sim-alibabacloud` 的代码调研发现，当前实现的 `_DP` 和 `_DP_EP` 通信组语义是**颠倒的**：

| AICB 后缀 | astra-sim 意图 | 当前 simai 实现 | 问题 |
|---|---|---|---|
| `ALLREDUCE_DP` | 跨所有模型副本的 all-reduce（`dp_total` 个 rank） | `old_dp` 个 rank（只含 ep 内） | 漏掉其他 ep 组，通信量偏小 |
| `ALLREDUCE_DP_EP` | 跨同一专家内副本的 all-reduce（`dp_ep = dp_total/ep` 个 rank） | `old_dp × ep` 个 rank（跨所有 ep 组） | 通信量偏大 |

### 举例说明

```python
# 配置：tp=2, ep=2, 总 GPU=8, pp=1
# 当前模型：dp=2（old_dp，即 ep 内的数据并行）
# 正确模型：dp_total=4（总模型副本数 = old_dp × ep = 2×2）

# 当前模型给出的 DP 组（"dp" context）：
#   [0,4], [1,5], [2,6], [3,7]       ← 每组只有 2 个 rank（ep 内）
# 正确的 DP 组应该为：
#   [0,2,4,6], [1,3,5,7]              ← 每组 4 个 rank（全量）

# 当前模型给出的 DP_EP 组（"dp_ep" context）：
#   [0,2,4,6], [1,3,5,7]              ← 每组 4 个 rank（跨所有 ep）
# 正确的 DP_EP 组应该为：
#   [0,4], [1,5], [2,6], [3,7]        ← 每组 2 个 rank（ep 内）
```

## 目标

将 EP 的处理方式从**独立乘法因子**重构为 **DP 的子划分**，使：

1. 排名布局从 `[PP][DP][EP][TP]`（4D）变为 `[PP][DP][TP]`（3D）
2. `dp` 的语义变为"总模型副本数（含 EP）"
3. EP 组和 DP_EP 组从连续/跨步 DP 索引派生
4. 通信组语义与 `astra-sim-alibabacloud` 对齐

## 新模型设计

### ParallelismConfig

```python
@dataclass
class ParallelismConfig:
    tp: int = 1
    dp: int = 1   # 总模型副本数（含 EP 分组在内）
    pp: int = 1
    ep: int = 1   # 专家分组数，dp % ep == 0

    # 派生属性
    # dp_ep = dp // ep   每个专家组内的数据并行副本数
```

**约束**：`tp × dp × pp == total_gpus`，不再包含 ep 的乘法。

### 排名布局

```
新布局：  [PP][DP][TP]    (3D)
索引公式： global_idx = pp_idx × (dp × tp) + dp_idx × tp + tp_idx
```

`dp` 在此处为"总模型副本数（含 EP 分组）"。

### 组派生规则（默认 Megatron 风格）

| 组类型 | 范围 | 构造规则 |
|---|---|---|
| TP | 一个 DP 副本内的 tp_size 个 rank | `get_tp_group(pp_idx, dp_idx)` |
| DP | 所有模型副本的同一 tp_idx | `get_dp_group(pp_idx, tp_idx)` |
| EP | 连续 ep 个 DP 副本的同一 tp_idx | `get_ep_group(pp_idx, dp_ep_idx, tp_idx)` |
| DP_EP | 每隔 ep 步长取一个 DP 副本的同一 tp_idx | `get_dp_ep_group(pp_idx, ep_idx, tp_idx)` |
| PP | 跨 PP stage 的对应 rank | `get_pp_rank(pp_idx, dp_idx, tp_idx)` |

### 3D 布局的灵活性：支持不同 EP 划分策略

3D 布局的核心优势是 **EP 组的派生逻辑与排名布局解耦**。`get_ep_group` 只是一个方法，具体实现可以根据使用场景替换：

| 策略 | 典型场景 | EP 组构造方式 |
|---|---|---|
| **Megatron 风格**（astra-sim，上图默认规则） | MoE 训练，专家分布在 `ep` 个连续模型副本上 | 连续 `ep` 个 DP 索引的同一 `tp_idx` |
| **vLLM 风格**（推理） | MoE 推理，专家分布在**所有** GPU 上（跨 DP + TP） | 整个 PP stage 内所有 rank 构成一个 EP 组 |
| **NCCL 风格**（自定义） | 自定义通信拓扑 | 按任意分组规则实现 |

**举例：vLLM 推理的 EP 划分**

vLLM 中 `ep_size = dp × tp`，MoE 专家的 all-to-all 通信在 PP stage 内的所有 GPU 间进行：

```python
class RankGrouper:
    # vLLM-style: EP spans ALL GPUs in the PP stage
    def get_ep_group(self, pp_idx):
        """Retrieve all ranks for a given PP stage."""
        return [
            self.nodes[pp_idx * (self.dp * self.tp) + dp_idx * self.tp + tp_idx]
            for dp_idx in range(self.dp)
            for tp_idx in range(self.tp)
        ]
```

这种场景下，EP 组大小 = `dp × tp`，不需要 `dp_ep_idx` 和 `tp_idx` 参数。

**为什么 4D 布局 `[PP][DP_EP][EP][TP]` 做不到？**

4D 布局中 EP 在排名网格中的位置是固定的（介于 DP_EP 和 TP 之间）。`get_ep_group` 只能迭代 EP 维度本身——它无法让 EP 组跨 DP_EP 和 TP 维度。如果要支持 vLLM 风格，需要重新排列维度顺序，而不同场景又需要不同的排列方式。3D 布局通过"派生"而非"内置"的方式避免了这个问题。

### 推理场景的 EP 配置

推理场景下，EP 的划分方式取决于具体引擎的实现：

- **训练类推理**（Megatron 风格）：沿用训练配置，EP 作为 DP 的子划分
- **vLLM 风格**：EP 跨所有 non-PP GPU，`get_ep_group` 返回整个 stage 的 rank 列表

当前 `InferenceTraceExpander` 采用 Megatron 风格（与训练一致），后续可根据需要扩展支持 vLLM 风格。`ParallelismConfig` 本身不需要为此增加字段——EP 的划分策略由 `RankGrouper` 的 `get_ep_group` 实现决定。

## 需要修改的文件

### 核心数据结构

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/workload_format/schema.py` | 小 | 给 `ParallelismConfig` 加 `dp_ep` property 和 `__post_init__` 校验 |

### 排名分组

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/workload_generator/rank_grouper.py` | **大** | 从 4D `[PP][DP][EP][TP]` 改为 3D `[PP][DP][TP]`，`get_dp_group` 返回所有副本，`get_ep_group` / `get_dp_ep_group` 按连续/跨步 DP 索引派生 |

### Workload 生成器

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/workload_generator/workload_builder.py` | 中 | `_iter_subgroups` 适配新 DP/DP_EP 语义；PP 流生成去掉 `ep_idx` 维度循环 |
| `src/workload_generator/inference_trace_expander.py` | 中 | 当前采用 Megatron 风格：`self._dp = ep`（`dp_ep = 1`）；`_world_size() = tp * dp * pp`；`_stage_ranks` 和 `_grouper` 接收新 ParallelismConfig。后续可扩展 vLLM 风格 |
| `src/workload_generator/aicb_parser.py` | 无 | `AicbHeader` 不变，suffix 解析不变；调用方构建 `ParallelismConfig` 时用新语义即可 |

### Job 切片与放置

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/workload_generator/job_slicer.py` | 中 | `TrainingJobSlicer`：`world_size = tp × pp`（不再 × ep）；`InferenceJobSlicer`：`ws = tp × pp` |
| `scripts/utils/job_placement.py` | 中 | `resolve_parallelism`：`total_gpus = tp × dp × pp`；`_replica_size` 去掉 ep；`_assignment_from_dp_servers` 改为 `[PP][DP][TP]` 布局 |

### 调度器

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/static_analysis/passes/hermod_metadata.py` | 小 | 重构完成后可设置 `reject_ep=False` 以启用 EP 验证 |
| `src/static_analysis/passes/mfs_context.py` | 无 | `EP_ALLTOALL` 已在 `_COLLECTIVE_TYPES` 中，无需修改 |
| `src/executor/dynamic/job_expander.py` | 小 | `_expand_inference` 中的 `ep=job.parallelism.ep` 不需要改，由 `InferenceTraceExpander` 内部推导 `dp=ep` |

### 序列化

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `src/workload_format/compact_workload.py` | 无 | `to_dict`/`from_dict` 中的 `ep` 字段保留，JSON 格式兼容 |

### 测试

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `tests/test_rank_grouper.py` | **大** | `TestWithEP` 类中 EP 组/DP_EP 组的预期结果需按新公式重算 |
| 其他测试文件 | 中 | 涉及 `ParallelismConfig(tp=..., dp=..., ep=...)` 的地方按 `new_dp = old_dp × old_ep` 更新 |

### 脚本层

| 文件 | 改动量 | 说明 |
|------|--------|------|
| `scripts/run_cassini_dynamic_e2e.py` | 小 | 间接通过 `resolve_parallelism` 和 `job_slicer` 受影响 |
| `scripts/utils/iteration_expansion.py` | 小 | 若有 `ws = tp * ep * pp` 类公式则更新 |

## 迁移指南

### 配置迁移规则

```
new_dp = old_dp × old_ep
```

**举例**：

| 旧配置 | 旧 total_gpus | 新配置 | 新 total_gpus |
|---|---|---|---|
| `tp=2, dp=2, ep=1` | 4 | `tp=2, dp=2, ep=1` | 4 |
| `tp=2, dp=2, ep=2` | 8 | `tp=2, dp=4, ep=2` | 8 |
| `tp=4, dp=8, ep=4` | 128 | `tp=4, dp=32, ep=4` | 128 |
| `tp=1, dp=1, ep=1` | 1 | `tp=1, dp=1, ep=1` | 1 |

### AICB Header 解析

AICB 文件中的 `all_gpus:` 字段保持不变。构建 `ParallelismConfig` 时：

```python
# 旧代码
dp = all_gpus // (tp * pp * ep)

# 新代码
dp = all_gpus // (tp * pp)   # dp 现在包含 ep 的因子
```

### 通信组语义对照

| 上下文 | 旧语义 | 新语义 |
|---|---|---|
| `context == "dp"` | `old_dp`（ep 内 DP） | `dp_total`（全量 DP） |
| `context == "dp_ep"` | `old_dp × ep`（跨 ep） | `dp_ep = dp // ep`（ep 内 DP） |
| `context == "ep"` | `ep`（独立维度） | `ep`（派生自连续 DP 索引） |

## 实施顺序

建议按以下次序逐步实施：

```
Phase 1 ─ schema.py → rank_grouper.py（核心基础设施）         ✅ 已完成
Phase 2 ─ workload_builder.py + inference_trace_expander.py（生成器适配） ✅ 已完成
Phase 3 ─ job_slicer.py + job_placement.py（切片与放置适配） ✅ 已完成
Phase 4 ─ tests/（全面更新测试）                              ✅ 已完成（803 个测试通过）
Phase 5 ─ hermod_metadata.py（解除 EP 封锁，逐步验证）        ⏭️ 已跳过
Phase 6 ─ 运行 e2e 脚本验证                                   ⬜ 待办
```

### Phase 5 跳过说明

Hermod 的 coflow 分类逻辑（`hermod_priority.py::classify_coflow_type`）已有
`EP_ALLTOALL → HermodCoflowType.EP` 分支，优先级计算（`_ctype_rank`）也给了 EP
最高优先级（rank 0）。但 EP 任务被 `hermod_metadata.py` 的 `reject_ep=True` 在
入口处拦截，永远到不了 `HermodPriorityAnalysis`。此外 `HermodEpMode` 枚举定义了
`REJECT`/`ENABLE` 但只是存储传递，从未被分支逻辑读取（死代码）。

后续若要启用 Hermod + EP：把 `reject_ep` 改为 `False`，让 `ep_mode` 真正生效或
废弃，并验证 EP 任务在 `apply()` 中走通用 GA 分支而非落入 `else: raise`。

每个 Phase 完成后建议运行现有测试确认回归覆盖。

## 附录：astra-sim-alibabacloud 参考实现

### 关键文件

| 文件 | 对应概念 |
|---|---|
| `astra-sim/system/MockNcclGroup.cc` | EP/DP_EP 组构建逻辑 |
| `astra-sim/system/Sys.cc` | `DP_size = total / (TP×PP)`，`EP_size` 从配置读取，`DP_EP_size = DP_size / EP_size` |
| `astra-sim/workload/Workload.cc` | AICB 文件解析，`_EP`/`_DP_EP` 后缀处理 |
| `astra-sim/workload/Layer.cc` | 按 group_type 分支的 busbw 计算、EP 独立的 overlap 参数 |

### 核心约束关系

```cpp
_TP_size * _DP_size * _PP_size = _ngpus    // EP 不计入 total
_EP_size * _DP_EP_size = _DP_size          // EP 分割 DP
```

### EP 组的物理含义

EP 组由**连续 `ep` 个 TP 组**中取相同 `tp_idx` 的 rank 构成。这反映了 MoE 的标准通信模式：每个 TP 组持有一个完整的模型副本，EP 则"正交地"跨多个 TP 副本分布专家参数。一个 EP all-to-all 操作将每个 token 路由到其对应的专家所在的 GPU。

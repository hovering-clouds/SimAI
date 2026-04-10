# Phase 2 - Task 2 开发记录：Rank 分组器

## 1. 目标与范围

Task 2 的目标是实现 Rank 分组器，根据 `assigned_nodes` 列表顺序和并行策略配置推导 TP/DP/EP/PP 分组。这是 WorkloadBuilder（Task 3）的关键依赖模块。

**包含**：
- `RankGrouper` 类实现，支持四维并行网格 `[PP][DP][EP][TP]`
- 5 种分组方法：`get_tp_group()`, `get_dp_group()`, `get_ep_group()`, `get_dp_ep_group()`, `get_pp_group()`
- 通过 rank 列表顺序控制分组的机制

**不包含**：Workload 构建（Task 3）、多任务合并（Task 4）。

---

## 2. 交付物

### 2.1 新增文件

```
simai-flow-scheduler/
├── src/workload_generator/
│   ├── __init__.py              # 更新：导出 RankGrouper
│   └── rank_grouper.py          # 新增：RankGrouper 实现
└── tests/
    └── test_rank_grouper.py     # 新增：17 个测试
```

### 2.2 核心 API

```python
class RankGrouper:
    """根据 assigned_nodes 列表顺序 + parallelism 配置推导并行分组。

    列表排列约定: [PP=pp][DP=dp][EP=ep][TP=tp]
    索引公式: global_idx = pp_idx*(dp*ep*tp) + dp_idx*(ep*tp) + ep_idx*tp + tp_idx
    """

    def __init__(self, assigned_nodes: list[int], parallelism: ParallelismConfig):
        ...

    def get_tp_group(self, pp_idx: int, dp_idx: int, ep_idx: int) -> list[int]:
        """获取指定 (PP, DP, EP) 下的 TP group ranks。"""

    def get_dp_group(self, pp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (PP, EP, TP) 下的 DP group ranks。"""

    def get_ep_group(self, pp_idx: int, dp_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (PP, DP, TP) 下的 EP group ranks。"""

    def get_dp_ep_group(self, pp_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (PP, TP) 下的 DP×EP 联合 group ranks。"""

    def get_pp_group(self, dp_idx: int, ep_idx: int, tp_idx: int) -> list[int]:
        """获取指定 (DP, EP, TP) 下的 PP group ranks。"""
```

---

## 3. 设计原理

### 3.1 Rank 列表顺序约定

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

### 3.2 索引公式

```
global_idx = pp_idx * (dp * ep * tp) + dp_idx * (ep * tp) + ep_idx * tp + tp_idx
```

这个公式确保了：
- TP ranks 在最内层连续分布
- DP ranks 跨越所有 TP ranks
- EP ranks 跨越所有 DP × TP ranks
- PP stages 在最外层划分

### 3.3 用户如何改变分组

如果想不同的分组，调整传入的 rank 列表顺序即可。

```python
# 想让 [0,4] 和 [1,5] 成为同一个 TP group:
ranks = [0, 4, 1, 5, 2, 6, 3, 7]  # 重排顺序
```

---

## 4. 与 AICB 通信类型的对应

| AICB comm 后缀 | parse_comm_type 返回 | RankGrouper 调用 | 获取的 ranks |
|----------------|---------------------|------------------|-------------|
| 无后缀 (TP) | `(base, "tp")` | `get_tp_group(pp_idx, dp_idx, ep_idx)` | 同一 PP/DP/EP 内的所有 TP rank |
| `_DP` | `(base, "dp")` | `get_dp_group(pp_idx, ep_idx, tp_idx)` | 同一 PP/EP/TP 内的所有 DP rank |
| `_EP` | `(base, "ep")` | `get_ep_group(pp_idx, dp_idx, tp_idx)` | 同一 PP/DP/TP 内的所有 EP rank |
| `_DP_EP` | `(base, "dp_ep")` | `get_dp_ep_group(pp_idx, tp_idx)` | 同一 PP/TP 内的所有 DP×EP rank |

---

## 5. 开发过程（TDD）

### 5.1 方法

严格遵循 TDD 流程：先编写测试 → RED → GREEN → REFACTOR。

### 5.2 测试覆盖

`tests/test_rank_grouper.py`（17 个测试）覆盖：

| 测试类 | 数量 | 覆盖内容 |
|--------|------|---------|
| `TestBasicTPDPPP` | 5 | 基本 TP/DP/PP 分组（无 EP） |
| `TestWithEP` | 3 | 四维分组含 EP |
| `TestCustomRankOrder` | 2 | 自定义 rank 顺序验证分组变化 |
| `TestEdgeCases` | 4 | 单 GPU、TP=1、维度不匹配断言 |
| `TestIntegration` | 3 | 与 `parse_comm_type()` 集成模式验证 |

### 5.3 测试结果

```
tests/test_rank_grouper.py: 17 passed
Total: 104 passed (含 Phase 1 和其他模块测试)
```

---

## 6. 发现的问题及修复

### 问题 1：测试期望值错误

**现象**：`test_pp1_only_one_pp_group` 失败，期望 `[0, 2]` 但实际得到 `[0]`。

**原因**：测试编写时对索引公式理解有误。当 `pp=1` 时，PP group 只有一个元素，因为 `pp_idx` 只能为 0。

正确计算（4 GPUs, tp=2, dp=2, pp=1）：
- `get_pp_group(dp_idx=0, ep_idx=0, tp_idx=0)`:
  - pp_idx=0 only: `0 * 4 + 0 * 2 + 0 * 2 + 0 = 0` → `[0]`

**修复**：修正测试期望值为 `[0]`，并补充更多验证点。

**教训**：对于数学公式驱动的代码，测试期望值应该手动计算验证，而不是凭直觉猜测。

---

## 7. 使用示例

### 7.1 基本用法

```python
from src.workload_generator.rank_grouper import RankGrouper
from src.workload_format.schema import ParallelismConfig

# 8 GPUs, TP=2, DP=2, PP=2
config = ParallelismConfig(tp=2, dp=2, pp=2, ep=1)
grouper = RankGrouper(assigned_nodes=[0, 1, 2, 3, 4, 5, 6, 7], parallelism=config)

# 获取 TP group for pp_idx=0, dp_idx=0
tp_ranks = grouper.get_tp_group(pp_idx=0, dp_idx=0, ep_idx=0)
# → [0, 1]

# 获取 DP group for pp_idx=0, tp_idx=0
dp_ranks = grouper.get_dp_group(pp_idx=0, ep_idx=0, tp_idx=0)
# → [0, 2]

# 获取 PP group for dp_idx=0, tp_idx=0
pp_ranks = grouper.get_pp_group(dp_idx=0, ep_idx=0, tp_idx=0)
# → [0, 4]
```

### 7.2 与 WorkloadBuilder 集成

```python
# 在 WorkloadBuilder 中，根据 AICB 通信类型后缀选择正确的分组方法
from src.workload_generator.aicb_parser import AicbParser

base_type, context = AicbParser.parse_comm_type(item.forward_comm)

if context == "tp":
    ranks = grouper.get_tp_group(pp_idx, dp_idx, ep_idx)
elif context == "dp":
    ranks = grouper.get_dp_group(pp_idx, ep_idx, tp_idx)
elif context == "ep":
    ranks = grouper.get_ep_group(pp_idx, dp_idx, tp_idx)
elif context == "dp_ep":
    ranks = grouper.get_dp_ep_group(pp_idx, tp_idx)
```

---

## 8. 后续依赖

Task 2 的输出将被以下模块使用：

1. **Task 3（WorkloadBuilder）**：根据 AICB 通信类型后缀，调用对应的 `get_*_group()` 方法获取参与通信的 rank 列表，然后传递给 CollectiveExpander 展开为 P2P flows。
2. **Task 5（端到端示例）**：直接创建 RankGrouper 实例进行分组推导。

---

*开发时间：2026-04-10*
*测试状态：17 passed*
*开发方法：TDD（RED → GREEN → REFACTOR）*

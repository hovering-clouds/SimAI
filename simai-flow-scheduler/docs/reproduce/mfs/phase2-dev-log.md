# Phase 2 开发记录：Deadline-Aware P2D Promotion

## 概述

Phase 2 在 Phase 1 的基础上引入了 request-level 的 TTFT SLO 元数据，使 MFS allocator 能够根据 MLU（Minimal Link Utilization）动态决定 P2D 流的提升时机。

Phase 2 经历了初始实现和后续架构重构两个阶段。重构解决了三个设计问题：RLI 计算位置不当、时间域不对齐、不必要的向后兼容代码。

**最终结果：**
- 全部 555 个测试通过（含新增 Phase 2 测试），无回归
- RLI 在 allocator 中按需计算，由 policy 动态更新 current_layer
- 使用相对 SLO（ttft_slo_us）避免 Vidur 与 executor 时间域不对齐问题
- 移除了所有向后兼容代码，MLU 是唯一的 P2D 提升机制

---

## 开发顺序与具体内容

### Task 7: 扩展 Vidur TraceRecorder（`vidur-alibabacloud/vidur/trace_recorder.py`）

**目标：** 在 Vidur 的 trace 输出中为每个 request 添加 TTFT SLO 字段。

**实现：**

`TraceRecorder.__init__` 新增 `ttft_slo_us` 可选参数。当设置后，`record_request()` 在每个 request 条目中写入：

- `ttft_slo_us`：构造时传入的 SLO 值

**向后兼容：** `ttft_slo_us` 默认为 `None`，不传则不写入该字段。现有 Vidur 配置无需任何修改即可继续运行。

### Task 8: 解析 SLO 元数据（`mfs_context.py`）

**目标：** 在 MFS context 中解析并存储 request-level SLO 信息。

**实现：**

数据类简化为：

```python
@dataclass
class MfsRequestInfo:
    request_id: int
    ttft_slo_us: int | None
```

`MfsContext` 包含 `request_info: dict[int, MfsRequestInfo]` 字段。

`build_mfs_context()` 接受 `trace: dict` 参数（必选）。从 `trace["requests"]` 中提取 `ttft_slo_us`。缺少的字段自动为 `None`。

**测试：** 4 个测试覆盖带 SLO 的 trace 解析、缺少 SLO 字段的 None 降级、多 request 场景、空 requests。

### Task 9: 实现 MLU 提升（`mfs_allocator.py`）

**目标：** 根据 MLU 指标动态决定 P2D 流的优先级队列。

**实现：**

`MfsAllocatorConfig` 保留：

```python
p2d_mlu_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9)
```

移除了 `enable_deadline_promotion` 和 `p2d_promotion_delay_us`。

**RLI 动态计算：** 新增 `_compute_rli()` 方法，使用 `current_layer_by_stage` 字典按需计算 RLI，替代了静态的 `mfs_rli.py`。P2D 和 BACKGROUND 类型使用 sentinel 值（10000）。

**MLU 计算使用相对时间：** allocator 接受 policy 注入的 `request_start_time: dict[int, int]`。deadline = `request_start_time[rid] + ttft_slo_us`，确保在 executor 时间域内计算。

队列映射逻辑：
1. 计算剩余时间：`remaining_time = deadline - current_time`
2. 计算所需带宽：`required_bw = remaining_bits / (remaining_time * 1e3)`
3. 估计路径瓶颈带宽
4. 计算 MLU：`mlu = required_bw / bottleneck_bw`
5. 队列映射：
   - `remaining_time <= 0` → 紧急队列
   - `mlu >= 0.9` → 紧急队列
   - `mlu >= 0.75` → 中间队列
   - `mlu >= 0.5` → 较高队列
   - `mlu < 0.5` → 低队列

无 SLO 的 P2D 流保持在初始低优先级队列。

**测试：** 5 个 MLU 测试 + 4 个动态 RLI 测试覆盖宽松/紧迫/过期 deadline、无 SLO 降级、RLI 随 current_layer 推进等场景。

### Task 10: Policy 动态状态管理（`mfs_policy.py`）

**目标：** 在 executor 运行时跟踪 request 起始时间和计算层进度，注入 allocator。

**实现：**

`on_task_emitted()` 回调：当 request 的第一个 task 被发出时，记录 `request_start_time[rid] = current_time`。

`on_task_completed()` 回调：当 compute task 完成时，更新 allocator 的 `current_layer_by_stage[(job_id, stage_id)]` 为 `task.layer_id + 1`。这使得 allocator 中的 RLI 计算能反映真实的执行进度。

**测试：** 2 个测试验证 request_start_time 记录和 current_layer 推进。

### Task 11: E2E 脚本更新（`run_mfs_inference_e2e.py`）

**目标：** E2E 脚本适配新 API。

**实现：**

- 移除 `_has_deadlines()` 检测和 `enable_deadline_promotion` 配置
- 新增 `_has_slo()` 检测 trace 是否包含 SLO
- `_compute_qos()` 中，若有 SLO，从 executor 结果计算 `deadline_us = first_task_start + ttft_slo_us`，报告 `deadline_met`、`deadline_miss_us`、`p2d_earliness_us`
- 预构建 `p2d_tid_set` 在循环外，避免重复构建

---

## 重构：设计问题修复

Phase 2 初始实现后进行了架构重构，解决了三个设计问题：

### 问题 1: RLI 计算位置

**原问题：** RLI 在 static analysis 阶段预计算（`mfs_rli.py`），但 RLI 依赖 `current_layer` 这个动态状态。

**解决方案：** 废弃 `mfs_rli.py`，RLI 在 allocator 的 `_compute_rli()` 中按需计算。policy 的 `on_task_completed` 回调更新 allocator 的 `current_layer_by_stage`。

### 问题 2: 时间域不对齐

**原问题：** Vidur trace 中记录的 `arrival_time_us` / `deadline_us` 是 Vidur 仿真时间线，而 simai-flow-scheduler 的 executor 有自己的时间线，两者不对齐。

**解决方案：** Trace 只记录相对的 `ttft_slo_us`。Policy 在 `on_task_emitted` 中记录每个 request 在 executor 时间线上的首次出现时间（`request_start_time`）。Allocator 使用 `request_start_time + ttft_slo_us` 作为 deadline。

### 问题 3: 不必要的向后兼容

**原问题：** Phase 2 保留了 Phase 1 的 `enable_deadline_promotion`、`p2d_promotion_delay_us` 等兼容代码，增加了复杂度。

**解决方案：** 移除所有兼容代码。MLU 是唯一的 P2D 提升机制。无 SLO 的 P2D 流保持在低优先级队列。

---

## 文件清单

### 废弃文件

| 文件 | 状态 |
|---|---|
| `simai-flow-scheduler/src/static_analysis/passes/mfs_rli.py` | 标记为 deprecated，不再被导入 |

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `vidur-alibabacloud/vidur/trace_recorder.py` | 只输出 `ttft_slo_us`，移除 `arrival_time_us`/`deadline_us` |
| `simai-flow-scheduler/src/static_analysis/passes/mfs_context.py` | `MfsRequestInfo` 简化为只有 `ttft_slo_us`；`trace` 参数变为必选；移除未使用的 `TaskType` 导入和 `task_map` 变量 |
| `simai-flow-scheduler/src/static_analysis/passes/__init__.py` | 移除 `RliInfo`/`compute_static_rli` 导出 |
| `simai-flow-scheduler/src/static_analysis/strategies/mfs_strategy.py` | 移除 `rli_info` 字段和 `compute_static_rli` 调用；`trace` 参数变为必选 |
| `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py` | 移除 `rli_info` 参数、`enable_deadline_promotion`、`p2d_promotion_delay_us`；新增 `_compute_rli()` 动态计算；MLU 使用 `request_start_time + ttft_slo_us` |
| `simai-flow-scheduler/src/executor/policies/mfs_policy.py` | 新增 `on_task_emitted` 记录 `request_start_time`；`on_task_completed` 更新 allocator 的 `current_layer_by_stage` |
| `simai-flow-scheduler/scripts/run_mfs_inference_e2e.py` | 移除 `_has_deadlines`/`enable_deadline_promotion`；新增 `_has_slo`；QoS 报告使用相对 SLO |
| `simai-flow-scheduler/tests/test_mfs_context.py` | 移除 RLI 测试类，deadline 测试简化为 SLO 测试 |
| `simai-flow-scheduler/tests/test_mfs_allocator.py` | 移除 delay-based 测试，新增动态 RLI 测试，MLU 测试改用 `request_start_time` |
| `simai-flow-scheduler/tests/test_mfs_policy.py` | 所有测试适配新 API（trace 必选），新增 `request_start_time` 和 `current_layer` 验证 |

---

## 设计决策

1. **相对 SLO 而非绝对时间：** 只在 trace 中记录 `ttft_slo_us`（相对于 request 开始的时长），避免 Vidur 时间线和 executor 时间线的不对齐问题。Executor 侧通过 `request_start_time` 确定 request 开始时刻。

2. **RLI 按需计算：** RLI 依赖动态的 `current_layer` 状态，因此计算逻辑放在 allocator 中，由 policy 在每个 compute task 完成时更新。这比在 static analysis 中预计算更准确。

3. **不保留 Phase 1 的延迟提升机制：** MLU 是唯一的 P2D 提升机制。对于没有 SLO 的 trace，P2D 流保持在低优先级队列。这大幅简化了代码路径。

4. **`MfsRequestInfo` 最小化：** 只保留 `ttft_slo_us`。`arrival_time_us` 和 `deadline_us` 不再需要——前者属于 Vidur 时间线，后者由 executor 侧动态计算。

5. **Policy 职责清晰化：** Policy 负责跟踪运行时状态（`request_start_time`、`current_layer_by_stage`），Allocator 只负责带宽分配决策。两者通过 allocator 的公开属性通信。

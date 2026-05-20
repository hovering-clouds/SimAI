# Phase 2 开发记录：Deadline-Aware P2D Promotion

## 概述

Phase 2 在 Phase 1 的基础上引入了 request-level 的 TTFT deadline 元数据，使 MFS allocator 能够根据 MLU（Minimal Link Utilization）动态决定 P2D 流的提升时机，而非依赖固定的时间延迟。同时保持对无 deadline trace 的完全向后兼容。

**最终结果：**
- 全部 556 个测试通过（含 9 个新增 Phase 2 测试），无回归
- 无 deadline 的 trace 自动回退到 Phase 1 的延迟提升机制
- E2E 脚本自动检测 deadline 字段，选择对应的提升模式

---

## 开发顺序与具体内容

### Task 7: 扩展 Vidur TraceRecorder（`vidur-alibabacloud/vidur/trace_recorder.py`）

**目标：** 在 Vidur 的 trace 输出中为每个 request 添加 deadline 相关字段。

**实现：**

`TraceRecorder.__init__` 新增 `ttft_slo_us` 可选参数。当设置后，`record_request()` 在每个 request 条目中写入三个新字段：

- `arrival_time_us`：从 `request.arrival_time` 获取，若不存在则默认为 0
- `ttft_slo_us`：构造时传入的 SLO 值
- `deadline_us`：`arrival_time_us + ttft_slo_us`

**向后兼容：** `ttft_slo_us` 默认为 `None`，不传则不写入任何 deadline 字段。现有 Vidur 配置无需任何修改即可继续运行。

**注意：** 当前 `inference_trace_pp1.json` 是在不设置 `ttft_slo_us` 的条件下生成的，因此不包含 deadline 字段。要获得带 deadline 的 trace，需要在 Vidur 的 TraceRecorder 构造时传入 SLO 值。

### Task 8: 解析 Deadline 元数据（`mfs_context.py`）

**目标：** 在 MFS context 中解析并存储 request-level deadline 信息。

**实现：**

新增数据类：

```python
@dataclass
class MfsRequestInfo:
    request_id: int
    arrival_time_us: int | None
    ttft_slo_us: int | None
    deadline_us: int | None
```

`MfsContext` 新增 `request_info: dict[int, MfsRequestInfo]` 字段。

`build_mfs_context()` 新增 `trace: dict | None = None` 参数。当传入 trace 且 `trace["requests"]` 中包含 deadline 字段时，为每个 request 构建 `MfsRequestInfo`。缺少的字段自动为 `None`。

**中途遇到的问题：** 初始实现中 `MfsRequestInfo` 定义在 `MfsContext` 之后，导致 `MfsContext.request_info` 的类型注解引用了尚未定义的类（`NameError`）。修复为将 `MfsRequestInfo` 的定义移到 `MfsContext` 之前。

`MfsAnalyzer.analyze()` 同步新增 `trace` 参数，透传给 `build_mfs_context()`。

**测试：** 4 个测试覆盖带 deadline 的 trace 解析、无 trace 的空结果、缺少 deadline 字段的 None 降级、多 request 场景。

### Task 9: 实现 MLU 提升（`mfs_allocator.py`）

**目标：** 根据 MLU 指标动态决定 P2D 流的优先级队列。

**实现：**

`MfsAllocatorConfig` 新增两个字段：

```python
p2d_mlu_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9)
enable_deadline_promotion: bool = False
```

当 `enable_deadline_promotion=True` 且 P2D 流有关联的 request deadline 时，队列分配逻辑为：

1. 计算剩余时间：`remaining_time = deadline_us - current_time`
2. 计算所需带宽：`required_bw = remaining_bits / (remaining_time * 1e3)`（Gbps）
3. 估计路径瓶颈带宽
4. 计算 MLU：`mlu = required_bw / bottleneck_bw`
5. 队列映射：
   - `remaining_time <= 0` → 紧急队列
   - `mlu >= 0.9` → 紧急队列
   - `mlu >= 0.75` → 中间队列
   - `mlu >= 0.5` → 较高队列
   - `mlu < 0.5` → 低队列

若 P2D 流关联多个 request，取最早的 deadline。

**降级机制：** 当 `enable_deadline_promotion=False` 或 P2D 流没有关联 deadline 时，自动回退到 Phase 1 的固定延迟提升（`p2d_promotion_delay_us`）。

**中途遇到的问题：** 第一个 MLU 测试使用了 100 Gbps 链路，P2D 流仅 10KB 数据，deadline 100us。此时 MLU 计算结果极低（0.008），远低于阈值，导致 P2D 未被提升。原因在于 100 Gbps 链路上 10KB 数据在任何合理的 deadline 内都可以轻松传完。修复为使用 1 Gbps 链路 + 100KB 数据 + 100us 剩余时间，使 MLU 达到 8.0，触发紧急提升。

**测试：** 5 个测试覆盖宽松 deadline 保持低优先级、紧迫 deadline 提升到紧急、已过期 deadline 直接到紧急、无 deadline 回退到延迟模式、MLU 禁用时忽略 deadline。

### Task 10: Deadline 指标（`run_mfs_inference_e2e.py`）

**目标：** E2E 脚本在 trace 包含 deadline 时报告相关指标。

**实现：**

脚本新增 `_has_deadlines()` 函数检测 trace 是否包含 deadline 字段。根据检测结果：

- **MFS 策略配置：** `enable_deadline_promotion` 自动设为 trace 是否包含 deadline
- **QoS 报告新增字段：**
  - `deadline_us`：request 的 deadline
  - `deadline_met`：TTFT 是否满足 deadline（boolean）
  - `deadline_miss_us`：超出 deadline 的微秒数（正=超标）
  - `p2d_earliness_us`：P2D 完成时间相对 deadline 的提前量（正=提前完成）
- **comparison_report.json 新增字段：**
  - `promotion_mode`：`"mlu"` 或 `"delay"`
  - `deadline_aware`：boolean
  - per-request 中新增 `default_deadline_met`、`mfs_deadline_met`、`mfs_p2d_earliness_us` 等

同时修复了 Phase 1 中 `_compute_qos` 的一个低效问题：原实现在循环内对每个 collective flow 重复构建 `p2d_times_set`。改为在循环外预先构建一次。

---

## 文件清单

### 新建文件

无。

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `vidur-alibabacloud/vidur/trace_recorder.py` | 新增 `ttft_slo_us` 参数和 deadline 字段输出 |
| `simai-flow-scheduler/src/static_analysis/passes/mfs_context.py` | 新增 `MfsRequestInfo`、`request_info` 字段、`trace` 参数 |
| `simai-flow-scheduler/src/static_analysis/passes/__init__.py` | 导出 `MfsRequestInfo` |
| `simai-flow-scheduler/src/static_analysis/strategies/mfs_strategy.py` | `analyze()` 新增 `trace` 参数 |
| `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py` | MLU 提升逻辑、`_earliest_deadline()`、`_queue_for_mlu()` |
| `simai-flow-scheduler/scripts/run_mfs_inference_e2e.py` | deadline 检测、MLU 模式切换、deadline 指标报告 |
| `simai-flow-scheduler/tests/test_mfs_context.py` | 4 个 deadline 解析测试 |
| `simai-flow-scheduler/tests/test_mfs_allocator.py` | 5 个 MLU 提升测试 |

---

## 设计决策

1. **双向兼容：** `enable_deadline_promotion` 默认 `False`，不传 trace 的现有调用链完全不受影响。带 deadline 的 trace 配合 `enable_deadline_promotion=True` 激活 MLU；不带 deadline 的 trace 即使设为 `True` 也会自动回退到延迟模式。

2. **MLU 而非固定阈值：** 论文中 MLU 根据 `required_bw / bottleneck_bw` 动态计算，避免了固定阈值在链路带宽变化时失效的问题。Phase 1 的 `p2d_promotion_delay_us` 在无 deadline 时仍然可用。

3. **`MfsRequestInfo` 定义顺序：** 放在 `MfsContext` 之前，因为 dataclass 字段注解在定义时求值。

4. **E2E 自动检测：** 脚本自动判断 trace 是否包含 deadline，无需手动指定提升模式。

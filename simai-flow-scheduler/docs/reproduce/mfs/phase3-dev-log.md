# Phase 3 开发记录：RED-Based Inter-Request Ordering

## 概述

Phase 3 在 Phase 2 的基础上引入了 RED（Robust Effective Deadline）作为 EARLY RLI=0 flow 的队列分级依据。RED 通过在 batch 的多个 request deadline 中寻找最大相邻 gap，将 tight outlier 的影响降低，避免单个紧迫 request 拉高整个 batch 的优先级（piggyback 效果）。

Phase 3 经历了初始实现和后续重构两个阶段。重构将 RED 从 queue 内 strict sub-priority 改为 quantile 队列分级，修正了 MLU 队列映射错误，并修复了 P2D flow 的 request_ids 传播问题。

**最终结果：**
- 全部 563 个测试通过（含 8 个 Phase 3 测试），无回归
- 5 queue RMLQ：P2D initial / EARLY default / EARLY RLI=0 relaxed / EARLY RLI=0 urgent / P2D urgent
- EARLY RLI=0 flow 按 RED quantile 分级到 urgent/relaxed queue
- 所有 queue 内部统一 fair-share
- P2D 和 EARLY flow 完全隔离在不同 queue 中

---

## 开发顺序与具体内容

### Task 11: 运行时 RED 计算（`mfs_allocator.py`）

**目标：** 在 allocator 中动态计算 RED，替代 naive deadline-first 排序。

**设计原则：**

与 RLI 和 MLU 一致，RED 在运行时计算（不引入静态分析对象）。其输入依赖 executor 侧的 `request_start_time`，因此放在 allocator 中而非 static analysis。

**实现：**

新增 `_compute_red(request_ids: tuple[int, ...]) -> int | None` 方法：

1. 收集有 `request_start_time[rid] + ttft_slo_us` 的有效 deadline
2. 无有效 deadline → 返回 `None`
3. 单 request → 返回该 deadline
4. 多 request → 排序后找最大相邻 gap：
   - gap 前为 tight set，gap 后为 loose set
   - `f = len(tight) / n`
   - `RED = f * min(tight) + (1-f) * min(loose)`
   - loose set 为空时 `RED = min(tight)`

**测试：** 5 个测试覆盖单 request、均匀 tight batch、tight outlier 被 reduce、无 SLO 返回 None、无 request_info 返回 None。

### Task 12: RED Quantile 队列分级（`mfs_allocator.py`）

**目标：** 将 EARLY RLI=0 flow 按 RED 排序后 quantile 分级到不同 queue，每个 queue 内 fair-share。

初始实现使用 queue 内 per-flow strict sub-priority（按 RED 排序依次分配带宽），后续重构改为 quantile 队列分级。

**队列布局（5 queues）：**

```
queue 0: P2D initial / background
queue 1: EARLY default (RLI > 0)
queue 2: EARLY RLI=0, RED relaxed   (较高 RED，较不紧急)
queue 3: EARLY RLI=0, RED urgent    (较低 RED，较紧急)
queue 4: P2D urgent
```

**实现：**

Config 变更（`MfsAllocatorConfig`）：
- `num_queues: 4` → `5`
- `early_rli0_queue` 拆为 `early_rli0_relaxed_queue: int = 2` 和 `early_rli0_urgent_queue: int = 3`
- `urgent_p2d_queue: 3` → `4`
- 新增 `red_urgent_quantile: float = 0.5`
- `p2d_mlu_thresholds` 从 `tuple[float, ...] = (0.5, 0.75, 0.9)` 简化为 `float = 0.9`

`_queue_for()` 变更：
- EARLY RLI=0 flow 返回 `None`（延迟到 `allocate()` 统一处理）
- EARLY default (RLI > 0) → queue 1
- P2D → 通过 MLU 决定 queue 0 或 4

`allocate()` 变更（两阶段分类）：
- Phase 1: 逐 flow 调用 `_queue_for()`，RLI=0 flow 收集到 `rli0_flows` 列表
- Phase 2: 对 `rli0_flows` 计算 RED，按 RED 升序排序，取前 `red_urgent_quantile` 比例 → queue 3（urgent），其余 → queue 2（relaxed）
- Phase 3: 构建链路剩余容量
- Phase 4: 从最高 queue 到最低 queue 依次 fair-share 分配

删除的方法：
- `_flow_red_key()` — 不再需要 per-flow RED key
- `_is_early_queue()` — 所有 queue 统一 fair-share，无需特殊标记

**层级优先级总结：**

```
RMLQ queue priority (queue 4 > 3 > 2 > 1 > 0)
  └─ P2D urgent (queue 4, MLU-based)
  └─ EARLY RLI=0 urgent (queue 3, RED quantile top fraction)
  └─ EARLY RLI=0 relaxed (queue 2, RED quantile bottom fraction)
  └─ EARLY default (queue 1, RLI > 0)
  └─ P2D initial / background (queue 0)
```

**测试：** 3 个测试覆盖 lower RED 进 urgent queue 获得全部带宽、RED 不影响 P2D MLU、非 early queue 保持 fair-share。

---

## 重构：设计问题修复

Phase 3 初始实现后进行了架构重构，解决了三个设计问题：

### 问题 1: Strict sub-priority 不贴合实际网络调度

**原问题：** 初始实现使用 queue 内 per-flow strict sub-priority（按 RED 排序依次分配带宽，低 RED 的 flow 先拿完带宽，后续 flow 可能拿到 0）。实际网络中难以实现 per-flow 精确调度。

**解决方案：** 改为 quantile 队列分级。将 EARLY RLI=0 flow 按 RED 排序后分成两批，分别放入 `early_rli0_urgent_queue`（queue 3）和 `early_rli0_relaxed_queue`（queue 2），每个 queue 内 fair-share。配置项 `red_urgent_quantile: float = 0.5` 控制 top 比例。

不检测 RED variation，始终强制按 quantile 分割。这简化了逻辑，同时提供确定性的 tiebreak（task_id 排序）。

### 问题 2: MLU 队列映射导致 P2D 和 EARLY 混队

**原问题：** `_queue_for_mlu` 中 MLU 中间阈值（0.5、0.75）映射到 `early_rli0_queue` 和 `early_default_queue`，导致 P2D flow 被放入 EARLY queue，两类流量在 allocate 时混在一起。

**解决方案：** MLU 映射只使用 P2D 专用 queue（queue 0 和 queue 4）。同时将 `p2d_mlu_thresholds` 从多级阈值 `(0.5, 0.75, 0.9)` 简化为单阈值 `0.9`，P2D 只有两种状态：initial 或 urgent。

### 问题 3: P2D flow 的 request_ids 膨胀

**原问题：** `_expand_kv_transfer` 将所有 request 的 flow 合并为一个 `batch_task_map` 条目，`request_ids` 设为 decode batch 的全部 request。导致 P2D flow 的 `MfsTaskInfo.request_ids` 包含无关 request，`_earliest_deadline()` 和 `_compute_red()` 看到的是膨胀的 deadline 集合。

**解决方案：** `_expand_kv_transfer` 新增 `request_task_map: dict[int, list[int]]` 返回值，追踪每个 request 对应的 flow task_ids。Caller 为每个 request 创建独立的 `batch_task_map` 条目（key: `kv_{dep_id}_to_{bid}_req_{req_id}`），`request_ids` 只包含单个 request。

---

## 文件清单

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py` | RED 计算、quantile 分级、5 queue 配置、MLU 简化 |
| `simai-flow-scheduler/tests/test_mfs_allocator.py` | 适配新 queue 布局和 RED quantile 行为 |
| `simai-flow-scheduler/src/workload_generator/inference_trace_expander.py` | `_expand_kv_transfer` per-request batch_task_map 条目 |
| `simai-flow-scheduler/tests/test_inference_trace_expander.py` | 适配 kv key 格式变更 |
| `simai-flow-scheduler/tests/test_pp_inference_trace_expander.py` | 适配 kv key 格式变更 |

未修改 mfs_context、mfs_policy、E2E 脚本或其他文件。

---

## 设计决策

1. **RED 在运行时计算：** 与 RLI、MLU 一致，RED 依赖 `request_start_time` 和 `ttft_slo_us` 的动态状态，不适合放在 static analysis。不需要引入 `RedInfo` 数据类或新的 static analysis pass。

2. **Quantile 分级而非 strict sub-priority：** 初始设计使用 queue 内 per-flow strict sub-priority，但实际网络中难以实现 per-flow 精确调度。改为将 flow 按 RED quantile 分配到不同优先级的独立 queue，每个 queue 内 fair-share，更贴近真实网络行为。

3. **`red_urgent_quantile` 使用比例而非绝对阈值：** 不同场景下时间尺度差异大（毫秒到秒级），绝对时间阈值难以统一。MLU 式的归一化（`required_bw / bottleneck_bw`）对 EARLY flow 不适用，因为 EARLY flow 不是最后一跳。Quantile 比例（如 0.5）是 flow 间的相对排名，天然跨场景适用。

4. **RED 只影响 EARLY RLI=0 flow：** EARLY default (RLI > 0) 和 P2D flow 不参与 RED 分级。P2D 的 MLU 提升逻辑使用 `_earliest_deadline()`（最早 deadline），不受 RED 影响。EARLY default 不做 RED 分级，始终 fair-share。

5. **P2D 和 EARLY queue 完全隔离：** P2D MLU 映射只使用 queue 0（initial）和 queue 4（urgent），绝不借用 EARLY queue。两类流量在 allocate 时不会混在一起。

6. **MLU 简化为单阈值：** 多级 MLU 阈值增加了配置复杂度且中间级会借用 EARLY queue。简化为单阈值（0.9），P2D flow 要么在 initial queue 要么直接进 urgent queue。

7. **`_compute_red` 接受 request_ids 而非 batch_id：** Flow 通过 `MfsTaskInfo.request_ids` 关联到 request，不需要经过 batch 间接查找。P2D flow 的 request_ids 已修复为 per-request 粒度。

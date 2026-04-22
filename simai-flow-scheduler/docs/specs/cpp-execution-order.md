# C++ 参考实现执行顺序分析

## 概述

本文档记录 `astra-sim` C++ 参考实现（`Workload::iterate_hybrid_parallel_Transformer()`）中各操作的执行顺序和依赖关系，用于指导 Python workload 生成时正确建立 `deps` 依赖边。

---

## 单层执行顺序

### Forward Pass（正向传播）

```
等待 layer[i].wg_comm 完成（上一 GA 遗留）
  → layer[i].fwd_compute
  → layer[i].fwd_comm（Blocking，等待完成）
  → 进入 layer[i+1].Forward_Pass
```

### Input_Gradient（输入梯度）

```
layer[i].ig_compute
  → layer[i].ig_comm（Blocking，等待完成）
  → 进入 layer[i].Weight_Gradient
```

### Weight_Gradient（权重梯度）

```
layer[i].wg_compute
  → layer[i].wg_comm（Non-Blocking，立即返回，后台进行）
  → 等待 layer[i].ig_comm 完成（此时已完成，直接通过）
  → index--，进入 layer[i-1].Input_Gradient
```

---

## 完整单 GA 执行时序

```
Forward Pass（layer 0 → N-1）:
  [layer 0]  等待 GA[k-1].layer[0].wg_comm → fwd_compute → fwd_comm(Blocking)
  [layer 1]  等待 GA[k-1].layer[1].wg_comm → fwd_compute → fwd_comm(Blocking)
  ...
  [layer N-1] 等待 GA[k-1].layer[N-1].wg_comm → fwd_compute → fwd_comm(Blocking)

Backward Pass（layer N-1 → 0）:
  [layer N-1] ig_compute → ig_comm(Blocking) → wg_compute → wg_comm(Non-Blocking)
  [layer N-2] ig_compute → ig_comm(Blocking) → wg_compute → wg_comm(Non-Blocking)
  ...
  [layer 0]   ig_compute → ig_comm(Blocking) → wg_compute → wg_comm(Non-Blocking)
              → pass_counter++，进入下一 GA
```

---

## 依赖关系汇总

### GA 内部依赖

| 操作 | 依赖于 |
|------|--------|
| `layer[i].fwd_compute` | `layer[i-1].fwd_comm` 完成（前一层 fwd 通信） |
| `layer[i].fwd_comm` | `layer[i].fwd_compute` 完成 |
| `layer[i].ig_compute` | `layer[i+1].ig_comm` 完成（后一层 ig 通信，backward 方向） |
| `layer[i].ig_comm` | `layer[i].ig_compute` 完成 |
| `layer[i].wg_compute` | `layer[i].ig_comm` 完成 |
| `layer[i].wg_comm` 发起 | `layer[i].wg_compute` 完成 |

### GA 间依赖（跨 GA 同步）

| 操作 | 依赖于 |
|------|--------|
| `GA[k+1].layer[i].fwd_compute` | `GA[k].layer[i].wg_comm` 完成 |

**说明**：Forward Pass 进入每一层时都会检查 `layers[index]->is_weight_grad_comm_finished_blocking()`，`index` 即当前层编号，因此每层的 fwd_compute 各自等待上一 GA 同层的 wg_comm，而不是只等 layer[0]。

---

## WG 通信的重叠机制

WG 通信（DP AllReduce）以 Non-Blocking 方式发起，在后台与后续操作并发进行：

```
GA[k] backward:
  layer[N-1].wg_comm ──────────────────────────────────────► 完成
  layer[N-2].wg_comm ─────────────────────────────► 完成
  ...
  layer[0].wg_comm ──────────────────────────► 完成

GA[k+1] forward:
  layer[0].fwd_compute（等 layer[0].wg_comm 完成后开始）
  layer[1].fwd_compute（等 layer[1].wg_comm 完成后开始）
  ...
```

各层的 wg_comm 与 GA[k+1] 对应层的 fwd_compute 形成流水线重叠，这是 DP 通信隐藏延迟的核心机制。

---

## 对 workload 生成的指导

在生成 `deps` 字段时，需要显式建立以下依赖边：

1. **GA 内部**：按上表中的依赖关系建立，均已是数据依赖，应直接写入 `deps`。

2. **GA 间**：对每一层 `i`，将 `GA[k].layer[i].wg_comm` 的最后一个 flow task 加入 `GA[k+1].layer[i].fwd_compute` 的 `deps`。

3. **WG 通信的"最后一个 flow"**：wg_comm 展开为多个 P2P flow（Ring AllReduce），其中最后完成的 flow 即为 AllGather 阶段的最后一步，应作为依赖目标。

---

*文档创建日期：2026-04-21*

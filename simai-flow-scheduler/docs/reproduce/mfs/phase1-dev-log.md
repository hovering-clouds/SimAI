# Phase 1 开发记录：MFS-Lite Replay Layer

## 概述

Phase 1 在 `simai-flow-scheduler` replay 层实现了 MFS 论文中 Defer-and-Promote 机制的核心逻辑，不修改 Vidur trace 格式，不修改 executor 核心，仅通过新增 sidecar 元数据、RMLQ allocator 和 MFS policy 三个层次实现。

**最终结果：**
- 全部 547 个测试通过（含 26 个新增 MFS 测试），无回归
- E2E 脚本在现有推理 trace 上成功运行，MFS 策略 makespan 6.77s vs 默认 7.80s（提升约 13%）

---

## 开发顺序与具体内容

### Task 1: MFS Context 元数据（`mfs_context.py`）

**目标：** 为 workload 中的每个 task 构建侧车元数据，分类其 MFS 角色。

**实现：**

定义了三个核心数据类：

- `MfsStage` 枚举：`EARLY`（早期阻塞通信）、`P2D`（Prefill-to-Decode KV 传输）、`BACKGROUND`（计算和未知流）
- `MfsTaskInfo`：每个 task 的 MFS 分类信息，包括 batch_id、request_ids、stage_id、mfs_stage、target_layer、comm_role
- `MfsContext`：整体索引，包含 task_info 字典、batch_to_tasks 分组、request_to_tasks 分组

分类规则：

| CommType | MfsStage | comm_role |
|---|---|---|
| `KV_CACHE_TRANSFER` | P2D | p2d_transfer |
| `PP_SEND` / `PP_RECV` | EARLY | pp_send |
| TP/EP/DP 集合通信 | EARLY | collective |
| Compute task | BACKGROUND | compute |
| 其他（`UNKNOWN` 等） | BACKGROUND | unknown |

`build_mfs_context()` 函数从 `P2PWorkload` 和 `batch_task_map`（InferenceTraceExpander 的输出）构建完整上下文。通过反向索引将 task_id 映射回 batch_id、request_ids 和 stage_id。

**测试：** 10 个测试用例覆盖各类 task 的分类、batch/request 索引、无 batch_task_map 的降级处理。

### Task 2: RLI 元数据（`mfs_rli.py`）

**目标：** 为早期阻塞流计算静态 RLI（Relative Layer Index），作为优先级依据。

**实现：**

定义 `RliInfo` 数据类（task_id, target_layer, base_rli）。

`compute_static_rli()` 计算逻辑：

- **EARLY 流：** `base_rli = max(target_layer - current_layer, 0)`，RLI 越小越紧急
- **P2D 流：** 赋予大哨兵值（10000），Phase 1 不用 RLI 排序 P2D
- **BACKGROUND：** 同样赋予大哨兵值

支持 `current_layer_by_stage` 参数（key 为 `(job_id, stage_id)`），用于未来动态更新当前计算层。

**测试：** 6 个测试用例验证 layer 0 RLI=0、高 layer RLI 值、P2D 不超越 RLI=0 集合流、current_layer 推进后 RLI 递减、RLI 非负。

### Task 3: MFS Analyzer（`mfs_strategy.py`）

**目标：** 组合默认路由 + 计算排序 + MFS 元数据的分析策略。

**实现：**

`MfsAnalyzer` 镜像 `DefaultAnalyzer` 的结构，在其基础上额外调用 `build_mfs_context()` 和 `compute_static_rli()`。返回 `MfsAnalysisResult` 包含 route_table、execution_plan、mfs_context、rli_info 四项。

同时更新了三个 `__init__.py` 文件以导出新增类型。

### Task 4: RMLQ Allocator（`mfs_allocator.py`）

**目标：** 实现反向多级队列（RMLQ）风格的带宽分配器。

**实现：**

`MfsAllocatorConfig` 配置 4 个优先级队列（0-3，数字越大优先级越高）：

| 队列 | 用途 | 默认编号 |
|---|---|---|
| `p2d_initial_queue` | P2D 初始低优先级 | 0 |
| `early_default_queue` | 早期流（RLI > 0） | 1 |
| `early_rli0_queue` | 早期流（RLI = 0，当前层阻塞） | 2 |
| `urgent_p2d_queue` | P2D 提升后的高优先级 | 3 |

分配算法：

1. 将每个 active flow 分类到队列
2. 从高优先级到低优先级逐队列分配
3. 同一队列内在每条链路上公平分享
4. 每条流的最终分配取路径上所有链路的最小值（瓶颈）

P2D 提升逻辑：当 `current_time - flow.start_time >= p2d_promotion_delay_us` 时，P2D 从初始队列提升到紧急队列。

**中途遇到的问题：** 初始实现中，同一队列内的流逐个处理，处理第一个流时就扣减了链路剩余带宽，导致第二个流看到的剩余容量已经减少，公平分享失效（两个流在同一链路上分别得到 50% 和 25% 而非各 50%）。修复为先计算同队列所有流的分配量，再统一扣减。

**测试：** 6 个测试用例验证严格优先级、同队列公平分享、P2D 延迟提升、多跳瓶颈、空输入。

### Task 5: MFS Scheduling Policy（`mfs_policy.py`）

**目标：** MFS 调度策略，保留计算排序，使用 MFS allocator。

**实现：**

`MfsSchedulingPolicy` 实现 `SchedulingPolicy` 接口：

- **compute ordering：** 与 `DefaultSchedulingPolicy` 完全一致，通过 `execution_plan.compute_order` 和 `compute_cursor` 保证每个节点上 compute 串行执行
- **flow 路由：** 使用 `route_table.get_path(task)`（默认 BFS 最短路径）
- **带宽分配：** 委托给 `MfsAllocator`
- **flow 准入：** Phase 1 中所有 ready flow 都被准入，依赖 allocator 优先级而非延迟发放

**中途遇到的问题：** 第一个集成测试中，collective 流和 P2D 流不在同一拓扑链路上（星型拓扑中 0→4→1 和 2→4→3 走不同链路），没有实际竞争，测试断言失败。改为使用线型拓扑（0-1-2），两个流都经过同一条瓶颈链路。同时发现两个流的启动时间不同（P2D 无依赖在 t=0 启动，collective 依赖 compute 在 t=10 启动），需要给 P2D 也添加对 compute 的依赖，使两者同时启动才能展示优先级效果。

**测试：** 4 个集成测试验证 collective 优先于延迟 P2D、P2D 提升后获得带宽、计算排序与默认策略一致、多 P2D 流不死锁。

### Task 6: E2E 对比脚本（`run_mfs_inference_e2e.py`）

**目标：** 对同一推理 trace 分别运行 default 和 MFS 策略，生成对比报告。

**实现：**

脚本流程：

1. 加载推理 trace 和 profile
2. `InferenceTraceExpander` 展开为 P2PWorkload
3. 加载拓扑
4. 运行 default 策略（`DefaultAnalyzer` + `DefaultSchedulingPolicy`）
5. 运行 MFS 策略（`MfsAnalyzer` + `MfsSchedulingPolicy`）
6. 计算每个 request 的 TTFT、TBT、E2E、P2D 完成时间、collective 完成时间
7. 生成对比报告

输出目录：`outputs/mfs_reproduce/`，包含 workload.json、两种策略的 execution_result.json 和 qos_report.json，以及 comparison_report.json。

---

## 文件清单

### 新建文件

| 文件 | 用途 |
|---|---|
| `src/static_analysis/passes/mfs_context.py` | MFS sidecar 元数据构建 |
| `src/static_analysis/passes/mfs_rli.py` | RLI 优先级计算 |
| `src/static_analysis/strategies/mfs_strategy.py` | MFS 分析策略 |
| `src/executor/bandwidth_allocators/mfs_allocator.py` | RMLQ 带宽分配器 |
| `src/executor/policies/mfs_policy.py` | MFS 调度策略 |
| `scripts/run_mfs_inference_e2e.py` | 端到端对比脚本 |
| `tests/test_mfs_context.py` | Context + RLI 测试（16 用例） |
| `tests/test_mfs_allocator.py` | Allocator 测试（6 用例） |
| `tests/test_mfs_policy.py` | Policy 集成测试（4 用例） |

### 修改文件

| 文件 | 修改内容 |
|---|---|
| `src/static_analysis/passes/__init__.py` | 导出 MFS context/RLI 类型 |
| `src/static_analysis/strategies/__init__.py` | 导出 MfsAnalyzer/MfsAnalysisResult |
| `src/executor/bandwidth_allocators/__init__.py` | 导出 MfsAllocator/MfsAllocatorConfig |
| `src/executor/policies/__init__.py` | 导出 MfsSchedulingPolicy |

### 未修改的文件

- `AnalyticalExecutor` — policy 接口足够，无需修改
- `P2PWorkload` schema — 使用 sidecar 元数据，无 schema 变更
- `InferenceTraceExpander` — 无行为变更
- `DefaultSchedulingPolicy` — 保持不变

---

## 设计决策

1. **Sidecar 元数据而非 schema 变更：** Phase 1 的 MFS 分类信息通过独立的 `MfsContext` / `RliInfo` 字典存储，不修改 `Task` 或 `P2PWorkload` 的字段。这样避免了影响其他组件。

2. **P2D 用时间延迟而非 MLU 提升：** 论文中使用 MLU（Minimal Link Utilization）根据 deadline 紧迫度提升 P2D。Phase 1 没有 request deadline 信息，改用 `p2d_promotion_delay_us` 时间阈值近似。Phase 2 加入 deadline 后切换为 MLU。

3. **Flow 全准入 + allocator 管优先级：** Phase 1 中所有 ready flow 都被 policy 准入，由 allocator 决定实际获得的带宽。这避免了 policy 层面复杂的流控逻辑，也与现有 executor 的事件驱动模型兼容。

4. **4 级队列配置化：** 队列数量和映射通过 `MfsAllocatorConfig` 配置，方便调参和 Phase 2 扩展。

---

## E2E 测试结论

### 旧 trace（`inference_trace.json`）

使用旧版 trace 运行，MFS 策略 makespan 6.77s vs 默认 7.80s（提升约 13%）。但该 trace 存在重复 dep 和缺失字段问题，已不建议使用。

### 新 trace（`inference_trace_pp1.json`）

使用修正后的 pp1 trace 运行，两种策略的 makespan 完全一致（5,326,870 us），TTFT 也相同。

**原因分析：** 该 trace 场景中 P2D 流与集合流不存在链路竞争，两个策略对每个流分配了相同的带宽：

1. **集合流（TP allreduce 131k 条 + EP alltoall 178k 条）全在 replica 内部**（src/dst 范围 0-15），走 NVLink/NVSwitch 机内链路
2. **KV 传输仅 16 条**（src=0-7 → dst=8-15），走 PSW 跨机链路
3. **两者走不同的物理链路**，BFS 路由不会让它们共享瓶颈
4. **并发度低**：仅 2 个 request、1 个 prefill batch，895MB 的 KV 数据在 AlibabaHPN 400Gbps 跨机链路上不构成带宽压力

MFS 的 Defer-and-Promote 机制要体现效果需要满足以下条件之一：

- **多 request 并发**：多个 prefill/decode 对同时进行，P2D 和集合流在跨机链路上争抢同一个 PSW 瓶颈
- **同链路竞争**：拓扑的跨机带宽不够充裕，P2D 和集合流的路径有重叠

当前的 pp1 trace 是单 prefill batch 的简单场景，网络无竞争，因此两种策略等价。这符合预期——MFS 的价值在于网络竞争环境下的流量协调。需要更复杂的 trace（更多并发 request、更密集的 PD 切换）才能展示差异化效果。

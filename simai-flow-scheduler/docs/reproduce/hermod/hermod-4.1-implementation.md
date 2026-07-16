# Hermod §4.1 实现与实验说明

## 目的与范围

本实现复现 Hermod 论文 §4.1 的 **inter-coflow strict-priority** 思想：在共享链路上，
按 PP/DP coflow 的 Hermod 优先级分层服务；同一优先级层内使用 progressive-filling
max-min fair 分配，避免多链路 flow 的瓶颈造成可用容量闲置。

不在范围内：论文 §4.2 的 matching-based intra-coflow scheduling、EP 专用流量分配、
交换机 DSCP/队列部署、动态路由，以及生产集群参数拟合。

EP 的 collective/输入路径尚未单独验证，因此默认 `ep_mode=reject`。发现 EP flow 时，
Hermod 会明确失败，避免将未验证结果纳入实验结论。

## 组件与数据流

```text
AICB workload + JSON experiment config
  -> WorkloadBuilder
  -> HermodAicbMetadataAdapter
       GA step -> microbatch_id (MID)
       flattened AICB item position -> logical_layer_id (LID)
       writes hermod_metadata.json sidecar
  -> HermodAnalyzer (GPipe or 1F1B execution plan + BFS routes)
  -> HermodSchedulingPolicy / HermodAllocator
  -> AnalyticalExecutor
```

核心文件：

| 文件 | 职责 |
| --- | --- |
| `src/workload_generator/hermod_aicb_metadata.py` | 仅对 AICB 训练 workload 标注 MID/LID；不改变通用 `Task.iteration` 语义。 |
| `src/static_analysis/passes/hermod_priority.py` | PP/DP/EP 分类、优先级排序与 EP 模式校验。 |
| `src/static_analysis/strategies/hermod_strategy.py` | pipeline serializer、BFS routing 与 Hermod priority analysis。 |
| `src/executor/bandwidth_allocators/hermod_allocator.py` | coflow 间严格优先、层内公平共享。 |
| `src/executor/policies/hermod_policy.py` | 复用默认准入/路由接口，仅替换带宽分配。 |
| `scripts/run_hermod_e2e.py` | 配置驱动的 Default/Puppeteer/Hermod 对比入口。 |

`JobMerger` 会保留 Hermod metadata，并为 `coflow_id` 加入 job namespace，避免多 job
合并时碰撞。

## Pipeline 模式

`run_hermod_e2e.py` 支持：

- `gpipe`：`CppReferenceSerializer`，所有 forward 后执行 backward。
- `1f1b`：`OneFOneBSerializer`，按照 pipeline stage 的 warmup/steady/cooldown 顺序。

默认是 `1f1b`。新流水线模式应先在 `HermodAnalyzer` 注册对应 serializer，再加入 CLI/config
允许值；不要只增加一个名字。

## 配置文件与运行

示例：[`scripts/hermod_e2e_config.json`](../../../scripts/hermod_e2e_config.json)。

```powershell
cd D:\Code\SimAI\simai-flow-scheduler
python scripts/run_hermod_e2e.py --config scripts/hermod_e2e_config.json
```

配置是 JSON object：

```json
{
  "workload": "inputs/aicb-workload/<file>.txt",
  "topology": "inputs/topologies/<topology>",
  "dp": 2,
  "ep_mode": "reject",
  "pipeline": "1f1b",
  "variant": "conventional_1f1b",
  "modes": ["default", "puppeteer", "hermod"],
  "k_paths": 4,
  "output": "outputs/hermod_experiment"
}
```

CLI 参数覆盖配置，例如：

```powershell
python scripts/run_hermod_e2e.py --config scripts/hermod_e2e_config.json `
  --pipeline gpipe --modes default hermod
```

`dp` 是实验覆盖值：使用 AICB 的 TP/PP/EP 与指定 DP 重建 rank group，因此会生成 DP
collective；不会修改原始 AICB 文件。所需 GPU 数为 `tp * dp * pp * ep`，不得超过拓扑的
GPU 数。

## 输出与比较口径

输出目录包含：

- `workload.json`：带 Hermod metadata 的 P2P DAG；
- `hermod_metadata.json`：每个范围内 flow 的 MID/LID/coflow 来源 sidecar；
- `<mode>_execution_result.json`：每个已运行策略的 task 时间；
- `<mode>_task_meta.json`：dynamic 入口输出的 task Hermod metadata；
- `summary.json`：输入、并行度、coflow 表、类型计数与 makespan。

`default` 与 `hermod` 共享 BFS 路由和同一 compute execution plan，以隔离 Hermod allocator
的影响。`puppeteer` 使用自己的 Greedy route、TTE 分配及资源协调，因此是相关但不同的
网络调度基线；它不等同于 Hermod，也尚未实现两者的组合策略。

## Priority-tier and visualization notes

The allocator treats coflows equal on every paper-defined §4.1 dimension
(`job_id`, MID, CType, LID) as one priority tier and performs progressive
max-min sharing within that tier.  `coflow_id` is only a stable display key;
it must not invent an additional strict ordering.

Both E2E runners accept a JSON boolean `"visualize": true` (or CLI
`--visualize`).  The static runner writes `<mode>_trace.json` with the normal
Chrome Trace visualizer; the dynamic runner writes the same per-mode file with
task metadata, including Hermod MID/LID/coflow fields in each event's `args`.
Open the trace in `chrome://tracing` or Perfetto.  The trace is diagnostic: it
shows that bandwidth priorities were applied, but does not itself establish a
paper-equivalent end-to-end speedup.

### Performance interpretation

On the four-server Hermod topology, the runner's default contiguous rank
mapping places a complete `TP × DP` PP stage on each 8-GPU server for common
`pp=4,tp=4,dp=2` configurations.  PP therefore uses NICs between servers
while DP is local NVLink traffic, leaving little PP/DP contention for §4.1 to
resolve.  Multiple dynamic iterations are sequential dependencies, so they
smooth a measurement but do not create cross-iteration contention.  Results
from this placement must not be used to claim that Hermod has no benefit.

For contention diagnosis only, the dynamic runner also accepts
`"placement": "cyclic_pp_dp"` and `"gpus_per_server": 8`.  It keeps each TP
group inside one server, but rotates PP stages and DP replicas across the four
servers so both groups traverse NIC/leaf resources.  This is deliberately an
experimental stress placement, not a paper-validated placement.  Use it to
inspect traces and priority behavior; report it separately from the contiguous
placement baseline.

With strict LID and the supplied Hermod topology, useful existing AICB
single-iteration configurations are GPT-13B and GPT-22B `tp=4,pp=2,dp=4,ga=8`
with `gbs32/mbs4`: Default/Hermod makespans are respectively
`6,549,866 / 6,384,152 us` (+2.53%) and `9,854,039 / 9,686,335 us` (+1.70%).
The corresponding `pp=4` scans and the first cyclic-PP/DP GPT-13B run
regressed, so more contention alone is not a performance claim.

For GPT-style AICB traces, the Hermod-only adapter now reconstructs a local
model LID: the input `embedding_layer` is boundary LID 0, and each adjacent
`attention_layer` + `mlp_layer` pair shares the next Transformer-layer LID.
Post-GA DP traffic receives the final boundary LID.  The adapter validates the
sequence and fails for an unrecognized operation instead of silently applying
a flat position.  It does not change the generic parser, `Task.layer_id`, DAG,
or any non-Hermod scheduler.  Static and dynamic sidecars record the mapping
rule and source operation for audit.

The remaining reproduction limitations are material: this recovers the local
layer ordering but does not establish model-stage ownership or virtual-pipeline
chunk ownership, the placement is not yet a paper-validated distributed-
training placement, and §4.2 matching-based intra-coflow allocation/EP is
absent.  A result where Hermod is slower is therefore useful diagnosis, not
evidence of a faithful full-paper comparison.  Before reporting a speedup
study, validate a placement that makes the intended PP/DP coflows share the
NIC bottleneck, then compare its traces and critical-path stalls under the
same 1F1B plan and routes.

`interleaved_1f1b` currently selects only the §4.1 CType ordering variant.
The compute serializer is still the conventional `OneFOneBSerializer`; it is
not a virtual-pipeline/interleaved execution-model reproduction.  It must not
be used as an interleaved Hermod performance result until that serializer and
its VPP task DAG are implemented and validated.

## Dynamic executor

### Multi-job priority boundary

MID and LID restart for every dynamic training job. Hermod §4.1 priority
rules, including Case III, are therefore applied only between coflows of the
same `job_id`. For coflows from different jobs, the dynamic runner uses stable
ascending `job_id` order. This is a deterministic multi-job composition rule,
not a priority policy claimed by the Hermod paper. It prevents incomparable
per-job MID/LID values from creating a cyclic priority graph.

`scripts/run_hermod_dynamic_e2e.py` 支持按需展开多 job、多 iteration 的 AICB 训练实验。
配置示例为 [`scripts/hermod_dynamic_e2e_config.json`](../../../scripts/hermod_dynamic_e2e_config.json)：

```powershell
python scripts/run_hermod_dynamic_e2e.py --config scripts/hermod_dynamic_e2e_config.json
```

动态配置使用与 Cassini 一致的 `workloads` 列表，每项独立指定 `aicb`、`dp`、`num_jobs` 和
`num_iters`，例如：

```json
{
  "workloads": [
    {"aicb": "inputs/aicb-workload/model-a.txt", "dp": 2, "num_jobs": 2, "num_iters": 5},
    {"aicb": "inputs/aicb-workload/model-b.txt", "dp": 1, "num_jobs": 1, "num_iters": 3}
  ]
}
```

每个 spec 会生成 `num_jobs * num_iters` 个动态 job；同一逻辑 job 的 iteration 保持串行依赖，
所有 spec 的不同 job 则共享拓扑并可并发注入，因此可建模异构 workload 的网络竞争。每个按需展开
batch 都先完成 AICB Hermod metadata 标注，随后
`HermodSchedulingPolicy.update_analysis()` 合并路由、compute order、task-to-coflow 映射和
priority analysis；输出还包含 `<mode>_task_meta.json`，可审计每个动态任务的 MID/LID/coflow。
Hermod mode 还会写入聚合的 `hermod_metadata.json`，与静态入口的 sidecar 命名保持一致。

该 dynamic 路径不支持 EP；配置没有 `ep_mode`，固定为 `reject`。它也不替代单 iteration
静态复现，而是用于避免一次性物化大量训练 iteration，并研究多 job 竞争。

## 已验证配置与限制

已验证配置：现有 16-GPU AlibabaHPN 拓扑、GPT-13B AICB `tp=4, pp=2, ga=2`，以 `dp=2`
覆盖运行。该运行生成 4 个 PP coflow 与 16 个 DP coflow，并成功运行 Default、Puppeteer、
Hermod 三种模式；同一配置下 GPipe 与 1F1B 分支也可完成运行。

Hermod priority、allocator、AICB metadata、builder 与 job merger 的定向测试通过（46 tests）。
Dynamic Hermod 审查修复后，两个不同 AICB spec（第一项 `dp=2,num_jobs=1,num_iters=2`，第二项
`dp=1,num_jobs=1,num_iters=1`）在 16-GPU AlibabaHPN 上完成混合 E2E：共 3 个动态 job、27,416
个任务；Default/Hermod makespan 分别为 1,525,671 / 1,534,909 us。`hermod_metadata.json` 已验证
包含每个范围内 flow 的 `coflow_id`、MID 与 LID。
全量测试中 Spectrum-X 相关的既有测试仍依赖仓库外缺失的旧拓扑文件；这与 Hermod 无关。

当前实现的结论应限制为 PP/DP 的 §4.1 流级严格优先实验。AICB 当前只提供 GA 块内的扁平
item 位置，adapter 将其作为可审计的兼容 LID；它不是已由输入证实的论文逻辑模型层 ID。因此，
Case III 的真实层语义仍需含显式模型层标识的输入验证。EP 与 §4.2 完成前，不应报告为完整
Hermod 端到端复现。

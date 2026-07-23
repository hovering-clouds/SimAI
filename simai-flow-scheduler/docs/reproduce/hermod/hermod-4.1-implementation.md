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
       GPT embedding/attention+MLP structure -> audited local Transformer LID
       writes hermod_metadata.json sidecar
  -> HermodAnalyzer (registered pipeline execution plan + BFS routes)
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
- `interleaved_1f1b`：`InterleavedOneFOneBSerializer`，按 VPP chunk、micro-batch group
  和 stage-specific warmup 生成 `F -> B` steady-state 交错顺序；static/dynamic runner
  还会先生成 strategy-owned VPP workload overlay。
- `zero_bubble`：`ZeroBubbleSerializer`，把 `BACKWARD_INPUT` 视为 B、
  `BACKWARD_WEIGHT` 视为 W，优先传播 B 并延迟 W。
- `bidirectional`：`BidirectionalPipelineSerializer`，按 DualPipe 的八段本地执行结构合并
  down/up 两条逻辑 pipeline；static/dynamic runner 会先生成镜像双向 PP workload overlay。

默认是 `1f1b`。所有 analyzer 通过 `build_pipeline_serializer()` 使用同一注册表，避免
Default/Puppeteer/Hermod 对同名模式产生不同 compute order。高级模式的实现、输入约束和
阶段计划见 [`advanced-pipeline-schedules-plan.md`](../../extention/advanced-pipeline-schedules-plan.md)。

高级 serializer 不修改 `Task`。它们将 VPP chunk、B/W、pipeline direction、逻辑 stage、
目标 slot 和 DAG 合法化后的 GPU 本地位置写入 `dict[int, PipelineTaskInfo]` sidecar。策略先
生成目标序列，再投影到原始 DAG 的全局拓扑序，所以新增的 per-GPU resource edge 不会成环。

额外参数：

- `vpp` / `--vpp`：仅控制 `interleaved_1f1b` 的虚拟 chunk 数，默认 2；每个 GPU 的局部
  layer 数必须不少于 VPP。
- `interleave_group_size` / `--interleave-group-size`：VPP schedule table 的 micro-batch
  group 大小，默认等于 PP。
- `bidirectional` 要求 PP、GA 均为偶数且 `GA >= 2 * PP`；不满足时显式失败。

`interleaved_1f1b` 不修改通用 builder，而是深拷贝 workload，保留 compute 与 TP/DP/EP
task，只替换原物理 `PP_SEND` 和跨 chunk 的本地直连。overlay 按
`logical_stage = chunk_id * pp + physical_stage` 为每个 microbatch 和 `(dp,ep,tp)` lane
生成 activation/gradient PP 边，包括末尾物理 stage 回绕到开头物理 stage 的 chunk 间边。
每条 PP flow 都由发送端边界层的 compute/TP 输出触发，并作为接收端首个 compute 的依赖。
新 flow 在 Hermod metadata 中按边界 layer 拆分 coflow，避免一个 coflow 混合多个 LID。

这仍是从 AICB 局部操作序列推导出的连续 chunk ownership，不是从 Megatron checkpoint/model
module 读取的真实分片；尚未建模 activation buffer、CUDA stream 和异步 P2P API。因而可以用于
计算/通信 DAG 与网络竞争实验，但不能声称完整复现 Megatron runtime。

`bidirectional` overlay 将前半 GA micro-batch 建成 stage 递增的 down activation 链，将后半建成
stage 递减的 up activation 链，gradient 分别沿完全反向路径传播。每条 flow 依赖发送端边界
compute/TP completion，并解锁接收端边界 compute；Hermod coflow ID 使用 `pdown`/`pup` 审计方向。
两个 module replica 的参数/DP/optimizer task 仍复用现有物理 stage 投影，且 executor 不模拟
DualPipe 的融合 F/B kernel、异步收发 wait 和 buffer 生命周期，因此 overlap、显存与部分 DP 时间仍是
近似值。`zero_bubble` 未实现 optimizer post-validation 和自动 schedule search，模式名表示 ZB1P
风格 compute order，不能据此宣称任意配置实测 bubble 严格为零。

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
  "vpp": 2,
  "interleave_group_size": 2,
  "variant": "conventional_1f1b",
  "modes": ["default", "puppeteer", "hermod"],
  "k_paths": 4,
  "placement": "contiguous",
  "gpus_per_server": 8,
  "visualize": false,
  "output": "outputs/hermod_experiment"
}
```

CLI 参数覆盖配置，例如：

```powershell
python scripts/run_hermod_e2e.py --config scripts/hermod_e2e_config.json `
  --pipeline interleaved_1f1b --vpp 2 --interleave-group-size 2 `
  --modes default hermod
```

`dp` 是实验覆盖值：使用 AICB 的 TP/PP/EP 与指定 DP 重建 rank group，因此会生成 DP
collective；不会修改原始 AICB 文件。所需 GPU 数为 `tp * dp * pp * ep`，不得超过拓扑的
GPU 数。

## 输出与比较口径

输出目录包含：

- `workload.json`：带 Hermod metadata 的 P2P DAG；
- `hermod_metadata.json`：每个范围内 flow 的 MID/LID/coflow 来源 sidecar；
- `pipeline_task_metadata.json`：static 高级 pipeline 模式的 DAG overlay 与 compute schedule sidecar；
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

For contention diagnosis only, both runners accept
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

`HermodScheduleVariant.INTERLEAVED_1F1B` 与 pipeline mode 是两个独立维度：前者改变 §4.1
CType comparator，后者选择真实的 `InterleavedOneFOneBSerializer` compute order。当前已经
实现并测试 VPP compute serializer 与 strategy-specific workload overlay；后者提供推导式 chunk
ownership、chunk P2P 和反向对称边。由于 ownership 并非来自真实模型分片且 runtime buffer/stream
未建模，普通 AICB 结果仍不能表述为完整 Megatron interleaved 复现。

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

更精确的 Interleaved DAG 已在同一 GPT-13B AICB 的 `tp=4,pp=2,dp=1,ga=2,vpp=2`
配置验证：overlay 将 PP flow 扩展为 48 条，static Default 与 dynamic Default/Hermod 均无环、
无 cursor 死锁并完成执行；单 iteration smoke makespan 为 514,621 us。该数值只作为 static/dynamic
DAG 一致性回归信号，不作为吞吐收益结论。相关 golden DAG 测试覆盖 VPP 回绕边、TP 输出到 PP
producer、PP 到接收端 compute consumer、反向对称边、`PP=1/2/4`、`VPP=2/4`、非连续 rank
和动态全局 task-id 分配。

当前实现的结论应限制为 PP/DP 的 §4.1 流级严格优先实验。对于 GPT 风格 AICB，Hermod 专用
adapter 会从 `embedding_layer` 与相邻的 `attention_layer + mlp_layer` 对恢复可审计的局部
Transformer LID，而不是使用扁平 item 位置；但它尚未证明模型 stage ownership 或 VPP chunk
ownership 与真实模型配置一致，因此完整的 Case III 层语义仍待验证。EP 与 §4.2 完成前，不应报告为完整 Hermod
端到端复现。推荐场景、负例及结果解释见 `hermod-experiment-status.md`。

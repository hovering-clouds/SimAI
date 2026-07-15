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
- `summary.json`：输入、并行度、coflow 表、类型计数与 makespan。

`default` 与 `hermod` 共享 BFS 路由和同一 compute execution plan，以隔离 Hermod allocator
的影响。`puppeteer` 使用自己的 Greedy route、TTE 分配及资源协调，因此是相关但不同的
网络调度基线；它不等同于 Hermod，也尚未实现两者的组合策略。

## Dynamic executor

本入口当前使用静态 `AnalyticalExecutor`。`DynamicExecutor` 面向按 job/iteration 的按需展开与
注入；Hermod 尚未实现动态任务的增量 metadata/priority analysis/update-analysis，因此不能
把 dynamic mode 作为这个单 iteration 实验的替代。`HermodSchedulingPolicy.update_analysis()`
会显式拒绝动态注入，避免新任务静默退化为 background 流量。将来支持 dynamic Hermod 前，必须
保证每批注入任务都有 MID/LID/coflow metadata，并能安全更新 policy 的 priority analysis。

## 已验证配置与限制

已验证配置：现有 16-GPU AlibabaHPN 拓扑、GPT-13B AICB `tp=4, pp=2, ga=2`，以 `dp=2`
覆盖运行。该运行生成 4 个 PP coflow 与 16 个 DP coflow，并成功运行 Default、Puppeteer、
Hermod 三种模式；同一配置下 GPipe 与 1F1B 分支也可完成运行。

Hermod priority、allocator、AICB metadata、builder 与 job merger 的定向测试通过（46 tests）。
全量测试中 Spectrum-X 相关的既有测试仍依赖仓库外缺失的旧拓扑文件；这与 Hermod 无关。

当前实现的结论应限制为 PP/DP 的 §4.1 流级严格优先实验。AICB 当前只提供 GA 块内的扁平
item 位置，adapter 将其作为可审计的兼容 LID；它不是已由输入证实的论文逻辑模型层 ID。因此，
Case III 的真实层语义仍需含显式模型层标识的输入验证。EP 与 §4.2 完成前，不应报告为完整
Hermod 端到端复现。

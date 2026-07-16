# Hermod §4.1 最小侵入端到端复现计划（修订版）

> 日期：2026-07-15  
> 范围：论文 *Coflow Scheduling for LLM Training* §4.1；不实现 §4.2 的 matching 或 EP
> 流量分配。  
> 输入：`inputs/aicb-workload/` 与 `inputs/topologies/` 中现有文件。

## 结论与输入审计

本轮目标调整为 **PP/DP 的严格端到端复现**。现有 AICB 训练文件均为 `ep=1`，且当前
模拟器的 EP 路径尚待单独校验，因此 EP 不参与本轮调度、实验或性能结论。不能把 TP
AllReduce 当作 EP coflow。

EP 仅保留显式扩展接口：`HermodCoflowType.EP`、`classify_coflow_type()` 和策略配置
可以识别它，但默认 `ep_mode=reject`；只要真实 workload 出现 `EP_ALLTOALL`，Hermod
入口即报出可定位错误，而不是尝试使用未验证的优先级或带宽分配。待 EP collective、
元数据与输入共同验证后，再以独立阶段启用 `ep_mode=enable`。

现有可直接使用的拓扑只有以下三份，实验仅从中选择，**绝不调用**
`inputs/topologies/generate_cassini_paper_topo.py`：

| 拓扑 | 用途 |
| --- | --- |
| `AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100` | 16-GPU、A100 基线 |
| `Spectrum-X_16g_8gps_400Gbps_H100` | 16-GPU、400Gbps 敏感性实验 |
| `DCN+DualToR_64g_8gps_200Gbps_A100` | 32/64-GPU（按实际 workload world size）实验 |

`AicbHeader.ga` 是 gradient-accumulation 次数。对本仓库的 AICB **训练**输入，且仅在
工作负载的 `ga * vpp` 正常分块检查通过后，可将 builder 产生的 `iteration` 映射为
Hermod MID：它对应同一 training iteration 内的 microbatch 序号。这个映射只能存在于
Hermod 的 AICB 适配层；`Task.iteration` 的通用 IR 语义保持不变。

现有 `Task.layer_id` 是 GA 分块内的扁平 AICB item 序号，不能直接宣称是论文 LID。
适配层应由 AICB item 的 `(item_id, phase, forward/backward 顺序)` 导出独立的
`logical_layer_id`，并以生成的 task/DAG 表核验“较小 LID 的数据更早被消费”。

## 最小侵入设计

不改动事件循环、路由、collective 展开算法、默认/MFS/Puppeteer 策略；所有新增行为仅在
Hermod 路径生效。

```text
AICB + Header
  -> WorkloadBuilder（现有）
  -> HermodAicbMetadataAdapter（新增、纯后处理）
       - 验证 AICB 的 ga/vpp 分块
       - 写入 microbatch_id / logical_layer_id
       - 保留 builder 给出的 coflow_id
  -> HermodAnalyzer / HermodSchedulingPolicy（已有 Hermod 路径）
  -> AnalyticalExecutor（不修改）
```

必要的最小修正仅包括：

1. 新增 `src/workload_generator/hermod_aicb_metadata.py`，对已经构建的 `P2PWorkload`
   原地标注或返回深拷贝；不改 AICB parser 格式，也不重定义通用 `iteration/layer_id`。
2. 扩展 Hermod analysis 的诊断输出：每个 PP/DP coflow 的来源、MID、LID、CType、
   §4.1 case 与排序理由。未知/缺失元数据继续 fail-fast。
3. 让 `HermodAnalyzer` 使用已有的 1F1B compute serializer，而不是默认的 GPipe/C++ 顺序；
   选择 `conventional_1f1b` 或 `interleaved_1f1b` 只影响 Hermod 策略配置。
4. 在 `JobMerger` 复制 `coflow_id`、`microbatch_id`、`logical_layer_id`，并在 task/job
   remap 时为 `coflow_id` 加 job namespace，防止合并重复 iteration 后发生 coflow ID 冲突。
5. 新增一个独立入口 `scripts/run_hermod_e2e.py`。`run_e2e.py` 保持为默认基线入口。

不在 adapter 中：猜测 EP、基于 task ID 猜 MID/LID、改变 DAG、改变路由、修改带宽模型，
或实现 §4.2。

## 分阶段计划与验收

### Phase 0：冻结复现契约与选择输入

1. 从现有目录选择一个 `pp>1`、`ga>1` 的 AICB 文件作为 PP/MID fixture；若同时需要 DP，
   选择 `dp = all_gpus / (tp * pp * ep) > 1` 的文件并使用可覆盖其 GPU 数的现成拓扑。
2. 记录输入 header、`(tp, dp, pp, ep, ga, vpp)`、所用拓扑、schedule variant 和随机性
   （本模拟应无随机性）。
3. 运行构建器但不运行 Hermod，导出 `task_id -> item_id/phase/iteration/layer_id/comm_type`
   审计表，确认 AICB 中实际出现 PP 和 DP 流。若输入只有 `dp=1`，它只能验证 MID/PP。

验收：选定配置与可用拓扑相容；审计表可人工追踪一个 microbatch 的 forward、backward、
PP send 和 DP bucket。

### Phase 1：实现并验证 AICB 元数据适配器

1. 从 AICB header 和 builder 位置索引验证：每个普通 layer item 唯一对应
   `microbatch_id in [0, ga)`；pre/post item 不标为 Hermod MID。
2. 仅为 PP/DP flow 写入 `microbatch_id=iteration`，但仅作为 adapter 的显式、已验证产物；
   若遇到 EP flow，保留原始信息并交由 `ep_mode=reject` 报错。
3. 构造 `logical_layer_id`：以原始模型层/通信消费顺序为基准，为 forward 与 backward
   分别建立单调映射；同一 collective 展开的 flows 必须得到相同 LID。
4. PP flows 的 MID/LID 从其连接的 source/destination layer item 获取，而非仅依赖
   `_generate_pp_flows()` 的硬编码端点。
5. 输出 JSON sidecar（任务 ID、coflow ID、MID、LID、推导来源），并支持从 sidecar
   重放；这既是审计证据，也避免把 Hermod 专用字段扩散到 parser。

验收：所有范围内 PP/DP flow 都有三项元数据；同 coflow 完全一致；缺失/分块不一致/
PP 不能匹配层时清晰报错；EP flow 在默认模式下清晰拒绝。新增单测覆盖 `ga=1`、`ga>1`、
PP 边界、DP bucket 和 JSON round-trip。

### Phase 2：收紧 §4.1 优先级实现

1. 用 adapter 的 `schedule_context` 识别实际的 1F1B 竞争集；不从“当前活跃集合任意含
   DP 与非 DP”就推断 Case III。
2. 本轮显式实现并记录 PP/DP 可达规则：PP 跨 microbatch 的 Case I（MID）、PP/DP
   重叠时的 Case II（PP > DP）、以及跨 iteration PP/DP 的 Case III（LID）。EP 的
   `EP=PP>DP` 与交错 `EP>PP>DP` 仅保留枚举/配置接口和合成单元测试，不进入真实实验。
3. 对论文称为不可达的 `MID+CType`、`CType+LID`、三因子冲突配置实行 strict-mode
   拒绝；报告包含 coflow/task metadata。仅在 `--non-strict` 时使用稳定背景 fallback。
4. 保持严格 coflow 间优先、层内 max-min/fair-share；确认 allocator 不引入 §4.2 matching。

验收：单元测试针对真实 adapter 生成的数据断言每个 case/reason，不仅使用手写 task。

### Phase 3：端到端脚本与比较

`run_hermod_e2e.py` 参数：`--aicb`、`--topo`、`--variant`、`--output`、`--strict`。

单次运行固定生成：

- `workload.json`：标注后的 P2P workload；
- `hermod_metadata.json`：可重放 sidecar 与审计摘要；
- `priority_trace.json`：每次重分配的活跃 coflow、case、顺序、每 flow 带宽；
- `execution_result.json` 与 `summary.json`：makespan、按 coflow 的开始/完成时间。

同一输入至少运行 `DefaultSchedulingPolicy` 和 `HermodSchedulingPolicy`，保持相同路由、
compute order 和拓扑。结果比较仅解释共享链路上的带宽与完成顺序，不将绝对性能拟合宣称为
论文集群复现。

验收：共享链路上可观察到高层优先，链路隔离时低层不受影响；两个策略的 DAG、路径和
任务集完全一致。

### Phase 4：实验矩阵与完整性声明

1. 当前输入：严格完成 PP/MID（`pp>1, ga>1`），并选择 `dp>1` 的文件覆盖 PP/DP
   Case II/III；在 16-GPU AlibabaHPN 与 Spectrum-X，以及适用的 64-GPU DCN+ 上做
   敏感性运行。
2. 合成小型 fixture：PP/DP 规则必须进入 CI；EP 只保留识别、拒绝和未来启用的接口测试，
   不以其结果论证调度效果。
3. 后续 EP 阶段：先单独修复/验证 EP collective 展开、EP MID/LID 标注和 EP 输入，再
   打开 `ep_mode=enable`，补齐 EP/PP/DP Case II 与交错 1F1B 实验。

## 完成定义

本轮完成定义是“PP/DP 严格端到端复现”：真实 AICB 输入产生 PP、DP；MID/LID 由可审计
适配器提供；PP/DP 的 Case I、II、III 在真实 DAG 上验证；使用三份既有拓扑中的适配者
完成 Default-vs-Hermod 对比；EP 默认拒绝且不影响任何结果；§4.2 明确未实现。

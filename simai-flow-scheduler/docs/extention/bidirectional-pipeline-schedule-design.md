# Chimera Bidirectional Pipeline 扩展方案

> 状态：基础 Chimera 隔离实现，2026-07-23 从早期 DualPipe 近似收敛而来。
> 对外兼容名称仍为 `bidirectional`，验证入口仍为
> `scripts/run_e2e_bidirectional.py`。本模式不注册到
> Default、Hermod 或 Puppeteer，也不修改现有实验入口。

## 1. 目标与参考

本扩展以 Li 和 Hoefler 的 Chimera 为目标，而不是 DeepSeek DualPipe。基础模式在同一组
偶数个物理 PP stage 上放置两条完整模型流水线：

```text
down replica: logical 0 -> 1 -> ... -> D-1
up replica:   logical 0 -> 1 -> ... -> D-1
              physical D-1 -> ... -> 1 -> 0
```

偶数个 microbatch 均分给两条 pipeline，每条 pipeline 独立采用同步 1F1B，再把两个本地
序列合并到同一 GPU。Chimera 论文参考：

- https://arxiv.org/abs/2107.06925
- https://github.com/shigangli/chimera

当前实现不包含 DualPipe 的八阶段 schedule、F&B fused overlap、WeightGradStore 或
MoE attention/dispatch/MLP/combine 细粒度重叠。

## 2. 模型副本与放置

物理 stage `p` 上维护两个 stage replica：

```text
module replica 0 -> model shard p
module replica 1 -> model shard D-1-p
```

因此：

```text
down microbatch:
  module_replica_id = 0
  model_shard_id = physical_stage

up microbatch:
  module_replica_id = 1
  model_shard_id = D-1-physical_stage
```

这些字段只保存在 analyzer-owned
`dict[int, BidirectionalPipelineTaskInfo]` sidecar 中，不进入公共 `Task`。

## 3. PP workload DAG

对每个 microbatch、PP logical boundary 及 `(dp, ep, tp)` lane：

```text
down activation: physical s     -> physical s+1
down gradient:   physical s+1   -> physical s

up activation:   physical D-1-s -> physical D-2-s
up gradient:     physical D-2-s -> physical D-1-s
```

每条 activation/gradient 都必须保持：

```text
producer compute/TP completion
  -> PP_SEND(size=header.pp_comm_size)
  -> receiver compute
```

普通 PP 总 payload 不因双向放置而增加：

```text
2 * ga * (pp - 1) * dp * ep * tp * pp_comm_size
```

双向模式改变的是 microbatch 的流向、时序和链路竞争，而不是单个 microbatch 穿越的 stage 数。

## 4. Chimera 梯度同步

两条 pipeline 是同一模型的两个同步 replica。模型 shard `s` 的两个副本位于：

```text
replica 0: physical stage s
replica 1: physical stage D-1-s
```

两边负责的 microbatch 完成本地 W 后，必须在 optimizer 前同步梯度。若外部 DP degree
为 `d`，每个 `(model_shard, ep_lane, tp_lane)` 的 collective 参与者为：

```text
replica 0 的 d 个 DP ranks
+
replica 1 的 d 个 DP ranks
```

当前使用现有 Ring `DP_ALLREDUCE` 展开，不新增公共 `CommType`。collective 的输入依赖是：

- 该 replica 所有 microbatch 的 W completion；
- 如果 AICB 已有逐 microbatch DP communication，则等待其 receiver completion。

所有 post/optimizer compute 等待本 rank 两个本地 stage replica 的 gradient collective
完成。

`gradient_sync_bytes` 的含义是每个 model stage、每个 `(ep,tp)` lane 的梯度输入字节数。
优先由调用者显式传入；未传时只接受 AICB 中语义明确且为正的
`grad_param_comm.dp_comm_size`，不会从 `pp_comm_size` 或计算时长猜测。

## 5. Chimera Serializer

对每个方向分别按其 logical stage 构建普通同步 1F1B：

```text
warmup -> 1F1B steady state -> cooldown
```

同一 physical stage 上：

- 前半 stage 优先 down pipeline；
- 后半 stage 优先 up pipeline；
- 两个方向的 token 交替合并；
- 最终顺序通过完整 workload DAG 的 stable topological legalization。

基础 Chimera 将 backward input 和 weight gradient 作为同一个 `BW` operation 排序；
公共 workload 中仍保留独立 B/W task 和 `B -> W` DAG。

Chimera 支持 `ga < pp`；当前只要求：

- `pp` 为偶数；
- `ga` 为偶数；
- training AICB；
- `pp > 1` 时 `pp_comm_size > 0`；
- 存在明确的 gradient synchronization bytes。

## 6. 修改文件

| 文件 | 内容 |
|---|---|
| `src/workload_generator/bidirectional_pipeline_builder.py` | 双向 PP DAG、replica/shard sidecar、镜像梯度 AllReduce、optimizer barrier |
| `src/static_analysis/passes/pipeline_task_serializers.py` | 两条同步 1F1B 的 Chimera 本地合并 |
| `src/static_analysis/strategies/advanced_pipeline_strategies.py` | 隔离 analyzer 和 sidecar ownership |
| `src/executor/pipeline_job_expander.py` | 动态展开时传递梯度大小并重映射 sidecar task ID |
| `tests/test_bidirectional_pipeline_builder.py` | PP、镜像 replica、同步字节数及 DAG 测试 |
| `tests/test_pipeline_task_serializers.py` | Chimera 1F1B merge 和形状约束 |
| `scripts/run_e2e_bidirectional.py` | 隔离 E2E；兼容旧入口名 |

公共 `Task`、`P2PWorkload`、默认 `WorkloadBuilder`、Default/Hermod/Puppeteer 和通用
executor 不修改。

## 7. 验证标准

### 7.1 PP

- 每个 microbatch 有 `pp-1` activation 和 `pp-1` gradient。
- activation 与 gradient 的端点严格互逆。
- down/up 各占一半 microbatch。
- 每条 PP flow 的大小等于 `pp_comm_size`。
- 非连续 rank 按 `assigned_nodes` 映射。

### 7.2 Replica gradient synchronization

- 每个 model shard 有唯一的镜像 stage pair。
- group 包含两个 pipeline replica 以及全部外部 DP replicas。
- collective 的首轮 flow 等待对应 replica 的所有 W/DP terminal。
- optimizer/post task 等待本 rank 参与的所有同步 flow completion。
- Ring AllReduce 的逐 flow 字节数、总网络字节数和 task 数有黄金断言。

### 7.3 Schedule

- 每个方向单独投影后满足普通 1F1B warmup/steady/cooldown。
- 每个 F/B/W compute 恰好出现一次。
- up pipeline 使用镜像 logical stage。
- 合并 compute resource edge 后 DAG 无环。
- `ga < pp` 的偶数 microbatch 场景可运行。

### 7.4 E2E

- 独立真实 AICB workload 通过 `P2PWorkload.validate()`。
- 所有 compute/PP/DP tasks 完成，无死锁。
- summary 分别报告 PP flow 和 Chimera gradient-sync flow，makespan 仅作为回归信号。

## 8. 已知边界

- 当前是基础、双 pipeline 的 Chimera，不实现论文的多于两条 pipeline 扩展。
- serializer 使用 unit-operation preferred merge，不根据异构 F/B duration 自动搜索最优排程。
- 梯度同步使用 Ring AllReduce；论文中的 eager-sync / eager-sync-opt 时机优化尚未实现。
- 不模拟参数和 activation 的显存占用。
- 既有入口名 `bidirectional` 仅为兼容，不表示 DualPipe。

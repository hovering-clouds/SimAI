# SimAI Scheduler ZeRO/FSDP 扩展说明

本文面向使用和维护 `simai-flow-scheduler` 的开发者，说明如何将 AICB 生成的
DeepSpeed ZeRO workload 转换为计算和通信任务 DAG。当前实现主要覆盖 ZeRO-1/2/3；
其中 ZeRO-3 的参数分片通信模式可作为 FSDP 扩展的基础，但不宣称与任意 FSDP 配置逐项等价。

## 为什么需要专用路径

原有 builder 使用通用训练结构：

```text
forward compute -> backward-input compute -> backward-weight compute -> DP gradient communication
```

它适合普通 DP gradient all-reduce，但不能表示 ZeRO-3/FSDP 的参数通信：参数
all-gather 必须在 forward 或 backward 计算之前完成。若继续走通用路径，通信会被错误接在
weight-gradient 之后，实验结果不可信。

因此实现保留原有路径，并只对识别为 ZeRO 的 AICB item 路由到专用 builder。非 ZeRO
workload 的构建逻辑不变。

## 改动位置

- `src/workload_generator/zero_semantics.py`：按名称分类 ZeRO 行，包括参数/层级
  all-gather、计算、reduce-scatter、grad sync、step、初始化和 GA 边界。
- `src/workload_generator/workload_builder.py`：检测 ZeRO workload，按 GA group 和
  pipeline stage 构建专用 DAG。
- `src/workload_generator/collective_expander.py`：新增 `BROADCAST` 的 P2P 展开。
- `src/workload_format/schema.py`：新增 `CommType.DP_BROADCAST`。
- `tests/test_workload_builder.py`：覆盖参数级、层级、step/post、bucket、GA、初始化及
  PP + ZeRO 的关键依赖。

## DAG 建模方法

### ZeRO-3

对于同一参数或 module，基本因果关系为：

```text
forward all-gather
  -> forward compute
  -> backward all-gather
  -> backward-input compute
  -> backward weight-gradient compute
  -> gradient reduce-scatter
```

第一条 backward all-gather 会锚定到相应 micro-batch 的 forward 尾部，防止它在 forward
之前作为根 flow 启动。后续 backward all-gather 不强制等待前一个 reduce-scatter，因此在
AICB 文本顺序允许时，可以与前一个参数的梯度规约重叠。

参数级 bucket 的 reduce-scatter 会依赖该 bucket 中 **全部** weight-gradient compute，
而不是只依赖最近一个。这避免 bucket 尚有较早梯度未完成时就提前通信。

对于没有独立 weight-gradient compute 的通信条目，builder 使用当前 backward 进度作为
保守依赖，避免 reduce-scatter 成为根 flow。

### ZeRO-1 / ZeRO-2

ZeRO-1/2 的 grad sync 同样接在相关 weight-gradient compute 之后。step 阶段的 overflow、
grad norm、参数 all-gather，以及后续 `cross_entropy*` / `optimizer*` 被串为一个 post
链，并等待所有 GA micro-batch 的终止事件。

### Gradient Accumulation

AICB 的 `zero{stage}_ga_boundary` 是零成本元数据，不创建 scheduler task。builder 用它切分
各 accumulation group，而不是将所有 item 数量平均除以 `ga_num`。这是必要的：参数级
ZeRO-3 中 prefetch 或 bucket flush 会使不同 micro-batch 的条目数量不同。

旧 workload 没有边界行时，builder 仅在每个 GA group 条目数可均分时使用兼容回退；否则会
报错并提示重新生成 workload。

### 初始化 Broadcast

`zero{stage}_init_broadcast_model` 被展开为 DP root 到各非 root rank 的 P2P flow。它用于
可选的模型初始化同步，适用于 ZeRO-1/2/3，且并非 ZeRO 独有。

`FlowGroupResult` 现在有两类索引：

- `receiver_index`：某 rank 接收到数据的 flow，用于数据依赖。
- `completion_index`：某 rank 的 collective 完成事件，用于阶段屏障。

普通 collectives 中两者通常相同；broadcast root 没有接收 flow，但必须等发送结束后才能
进入训练，因此将 root 的发送 flow 仅记录为 `completion_index`。`add_completion()` 正是为
此类“collective 已完成但本 rank 无接收任务”的情况补充完成事件；`add_flow()` 原有的接收
数据语义未改变。

## 使用方法

1. 使用 AICB 生成 DeepSpeed ZeRO 文本 workload。推荐先从层级模式开始检查任务顺序：

```powershell
cd "D:\paper\flow scheduling\SimAI\aicb"
python -m workload_generator.SimAI_training_workload_generator `
  --frame DeepSpeed --stage 3 --simai_deepspeed_granularity layer `
  --gpu_type A100 --world_size 4 --global_batch 4 --micro_batch 1 `
  --num_layers 2 --hidden_size 16 --ffn_hidden_size 64 `
  --num_attention_heads 4 --model_name zero3_layer
```

2. 将 `aicb/results/workload/*.txt` 作为现有 scheduler 静态 workload 构建入口的输入。
   无需新增 JSON schema、executor 或 analyzer 参数；`WorkloadBuilder.build_from_aicb()` 会
   自动检测 `zero*` 名称并进入专用路径。

3. 正常运行既有 static-analysis / executor 工作流。生成的任务仍是标准的 `P2PWorkload`：
   compute task 与由 collective expander 生成的 P2P flow task。


## 注意事项和限制

- 这不是完整的 DeepSpeed 或 PyTorch FSDP runtime 重放。AICB 没有精确预取参数集合、
  live-parameter 阈值和 bucket 成员的 metadata，builder 只能按 item 顺序建模。
- ZeRO-3 层级模式是 unbucketed module approximation，适合通信量、阶段和粗粒度重叠分析；
  若研究 bucket flush、持久化参数或预取窗口，应优先使用参数级模式。
- `--simai_include_non_amp_init` 会加入一次性初始化通信。进行稳态 iteration 对比时不要混入
  该开销；研究启动时间时才启用。
- ZeRO-2 contiguous-gradients 的 `REDUCESCATTER` 是 AICB 输入格式的近似，不能表达实际
  定向 reduce 的 destination-rank 竞争。
- 当前 PP + ZeRO 覆盖基础连接；复杂的 interleaved virtual pipeline、精确 ZeRO prefetch
  与 PP 重叠仍需单独设计和测试。

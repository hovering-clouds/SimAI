# Puppeteer 调度策略解读

本文档用于梳理论文 `Puppeteer: A Network Planner for AI Training Workloads`
中与调度策略直接相关的核心机制，帮助后续判断哪些部分值得在
`simai-flow-scheduler` 中复现，哪些部分可以先做近似。

本文重点不是逐段翻译论文，而是回答三个更实际的问题：

1. Puppeteer 到底在调度什么？
2. 它是怎么做路由、带宽分配和优先级判断的？
3. 哪些机制是论文收益的核心，哪些机制实现成本较高、可以晚点做？

## 1. Puppeteer 的总体思路

Puppeteer 的基本立场是：AI 训练的跨节点通信高度可预测，所以没有必要把
网络当成一个完全在线、完全被动的系统来处理。它选择在训练开始前离线生成
一份网络执行计划，然后在每次 iteration 中重复使用。

这份计划至少包含三类信息：

1. 每条 flow 走哪条路径（route）
2. 每条 flow 在不同阶段使用多大带宽（rate）
3. 哪些 flow 在运行时需要额外同步，避免因为抖动破坏计划

所以 Puppeteer 本质上不是一个简单的带宽分配器，而是一个
"workload-aware network planner"。它把训练 DAG、物理拓扑、collective 展开方式
一起纳入决策，然后输出一份面向 flow 的执行计划。

## 2. 输入假设和建模边界

Puppeteer 的方案建立在几个很强的前提上：

### 2.1 只关注 inter-node 的 scale-out 网络

论文默认 TP 通信尽量留在节点内高带宽域中处理，真正需要规划的是
PP 和 DP 等跨节点通信。这个假设很重要，因为它显著降低了问题规模。

### 2.2 输入必须能展开到 flow 级别

Puppeteer 不直接对 "all-reduce" 这样的 collective 原语做抽象调度，
而是先根据具体 collective 算法把它展开为一组点对点 flow，然后再做规划。

也就是说，它的最小决策单元不是 collective，而是 flow。

### 2.3 workload 是周期性的

论文反复强调 AI 训练 iteration 的重复性。Puppeteer 的离线开销之所以成立，
是因为计划不是只用一次，而是可以在很多 iteration 中复用。

## 3. 核心调度流程

从论文描述看，Puppeteer 的调度过程可以概括为下面四步：

1. 读取 workload DAG，得到 compute/communication 依赖关系
2. 把 collective 展开成 P2P flows
3. 按时间顺序遍历 flow 的 ready/start/completion 事件
4. 在每个 flow 启动时决定其 route 和 rate，并在 flow 完成时释放资源

这里最关键的一点是：Puppeteer 的规划逻辑是事件驱动的，而不是一次性静态求解
所有 flow 的最终状态。论文虽然把它称为离线 planner，但内部做法非常接近
一个离散事件模拟器。

这点和 `simai-flow-scheduler/src/executor/analytical.py` 的整体骨架很像：
都是维护 active flows，在 flow ready / flow completion 时更新网络状态。

## 4. 路由策略：Greedy Least-Active Path Selection

### 4.1 目标

Puppeteer 不追求 ILP 级别的全局最优路由，而是追求：

1. 比 ECMP 更稳定
2. 比 packet spraying 更容易实现
3. 比复杂求解器更快

所以它选择了一个贪心策略。

### 4.2 核心状态

论文提到它维护两类运行时状态：

1. `link -> active_flows`
2. `switch -> bucketed outgoing links by active flow count`

直观地说，就是：

- 知道每条链路现在有多少活跃 flow
- 知道在每个交换机往外走时，哪条出链路当前最空

### 4.3 路由决策逻辑

当一个 flow ready to start 时，Puppeteer 会先判断它属于哪种范围：

1. intra-node
2. intra-pod
3. inter-pod

然后沿着 Clos 层级逐跳选择当前 "least-active" 的出链路。

论文的味道不是"预先为每对 src/dst 固定唯一 shortest path"，而是：

- flow 真正 ready 的那一刻才做路径选择
- 路径选择要参考当时已经活跃的 flows
- 如果某条链路已经更忙，就尽量把新 flow 导向更空的路径

### 4.4 和常见方案的区别

#### 相对 ECMP

ECMP 依赖 hash，属于概率均衡；Puppeteer 是显式、确定性选路。

#### 相对 packet spraying

packet spraying 把负载均衡粒度下沉到 packet，但实现和重排成本高。
Puppeteer 仍然在 flow 级别做决策，但由于 AI workload 的 flow 稀疏，
它认为这种显式规划已经足够接近理想均衡。

### 4.5 这一策略的本质

这其实是一种"当前时刻局部最优"的 route assignment：

- 不解全局最优
- 不提前锁死整轮所有路径
- 在每个 flow 启动时基于当前 active flow 负载做贪心选择

因此它特别适合放进事件驱动执行器里实现。

## 5. 带宽分配策略：不是简单公平，而是按关键性分配

### 5.1 论文反对什么

论文明确认为，单纯的 max-min fairness 对 AI 训练并不够好。

原因不是公平性不对，而是目标函数不对。

普通网络更关心：

- individual flow completion time
- throughput fairness

但 AI 训练更关心：

- 哪些通信真的暴露在关键路径上
- 哪些通信即使晚一点，也不会拖慢 iteration time

所以 Puppeteer 想优化的是 job completion / iteration completion，
不是每条 flow 的 FCT。

### 5.2 论文提出的关键指标：TTE

Puppeteer 用 `Time-to-Exposed (TTE)` 来衡量一条 flow 的"可拖延空间"。

可把它理解为：

> 这条 flow 最多还能被拖多久，才会真正暴露成 compute stall。

TTE 小，说明这条 flow 更关键；
TTE 大，说明这条 flow 还有 slack，可以让路。

### 5.3 TTE 的定义

论文给出的核心式子是：

`TTE(flow) = Start(child) - Finish(flow)`

这里的 `child` 是 flow 的下游依赖节点。更准确地说：

- 先看这个 flow 的某个 dependent child
- child 的启动时间由所有 parent 中最晚完成的那个决定
- 如果当前 flow 正好就是卡住 child 启动的最后一个 parent，那么 TTE = 0
- 如果 child 其实被别的更慢 parent 卡住，那么当前 flow 有正的 slack

所以 TTE 不只是看本 flow 本身，而是看它在 DAG 中对后续计算的实际阻塞程度。

### 5.4 TTE 的计算思路

这里有个很漂亮的小技巧。

严格来说，TTE 依赖 flow finish time；
而 finish time 又依赖带宽分配；
而带宽分配又想由 TTE 决定。

这是个循环依赖。

论文的处理方式是做一个 optimistic pass：

1. 用理想化条件跑一遍 DAG
2. 假设通信都以 line rate 或无限带宽执行
3. 由这次理想执行得到每个节点的 start/finish 时间
4. 再据此估计每条 flow 的 TTE

因此，TTE 不是精确的"真实运行 slack"，而是一个结构化近似：
它揭示的是 workload 结构上的关键路径，而不是拥塞后的真实关键路径。

### 5.5 TTE 驱动的带宽分配

当多条 flow 共享链路时，Puppeteer 不再默认平均分，而是倾向于：

- 优先给 `TTE ~= 0` 的 flow 更多带宽
- 把 `TTE` 很大的 flow 适当延后或限速

论文最强调的例子是 PP 和 DP 冲突：

1. PP 的 P2P transfer 往往直接阻塞下一 pipeline stage 的 forward
2. 早期 stage 的 DP all-reduce 往往并不立刻暴露，因为最终 optimizer
   还要等更晚的 stage

因此 PP flow 往往 TTE 更接近 0，而某些 DP flow 有明显 slack。
Puppeteer 就会让 PP 先走，把 DP 往后压。

### 5.6 这一策略真正改变了什么

它把网络调度从

"谁在共享链路，大家平均分"

变成了

"谁真的会拖慢 iteration，谁优先"

这正是论文中最值得复现的调度思想之一。

## 6. Runtime Uncertainty：为什么论文还要做同步机制

如果只是离线算出某个时间表，然后要求所有 GPU 按时发包，理论上很好看，
但现实里很容易被 compute jitter 打坏。

论文给出的担忧很直接：

- 某个 rank 稍微慢一点
- 另一个 rank 按原计划提前发送
- 原本应该共同 share 某条链路的 flow 在时间上错开
- 瞬时就会破坏 zero-queue 假设

所以论文认为，只做"时间表"不够，还需要一个运行时校验机制。

## 7. 资源依赖 / Handshake Barrier 机制

### 7.1 核心思想

Puppeteer 会找出那些"本来在 workload DAG 中没有直接依赖，但在网络资源上会相遇"
的 peer flows。

对于这类 flow，发送前需要做一个轻量同步：

- 发送方先发一个 handshake
- 对端确认它也到达了计划中的相应状态
- 双方都 ready 后再开始真正的数据传输

论文把这个机制叫作 resource dependency。

### 7.2 它在解决什么问题

它不是为了加快速度，而是为了维持计划的有效性。

也就是说，这个机制的首要目标是：

- 不让快的 sender 提前把链路灌爆
- 不让运行时小抖动演变成真实排队和拥塞

它是一种"同步后再发"的纪律约束，而不是吞吐优化本身。

### 7.3 论文中的实现建议

论文提出两种表达方式：

1. 在计划里记录 peer-flow dependency，运行时由 endpoint 协议执行
2. 在模拟/执行图里直接插入 synthetic dependency edges

第二种对模拟器尤其友好，因为它不一定需要真的实现 NIC/transport 层握手，
只要在 workload DAG 里插入额外 barrier 就能近似表达其效果。

### 7.4 这一机制的重要性

如果目标只是复现论文的"优先级调度策略"，它不是第一优先级；
但如果目标是更认真地接近论文的 zero-queue 叙事，它又非常关键。

简单说：

- `TTE + greedy routing` 决定性能收益的主要方向
- `resource dependency` 决定计划在 runtime jitter 下是否还能站得住

## 8. 论文里的 zero-queue 是怎么成立的

Puppeteer 的 zero-queue 不是靠交换机里更强的拥塞控制来达成的，
而是靠 endpoint 侧三件事共同保证：

1. 路由尽量避开冲突
2. 带宽分配不让链路超载
3. 运行时用 resource dependency 防止时序漂移

因此 zero-queue 的含义不是"链路永远只有一条 flow"，
而是"任何时刻每条链路上的总发送速率都不超过容量，且并发关系是受控的"。

## 9. 哪些是 Puppeteer 最值得复现的核心策略

如果只从论文贡献和实现投入的性价比来看，我认为有三层：

### 第一层：必须抓住的核心

1. flow 级建模
2. 基于运行时 active flow 状态的 greedy route assignment
3. 基于 TTE / slack 的非公平带宽分配

没有这三点，就很难说是在复现 Puppeteer 的主要策略。

### 第二层：强烈建议有

4. 用一次 optimistic pass 计算 flow criticality
5. 区分 PP / DP 等不同通信在 DAG 中的暴露程度
6. 在 flow start / completion 事件上动态重算剩余 flow 带宽

这一层基本决定模拟结果是否能体现论文中的关键路径优先级。

### 第三层：更像论文，但成本更高

7. resource dependency / handshake barrier
8. 显式建模运行时 jitter 对静态计划的破坏
9. 更接近 Clos 分层结构的路径选择，而不是简单 shortest path

这一层做上去以后，方案会明显更像论文完整系统，而不是一个
"Puppeteer-like allocator"。

## 10. 从复现角度看，最容易误解的地方

### 10.1 Puppeteer 不是单纯的 priority queue

它不是给 flow 打一个静态优先级然后排队，而是把：

- DAG 依赖
- 运行时链路占用
- 路径选择
- 带宽重新分配

放在同一个 planner 里联合考虑。

### 10.2 TTE 不是普通 slack 的直接同义词

二者很接近，但论文关心的是 communication 对 exposed compute stall 的影响。
如果实现时直接拿"越早开始越重要"之类的启发式替代，可能会偏离论文精神。

### 10.3 Resource dependency 不是 workload 原生依赖

它是为了维护网络计划而额外引入的依赖，不是模型语义本身要求的依赖。
这一点在设计 workload IR 时尤其要分清。

## 11. 对 `simai-flow-scheduler` 的启发

结合当前项目结构，论文策略和现有模块的关系可以先粗略理解成：

### 11.1 已经比较接近的部分

- `workload_generator/`
  已经能把 collective 展开为 flow，符合 Puppeteer 的 flow-level planning 前提

- `static_analysis/critical_path.py`
  已经在做 DAG timing 分析，是实现 TTE 的天然落点

- `static_analysis/contention_analysis.py`
  已经能看链路共享和时间窗，是做 flow criticality + contention reasoning 的好基础

- `executor/analytical.py`
  已经是事件驱动执行器，天然适合承接 route/rate 规划

### 11.2 当前和论文仍有明显距离的部分

- `routing_hints.py` 现在更像静态 shortest-path cache，
  还不是 flow start 时基于 active load 的在线贪心选路

- `bandwidth.py` 现在默认是 fair share，
  还没有 TTE-aware prioritisation

- 当前 workload / executor 里还没有明确表达 resource dependency /
  handshake barrier 这类"非数据依赖、但为网络计划服务"的同步边

## 12. 一句话总结

如果把 Puppeteer 压缩成一句话，它做的是：

> 先用 workload DAG 找出哪些通信真的会暴露成训练瓶颈，再在每个 flow 启动时
> 基于当前网络状态为它选择路径和带宽，并在必要时加入额外同步，确保这个计划
> 在真实运行中不被小抖动轻易打坏。

对复现来说，最有代表性的不是"zero-queue"这个口号本身，而是下面三件事：

1. `flow-aware greedy routing`
2. `TTE-aware bandwidth allocation`
3. `resource-dependency-based runtime coordination`

其中前两项最适合作为第一阶段复现目标，第三项更适合作为增强项。

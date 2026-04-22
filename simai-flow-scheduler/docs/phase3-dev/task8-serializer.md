# Phase 3 - Task 8 开发记录：Task Serializer（任务序列化器）

## 1. 目标与范围

Task 8 的目标是为每个节点预计算 compute 任务的执行顺序，输出为 JSON 格式的 `ExecutionPlan` 文件。

**包含**：
- `ExecutionPlan` 数据类：每个节点的 compute 排序
- `OrderingStrategy` 接口：排序策略抽象
- `CppReferenceOrdering` 默认实现：复现 C++ 参考执行顺序
- `TaskSerializer` 类：统一序列化入口
- DAG 环检测验证

**不包含**：
- Flow task 调度（由执行器根据 deps 动态决定）
- 执行器实现（Phase 4 的职责）

---

## 2. 交付物

### 2.1 新增/修改文件

```
simai-flow-scheduler/
├── src/static_analysis/
│   ├── __init__.py                # 更新：导出 ExecutionPlan, OrderingStrategy, CppReferenceOrdering, TaskSerializer
│   └── task_serializer.py         # 新增：任务序列化器完整实现
├── tests/
│   └── test_task_serializer.py    # 新增：22 个测试
```

### 2.2 数据结构

```python
@dataclass
class ExecutionPlan:
    version: str = "1.0"
    source_workload: str                       # 原始 P2PWorkload 文件路径
    compute_order: dict[int, list[int]]        # 节点 ID → compute task_id 有序列表
```

### 2.3 API

```python
# 主入口
class TaskSerializer:
    def __init__(self, topology: NetworkTopology, strategy: Optional[OrderingStrategy] = None)
    def serialize(self, workload: P2PWorkload, analysis: Optional[WorkloadAnalysisResult] = None) -> ExecutionPlan
    def validate(self, workload: P2PWorkload, compute_order: dict[int, list[int]]) -> list[str]
    def to_json(self, plan: ExecutionPlan, path: str) -> None
    @staticmethod
    def from_json(path: str) -> ExecutionPlan

# 排序策略接口
class OrderingStrategy(ABC):
    def order(self, workload: P2PWorkload, analysis: Optional[WorkloadAnalysisResult] = None) -> dict[int, list[int]]

# 默认实现
class CppReferenceOrdering(OrderingStrategy):
    def order(self, workload, analysis=None) -> dict[int, list[int]]
```

---

## 3. 设计原理

### 3.1 为什么需要 Task Serializer

GPU 训练代码中计算任务的执行顺序是固定的，无法在运行时动态调整。因此需要预计算每个节点上 compute 任务的执行顺序。

**计算与通信的并行模型**：
- **Compute 管道**：每个节点严格按序列执行，Serializer 决定顺序
- **Flow 管道**：只要 deps 满足就立即发出，多条通信可同时在网络中传输
- Flow task 的调度由执行器在运行时根据网络竞争状态动态决定

### 3.2 CppReferenceOrdering 排序规则

默认排序严格复现 C++ 参考实现的执行顺序（参见 [cpp-execution-order.md](../specs/cpp-execution-order.md)）：

排序优先级（从高到低）：
1. **iteration**（GA 序号）：先执行较早的 GA
2. **phase**（阶段）：
   - FORWARD (0)
   - BACKWARD_INPUT (1)
   - BACKWARD_WEIGHT (2)
   - OPTIMIZER (3)
3. **layer_id**：按顺序
4. **item_id**：按顺序

### 3.3 DAG 环检测验证

`validate()` 方法验证 compute_order 不会在原始 DAG 中产生循环依赖：
- 对每个节点的相邻 compute 任务添加隐式依赖边
- 与原始 DAG 的 deps 边合并后检测环
- 如果会形成环，返回错误信息

### 3.4 JSON 序列化

- `to_json()`：将 ExecutionPlan 写入 JSON 文件
- `from_json()`：从 JSON 文件读取 ExecutionPlan

---

## 4. 与设计文档的偏差

### 4.1 与计划一致

数据结构设计、API 接口、排序规则均与 [phase3-extend.md](phase3-extend.md) 计划文档一致。

---

## 5. 测试结果

```
tests/test_task_serializer.py: 22 passed
tests/ (完整回归): 372 passed
```

### 测试覆盖范围

| 测试类 | 数量 | 覆盖内容 |
|--------|------|----------|
| `TestExecutionPlan` | 2 | 默认值、自定义值 |
| `TestCppReferenceOrdering` | 9 | 空workload、单task、按iteration/phase/layer_id/item_id排序、多节点、跳过flow tasks |
| `TestTaskSerializer` | 10 | 空/单task/多task、验证通过/检测环、JSON序列化、多节点独立、Optimizer相位、Backward倒序、复杂workload |
| `TestCustomOrderingStrategy` | 1 | 自定义策略 |

---

## 6. 后续依赖

Task 8 的输出 (`ExecutionPlan`) 将被以下模块使用：

1. **Phase 4（Scheduler/Executor）**：
   - 读取 ExecutionPlan 获取每个节点的 compute 顺序
   - Flow task 的调度根据原始 workload 的 deps 动态决定

---

*开发时间：2026-04-22*
*测试状态：22 passed（task serializer）+ 350 passed（原有）= 372 total*
*关键设计：C++参考顺序排序、DAG环检测验证、计算与通信并行模型*
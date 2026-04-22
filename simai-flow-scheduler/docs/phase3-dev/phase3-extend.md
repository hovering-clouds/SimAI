# Phase 3 扩展: Task Serializer

## 目标

确定 P2PWorkload 中每个节点上 compute 任务的执行顺序，输出为 JSON 格式的 `ExecutionPlan` 文件。Flow task 的调度不在本阶段处理，由执行器根据 deps 动态决定发出时机。

## 设计理由

- GPU 训练代码中计算任务的执行顺序是固定的，无法在运行时动态调整
- **通信任务与计算任务可以并行**：GPU 上 SM（计算单元）和 NIC/NVLink（通信单元）是独立硬件，通信操作交由通信库异步处理
- 因此 Serializer **只决定计算任务的顺序**，通信任务在 deps 满足后即发出，多条通信可同时在网络中传输
- 不重复定义新的 task 数据结构，完全沿用原始 P2PWorkload 的 task 定义和 deps，Serializer 仅输出 compute 排序这一增量信息

## 计算与通信的并行模型

```
GPU 节点上的两类管道（独立并行）：

Compute 管道（严格串行，Serializer 控制）:
  [fwd_compute_0] → [fwd_compute_1] → [ig_compute_1] → ...
       ↓                 ↓                  ↓
Comm 管道（异步发出，多条可同时在网）:
  flow_0           flow_1             flow_1(ig)
  (deps: comp_0)   (deps: comp_1)     (deps: ig_comp_1)
       ↕                 ↕                  ↕
  ───────── 可以同时在网络中传输（带宽共享）──────────
```

- **Compute 任务**：每个节点严格按序列执行，Serializer 决定顺序
- **Flow 任务**：只要 deps 满足就立即发出，多条通信可同时在网络中传输
- **Flow 任务的调度**由执行器在运行时根据网络竞争状态动态决定

## ExecutionPlan 数据结构

Serializer 不重复定义 task 信息，仅输出每个节点上 compute task 的执行顺序：

```python
@dataclass
class ExecutionPlan:
    """每个节点的计算任务执行顺序。"""
    version: str
    source_workload: str                       # 原始 P2PWorkload 文件路径
    compute_order: dict[int, list[int]]        # 节点 ID → compute task_id 有序列表
```

## JSON 输出示例

```json
{
  "version": "1.0",
  "source_workload": "workload.json",
  "compute_order": {
    "0": [0, 6, 18, 24],
    "1": [3, 9, 21, 27],
    "2": [1, 7, 19, 25],
    "3": [4, 10, 22, 28]
  }
}
```

**执行器使用方式**：
- 读取原始 P2PWorkload（task 定义 + deps）
- 读取 ExecutionPlan（compute 排序）
- Flow task 的调度完全由原始 workload 的 deps 驱动，不在 ExecutionPlan 中体现

## 默认排序规则

默认排序严格复现 C++ 参考实现（`iterate_hybrid_parallel_Transformer()`）的执行顺序。具体执行顺序参见 [cpp-execution-order.md](../specs/cpp-execution-order.md)。

排序策略可通过接口自定义，但默认实现必须与 C++ 参考实现一致，以确保 Python 生成结果的可验证性。

## TaskSerializer 类设计

将排序逻辑、环检测验证、JSON 读写统一放在一个类中：

```python
class TaskSerializer:
    """计算任务序列化器：确定每个节点上 compute 任务的执行顺序。"""

    def __init__(
        self,
        topology: NetworkTopology,
        strategy: Optional[OrderingStrategy] = None,
    ):
        """
        Args:
            topology: 网络拓扑（用于路由查询等）
            strategy: 排序策略，默认为 C++ 参考实现的执行顺序
        """
        self.topology = topology
        self.strategy = strategy or CppReferenceOrdering()

    def serialize(
        self,
        workload: P2PWorkload,
        analysis: Optional[WorkloadAnalysisResult] = None,
    ) -> ExecutionPlan:
        """
        为 workload 生成 ExecutionPlan。

        Args:
            workload: 原始 P2P Workload
            analysis: Phase 3 静态分析结果（可选，供排序策略参考）

        Returns:
            ExecutionPlan（包含每个节点的 compute 排序）

        Raises:
            ValueError: 如果排序结果在 DAG 中引入环
        """
        compute_order = self.strategy.order(
            workload=workload,
            analysis=analysis,
        )
        errors = self.validate(workload, compute_order)
        if errors:
            raise ValueError(f"ExecutionPlan validation failed: {errors}")
        return ExecutionPlan(
            version="1.0",
            source_workload="",
            topology_file="",
            num_nodes=workload.meta.num_nodes,
            compute_order=compute_order,
        )

    def validate(
        self,
        workload: P2PWorkload,
        compute_order: dict[int, list[int]],
    ) -> list[str]:
        """
        验证 compute_order 不会在原始 DAG 中产生循环依赖。

        对每个节点的 compute_order，将相邻 compute 任务之间
        添加隐式依赖边，与原始 DAG 的 deps 边合并后检测环。

        Args:
            workload: 原始 P2PWorkload
            compute_order: 节点 ID → compute task_id 有序列表

        Returns:
            错误列表，空列表表示验证通过
        """
        ...

    def to_json(self, plan: ExecutionPlan, path: str) -> None:
        """将 ExecutionPlan 写入 JSON 文件。"""
        ...

    @staticmethod
    def from_json(path: str) -> ExecutionPlan:
        """从 JSON 文件读取 ExecutionPlan。"""
        ...
```

## 排序策略接口

策略接口接收完整上下文（workload + topology + 静态分析结果），综合产生执行序列：

```python
class OrderingStrategy(ABC):
    """计算任务排序策略接口。"""

    @abstractmethod
    def order(
        self,
        workload: P2PWorkload,
        analysis: Optional[WorkloadAnalysisResult] = None,
    ) -> dict[int, list[int]]:
        """
        为每个节点生成 compute task 的执行顺序。

        Args:
            workload: 完整的 P2P Workload
            analysis: Phase 3 静态分析结果（可选）

        Returns:
            节点 ID → compute task_id 有序列表
        """
        pass
```

**默认实现 `CppReferenceOrdering`**：严格复现 C++ 参考实现的执行顺序。

## 文件结构

```
src/
  static_analysis/
    ...                   # Phase 3 已有模块
    task_serializer.py   # ExecutionPlan 数据结构 + TaskSerializer 类 + OrderingStrategy + 默认实现
```

## 验收标准

- 输出的 ExecutionPlan 满足原始 workload 的所有 deps 约束
- DAG 环检测通过：排序后的隐式依赖边不与原始 deps 冲突
- 每个节点的 compute 任务列表是合法的线性序列（满足 deps 偏序）
- JSON 文件格式可被两种执行后端解析

---

## 相关文档

- [phase4-plan.md](../phase4-dev/phase4-plan.md) — Phase 4 开发规划
- [cpp-execution-order.md](../specs/cpp-execution-order.md) — C++ 参考实现执行顺序分析

---

*文档创建日期：2026-04-22*

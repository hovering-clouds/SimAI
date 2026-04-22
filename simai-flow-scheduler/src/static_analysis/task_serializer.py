"""
Task Serializer - 确定每个节点上 compute 任务的执行顺序。

输出 ExecutionPlan（JSON 格式），包含每个节点的 compute 排序。
Flow task 的调度由执行器根据 deps 动态决定，不在本模块处理。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
from typing import Optional

from src.static_analysis.analyzer import WorkloadAnalysisResult

from src.workload_format.schema import P2PWorkload, Task, TaskType, Phase


@dataclass
class ExecutionPlan:
    """
    每个节点的计算任务执行顺序。

    Serializer 不重复定义 task 信息，仅输出每个节点上
    compute task 的执行顺序。
    """
    version: str = "1.0"
    compute_order: dict[int, list[int]] = field(default_factory=dict)

    def to_json(self, path: str) -> None:
        """将 ExecutionPlan 写入 JSON 文件。"""
        output = {
            "version": self.version,
            "compute_order": {
                str(node_id): task_ids
                for node_id, task_ids in self.compute_order.items()
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "ExecutionPlan":
        """从 JSON 文件读取 ExecutionPlan。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            version=data.get("version", "1.0"),
            compute_order={
                int(node_id): task_ids
                for node_id, task_ids in data.get("compute_order", {}).items()
            },
        )


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


class CppReferenceOrdering(OrderingStrategy):
    """
    默认排序：严格复现 C++ 参考实现的执行顺序。

    根据 cpp-execution-order.md 中的分析：
    - Forward Pass: layer[0] → layer[N-1]（正序）
    - Backward Pass: layer[N-1] → layer[0]（倒序）
    - WG 通信（DP AllReduce）以 Non-Blocking 方式发起，与下一 GA 的 Forward Pass 并行

    排序优先级（从高到低）：
    1. iteration（GA 序号）：先执行较早的 GA
    2. phase（阶段）：
       - FORWARD
       - BACKWARD_INPUT
       - BACKWARD_WEIGHT
       - OPTIMIZER
    3. layer_id：Forward 正序，Backward 倒序
    4. item_id：按顺序
    """

    def order(
        self,
        workload: P2PWorkload,
        analysis: Optional[WorkloadAnalysisResult] = None,
    ) -> dict[int, list[int]]:
        """
        按 C++ 参考实现的顺序生成 compute 任务排序。

        排序规则：
        - 首先按 iteration 排序（GA 序号）
        - 同一 iteration 内，按 phase 排序：FORWARD → BACKWARD_INPUT → BACKWARD_WEIGHT → OPTIMIZER
        - 同一 phase 内：
          - FORWARD：layer_id 正序
          - BACKWARD_INPUT / BACKWARD_WEIGHT：layer_id 倒序
          - OPTIMIZER：按 layer_id
        - 同一 layer 内，按 item_id 排序
        """
        # 按节点分组 compute 任务
        compute_by_node: dict[int, list[Task]] = {}

        for task in workload.tasks:
            if task.is_compute():
                node_id = task.node
                if node_id not in compute_by_node:
                    compute_by_node[node_id] = []
                compute_by_node[node_id].append(task)

        # 对每个节点的 compute 任务排序
        result: dict[int, list[int]] = {}
        for node_id, tasks in compute_by_node.items():
            sorted_tasks = sorted(
                tasks,
                key=self._compute_sort_key,
            )
            result[node_id] = [task.task_id for task in sorted_tasks]

        return result

    def _compute_sort_key(self, task: Task) -> tuple:
        """
        生成排序键。

        排序规则：
        - iteration：主要排序键
        - phase：次要排序键
        - layer_id：第三排序键，Forward 正序，Backward 倒序
        - item_id：第四排序键

        phase 顺序：FORWARD(0) → BACKWARD_INPUT(1) → BACKWARD_WEIGHT(2) → OPTIMIZER(3)
        """
        phase_order = {
            Phase.FORWARD: 0,
            Phase.BACKWARD_INPUT: 1,
            Phase.BACKWARD_WEIGHT: 2,
            Phase.OPTIMIZER: 3,
        }

        # Backward 阶段需要倒序排列 layer
        # 使用 -layer_id 实现倒序，正序用 layer_id
        if task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT):
            layer_sort = -task.layer_id
        else:
            layer_sort = task.layer_id

        return (
            task.iteration,
            phase_order.get(task.phase, 999),
            layer_sort,
            task.item_id,
        )


class TaskSerializer:
    """
    计算任务序列化器：确定每个节点上 compute 任务的执行顺序。

    Flow task 的调度不在本阶段处理，由执行器根据 deps 动态决定发出时机。
    """

    def __init__(self, strategy: Optional[OrderingStrategy] = None):
        """
        Args:
            strategy: 排序策略，默认为 C++ 参考实现的执行顺序
        """
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

        # 验证排序结果
        errors = self.validate(workload, compute_order)
        if errors:
            raise ValueError(f"ExecutionPlan validation failed: {errors}")

        return ExecutionPlan(compute_order=compute_order)

    def validate(
        self,
        workload: P2PWorkload,
        compute_order: dict[int, list[int]],
    ) -> list[str]:
        """
        验证 compute_order 不会在原始 DAG 中产生循环依赖。

        构建合并图（原始 deps 边 + compute_order 隐式顺序边），
        用 Kahn 算法（拓扑排序）一次性检测环。

        Args:
            workload: 原始 P2PWorkload
            compute_order: 节点 ID → compute task_id 有序列表

        Returns:
            错误列表，空列表表示验证通过
        """
        # 收集所有 task_id
        all_task_ids = {task.task_id for task in workload.tasks}

        # 构建合并图的后继列表：A → {B, C, ...} 表示 A 完成后 B、C 才能开始
        successors: dict[int, set[int]] = {tid: set() for tid in all_task_ids}

        # 添加原始 deps 边：task.deps 中的每个 dep 必须在 task 之前完成
        for task in workload.tasks:
            for dep in task.deps:
                successors[dep].add(task.task_id)

        # 添加 compute_order 隐式顺序边：task[i] 必须在 task[i+1] 之前完成
        for task_ids in compute_order.values():
            for i in range(len(task_ids) - 1):
                successors[task_ids[i]].add(task_ids[i + 1])

        # Kahn 算法：拓扑排序，若无法处理所有节点则存在环
        in_degree = {tid: 0 for tid in all_task_ids}
        for succs in successors.values():
            for succ in succs:
                in_degree[succ] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        processed = 0

        while queue:
            current = queue.pop()
            processed += 1
            for succ in successors[current]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        if processed < len(all_task_ids):
            cycle_tasks = sorted(tid for tid, deg in in_degree.items() if deg > 0)
            return [
                f"Cycle detected in combined dependency graph. "
                f"Tasks involved: {cycle_tasks}"
            ]

        return []


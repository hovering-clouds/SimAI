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
    默认排序：复现 C++ 参考实现的执行顺序。

    C++ iterate_hybrid_parallel_Transformer_fwd_in_bckwd() 的执行模式
    （TOTAL_PASS=1，所有 item 一遍）：
      Forward:  所有 item 的 fwd，从 index 0 到 SIZE-1
      Backward: 所有 item 的 ig/wg，从 index SIZE-1 到 0

    AICB 文件中 item 按顺序排列为 pre → GA[0] layers → GA[1] layers → post，
    因此 C++ 的实际执行顺序等价于：
      Forward:  GA[0] fwd → GA[1] fwd（GA 升序）
      Backward: GA[1] ig/wg → GA[0] ig/wg（GA 降序，逐层 ig→wg 交替）

    排序优先级（从高到低）：
    1. direction: FORWARD(0) vs BACKWARD(1) — 保证所有 fwd 在 bwd 之前
    2. iteration: Forward 阶段 GA 升序，Backward 阶段 GA 降序
    3. layer_id: Forward 正序，Backward 倒序
    4. sub_phase: ig(0) → wg(1) — backward 逐层 ig→wg 交替
    5. item_id

    注：pre items（iteration=-1）和 post items（iteration=ga）也参与排序，
    通过 ga_sort 自然地排在正确位置（pre fwd 最先、pre bkwd 最后，反之亦然）。
    """

    def order(
        self,
        workload: P2PWorkload,
        analysis: Optional[WorkloadAnalysisResult] = None,
    ) -> dict[int, list[int]]:
        """
        按 C++ 参考实现的顺序生成 compute 任务排序。

        包含所有 compute 任务（pre/GA/post），排序规则见类 docstring。
        """
        # 按节点分组所有 compute 任务（含 pre/GA/post）
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
        生成排序键，复现 C++ 参考实现的执行顺序。

        C++ backward 逐层执行 ig→wg（见 docs/specs/cpp-execution-order.md）：
          [layer N-1] ig → wg
          [layer N-2] ig → wg
          ...

        排序规则：
        1. direction：FORWARD(0) → BACKWARD(1)
        2. iteration：Forward 正序（GA 升序），Backward 倒序（GA 降序）
        3. layer_id：Forward 正序，Backward 倒序
        4. sub_phase：ig(0) → wg(1)，仅对 backward 有意义
        5. item_id：兜底
        """
        # FORWARD vs BACKWARD 分组（不再按 ig/wg 分开）
        if task.phase == Phase.FORWARD:
            direction = 0
        elif task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT):
            direction = 1
        else:
            direction = 2

        is_backward = task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT)

        # GA ordering: forward ascending, backward descending
        ga_sort = -task.iteration if is_backward else task.iteration
        # Layer ordering: forward ascending, backward descending
        layer_sort = -task.layer_id if is_backward else task.layer_id

        # Within same layer: ig before wg
        if task.phase == Phase.BACKWARD_INPUT:
            sub_phase = 0
        elif task.phase == Phase.BACKWARD_WEIGHT:
            sub_phase = 1
        else:
            sub_phase = 0

        return (
            direction,
            ga_sort,
            layer_sort,
            sub_phase,
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


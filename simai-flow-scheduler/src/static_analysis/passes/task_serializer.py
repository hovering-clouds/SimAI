"""
Task Serializer - 确定每个节点上 compute 任务的执行顺序。

输出 ExecutionPlan（JSON 格式），包含每个节点的 compute 排序。
Flow task 的调度由执行器根据 deps 动态决定，不在本模块处理。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json

from ...workload_format.schema import P2PWorkload, Task, Phase


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


class TaskSerializer(ABC):
    """计算任务序列化器基类。

    子类实现 serialize() 方法以提供不同的 compute 任务排序策略。
    """

    @abstractmethod
    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        """
        为 workload 生成 ExecutionPlan。

        Args:
            workload: 原始 P2P Workload

        Returns:
            ExecutionPlan（包含每个节点的 compute 排序）

        Raises:
            ValueError: 如果排序结果在 DAG 中引入环
        """
        ...

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
        all_task_ids = {task.task_id for task in workload.tasks}

        successors: dict[int, set[int]] = {tid: set() for tid in all_task_ids}

        for task in workload.tasks:
            for dep in task.deps:
                successors[dep].add(task.task_id)

        for task_ids in compute_order.values():
            for i in range(len(task_ids) - 1):
                successors[task_ids[i]].add(task_ids[i + 1])

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


class CppReferenceSerializer(TaskSerializer):
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

    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        """
        按 C++ 参考实现的顺序生成 compute 任务排序。

        包含所有 compute 任务（pre/GA/post），排序规则见类 docstring。
        """
        compute_by_node: dict[int, list[Task]] = {}

        for task in workload.tasks:
            if task.is_compute():
                node_id = task.node
                if node_id not in compute_by_node:
                    compute_by_node[node_id] = []
                compute_by_node[node_id].append(task)

        result: dict[int, list[int]] = {}
        for node_id, tasks in compute_by_node.items():
            sorted_tasks = sorted(
                tasks,
                key=self._compute_sort_key,
            )
            result[node_id] = [task.task_id for task in sorted_tasks]

        errors = self.validate(workload, result)
        if errors:
            raise ValueError(f"ExecutionPlan validation failed: {errors}")

        return ExecutionPlan(compute_order=result)

    def _compute_sort_key(self, task: Task) -> tuple:
        """
        生成排序键，复现 C++ 参考实现的执行顺序。

        C++ backward 逐层执行 ig→wg（见 docs/specs/cpp-execution-order.md）：
          [layer N-1] ig → wg
          [layer N-2] ig → wg
          ...

        排序规则：
        1. direction：FORWARD(0) → BACKWARD(1) → INFERENCE(2)
        2. iteration：Forward 正序（GA 升序），Backward 倒序（GA 降序）
        3. layer_id：Forward 正序，Backward 倒序
        4. sub_phase：ig(0) → wg(1)，仅对 backward 有意义
        5. item_id：兜底

        推理任务（PREFILL/DECODE）：按 task_id 排序，保留 expander 创建顺序
        （即 batch 内按层顺序，batch 间按 depends_on 顺序），避免跨 batch
        的层交错与 depends_on 依赖产生环。
        """
        if task.phase == Phase.FORWARD:
            direction = 0
        elif task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT):
            direction = 1
        else:
            direction = 2

        if task.phase in (Phase.PREFILL, Phase.DECODE):
            return (direction, task.task_id)

        is_backward = task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT)

        ga_sort = -task.iteration if is_backward else task.iteration
        layer_sort = -task.layer_id if is_backward else task.layer_id

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


class OneFOneBSerializer(TaskSerializer):
    """1F1B 排序：每个 GA 步的 backward 紧随其 forward，GA 步间交替。

    对 Stage k (0-indexed)，warmup = pp - 1 - k：
        1F1B 序列 = [F0 ... F_{warmup-1}]  → warmup
                    [B0 F_{warmup} B1 F_{warmup+1} ...]  → alternating
                    [B_{ga-warmup} ... B_{ga-1}]  → cooldown

    排序优先级（从高到低）：
    1. pre/post：pre 最前，post 最后
    2. 1F1B 逻辑位置：forward i → i, backward i → warmup + i（经交替展开后）
    3. layer_id：Forward 正序，Backward 倒序
    4. sub_phase：ig(0) → wg(1)
    5. item_id：兜底
    """

    def __init__(self, pp: int, node_to_stage: dict[int, int]):
        """
        Args:
            pp: 流水线并行度。
            node_to_stage: 每个 node → stage_id 的映射。
                从 Job.assigned_nodes + Job.parallelism 推导：
                stage_id = node_idx // (dp * ep * tp)
        """
        self.pp = pp
        self._node_to_stage = node_to_stage
        # 缓存 per-stage 的 position map
        self._position_cache: dict[int, dict[tuple, int]] = {}

    def serialize(self, workload: P2PWorkload) -> ExecutionPlan:
        """按 1F1B 顺序生成 compute 任务排序。"""
        ga = self._infer_ga(workload)

        compute_by_node: dict[int, list[Task]] = {}
        for task in workload.tasks:
            if task.is_compute():
                node_id = task.node
                if node_id not in compute_by_node:
                    compute_by_node[node_id] = []
                compute_by_node[node_id].append(task)

        result: dict[int, list[int]] = {}
        for node_id, tasks in compute_by_node.items():
            stage_id = self._node_to_stage.get(node_id, 0)
            position_map = self._get_position_map(ga, stage_id)

            sorted_tasks = sorted(
                tasks,
                key=lambda t: self._compute_sort_key(t, position_map, ga),
            )
            result[node_id] = [task.task_id for task in sorted_tasks]

        errors = self.validate(workload, result)
        if errors:
            raise ValueError(f"ExecutionPlan validation failed: {errors}")

        return ExecutionPlan(compute_order=result)

    def _infer_ga(self, workload: P2PWorkload) -> int:
        """从 workload 中推断 GA 步数。

        GA 步的 iteration 范围为 0..ga-1，post items 的 iteration = ga，
        因此 max(iteration) = ga。
        """
        return max(
            (t.iteration for t in workload.tasks if t.iteration >= 0),
            default=0,
        )

    def _get_position_map(
        self, ga: int, stage_id: int
    ) -> dict[tuple[str, int], int]:
        """构建 (type, iteration) → 1F1B 时间轴位置的映射。

        type: 'F' = forward, 'B' = backward_input/backward_weight
        """
        cache_key = (ga, stage_id)
        if cache_key in self._position_cache:
            return self._position_cache[cache_key]

        warmup = max(0, self.pp - 1 - stage_id)
        seq: list[tuple[str, int]] = []

        # Warmup phase: forwards only (up to warmup microbatches)
        for i in range(min(warmup, ga)):
            seq.append(('F', i))

        # Alternating steady state: B_i, F_{warmup+i}
        for i in range(ga - warmup):
            seq.append(('B', i))
            if warmup + i < ga:
                seq.append(('F', warmup + i))

        # Cooldown: remaining backwards not yet placed
        placed_b = {iter_ for typ, iter_ in seq if typ == 'B'}
        for i in range(ga):
            if i not in placed_b:
                seq.append(('B', i))

        result = {key: idx for idx, key in enumerate(seq)}
        self._position_cache[cache_key] = result
        return result

    def _compute_sort_key(
        self,
        task: Task,
        position_map: dict[tuple[str, int], int],
        ga: int,
    ) -> tuple:
        """生成 1F1B 排序键。"""
        # 推理任务按 task_id 保留 expander 创建顺序
        if task.phase in (Phase.PREFILL, Phase.DECODE):
            return (2, task.task_id)

        # Pre items: 排在最前
        if task.iteration == -1:
            return (-1000, task.layer_id, 0, task.item_id)

        # Post items: 排在最后
        if task.iteration >= ga:
            return (1000 + task.layer_id, 0, 0, task.item_id)

        # GA items
        is_backward = task.phase in (Phase.BACKWARD_INPUT, Phase.BACKWARD_WEIGHT)
        typ = 'B' if is_backward else 'F'
        pos = position_map.get((typ, task.iteration), 999999)

        if is_backward:
            layer_sort = -task.layer_id
            sub_phase = 0 if task.phase == Phase.BACKWARD_INPUT else 1
        else:
            layer_sort = task.layer_id
            sub_phase = 0

        return (0, pos, layer_sort, sub_phase, task.item_id)

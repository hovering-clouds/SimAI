"""
Job policy interface — plugin-style scheduling decisions at the Job level.

Follows the same pattern as the existing SchedulingPolicy ABC.
Users implement JobPolicy to customise Job-level scheduling behaviour.
"""

from abc import ABC, abstractmethod

from ..workload_format.schema import Job
from ..workload_format.compact_workload import SimulationState


class JobPolicy(ABC):
    """Job 级调度策略的统一接口。

    用户通过实现此类来自定义 Job 展开与调度的各个方面。
    """

    @abstractmethod
    def can_emit(self, job: Job, sim_state: SimulationState) -> bool:
        """此 Job 是否可以展开并注入 executor。

        返回 False 表示暂缓，下一轮重新检查。
        """

    @abstractmethod
    def get_placement(self, job: Job, sim_state: SimulationState) -> list[int]:
        """决定此 Job 分配到哪些物理节点。

        返回 job.assigned_nodes 表示使用默认值。
        """

    @abstractmethod
    def get_delay_us(self, job: Job, sim_state: SimulationState) -> int:
        """为此 Job 的 entry tasks 施加额外启动延迟（微秒）。

        返回 0 表示无延迟。
        """

    @abstractmethod
    def order_expansion(self, eligible_jobs: list[Job],
                        sim_state: SimulationState) -> list[Job]:
        """对多个 eligible jobs 排序，决定展开优先级。

        返回排序后的列表。
        """


class FifoJobPolicy(JobPolicy):
    """FIFO 默认策略：按 Job ID 顺序展开，不做任何调整。"""

    def can_emit(self, job: Job, sim_state: SimulationState) -> bool:
        return True

    def get_placement(self, job: Job, sim_state: SimulationState) -> list[int]:
        return job.assigned_nodes

    def get_delay_us(self, job: Job, sim_state: SimulationState) -> int:
        return 0

    def order_expansion(self, eligible_jobs: list[Job],
                        sim_state: SimulationState) -> list[Job]:
        return sorted(eligible_jobs, key=lambda j: j.job_id)


class DelayByJobPolicy(JobPolicy):
    """按 job_id 为不同 Job 附加启动延迟的装饰策略。

    将任意 JobPolicy 包装一层，在 get_delay_us() 中从预计算的
    job_id → delay_us 映射查表返回延迟值。

    Args:
        delay_by_job: job_id → 启动延迟(us) 的映射。缺失的 job_id 返回 0。
        base: 被包装的 JobPolicy（默认 FifoJobPolicy）。
    """

    def __init__(
        self,
        delay_by_job: dict[int, int],
        base: JobPolicy | None = None,
    ):
        self._base = base if base is not None else FifoJobPolicy()
        self._delay_by_job = delay_by_job

    def can_emit(self, job: Job, sim_state: SimulationState) -> bool:
        return self._base.can_emit(job, sim_state)

    def get_placement(self, job: Job, sim_state: SimulationState) -> list[int]:
        return self._base.get_placement(job, sim_state)

    def get_delay_us(self, job: Job, sim_state: SimulationState) -> int:
        return self._delay_by_job.get(job.job_id, 0)

    def order_expansion(
        self, eligible_jobs: list[Job], sim_state: SimulationState
    ) -> list[Job]:
        return self._base.order_expansion(eligible_jobs, sim_state)

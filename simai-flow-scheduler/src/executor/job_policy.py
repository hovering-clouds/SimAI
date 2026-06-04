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

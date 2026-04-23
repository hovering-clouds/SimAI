"""Execution result data structures."""
from dataclasses import dataclass


@dataclass
class TaskTiming:
    """单个 task 的时间信息。"""
    task_id: int
    node: int          # compute 任务的节点；flow 任务用 src
    task_type: str     # "compute" | "flow"
    start_time_us: int
    end_time_us: int


@dataclass
class ExecutionResult:
    """执行结果。"""
    per_task: dict[int, TaskTiming]
    job_iteration_times: dict[int, int]   # job_id → 单次 iteration 时间
    total_time_us: int                    # 所有 task 的最晚 end_time
    makespan_us: int                      # max(end_time) - min(start_time)

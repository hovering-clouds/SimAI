"""Execution result data structures."""
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskTiming:
    """单个 task 的时间信息。"""
    task_id: int
    node: int          # compute 任务的节点；flow 任务用 src
    task_type: str     # "compute" | "flow"
    start_time_us: int
    end_time_us: int

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "node": self.node,
            "task_type": self.task_type,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TaskTiming":
        return cls(**data)


@dataclass
class ExecutionResult:
    """执行结果。"""
    per_task: dict[int, TaskTiming]
    job_iteration_times: dict[int, int]   # job_id → 单次 iteration 时间
    total_time_us: int                    # 所有 task 的最晚 end_time
    makespan_us: int                      # max(end_time) - min(start_time)

    def to_json(self, path: str | Path) -> None:
        data = {
            "per_task": {
                str(tid): timing.to_dict()
                for tid, timing in self.per_task.items()
            },
            "job_iteration_times": self.job_iteration_times,
            "total_time_us": self.total_time_us,
            "makespan_us": self.makespan_us,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | Path) -> "ExecutionResult":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            per_task={
                int(tid): TaskTiming.from_dict(td)
                for tid, td in data["per_task"].items()
            },
            job_iteration_times={int(k): v for k, v in data["job_iteration_times"].items()},
            total_time_us=data["total_time_us"],
            makespan_us=data["makespan_us"],
        )

"""
Compact Workload - Job DAG level workload representation for dynamic expansion.

Mirrors the P2PWorkload structure but replaces the full task list with
per-job expansion metadata and dependency information, enabling on-demand
task expansion during simulation.
"""

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from .schema import Meta, Network, Job, Task


# ── Per-job expansion metadata ──────────────────────────────────────────────


@dataclass
class JobExpansionInfo:
    """单个 Job 的展开信息（不含现有 Job 类已有字段）。

    在 CompactWorkload 中按 job_id 索引，与现有 Job 类配合使用。
    """
    depends_on: list[int] = field(default_factory=list)
    job_type: str = "inference"              # "inference" | "training"
    trace_src: str = ""                      # trace 源文件路径
    trace_job_index: int = 0                 # 在 trace 中的序号（batch_idx / iter_idx）


# ── Compact workload ────────────────────────────────────────────────────────


@dataclass
class CompactWorkload:
    """紧凑 Workload — 镜像 P2PWorkload，用 Job DAG 替代全量 tasks。

    对比 P2PWorkload:
      version, meta, network, jobs: list[Job]   ← 完全相同
      tasks: list[Task]                          ← 删掉
      job_expansion_info                         ← 新增（展开元数据 + 依赖关系）
    """

    version: str
    meta: Meta
    network: Optional[Network] = None
    jobs: list[Job] = field(default_factory=list)

    # 按 job_id → JobExpansionInfo（替代 P2PWorkload 的全量 tasks）
    job_expansion_info: dict[int, JobExpansionInfo] = field(default_factory=dict)

    # ── JSON 序列化 ─────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """序列化为 JSON 可序列化 dict。"""
        meta_dict = {"num_jobs": self.meta.num_jobs, "num_nodes": self.meta.num_nodes}
        network_dict = {"topology_file": self.network.topology_file} if self.network else None

        jobs_list = []
        for j in self.jobs:
            jd = {"job_id": j.job_id, "assigned_nodes": j.assigned_nodes}
            jd["parallelism"] = {
                "tp": j.parallelism.tp, "dp": j.parallelism.dp,
                "pp": j.parallelism.pp, "ep": j.parallelism.ep,
            }
            if j.name is not None:
                jd["name"] = j.name
            if j.model is not None:
                jd["model"] = j.model
            jobs_list.append(jd)

        info_dict = {}
        for jid, info in self.job_expansion_info.items():
            info_dict[str(jid)] = {
                "depends_on": info.depends_on,
                "job_type": info.job_type,
                "trace_src": info.trace_src,
                "trace_job_index": info.trace_job_index,
            }

        return {
            "version": self.version,
            "meta": meta_dict,
            "network": network_dict,
            "jobs": jobs_list,
            "job_expansion_info": info_dict,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CompactWorkload":
        """从 dict 反序列化。"""
        meta = Meta(**data.get("meta", {}))
        network = Network(**data["network"]) if data.get("network") else None

        jobs = []
        for jd in data.get("jobs", []):
            p = jd.get("parallelism", {})
            j = Job(
                job_id=jd.get("job_id", 0),
                name=jd.get("name"),
                model=jd.get("model"),
                assigned_nodes=jd.get("assigned_nodes", []),
                parallelism=p if isinstance(p, dict) else {},
            )
            jobs.append(j)

        info = {}
        for jid_str, v in data.get("job_expansion_info", {}).items():
            info[int(jid_str)] = JobExpansionInfo(**v)

        return cls(
            version=data.get("version", "1.0"),
            meta=meta,
            network=network,
            jobs=jobs,
            job_expansion_info=info,
        )

    def to_json(self, path: str):
        """写入 JSON 文件。"""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "CompactWorkload":
        """从 JSON 文件加载。"""
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ── Job DAG runtime index (not serialized) ──────────────────────────────────


class JobDAG:
    """从 CompactWorkload.jobs 构建的运行时 DAG 索引。

    纯数据结构，不入序列化格式。
    """

    def __init__(
        self,
        jobs: dict[int, Job],
        dependents: dict[int, list[int]],
        dep_count: dict[int, int],
        root_jobs: list[int],
    ):
        self.jobs = jobs
        self.dependents = dependents
        self.dep_count = dep_count
        self.root_jobs = root_jobs

    @classmethod
    def from_compact(cls, wl: CompactWorkload) -> "JobDAG":
        """从 CompactWorkload 构建 JobDAG。"""
        jobs: dict[int, Job] = {j.job_id: j for j in wl.jobs}
        dependents: dict[int, list[int]] = defaultdict(list)
        dep_count: dict[int, int] = {}

        for jid, info in wl.job_expansion_info.items():
            dep_count[jid] = len(info.depends_on)
            for d in info.depends_on:
                dependents[d].append(jid)

        root_jobs = [jid for jid, cnt in dep_count.items() if cnt == 0]

        return cls(
            jobs=jobs,
            dependents=dict(dependents),
            dep_count=dep_count,
            root_jobs=root_jobs,
        )

    def mark_completed(self, job_id: int) -> None:
        """标记 Job 完成，将依赖此 Job 的 dep_count 减 1。为了正确性，每个job_id只能调用一次。"""
        for dep_id in self.dependents.get(job_id, []):
            self.dep_count[dep_id] -= 1

    def get_eligible_jobs(self) -> list[int]:
        """返回所有 dep_count == 0 的 job_ids。"""
        return [jid for jid, cnt in self.dep_count.items() if cnt == 0]

    def get_job_info(self, wl: CompactWorkload, job_id: int) -> JobExpansionInfo:
        """从 CompactWorkload 中获取指定 job 的 expansion info。"""
        return wl.job_expansion_info[job_id]


# ── Auxiliary types ─────────────────────────────────────────────────────────


@dataclass
class ExpandedJob:
    """单个 Job 的展开结果 — JobManager 提取后即可丢弃。"""
    job_id: int
    tasks: list[Task]
    entry_task_ids: list[int]
    terminal_task_ids: list[int]


@dataclass
class SimulationState:
    """传递给 JobPolicy 的模拟器状态快照。"""
    current_time_us: int
    completed_job_ids: set[int] = field(default_factory=set)


@dataclass
class SlicerConfig:
    """Job 切片配置。"""
    granularity: str = "auto"           # "batch" | "wave" | "iteration" | "custom"
    wave_size: Optional[int] = None


class TaskIdAllocator:
    """全局单调递增的 task_id 分配器。"""

    def __init__(self, start: int = 0):
        self._next_id = start

    def allocate(self, count: int) -> tuple[int, int]:
        """分配 count 个连续 ID，返回 (start_id, end_id)。"""
        start = self._next_id
        self._next_id += count
        return start, self._next_id - 1

    @property
    def next_id(self) -> int:
        return self._next_id

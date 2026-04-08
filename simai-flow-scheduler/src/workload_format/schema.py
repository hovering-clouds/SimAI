"""
JSON Schema definitions for P2P Workload format.

This module defines the dataclasses and JSON Schema for the P2P Workload IR.
See docs/WORKLOAD_FORMAT.md for detailed specification.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class TaskType(str, Enum):
    """Task type: compute or flow."""
    COMPUTE = "compute"
    FLOW = "flow"


class Phase(str, Enum):
    """Training phase."""
    FORWARD = "forward"
    BACKWARD_INPUT = "backward_input"
    BACKWARD_WEIGHT = "backward_weight"
    OPTIMIZER = "optimizer"


class CommType(str, Enum):
    """Communication type for flow tasks."""
    # Tensor Parallel
    TP_ALLREDUCE_RING = "tp_allreduce_ring"
    TP_ALLREDUCE_TREE = "tp_allreduce_tree"
    TP_ALLGATHER_RING = "tp_allgather_ring"
    TP_ALLGATHER_TREE = "tp_allgather_tree"
    TP_REDUCESCATTER_RING = "tp_reducescatter_ring"
    TP_REDUCESCATTER_TREE = "tp_reducescatter_tree"
    TP_ALLTOALL = "tp_alltoall"
    # Data Parallel
    DP_ALLREDUCE = "dp_allreduce"
    DP_ALLGATHER = "dp_allgather"
    DP_REDUCESCATTER = "dp_reducescatter"
    DP_ALLTOALL = "dp_alltoall"
    # Expert Parallel
    EP_ALLTOALL = "ep_alltoall"
    # Pipeline Parallel
    PP_SEND = "pp_send"
    PP_RECV = "pp_recv"
    # Unknown/unspecified
    UNKNOWN = "unknown"


@dataclass
class Meta:
    """Metadata for the workload file."""
    num_jobs: int
    num_nodes: int
    generated_at: Optional[str] = None
    generator_version: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Network:
    """Network topology reference."""
    topology_file: str
    bandwidth_gbps: Optional[float] = None
    latency_us: Optional[float] = None


@dataclass
class ParallelismConfig:
    """Parallelism configuration for a job."""
    tp: int = 1
    dp: int = 1
    pp: int = 1
    ep: int = 1


@dataclass
class Job:
    """Job configuration."""
    job_id: int
    name: Optional[str] = None
    model: Optional[str] = None
    assigned_nodes: list[int] = field(default_factory=list)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)

    def __post_init__(self):
        if self.parallelism is None:
            self.parallelism = ParallelismConfig()
        if isinstance(self.parallelism, dict):
            self.parallelism = ParallelismConfig(**self.parallelism)


@dataclass
class Task:
    """
    Task definition - either compute or flow.

    For compute tasks:
    - node: the node executing the computation
    - duration_us: estimated compute duration

    For flow tasks:
    - src: source node rank
    - dst: destination node rank
    - size_bytes: amount of data to transfer
    - chunk_id, num_chunks: for collective decomposition
    """
    task_id: int
    job_id: int
    type: TaskType
    iteration: int = 0
    phase: Phase = Phase.FORWARD
    layer_id: int = 0
    deps: list[int] = field(default_factory=list)

    # Compute-specific fields
    node: Optional[int] = None
    duration_us: Optional[int] = None

    # Flow-specific fields
    src: Optional[int] = None
    dst: Optional[int] = None
    size_bytes: Optional[int] = None
    comm_type: CommType = CommType.UNKNOWN
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None

    def __post_init__(self):
        if isinstance(self.type, str):
            self.type = TaskType(self.type)
        if isinstance(self.phase, str):
            self.phase = Phase(self.phase)
        if isinstance(self.comm_type, str):
            self.comm_type = CommType(self.comm_type)

    def is_compute(self) -> bool:
        """Check if this is a compute task."""
        return self.type == TaskType.COMPUTE

    def is_flow(self) -> bool:
        """Check if this is a flow task."""
        return self.type == TaskType.FLOW

    def validate(self) -> list[str]:
        """Validate task fields based on type."""
        errors = []

        if self.is_compute():
            if self.node is None:
                errors.append(f"Task {self.task_id}: compute task must have 'node' field")
            if self.duration_us is None:
                errors.append(f"Task {self.task_id}: compute task must have 'duration_us' field")
            if self.src is not None or self.dst is not None or self.size_bytes is not None:
                errors.append(f"Task {self.task_id}: compute task should not have flow fields (src, dst, size_bytes)")
        else:
            if self.src is None:
                errors.append(f"Task {self.task_id}: flow task must have 'src' field")
            if self.dst is None:
                errors.append(f"Task {self.task_id}: flow task must have 'dst' field")
            if self.size_bytes is None:
                errors.append(f"Task {self.task_id}: flow task must have 'size_bytes' field")
            if self.node is not None or self.duration_us is not None:
                errors.append(f"Task {self.task_id}: flow task should not have compute fields (node, duration_us)")

        return errors


@dataclass
class P2PWorkload:
    """
    P2P Workload - the intermediate representation for flow scheduling.

    This is the main entry point for workload serialization/deserialization.
    """
    version: str
    meta: Meta
    network: Optional[Network] = None
    jobs: list[Job] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def __post_init__(self):
        if self.meta is None:
            self.meta = Meta(num_jobs=len(self.jobs), num_nodes=0)
        if self.network is None:
            self.network = Network(topology_file="")
        if isinstance(self.meta, dict):
            self.meta = Meta(**self.meta)
        if isinstance(self.network, dict):
            self.network = Network(**self.network)

    def get_tasks_by_job(self, job_id: int) -> list[Task]:
        """Get all tasks for a specific job."""
        return [t for t in self.tasks if t.job_id == job_id]

    def get_flow_tasks(self) -> list[Task]:
        """Get all flow tasks."""
        return [t for t in self.tasks if t.is_flow()]

    def get_compute_tasks(self) -> list[Task]:
        """Get all compute tasks."""
        return [t for t in self.tasks if t.is_compute()]

    def validate(self) -> list[str]:
        """Validate the entire workload."""
        errors = []

        # Validate meta
        if self.meta.num_jobs != len(self.jobs):
            errors.append(f"meta.num_jobs ({self.meta.num_jobs}) != len(jobs) ({len(self.jobs)})")

        # Validate task IDs are unique
        task_ids = [t.task_id for t in self.tasks]
        if len(task_ids) != len(set(task_ids)):
            errors.append("Task IDs are not unique")

        # Validate each task
        for task in self.tasks:
            errors.extend(task.validate())

        # Validate DAG integrity (no cycles, deps exist)
        errors.extend(self._validate_dag())

        return errors

    def _validate_dag(self) -> list[str]:
        """Validate DAG integrity - no cycles and all deps exist."""
        errors = []
        task_id_set = set(t.task_id for t in self.tasks)

        for task in self.tasks:
            for dep in task.deps:
                if dep not in task_id_set:
                    errors.append(f"Task {task.task_id}: dependency {dep} does not exist")

        # Cycle detection using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(task_id: int) -> bool:
            visited.add(task_id)
            rec_stack.add(task_id)

            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if task:
                for dep in task.deps:
                    if dep not in visited:
                        if has_cycle(dep):
                            return True
                    elif dep in rec_stack:
                        return True

            rec_stack.remove(task_id)
            return False

        for task in self.tasks:
            if task.task_id not in visited:
                if has_cycle(task.task_id):
                    errors.append("DAG contains cycles")
                    break

        return errors


# JSON Schema for validation
P2P_WORKLOAD_JSON_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "$id": "https://example.com/p2p-workload.schema.json",
    "title": "P2P Workload",
    "description": "Point-to-Point Workload format for AI training flow scheduling",
    "type": "object",
    "required": ["version", "meta", "tasks"],
    "properties": {
        "version": {
            "type": "string",
            "description": "Schema version"
        },
        "meta": {
            "type": "object",
            "required": ["num_jobs", "num_nodes"],
            "properties": {
                "num_jobs": {"type": "integer", "minimum": 1},
                "num_nodes": {"type": "integer", "minimum": 1},
                "generated_at": {"type": "string"},
                "generator_version": {"type": "string"},
                "description": {"type": "string"}
            }
        },
        "network": {
            "type": "object",
            "required": ["topology_file"],
            "properties": {
                "topology_file": {"type": "string"},
                "bandwidth_gbps": {"type": "number"},
                "latency_us": {"type": "number"}
            }
        },
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["job_id"],
                "properties": {
                    "job_id": {"type": "integer"},
                    "name": {"type": "string"},
                    "model": {"type": "string"},
                    "assigned_nodes": {
                        "type": "array",
                        "items": {"type": "integer"}
                    },
                    "parallelism": {
                        "type": "object",
                        "properties": {
                            "tp": {"type": "integer", "minimum": 1},
                            "dp": {"type": "integer", "minimum": 1},
                            "pp": {"type": "integer", "minimum": 1},
                            "ep": {"type": "integer", "minimum": 1}
                        }
                    }
                }
            }
        },
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["task_id", "job_id", "type"],
                "properties": {
                    "task_id": {"type": "integer"},
                    "job_id": {"type": "integer"},
                    "type": {"type": "string", "enum": ["compute", "flow"]},
                    "iteration": {"type": "integer"},
                    "phase": {
                        "type": "string",
                        "enum": ["forward", "backward_input", "backward_weight", "optimizer"]
                    },
                    "layer_id": {"type": "integer"},
                    "node": {"type": "integer"},
                    "duration_us": {"type": "integer"},
                    "src": {"type": "integer"},
                    "dst": {"type": "integer"},
                    "size_bytes": {"type": "integer"},
                    "comm_type": {"type": "string"},
                    "chunk_id": {"type": "integer"},
                    "num_chunks": {"type": "integer"},
                    "deps": {
                        "type": "array",
                        "items": {"type": "integer"}
                    }
                }
            }
        }
    }
}

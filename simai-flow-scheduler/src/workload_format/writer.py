"""
Workload reader/writer - Serialize and deserialize P2P Workload files.
"""

import json
from pathlib import Path
from typing import Optional, Union

from .schema import P2PWorkload, Meta, Network, Job, ParallelismConfig, Task, TaskType, Phase, CommType


class WorkloadWriter:
    """
    Writer for P2P Workload files.
    """

    def __init__(self, indent: int = 2):
        """
        Initialize writer.

        Args:
            indent: JSON indentation level. Default is 2.
        """
        self.indent = indent

    def write(self, workload: P2PWorkload, file_path: Union[str, Path]) -> None:
        """
        Write workload to a JSON file.

        Args:
            workload: P2PWorkload to write.
            file_path: Output file path.
        """
        data = self._serialize(workload)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=self.indent)

    def write_string(self, workload: P2PWorkload) -> str:
        """
        Serialize workload to JSON string.

        Args:
            workload: P2PWorkload to serialize.

        Returns:
            JSON string.
        """
        data = self._serialize(workload)
        return json.dumps(data, indent=self.indent)

    def _serialize(self, workload: P2PWorkload) -> dict:
        """
        Serialize P2PWorkload to JSON-serializable dict.

        Only includes non-None optional fields to keep output clean.

        Args:
            workload: P2PWorkload to serialize.

        Returns:
            JSON-serializable dictionary.
        """
        # Build meta - only include non-None optional fields
        meta_dict = {
            "num_jobs": workload.meta.num_jobs,
            "num_nodes": workload.meta.num_nodes,
        }
        if workload.meta.generated_at is not None:
            meta_dict["generated_at"] = workload.meta.generated_at
        if workload.meta.generator_version is not None:
            meta_dict["generator_version"] = workload.meta.generator_version
        if workload.meta.description is not None:
            meta_dict["description"] = workload.meta.description

        # Build network - only include non-None optional fields
        network_dict = {
            "topology_file": workload.network.topology_file,
        }
        if workload.network.bandwidth_gbps is not None:
            network_dict["bandwidth_gbps"] = workload.network.bandwidth_gbps
        if workload.network.latency_us is not None:
            network_dict["latency_us"] = workload.network.latency_us

        return {
            "version": workload.version,
            "meta": meta_dict,
            "network": network_dict,
            "jobs": [self._serialize_job(job) for job in workload.jobs],
            "tasks": [self._serialize_task(task) for task in workload.tasks],
        }

    def _serialize_job(self, job) -> dict:
        """Serialize a Job object."""
        result = {
            "job_id": job.job_id,
            "assigned_nodes": job.assigned_nodes,
        }
        if job.name is not None:
            result["name"] = job.name
        if job.model is not None:
            result["model"] = job.model

        # Only include parallelism if non-default values
        parallelism = {
            "tp": job.parallelism.tp,
            "dp": job.parallelism.dp,
            "pp": job.parallelism.pp,
            "ep": job.parallelism.ep,
        }
        # Only include parallelism if any value is not 1
        if any(v != 1 for v in parallelism.values()):
            result["parallelism"] = parallelism

        return result

    def _serialize_task(self, task) -> dict:
        """Serialize a Task object."""
        result = {
            "task_id": task.task_id,
            "job_id": task.job_id,
            "type": task.type.value,
            "iteration": task.iteration,
            "phase": task.phase.value,
            "layer_id": task.layer_id,
            "item_id": task.item_id,
            "deps": task.deps,
        }

        if task.is_compute():
            result["node"] = task.node
            result["duration_us"] = task.duration_us
        else:
            result["src"] = task.src
            result["dst"] = task.dst
            result["size_bytes"] = task.size_bytes
            result["comm_type"] = task.comm_type.value
            result["chunk_id"] = task.chunk_id
            result["num_chunks"] = task.num_chunks

        return result


class WorkloadReader:
    """
    Reader for P2P Workload files.
    """

    def __init__(self):
        """Initialize reader."""
        pass

    def read(self, file_path: Union[str, Path]) -> P2PWorkload:
        """
        Read workload from a JSON file.

        Args:
            file_path: Input file path.

        Returns:
            P2PWorkload object.
        """
        with open(file_path, 'r') as f:
            data = json.load(f)
        return self._deserialize(data)

    def read_string(self, json_string: str) -> P2PWorkload:
        """
        Deserialize workload from JSON string.

        Args:
            json_string: JSON string to parse.

        Returns:
            P2PWorkload object.
        """
        data = json.loads(json_string)
        return self._deserialize(data)

    def _deserialize(self, data: dict) -> P2PWorkload:
        """
        Deserialize JSON data to P2PWorkload object.

        Args:
            data: Parsed JSON data.

        Returns:
            P2PWorkload object.
        """
        # Parse meta
        meta_data = data.get("meta", {})
        meta = Meta(
            num_jobs=meta_data.get("num_jobs", 0),
            num_nodes=meta_data.get("num_nodes", 0),
            generated_at=meta_data.get("generated_at"),
            generator_version=meta_data.get("generator_version"),
            description=meta_data.get("description"),
        )

        # Parse network
        network_data = data.get("network")
        if network_data:
            network = Network(
                topology_file=network_data.get("topology_file", ""),
                bandwidth_gbps=network_data.get("bandwidth_gbps"),
                latency_us=network_data.get("latency_us"),
            )
        else:
            network = Network(topology_file="")

        # Parse jobs
        jobs = []
        for job_data in data.get("jobs", []):
            parallelism_data = job_data.get("parallelism", {})
            parallelism = ParallelismConfig(
                tp=parallelism_data.get("tp", 1),
                dp=parallelism_data.get("dp", 1),
                pp=parallelism_data.get("pp", 1),
                ep=parallelism_data.get("ep", 1),
            )
            job = Job(
                job_id=job_data.get("job_id", 0),
                name=job_data.get("name"),
                model=job_data.get("model"),
                assigned_nodes=job_data.get("assigned_nodes", []),
                parallelism=parallelism,
            )
            jobs.append(job)

        # Parse tasks
        tasks = []
        for task_data in data.get("tasks", []):
            task = Task(
                task_id=task_data.get("task_id", 0),
                job_id=task_data.get("job_id", 0),
                type=task_data.get("type", "compute"),
                iteration=task_data.get("iteration", 0),
                phase=task_data.get("phase", "forward"),
                layer_id=task_data.get("layer_id", 0),
                item_id=task_data.get("item_id", 0),
                deps=task_data.get("deps", []),
                node=task_data.get("node"),
                duration_us=task_data.get("duration_us"),
                src=task_data.get("src"),
                dst=task_data.get("dst"),
                size_bytes=task_data.get("size_bytes"),
                comm_type=task_data.get("comm_type", "unknown"),
                chunk_id=task_data.get("chunk_id"),
                num_chunks=task_data.get("num_chunks"),
            )
            tasks.append(task)

        return P2PWorkload(
            version=data.get("version", "1.0"),
            meta=meta,
            network=network,
            jobs=jobs,
            tasks=tasks,
        )

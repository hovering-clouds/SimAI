"""
Workload reader/writer - Serialize and deserialize P2P Workload files.
"""

import json
from pathlib import Path
from typing import Optional, Union

from .schema import P2PWorkload


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

        Args:
            workload: P2PWorkload to serialize.

        Returns:
            JSON-serializable dictionary.
        """
        return {
            "version": workload.version,
            "meta": {
                "num_jobs": workload.meta.num_jobs,
                "num_nodes": workload.meta.num_nodes,
                "generated_at": workload.meta.generated_at,
                "generator_version": workload.meta.generator_version,
                "description": workload.meta.description,
            },
            "network": {
                "topology_file": workload.network.topology_file,
                "bandwidth_gbps": workload.network.bandwidth_gbps,
                "latency_us": workload.network.latency_us,
            } if workload.network else None,
            "jobs": [self._serialize_job(job) for job in workload.jobs],
            "tasks": [self._serialize_task(task) for task in workload.tasks],
        }

    def _serialize_job(self, job) -> dict:
        """Serialize a Job object."""
        return {
            "job_id": job.job_id,
            "name": job.name,
            "model": job.model,
            "assigned_nodes": job.assigned_nodes,
            "parallelism": {
                "tp": job.parallelism.tp,
                "dp": job.parallelism.dp,
                "pp": job.parallelism.pp,
                "ep": job.parallelism.ep,
            },
        }

    def _serialize_task(self, task) -> dict:
        """Serialize a Task object."""
        result = {
            "task_id": task.task_id,
            "job_id": task.job_id,
            "type": task.type.value,
            "iteration": task.iteration,
            "phase": task.phase.value,
            "layer_id": task.layer_id,
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
        return P2PWorkload(
            version=data.get("version", "1.0"),
            meta=data.get("meta", {}),
            network=data.get("network"),
            jobs=data.get("jobs", []),
            tasks=data.get("tasks", [])
        )

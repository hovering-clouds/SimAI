"""
Workload validator - validates P2P workload files against the schema.
"""

import json
from typing import Optional
from jsonschema import validate, ValidationError, Draft7Validator

from .schema import P2PWorkload, P2P_WORKLOAD_JSON_SCHEMA, Meta, Network, Job, ParallelismConfig, Task


class WorkloadValidator:
    """
    Validator for P2P Workload files.

    Performs two levels of validation:
    1. JSON Schema validation - checks structure and required fields
    2. Semantic validation - checks DAG integrity, node ranges, etc.
    """

    def __init__(self, schema: Optional[dict] = None):
        """
        Initialize validator with optional custom schema.

        Args:
            schema: Optional JSON Schema to use. Defaults to P2P_WORKLOAD_JSON_SCHEMA.
        """
        self.schema = schema or P2P_WORKLOAD_JSON_SCHEMA

    def validate_json(self, data: dict) -> list[str]:
        """
        Validate JSON data against schema.

        Args:
            data: Parsed JSON data to validate.

        Returns:
            List of validation error messages. Empty if valid.
        """
        errors = []

        validator = Draft7Validator(self.schema)
        for error in validator.iter_errors(data):
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"Schema validation error at '{path}': {error.message}")

        return errors

    def validate_workload(self, workload: P2PWorkload) -> list[str]:
        """
        Validate a P2PWorkload object.

        Args:
            workload: P2PWorkload to validate.

        Returns:
            List of validation error messages. Empty if valid.
        """
        return workload.validate()

    def validate_file(self, file_path: str) -> tuple[bool, list[str], Optional[P2PWorkload]]:
        """
        Validate a workload file.

        Args:
            file_path: Path to the JSON file to validate.

        Returns:
            Tuple of (is_valid, errors, workload_or_none)
        """
        errors = []
        workload = None

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            return False, [f"File not found: {file_path}"], None
        except json.JSONDecodeError as e:
            return False, [f"Invalid JSON: {e}"], None

        # Schema validation
        schema_errors = self.validate_json(data)
        if schema_errors:
            return False, schema_errors, None

        # Deserialize to P2PWorkload
        try:
            workload = self._deserialize(data)
        except Exception as e:
            errors.append(f"Deserialization error: {e}")
            return False, errors, None

        # Semantic validation
        semantic_errors = workload.validate()
        if semantic_errors:
            return False, semantic_errors, workload

        return True, [], workload

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
                coflow_id=task_data.get("coflow_id"),
                microbatch_id=task_data.get("microbatch_id"),
                logical_layer_id=task_data.get("logical_layer_id"),
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

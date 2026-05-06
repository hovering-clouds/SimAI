"""
Tests for P2P Workload schema and serialization.
"""

import pytest
from src.workload_format.schema import (
    P2PWorkload,
    Task,
    Job,
    Meta,
    TaskType,
    Phase,
    CommType,
    ParallelismConfig,
    P2P_WORKLOAD_JSON_SCHEMA,
)
from src.workload_format.writer import WorkloadWriter, WorkloadReader
from src.workload_format.validator import WorkloadValidator


class TestTask:
    """Tests for Task dataclass."""

    def test_create_compute_task(self):
        """Test creating a compute task."""
        task = Task(
            task_id=0,
            job_id=0,
            type=TaskType.COMPUTE,
            node=2,
            duration_us=1500,
        )
        assert task.is_compute()
        assert not task.is_flow()
        assert task.node == 2
        assert task.duration_us == 1500

    def test_create_flow_task(self):
        """Test creating a flow task."""
        task = Task(
            task_id=1,
            job_id=0,
            type=TaskType.FLOW,
            src=2,
            dst=3,
            size_bytes=134217728,
            comm_type=CommType.TP_ALLREDUCE_RING,
            chunk_id=0,
            num_chunks=8,
        )
        assert task.is_flow()
        assert not task.is_compute()
        assert task.src == 2
        assert task.dst == 3
        assert task.size_bytes == 134217728

    def test_compute_task_validation(self):
        """Test compute task field validation."""
        task = Task(
            task_id=0,
            job_id=0,
            type=TaskType.COMPUTE,
            node=2,
            duration_us=1500,
        )
        errors = task.validate()
        assert len(errors) == 0

    def test_compute_task_missing_fields(self):
        """Test compute task validation fails without required fields."""
        task = Task(
            task_id=0,
            job_id=0,
            type=TaskType.COMPUTE,
        )
        errors = task.validate()
        assert any("node" in e for e in errors)
        assert any("duration_us" in e for e in errors)

    def test_flow_task_validation(self):
        """Test flow task field validation."""
        task = Task(
            task_id=1,
            job_id=0,
            type=TaskType.FLOW,
            src=2,
            dst=3,
            size_bytes=1024,
        )
        errors = task.validate()
        assert len(errors) == 0

    def test_flow_task_missing_fields(self):
        """Test flow task validation fails without required fields."""
        task = Task(
            task_id=1,
            job_id=0,
            type=TaskType.FLOW,
        )
        errors = task.validate()
        assert any("src" in e for e in errors)
        assert any("dst" in e for e in errors)
        assert any("size_bytes" in e for e in errors)


class TestP2PWorkload:
    """Tests for P2PWorkload dataclass."""

    def test_create_workload(self):
        """Test creating a basic workload."""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            jobs=[
                Job(
                    job_id=0,
                    name="test-job",
                    assigned_nodes=[0, 1, 2, 3],
                    parallelism=ParallelismConfig(tp=4),
                )
            ],
            tasks=[],
        )
        assert workload.version == "1.0"
        assert workload.meta.num_jobs == 1
        assert len(workload.jobs) == 1

    def test_validate_workload(self):
        """Test workload validation."""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            jobs=[
                Job(
                    job_id=0,
                    assigned_nodes=[0, 1, 2, 3],
                )
            ],
            tasks=[
                Task(
                    task_id=0,
                    job_id=0,
                    type=TaskType.COMPUTE,
                    node=2,
                    duration_us=1500,
                ),
                Task(
                    task_id=1,
                    job_id=0,
                    type=TaskType.FLOW,
                    src=2,
                    dst=3,
                    size_bytes=1024,
                    deps=[0],
                ),
            ],
        )
        errors = workload.validate()
        assert len(errors) == 0

    def test_validate_task_id_uniqueness(self):
        """Test that duplicate task IDs are detected."""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            tasks=[
                Task(
                    task_id=0,
                    job_id=0,
                    type=TaskType.COMPUTE,
                    node=2,
                    duration_us=1500,
                ),
                Task(
                    task_id=0,  # Duplicate!
                    job_id=0,
                    type=TaskType.FLOW,
                    src=2,
                    dst=3,
                    size_bytes=1024,
                ),
            ],
        )
        errors = workload.validate()
        assert any("unique" in e for e in errors)


class TestWorkloadIO:
    """Tests for WorkloadReader and WorkloadWriter."""

    def test_write_and_read(self, tmp_path):
        """Test writing and reading a workload."""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            tasks=[
                Task(
                    task_id=0,
                    job_id=0,
                    type=TaskType.COMPUTE,
                    node=2,
                    duration_us=1500,
                ),
            ],
        )

        file_path = tmp_path / "workload.json"
        writer = WorkloadWriter()
        writer.write(workload, file_path)

        reader = WorkloadReader()
        loaded = reader.read(file_path)

        assert loaded.version == workload.version
        assert loaded.meta.num_jobs == workload.meta.num_jobs
        assert len(loaded.tasks) == len(workload.tasks)

    def test_validator(self, tmp_path):
        """Test WorkloadValidator."""
        from src.workload_format.schema import Job, ParallelismConfig

        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            jobs=[
                Job(
                    job_id=0,
                    name="test-job",
                    assigned_nodes=[0, 1, 2, 3],
                    parallelism=ParallelismConfig(tp=4),
                )
            ],
            tasks=[
                Task(
                    task_id=0,
                    job_id=0,
                    type=TaskType.COMPUTE,
                    node=2,
                    duration_us=1500,
                ),
            ],
        )

        file_path = tmp_path / "workload.json"
        writer = WorkloadWriter()
        writer.write(workload, file_path)

        validator = WorkloadValidator()
        is_valid, errors, loaded = validator.validate_file(str(file_path))

        assert is_valid
        assert len(errors) == 0
        assert loaded is not None


class TestNewInferenceEnums:
    """Tests for new inference enum values added in Phase 5."""

    def test_prefill_phase_creation(self):
        """Test creating a task with PREFILL phase."""
        task = Task(
            task_id=0,
            job_id=0,
            type=TaskType.COMPUTE,
            node=2,
            duration_us=1500,
            phase=Phase.PREFILL,
        )
        assert task.phase == Phase.PREFILL
        assert task.phase.value == "prefill"

    def test_decode_phase_creation(self):
        """Test creating a task with DECODE phase."""
        task = Task(
            task_id=0,
            job_id=0,
            type=TaskType.COMPUTE,
            node=2,
            duration_us=500,
            phase=Phase.DECODE,
        )
        assert task.phase == Phase.DECODE
        assert task.phase.value == "decode"

    def test_kv_cache_transfer_comm_type(self):
        """Test creating a flow task with KV_CACHE_TRANSFER comm type."""
        task = Task(
            task_id=0,
            job_id=0,
            type=TaskType.FLOW,
            src=2,
            dst=3,
            size_bytes=12345678,
            comm_type=CommType.KV_CACHE_TRANSFER,
        )
        assert task.comm_type == CommType.KV_CACHE_TRANSFER
        assert task.comm_type.value == "kv_cache_transfer"

    def test_new_enum_serialization_round_trip(self, tmp_path):
        """Test that new enum values survive write → read round-trip."""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=8),
            tasks=[
                Task(
                    task_id=0,
                    job_id=0,
                    type=TaskType.COMPUTE,
                    node=0,
                    duration_us=1000,
                    phase=Phase.PREFILL,
                ),
                Task(
                    task_id=1,
                    job_id=0,
                    type=TaskType.COMPUTE,
                    node=4,
                    duration_us=500,
                    phase=Phase.DECODE,
                ),
                Task(
                    task_id=2,
                    job_id=0,
                    type=TaskType.FLOW,
                    src=0,
                    dst=4,
                    size_bytes=123456,
                    comm_type=CommType.KV_CACHE_TRANSFER,
                    deps=[0],
                ),
            ],
        )

        file_path = tmp_path / "inference_workload.json"
        WorkloadWriter().write(workload, file_path)

        loaded = WorkloadReader().read(file_path)
        assert len(loaded.tasks) == 3
        assert loaded.tasks[0].phase == Phase.PREFILL
        assert loaded.tasks[1].phase == Phase.DECODE
        assert loaded.tasks[2].comm_type == CommType.KV_CACHE_TRANSFER

    def test_new_enum_validation(self):
        """Test that JSON schema validates new phase values."""
        from src.workload_format.validator import WorkloadValidator

        validator = WorkloadValidator()

        # Valid: prefill phase
        data = {
            "version": "1.0",
            "meta": {"num_jobs": 1, "num_nodes": 4},
            "tasks": [
                {"task_id": 0, "job_id": 0, "type": "compute", "phase": "prefill", "node": 0, "duration_us": 1000}
            ],
        }
        errors = validator.validate_json(data)
        assert len(errors) == 0

        # Valid: decode phase
        data["tasks"][0]["phase"] = "decode"
        errors = validator.validate_json(data)
        assert len(errors) == 0

        # Invalid: unknown phase
        data["tasks"][0]["phase"] = "invalid_phase"
        errors = validator.validate_json(data)
        assert len(errors) > 0

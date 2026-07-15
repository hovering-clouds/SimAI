"""Tests for JobMerger - multi-job workload merging."""

import pytest
from src.workload_generator.job_merger import JobMerger, MergeResult
from src.workload_format.schema import (
    P2PWorkload, Task, TaskType, Phase, Job, ParallelismConfig,
    Meta, Network, CommType
)


def create_simple_workload(
    job_id: int,
    num_tasks: int = 3,
    num_nodes: int = 4,
    topology_file: str = "topo.json",
) -> P2PWorkload:
    """Create a simple single-job workload for testing.

    Creates tasks with simple dependency chain: task0 → task1 → task2
    """
    job = Job(
        job_id=job_id,
        name=f"test-job-{job_id}",
        model="test-model",
        assigned_nodes=list(range(num_nodes)),
        parallelism=ParallelismConfig(tp=2, dp=2, pp=1, ep=1),
    )

    tasks = []
    for i in range(num_tasks):
        task = Task(
            task_id=i,
            job_id=job_id,
            type=TaskType.COMPUTE,
            iteration=0,
            phase=Phase.FORWARD,
            layer_id=i,
            node=i % num_nodes,
            duration_us=100,
            deps=[i - 1] if i > 0 else [],
        )
        tasks.append(task)

    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=num_nodes),
        network=Network(topology_file=topology_file),
        jobs=[job],
        tasks=tasks,
    )


class TestJobMergerBasic:
    """Test basic JobMerger functionality."""

    def test_merge_single_workload(self):
        """Merging a single workload should preserve structure."""
        w1 = create_simple_workload(job_id=0, num_tasks=3, num_nodes=4)
        merger = JobMerger()
        result = merger.merge([w1])

        assert result.merged_workload.meta.num_jobs == 1
        assert result.merged_workload.meta.num_nodes == 4
        assert len(result.merged_workload.tasks) == 3
        assert len(result.merged_workload.jobs) == 1
        # Task IDs should be remapped starting from 0
        task_ids = [t.task_id for t in result.merged_workload.tasks]
        assert task_ids == [0, 1, 2]
        # Dependencies should be preserved
        assert result.merged_workload.tasks[1].deps == [0]
        assert result.merged_workload.tasks[2].deps == [1]

    def test_merge_two_workloads(self):
        """Merging two workloads should remap task_ids and preserve deps."""
        w1 = create_simple_workload(job_id=0, num_tasks=3, num_nodes=4)
        w2 = create_simple_workload(job_id=1, num_tasks=2, num_nodes=4)
        merger = JobMerger()
        result = merger.merge([w1, w2])

        assert result.merged_workload.meta.num_jobs == 2
        assert result.merged_workload.meta.num_nodes == 4
        assert len(result.merged_workload.tasks) == 5  # 3 + 2
        assert len(result.merged_workload.jobs) == 2

        # Check task_id mapping
        assert 0 in result.task_id_mapping  # workload 0
        assert 1 in result.task_id_mapping  # workload 1

        # Workload 0 tasks: 0, 1, 2
        # Workload 1 tasks: 3, 4 (remapped from 0, 1)
        w0_tasks = [t for t in result.merged_workload.tasks if t.job_id == 0]
        w1_tasks = [t for t in result.merged_workload.tasks if t.job_id == 1]
        assert len(w0_tasks) == 3
        assert len(w1_tasks) == 2

        # Check dependencies within each job
        for task in w0_tasks:
            if task.task_id > 0:
                assert all(d < task.task_id for d in task.deps)
        for task in w1_tasks:
            if task.task_id > min(t.task_id for t in w1_tasks):
                assert all(d < task.task_id for d in task.deps)

    def test_merge_empty_list_raises(self):
        """Merging an empty list should raise ValueError."""
        merger = JobMerger()
        with pytest.raises(ValueError, match="Cannot merge empty"):
            merger.merge([])

    def test_missing_dependency_is_rejected_instead_of_dropped(self):
        workload = create_simple_workload(job_id=0, num_tasks=1)
        workload.tasks[0].deps = [999]
        with pytest.raises(ValueError, match="missing dependencies"):
            JobMerger().merge([workload])

    def test_merge_multi_job_workload(self):
        """Merging a workload with multiple jobs should succeed with job_id remapping."""
        job0 = Job(
            job_id=0,
            name="job-0",
            model="test",
            assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2),
        )
        job1 = Job(
            job_id=1,
            name="job-1",
            model="test",
            assigned_nodes=[2, 3],
            parallelism=ParallelismConfig(tp=2),
        )
        w = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=2, num_nodes=4),
            jobs=[job0, job1],
            tasks=[],
        )
        merger = JobMerger()
        result = merger.merge([w])

        assert len(result.merged_workload.jobs) == 2
        # job_ids should be remapped to 0, 1
        job_ids = sorted(j.job_id for j in result.merged_workload.jobs)
        assert job_ids == [0, 1]


class TestTaskIdRemapping:
    """Test task ID remapping logic."""

    def test_task_ids_are_contiguous(self):
        """Task IDs should be contiguous across merged workloads."""
        w1 = create_simple_workload(job_id=0, num_tasks=2)
        w2 = create_simple_workload(job_id=1, num_tasks=3)
        w3 = create_simple_workload(job_id=2, num_tasks=4)

        merger = JobMerger()
        result = merger.merge([w1, w2, w3])

        all_ids = sorted(t.task_id for t in result.merged_workload.tasks)
        expected = list(range(2 + 3 + 4))  # 0..8
        assert all_ids == expected

    def test_mapping_preserves_original_refs(self):
        """Task ID mapping should correctly track original → new mappings."""
        w1 = create_simple_workload(job_id=0, num_tasks=2)
        w2 = create_simple_workload(job_id=1, num_tasks=2)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        # w1's task 0 → new task 0, task 1 → new task 1
        assert result.task_id_mapping[0][0] == 0
        assert result.task_id_mapping[0][1] == 1
        # w2's task 0 → new task 2, task 1 → new task 3
        assert result.task_id_mapping[1][0] == 2
        assert result.task_id_mapping[1][1] == 3

    def test_deps_updated_with_new_ids(self):
        """Dependencies should reference new task IDs after remapping."""
        # Create workload with explicit dependency: task1 depends on task0
        w1 = create_simple_workload(job_id=0, num_tasks=2)
        w2 = create_simple_workload(job_id=1, num_tasks=2)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        # Find the tasks that were originally from w2
        w2_tasks = [t for t in result.merged_workload.tasks if t.job_id == 1]
        w2_tasks.sort(key=lambda t: t.task_id)

        # Second task of w2 should depend on first task of w2
        if len(w2_tasks) > 1 and w2_tasks[1].deps:
            assert w2_tasks[1].deps[0] == w2_tasks[0].task_id


class TestMetadataMerging:
    """Test metadata merging logic."""

    def test_num_jobs_is_sum(self):
        """num_jobs should be sum of all workloads' num_jobs."""
        w1 = create_simple_workload(job_id=0)
        w2 = create_simple_workload(job_id=1)
        w3 = create_simple_workload(job_id=2)

        merger = JobMerger()
        result = merger.merge([w1, w2, w3])

        assert result.merged_workload.meta.num_jobs == 3

    def test_num_nodes_is_unique_count(self):
        """num_nodes should be the count of unique node IDs across all jobs."""
        w1 = create_simple_workload(job_id=0, num_nodes=4)
        w2 = create_simple_workload(job_id=1, num_nodes=8)
        w3 = create_simple_workload(job_id=2, num_nodes=2)

        merger = JobMerger()
        result = merger.merge([w1, w2, w3])

        # w1: [0,1,2,3], w2: [0,1,2,3,4,5,6,7], w3: [0,1] → unique = 8
        assert result.merged_workload.meta.num_nodes == 8

    def test_num_nodes_with_overlapping_nodes(self):
        """num_nodes counts unique nodes, overlapping nodes counted once."""
        # job1 uses [1,2,3], job2 uses [0,1,4] → unique = {0,1,2,3,4} → 5
        j1 = Job(
            job_id=0, name="j1", model="m", assigned_nodes=[1, 2, 3],
            parallelism=ParallelismConfig(tp=1, dp=1, pp=1, ep=1),
        )
        j2 = Job(
            job_id=1, name="j2", model="m", assigned_nodes=[0, 1, 4],
            parallelism=ParallelismConfig(tp=1, dp=1, pp=1, ep=1),
        )
        w1 = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            network=None, jobs=[j1], tasks=[],
        )
        w2 = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            network=None, jobs=[j2], tasks=[],
        )

        merger = JobMerger()
        result = merger.merge([w1, w2])

        assert result.merged_workload.meta.num_jobs == 2
        assert result.merged_workload.meta.num_nodes == 5


class TestNetworkMerging:
    """Test network topology merging logic."""

    def test_topology_file_override(self):
        """Provided topology_file should override workload networks."""
        w1 = create_simple_workload(job_id=0, topology_file="topo1.json")
        w2 = create_simple_workload(job_id=1, topology_file="topo2.json")

        merger = JobMerger()
        result = merger.merge([w1, w2], topology_file="merged_topo.json")

        assert result.merged_workload.network is not None
        assert result.merged_workload.network.topology_file == "merged_topo.json"

    def test_first_network_used_when_no_override(self):
        """First non-None network should be used when no topology_file provided."""
        w1 = create_simple_workload(job_id=0, topology_file="first.json")
        w2 = create_simple_workload(job_id=1, topology_file="second.json")

        merger = JobMerger()
        result = merger.merge([w1, w2])

        assert result.merged_workload.network.topology_file == "first.json"

    def test_none_network_when_all_none(self):
        """Network should have empty topology_file when all workloads have None network."""
        job = Job(
            job_id=0,
            name="test-job",
            model="test-model",
            assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2),
        )
        w_with_none = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=2),
            network=None,  # Explicitly None, will be converted to empty Network
            jobs=[job],
            tasks=[],
        )

        merger = JobMerger()
        result = merger.merge([w_with_none])

        # P2PWorkload.__post_init__ converts None to Network(topology_file="")
        assert result.merged_workload.network is not None
        assert result.merged_workload.network.topology_file == ""


class TestJobMerging:
    """Test jobs list merging."""

    def test_jobs_preserved(self):
        """All jobs should be preserved in merged workload."""
        job0 = Job(
            job_id=0,
            name="llama-7b",
            model="llama-7b",
            assigned_nodes=[0, 1, 2, 3],
            parallelism=ParallelismConfig(tp=2, dp=2),
        )
        job1 = Job(
            job_id=1,
            name="deepseek-moe",
            model="deepseek-moe",
            assigned_nodes=[4, 5, 6, 7],
            parallelism=ParallelismConfig(tp=2, ep=2),
        )
        w1 = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            jobs=[job0],
            tasks=[],
        )
        w2 = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=4),
            jobs=[job1],
            tasks=[],
        )

        merger = JobMerger()
        result = merger.merge([w1, w2])

        assert len(result.merged_workload.jobs) == 2
        assert result.merged_workload.jobs[0].name == "llama-7b"
        assert result.merged_workload.jobs[1].name == "deepseek-moe"

    def test_job_mapping_tracks_ids(self):
        """Job mapping should track old→new job_id for each workload."""
        w1 = create_simple_workload(job_id=10)
        w2 = create_simple_workload(job_id=20)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        # job_mapping is now {workload_idx: {old_job_id: new_job_id}}
        assert result.job_mapping[0] == {10: 0}
        assert result.job_mapping[1] == {20: 1}


class TestMergeResult:
    """Test MergeResult structure."""

    def test_result_has_all_fields(self):
        """MergeResult should have merged_workload, task_id_mapping, job_mapping."""
        w1 = create_simple_workload(job_id=0)
        w2 = create_simple_workload(job_id=1)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        assert isinstance(result, MergeResult)
        assert isinstance(result.merged_workload, P2PWorkload)
        assert isinstance(result.task_id_mapping, dict)
        assert isinstance(result.job_mapping, dict)

    def test_result_is_valid_workload(self):
        """Merged workload should pass validation."""
        w1 = create_simple_workload(job_id=0, num_tasks=3)
        w2 = create_simple_workload(job_id=1, num_tasks=2)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        errors = result.merged_workload.validate()
        assert not errors, f"Merged workload validation failed: {errors}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_merge_workloads_with_no_tasks(self):
        """Merging workloads with no tasks should work."""
        job = Job(
            job_id=0,
            name="empty-job",
            model="test",
            assigned_nodes=[0, 1],
            parallelism=ParallelismConfig(tp=2),
        )
        w = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=2),
            jobs=[job],
            tasks=[],
        )

        merger = JobMerger()
        result = merger.merge([w])

        assert result.merged_workload.meta.num_jobs == 1
        assert len(result.merged_workload.tasks) == 0

    def test_merge_workloads_with_complex_deps(self):
        """Merging workloads with complex dependency chains."""
        # w1: task0 → task1 → task2
        # w2: task0 → task1 (diamond: task1 depends on task0)
        w1 = create_simple_workload(job_id=0, num_tasks=3)
        w2 = create_simple_workload(job_id=1, num_tasks=2)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        # Verify no cross-job dependencies
        for task in result.merged_workload.tasks:
            for dep in task.deps:
                dep_task = next(
                    t for t in result.merged_workload.tasks if t.task_id == dep
                )
                assert dep_task.job_id == task.job_id, \
                    "Cross-job dependency detected!"

    def test_three_way_merge(self):
        """Test merging three workloads simultaneously."""
        w1 = create_simple_workload(job_id=0, num_tasks=2, num_nodes=4)
        w2 = create_simple_workload(job_id=1, num_tasks=3, num_nodes=8)
        w3 = create_simple_workload(job_id=2, num_tasks=1, num_nodes=2)

        merger = JobMerger()
        result = merger.merge([w1, w2, w3])

        assert result.merged_workload.meta.num_jobs == 3
        assert result.merged_workload.meta.num_nodes == 8  # max(4, 8, 2)
        assert len(result.merged_workload.tasks) == 6  # 2 + 3 + 1
        assert len(result.task_id_mapping) == 3


class TestJobIdRemapping:
    """Test job_id remapping when workloads have conflicting job_ids."""

    def _create_multi_job_workload(
        self,
        job_ids: list[int],
        tasks_per_job: int = 2,
        num_nodes: int = 4,
    ) -> P2PWorkload:
        """Create a workload with multiple jobs and tasks."""
        jobs = []
        tasks = []
        task_id = 0

        for jid in job_ids:
            job = Job(
                job_id=jid,
                name=f"job-{jid}",
                model="test",
                assigned_nodes=list(range(num_nodes)),
                parallelism=ParallelismConfig(tp=2, dp=2),
            )
            jobs.append(job)

            for i in range(tasks_per_job):
                tasks.append(Task(
                    task_id=task_id,
                    job_id=jid,
                    type=TaskType.COMPUTE,
                    phase=Phase.FORWARD,
                    layer_id=i,
                    node=i % num_nodes,
                    duration_us=100,
                    deps=[task_id - 1] if i > 0 else [],
                ))
                task_id += 1

        return P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=len(job_ids), num_nodes=num_nodes),
            jobs=jobs,
            tasks=tasks,
        )

    def test_conflicting_job_ids_across_workloads(self):
        """Two workloads both using job_id=0 should get remapped to 0 and 1."""
        w1 = create_simple_workload(job_id=0)
        w2 = create_simple_workload(job_id=0)  # Same job_id!

        merger = JobMerger()
        result = merger.merge([w1, w2])

        job_ids = sorted(j.job_id for j in result.merged_workload.jobs)
        assert job_ids == [0, 1]

        # job_mapping should show the remapping
        assert result.job_mapping[0] == {0: 0}
        assert result.job_mapping[1] == {0: 1}

    def test_conflicting_job_ids_in_same_workload(self):
        """A single workload with jobs [0, 0] is invalid at input but
        we just remap them sequentially (0→0, second 0→1)."""
        job_a = Job(job_id=5, name="a", model="t", assigned_nodes=[0],
                    parallelism=ParallelismConfig())
        job_b = Job(job_id=5, name="b", model="t", assigned_nodes=[1],
                    parallelism=ParallelismConfig())
        w = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=2, num_nodes=2),
            jobs=[job_a, job_b],
            tasks=[],
        )

        merger = JobMerger()
        result = merger.merge([w])

        # Both get remapped to unique IDs
        merged_ids = sorted(j.job_id for j in result.merged_workload.jobs)
        assert merged_ids == [0, 1]
        assert result.job_mapping[0] == {5: 0, 5: 1}  # last one wins in dict

    def test_multi_job_workload_plus_single_job_workload(self):
        """Merge a multi-job workload [0, 1] with a single-job workload [0]."""
        w_multi = self._create_multi_job_workload(job_ids=[0, 1])
        w_single = create_simple_workload(job_id=0)

        merger = JobMerger()
        result = merger.merge([w_multi, w_single])

        assert len(result.merged_workload.jobs) == 3
        merged_job_ids = sorted(j.job_id for j in result.merged_workload.jobs)
        assert merged_job_ids == [0, 1, 2]

        # Check job_mapping
        assert result.job_mapping[0] == {0: 0, 1: 1}
        assert result.job_mapping[1] == {0: 2}

    def test_task_job_ids_updated_after_remap(self):
        """Tasks should reference the new job_ids after remapping."""
        # w1 has job_id=10, w2 has job_id=10 (conflict)
        w1 = create_simple_workload(job_id=10, num_tasks=2)
        w2 = create_simple_workload(job_id=10, num_tasks=2)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        # Tasks from w1 should have job_id=0
        w1_tasks = [t for t in result.merged_workload.tasks
                    if t.task_id in result.task_id_mapping[0].values()]
        for t in w1_tasks:
            assert t.job_id == 0

        # Tasks from w2 should have job_id=1
        w2_tasks = [t for t in result.merged_workload.tasks
                    if t.task_id in result.task_id_mapping[1].values()]
        for t in w2_tasks:
            assert t.job_id == 1

    def test_multi_job_workload_preserves_per_job_deps(self):
        """Dependencies within each job should be preserved after remap."""
        w = self._create_multi_job_workload(job_ids=[100, 200], tasks_per_job=3)

        merger = JobMerger()
        result = merger.merge([w])

        # Group tasks by job_id
        by_job: dict[int, list[Task]] = {}
        for t in result.merged_workload.tasks:
            by_job.setdefault(t.job_id, []).append(t)

        assert len(by_job) == 2
        for job_id, tasks in by_job.items():
            tasks.sort(key=lambda t: t.task_id)
            # task[1] depends on task[0]
            assert tasks[1].deps == [tasks[0].task_id]
            # task[2] depends on task[1]
            assert tasks[2].deps == [tasks[1].task_id]

    def test_multi_job_workload_validates(self):
        """Merged result from multi-job workloads should pass validation."""
        w1 = self._create_multi_job_workload(job_ids=[0, 1])
        w2 = self._create_multi_job_workload(job_ids=[0, 1, 2])

        merger = JobMerger()
        result = merger.merge([w1, w2])

        errors = result.merged_workload.validate()
        assert not errors, f"Validation failed: {errors}"

    def test_two_multi_job_workloads_no_cross_deps(self):
        """Merging two multi-job workloads should have no cross-job deps."""
        w1 = self._create_multi_job_workload(job_ids=[0, 1], tasks_per_job=2)
        w2 = self._create_multi_job_workload(job_ids=[0, 1], tasks_per_job=2)

        merger = JobMerger()
        result = merger.merge([w1, w2])

        for task in result.merged_workload.tasks:
            for dep in task.deps:
                dep_task = next(
                    t for t in result.merged_workload.tasks if t.task_id == dep
                )
                assert dep_task.job_id == task.job_id, \
                    f"Cross-job dep: task {task.task_id} (job {task.job_id}) " \
                    f"depends on task {dep_task.task_id} (job {dep_task.job_id})"

    def test_job_id_contiguous_across_all_workloads(self):
        """Job IDs should be contiguous across all merged workloads."""
        w1 = self._create_multi_job_workload(job_ids=[10, 20])
        w2 = self._create_multi_job_workload(job_ids=[30])
        w3 = create_simple_workload(job_id=5)

        merger = JobMerger()
        result = merger.merge([w1, w2, w3])

        job_ids = sorted(j.job_id for j in result.merged_workload.jobs)
        assert job_ids == [0, 1, 2, 3]  # 2 + 1 + 1 = 4 contiguous jobs

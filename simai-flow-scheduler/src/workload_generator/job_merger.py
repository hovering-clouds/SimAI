"""
Job Merger - Merge multiple P2P workloads into one multi-job workload.

This module provides functionality to combine multiple P2P workloads
into a unified multi-job workload that can be simulated together, sharing the
same network topology. Each input workload may contain one or more jobs.

Key operations:
1. Remap job_ids globally to avoid conflicts
2. Remap task_ids globally to avoid conflicts
3. Update job_id references in tasks and dependency references
4. Merge meta information (num_jobs = sum, num_nodes = unique node count)
5. Merge network topology references
6. Merge jobs and tasks lists
"""

from dataclasses import dataclass, field
from typing import Optional

from src.workload_format.schema import P2PWorkload, Job, Task, Meta, Network


@dataclass
class MergeResult:
    """Result of merging multiple workloads."""

    merged_workload: P2PWorkload
    task_id_mapping: dict[int, dict[int, int]]  # {workload_idx: {old_task_id: new_task_id}}
    job_mapping: dict[int, dict[int, int]]  # {workload_idx: {old_job_id: new_job_id}}


class JobMerger:
    """Merge multiple P2P workloads into one multi-job workload.

    Each input workload may contain one or more jobs. Job IDs and task IDs
    are globally remapped to avoid conflicts.

    Usage:
        merger = JobMerger()
        result = merger.merge([w1, w2, w3], topology_file="topo.json")
        # result.merged_workload contains the unified workload
        # result.task_id_mapping shows how task_ids were remapped
        # result.job_mapping shows how job_ids were remapped
    """

    def merge(
        self,
        workloads: list[P2PWorkload],
        topology_file: str = "",
    ) -> MergeResult:
        """
        Merge multiple workloads into one multi-job workload.

        Rules:
        1. job_id 全局重新编号（避免冲突）
        2. task_id 全局重新编号（避免冲突）
        3. tasks 中的 job_id 引用和 deps 引用用新 ID 替换
        4. meta.num_jobs = sum of all workloads' num_jobs
        5. meta.num_nodes = unique node count across all jobs' assigned_nodes
        6. network topology 合并（使用第一个非空 topology，或使用传入的 topology_file）
        7. jobs 列表合并
        8. tasks 列表合并（含 job_id + task_id 重映射）

        Args:
            workloads: List of workloads to merge (each may have one or more jobs)
            topology_file: Optional topology file path to use for merged workload

        Returns:
            MergeResult containing merged workload and mappings
        """
        if not workloads:
            raise ValueError("Cannot merge empty workload list")

        # Step 1: Remap job_ids and task_ids
        remapped_workloads, task_id_mapping, job_mapping = self._remap_ids(workloads)

        # Step 2: Merge meta
        merged_meta = self._merge_meta(remapped_workloads)

        # Step 3: Merge network
        merged_network = self._merge_network(remapped_workloads, topology_file)

        # Step 4: Merge jobs
        merged_jobs = self._merge_jobs(remapped_workloads)

        # Step 5: Merge tasks
        merged_tasks = self._merge_tasks(remapped_workloads)

        # Step 6: Build merged workload
        merged_workload = P2PWorkload(
            version="1.0",
            meta=merged_meta,
            network=merged_network,
            jobs=merged_jobs,
            tasks=merged_tasks,
        )

        return MergeResult(
            merged_workload=merged_workload,
            task_id_mapping=task_id_mapping,
            job_mapping=job_mapping,
        )

    def _remap_ids(
        self,
        workloads: list[P2PWorkload],
    ) -> tuple[list[P2PWorkload], dict[int, dict[int, int]], dict[int, dict[int, int]]]:
        """
        Remap job_ids and task_ids across workloads to ensure global uniqueness.

        Job IDs are assigned contiguously across all workloads.
        Task IDs are assigned contiguously per workload.

        Returns:
            Tuple of (remapped_workloads, task_id_mapping, job_mapping)
            where:
              task_id_mapping[workload_idx][old_task_id] = new_task_id
              job_mapping[workload_idx][old_job_id] = new_job_id
        """
        remapped = []
        task_mappings: dict[int, dict[int, int]] = {}
        job_mappings: dict[int, dict[int, int]] = {}
        next_task_id = 0
        next_job_id = 0

        for idx, workload in enumerate(workloads):
            # --- Remap job_ids ---
            job_old_to_new: dict[int, int] = {}
            new_jobs = []
            for job in workload.jobs:
                job_old_to_new[job.job_id] = next_job_id
                new_job = Job(
                    job_id=next_job_id,
                    name=job.name,
                    model=job.model,
                    assigned_nodes=job.assigned_nodes,
                    parallelism=job.parallelism,
                )
                new_jobs.append(new_job)
                next_job_id += 1
            job_mappings[idx] = job_old_to_new

            # --- Remap task_ids ---
            task_old_to_new: dict[int, int] = {}
            for task in workload.tasks:
                task_old_to_new[task.task_id] = next_task_id
                next_task_id += 1
            task_mappings[idx] = task_old_to_new

            # Create new tasks with both task_id and job_id remapped
            new_tasks = []
            for task in workload.tasks:
                new_task = Task(
                    task_id=task_old_to_new[task.task_id],
                    job_id=job_old_to_new[task.job_id],
                    type=task.type,
                    iteration=task.iteration,
                    phase=task.phase,
                    layer_id=task.layer_id,
                    item_id=task.item_id,
                    coflow_id=(f"job{job_old_to_new[task.job_id]}:{task.coflow_id}"
                               if task.coflow_id is not None else None),
                    microbatch_id=task.microbatch_id,
                    logical_layer_id=task.logical_layer_id,
                    node=task.node,
                    duration_us=task.duration_us,
                    src=task.src,
                    dst=task.dst,
                    size_bytes=task.size_bytes,
                    comm_type=task.comm_type,
                    chunk_id=task.chunk_id,
                    num_chunks=task.num_chunks,
                    deps=[task_old_to_new[dep] for dep in task.deps if dep in task_old_to_new],
                )
                new_tasks.append(new_task)

            # Create new workload with remapped data
            new_workload = P2PWorkload(
                version=workload.version,
                meta=workload.meta,
                network=workload.network,
                jobs=new_jobs,
                tasks=new_tasks,
            )
            remapped.append(new_workload)

        return remapped, task_mappings, job_mappings

    def _merge_meta(self, workloads: list[P2PWorkload]) -> Meta:
        """Merge metadata from multiple workloads.

        num_nodes is the count of unique node IDs across all jobs'
        assigned_nodes. Jobs share the same cluster namespace, so
        overlapping nodes are counted only once.
        """
        total_jobs = sum(w.meta.num_jobs for w in workloads)
        unique_nodes: set[int] = set()
        for w in workloads:
            for job in w.jobs:
                unique_nodes.update(job.assigned_nodes)

        return Meta(
            num_jobs=total_jobs,
            num_nodes=len(unique_nodes),
        )

    def _merge_network(
        self,
        workloads: list[P2PWorkload],
        topology_file: str = "",
    ) -> Optional[Network]:
        """Merge network topology references.

        Strategy:
        1. If topology_file is provided, use it
        2. Otherwise, use the first non-None network from workloads
        3. If no network info available, return None
        """
        if topology_file:
            return Network(topology_file=topology_file)

        # Find first non-None network
        for w in workloads:
            if w.network is not None:
                return w.network

        return None

    def _merge_jobs(self, workloads: list[P2PWorkload]) -> list[Job]:
        """Merge jobs from all workloads."""
        merged = []
        for w in workloads:
            merged.extend(w.jobs)
        return merged

    def _merge_tasks(self, workloads: list[P2PWorkload]) -> list[Task]:
        """Merge tasks from all workloads (already remapped)."""
        merged = []
        for w in workloads:
            merged.extend(w.tasks)
        return merged

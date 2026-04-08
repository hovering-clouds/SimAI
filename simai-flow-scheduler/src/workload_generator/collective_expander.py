"""
Collective communication to P2P flow expanders.

This module provides the base classes and implementations for expanding
collective communication operations (AllReduce, AllGather, ReduceScatter, AlltoAll)
into point-to-point flow tasks.

Reference: MockNcclGroup.cc in astra-sim-alibabacloud
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..workload_format.schema import Task, TaskType, Phase, CommType


@dataclass
class FlowTask:
    """
    Internal representation of a flow task during expansion.

    This is a simpler structure used during the expansion process.
    It will be converted to Task objects when building the final workload.
    """
    task_id: int
    job_id: int
    type: TaskType
    src: Optional[int] = None
    dst: Optional[int] = None
    size_bytes: Optional[int] = None
    comm_type: CommType = CommType.UNKNOWN
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None
    deps: list[int] = field(default_factory=list)

    # Compute-specific fields
    node: Optional[int] = None
    duration_us: Optional[int] = None
    iteration: int = 0
    phase: Phase = Phase.FORWARD
    layer_id: int = 0

    def to_task(self) -> Task:
        """Convert FlowTask to Task object."""
        return Task(
            task_id=self.task_id,
            job_id=self.job_id,
            type=self.type,
            iteration=self.iteration,
            phase=self.phase,
            layer_id=self.layer_id,
            node=self.node,
            duration_us=self.duration_us,
            src=self.src,
            dst=self.dst,
            size_bytes=self.size_bytes,
            comm_type=self.comm_type,
            chunk_id=self.chunk_id,
            num_chunks=self.num_chunks,
            deps=self.deps,
        )


class CollectiveExpander(ABC):
    """
    Base class for collective communication expanders.

    Expands collective operations into sequences of P2P flow tasks.
    Reference implementation: MockNcclGroup.cc
    """

    @abstractmethod
    def expand_allreduce(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """
        Expand AllReduce into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes.
            algo: Algorithm to use ("ring", "tree", "nvls").
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.

        Returns:
            List of FlowTask objects representing the P2P flows.
        """
        pass

    @abstractmethod
    def expand_allgather(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """
        Expand AllGather into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes.
            algo: Algorithm to use ("ring", "tree").
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.

        Returns:
            List of FlowTask objects representing the P2P flows.
        """
        pass

    @abstractmethod
    def expand_reducescatter(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """
        Expand ReduceScatter into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes.
            algo: Algorithm to use ("ring", "tree").
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.

        Returns:
            List of FlowTask objects representing the P2P flows.
        """
        pass

    @abstractmethod
    def expand_alltoall(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """
        Expand AlltoAll into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes (per-destination).
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.

        Returns:
            List of FlowTask objects representing the P2P flows.
        """
        pass


class RingAllReduceExpander(CollectiveExpander):
    """
    Ring AllReduce expander.

    Implements the two-phase Ring AllReduce algorithm:
    1. Reduce-Scatter phase: N-1 steps, each rank sends one chunk to next rank
    2. AllGather phase: N-1 steps, each rank receives chunks from previous rank

    Reference: MockNcclGroup.cc::genAllReduceFlowModels
    """

    def expand_allreduce(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """
        Expand Ring AllReduce into P2P flows.

        Algorithm:
        - Split data into N chunks (N = number of ranks)
        - Reduce-Scatter: N-1 steps, each step N flows
        - AllGather: N-1 steps, each step N flows
        - Total: 2 * (N-1) * N flows

        Args:
            ranks: List of global ranks.
            data_size: Total data size in bytes.
            algo: Must be "ring".
            job_id: Job ID.
            task_id_start: Starting task ID.

        Returns:
            List of FlowTask objects.
        """
        if algo != "ring":
            raise ValueError(f"RingAllReduceExpander only supports 'ring' algo, got '{algo}'")

        n = len(ranks)
        if n < 2:
            return []

        chunk_size = data_size // n
        tasks: list[FlowTask] = []
        task_id = task_id_start

        # Reduce-Scatter phase
        # Each rank i sends chunk i to rank (i+1) % n
        for step in range(n - 1):
            for rank_idx in range(n):
                src = ranks[rank_idx]
                dst = ranks[(rank_idx + 1) % n]
                chunk = rank_idx

                deps = []
                if step > 0:
                    # Depend on the same chunk from previous step
                    prev_task_id = task_id - n
                    deps.append(prev_task_id)

                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=src,
                    dst=dst,
                    size_bytes=chunk_size,
                    comm_type=CommType.TP_ALLREDUCE_RING,
                    chunk_id=chunk,
                    num_chunks=n,
                    deps=deps,
                ))
                task_id += 1

        # AllGather phase
        # Each rank (i+1) % n sends chunk (i+1) % n to rank i
        rs_base = task_id_start  # Base task ID of Reduce-Scatter phase
        for step in range(n - 1):
            for rank_idx in range(n):
                src = ranks[(rank_idx + 1) % n]
                dst = ranks[rank_idx]
                chunk = (rank_idx + 1) % n

                deps = []
                # Depend on the same chunk from previous AllGather step
                if step > 0:
                    prev_task_id = task_id - n
                    deps.append(prev_task_id)
                # Depend on the corresponding Reduce-Scatter task completing
                rs_task_id = rs_base + ((n - 1 - 1) * n + rank_idx)
                deps.append(rs_task_id)

                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=src,
                    dst=dst,
                    size_bytes=chunk_size,
                    comm_type=CommType.TP_ALLREDUCE_RING,
                    chunk_id=chunk,
                    num_chunks=n,
                    deps=deps,
                ))
                task_id += 1

        return tasks

    def expand_allgather(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """Expand Ring AllGather into P2P flows."""
        # TODO: Implement Ring AllGather
        raise NotImplementedError("Ring AllGather not yet implemented")

    def expand_reducescatter(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """Expand Ring ReduceScatter into P2P flows."""
        # TODO: Implement Ring ReduceScatter
        raise NotImplementedError("Ring ReduceScatter not yet implemented")

    def expand_alltoall(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int = 0,
        task_id_start: int = 0,
    ) -> list[FlowTask]:
        """Expand AlltoAll into P2P flows."""
        # TODO: Implement AlltoAll
        raise NotImplementedError("AlltoAll not yet implemented")

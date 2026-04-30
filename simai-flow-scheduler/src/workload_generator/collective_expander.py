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


# ── CommType prefix mapping ──

_COMM_TYPE_MAP: dict[tuple[str, str], CommType] = {
    # context="tp" (default)
    ("ALLREDUCE", "tp"): CommType.TP_ALLREDUCE_RING,
    ("ALLGATHER", "tp"): CommType.TP_ALLGATHER_RING,
    ("REDUCESCATTER", "tp"): CommType.TP_REDUCESCATTER_RING,
    ("ALLTOALL", "tp"): CommType.TP_ALLTOALL,
    # context="dp"
    ("ALLREDUCE", "dp"): CommType.DP_ALLREDUCE,
    ("ALLGATHER", "dp"): CommType.DP_ALLGATHER,
    ("REDUCESCATTER", "dp"): CommType.DP_REDUCESCATTER,
    ("ALLTOALL", "dp"): CommType.DP_ALLTOALL,
    # context="ep"
    ("ALLTOALL", "ep"): CommType.EP_ALLTOALL,
}


def _make_comm_type(base: str, context: str) -> CommType:
    """根据 base 类型和并行上下文，返回对应的 CommType。"""
    return _COMM_TYPE_MAP.get((base, context), CommType.UNKNOWN)


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
    item_id: int = 0

    def to_task(self) -> Task:
        """Convert FlowTask to Task object."""
        return Task(
            task_id=self.task_id,
            job_id=self.job_id,
            type=self.type,
            iteration=self.iteration,
            phase=self.phase,
            layer_id=self.layer_id,
            item_id=self.item_id,
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


def _build_ring_topology(ranks: list[int]) -> dict[int, dict]:
    """
    Build the ring channel structure.

    For single-node (no PXN), each rank has:
    - prev: the rank that sends TO this rank (= ranks[(i-1) % n])
    - next: the rank this rank sends TO (= ranks[(i+1) % n])

    Returns:
        Dict mapping rank -> {prev, next}
    """
    n = len(ranks)
    ring = {}
    for i, rank in enumerate(ranks):
        prev_rank = ranks[(i - 1) % n]
        next_rank = ranks[(i + 1) % n]
        ring[rank] = {
            "prev": prev_rank,
            "next": next_rank,
        }
    return ring


class CollectiveExpander(ABC):
    """
    Base class for collective communication expanders.

    Expands collective operations into sequences of P2P flow tasks.
    Reference implementation: MockNcclGroup.cc

    Concrete implementations:
    - AllReduceExpander: expand_allreduce with algo dispatch (ring, tree, nvls)
    - AllGatherExpander: expand_allgather with algo dispatch (ring, tree)
    """

    @abstractmethod
    def expand_allreduce(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
        context: str = "tp",
    ) -> list[FlowTask]:
        """
        Expand AllReduce into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes.
            algo: Algorithm to use ("ring", "tree", "nvls").
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.
            context: Parallelism context ("tp", "dp", "ep").

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
        context: str = "tp",
    ) -> list[FlowTask]:
        """
        Expand AllGather into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes.
            algo: Algorithm to use ("ring", "tree").
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.
            context: Parallelism context ("tp", "dp", "ep").

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
        context: str = "tp",
    ) -> list[FlowTask]:
        """
        Expand ReduceScatter into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes.
            algo: Algorithm to use ("ring", "tree").
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.
            context: Parallelism context ("tp", "dp", "ep").

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
        context: str = "tp",
    ) -> list[FlowTask]:
        """
        Expand AlltoAll into P2P flows.

        Args:
            ranks: List of global ranks participating in the collective.
            data_size: Total data size in bytes (per-destination).
            job_id: Job ID to assign to generated tasks.
            task_id_start: Starting task ID.
            context: Parallelism context ("tp", "dp", "ep").

        Returns:
            List of FlowTask objects representing the P2P flows.
        """
        pass


class AllReduceExpander(CollectiveExpander):
    """
    AllReduce expander. Dispatches to algorithm-specific implementations.

    Supported algorithms:
    - "ring": Ring AllReduce (MockNcclGroup.cc::genAllReduceRingFlowModels)
    """

    def expand_allreduce(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
        context: str = "tp",
    ) -> list[FlowTask]:
        if algo == "ring":
            return self._expand_ring(ranks, data_size, job_id, task_id_start, context)
        raise ValueError(f"AllReduceExpander: unsupported algo '{algo}'")

    def _expand_ring(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int,
        task_id_start: int,
        context: str,
    ) -> list[FlowTask]:
        """
        Ring AllReduce implementation.

        Algorithm structure (for n ranks):
        - Phase 1: Initial chunk (chunk_id=0), 1 step — RS step 1
          * n flows, all ranks send simultaneously, no dependencies
        - Phase 2: Reduce-Scatter iterations, n-2 steps (chunk_id=1 to n-2)
          * Each step: n flows with diagonal dependency on previous rank's flow
        - Phase 3: AllGather iterations, n-1 steps (chunk_id=n-1 to 2n-3)
          * Starts AFTER RS completes, independent n-1 steps
          * Same diagonal dependency pattern as Phase 2

        Total flows: n + n*(n-2) + n*(n-1) = n * 2*(n-1)
        For 4 ranks: 4 + 8 + 12 = 24 flows.

        Reference: MockNcclGroup.cc lines 1047-1310
        """
        n = len(ranks)
        if n < 2:
            return []

        chunk_size = data_size // n
        chunk_count = 2 * (n - 1)
        tasks: list[FlowTask] = []
        task_id = task_id_start
        ring = _build_ring_topology(ranks)
        comm = _make_comm_type("ALLREDUCE", context)

        # --- Phase 1: Initial chunk (chunk_id = 0) ---
        task_list: dict[int, int] = {}
        for rank in ranks:
            rank_info = ring[rank]
            tasks.append(FlowTask(
                task_id=task_id,
                job_id=job_id,
                type=TaskType.FLOW,
                src=rank,
                dst=rank_info["next"],
                size_bytes=chunk_size,
                comm_type=comm,
                chunk_id=0,
                num_chunks=chunk_count,
                deps=[],
            ))
            task_list[rank] = task_id
            task_id += 1

        # --- Phase 2: Reduce-Scatter iterations (n-2 steps) ---
        for step in range(n - 2):
            task_list2: dict[int, int] = {}
            for rank in ranks:
                rank_info = ring[rank]
                prev_rank = rank_info["prev"]
                partner_task_id = task_list[prev_rank]

                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=rank,
                    dst=rank_info["next"],
                    size_bytes=chunk_size,
                    comm_type=comm,
                    chunk_id=1 + step,
                    num_chunks=chunk_count,
                    deps=[partner_task_id],
                ))
                task_list2[rank] = task_id
                task_id += 1
            task_list = task_list2

        # --- Phase 3: AllGather iterations (n-1 steps) ---
        for step in range(n - 1):
            task_list2: dict[int, int] = {}
            for rank in ranks:
                rank_info = ring[rank]
                prev_rank = rank_info["prev"]
                partner_task_id = task_list[prev_rank]

                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=rank,
                    dst=rank_info["next"],
                    size_bytes=chunk_size,
                    comm_type=comm,
                    chunk_id=(n - 1) + step,
                    num_chunks=chunk_count,
                    deps=[partner_task_id],
                ))
                task_list2[rank] = task_id
                task_id += 1
            task_list = task_list2

        return tasks

    def expand_allgather(
        self,
        ranks,
        data_size,
        algo="ring",
        job_id=0,
        task_id_start=0,
        _context="tp",
    ):
        raise NotImplementedError("Use AllGatherExpander for AllGather")

    def expand_reducescatter(
        self,
        ranks,
        data_size,
        algo="ring",
        job_id=0,
        task_id_start=0,
        _context="tp",
    ):
        raise NotImplementedError("ReduceScatter not yet implemented")

    def expand_alltoall(
        self,
        ranks,
        data_size,
        job_id=0,
        task_id_start=0,
        _context="tp",
    ):
        raise NotImplementedError("AlltoAll not yet implemented")


class AllGatherExpander(CollectiveExpander):
    """
    AllGather expander. Dispatches to algorithm-specific implementations.

    Supported algorithms:
    - "ring": Ring AllGather (MockNcclGroup.cc::genAllGatherFlowModels)
    """

    def expand_allgather(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
        context: str = "tp",
    ) -> list[FlowTask]:
        if algo == "ring":
            return self._expand_ring(ranks, data_size, job_id, task_id_start, context)
        raise ValueError(f"AllGatherExpander: unsupported algo '{algo}'")

    def _expand_ring(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int,
        task_id_start: int,
        context: str,
    ) -> list[FlowTask]:
        """
        Ring AllGather implementation.

        Algorithm structure (for n ranks):
        - Phase 1: Initial send (chunk_id=0), 1 step
          * n flows, each rank sends its own data to next rank, no dependencies
        - Phase 2: Forwarding iterations, n-2 steps (chunk_id=1 to n-2)
          * Each step: n flows with diagonal dependency on previous rank's flow

        Total flows: n + n*(n-2) = n*(n-1)
        For 4 ranks: 4 + 8 = 12 flows.

        Reference: MockNcclGroup.cc genAllGatherFlowModels
        """
        n = len(ranks)
        if n < 2:
            return []

        chunk_size = data_size // n
        chunk_count = n - 1
        tasks: list[FlowTask] = []
        task_id = task_id_start
        ring = _build_ring_topology(ranks)
        comm = _make_comm_type("ALLGATHER", context)

        # --- Phase 1: Initial send (chunk_id = 0) ---
        task_list: dict[int, int] = {}
        for rank in ranks:
            rank_info = ring[rank]
            tasks.append(FlowTask(
                task_id=task_id,
                job_id=job_id,
                type=TaskType.FLOW,
                src=rank,
                dst=rank_info["next"],
                size_bytes=chunk_size,
                comm_type=comm,
                chunk_id=0,
                num_chunks=chunk_count,
                deps=[],
            ))
            task_list[rank] = task_id
            task_id += 1

        # --- Phase 2: Forwarding (n-2 steps) ---
        for step in range(1, n - 1):
            task_list2: dict[int, int] = {}
            for rank in ranks:
                rank_info = ring[rank]
                prev_rank = rank_info["prev"]
                partner_task_id = task_list[prev_rank]

                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=rank,
                    dst=rank_info["next"],
                    size_bytes=chunk_size,
                    comm_type=comm,
                    chunk_id=step,
                    num_chunks=chunk_count,
                    deps=[partner_task_id],
                ))
                task_list2[rank] = task_id
                task_id += 1
            task_list = task_list2

        return tasks

    def expand_allreduce(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllReduceExpander for AllReduce")

    def expand_reducescatter(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use ReduceScatterExpander for ReduceScatter")

    def expand_alltoall(self, ranks, data_size, job_id=0, task_id_start=0):
        raise NotImplementedError("AlltoAll not yet implemented")


class ReduceScatterExpander(CollectiveExpander):
    """
    ReduceScatter expander. Dispatches to algorithm-specific implementations.

    Supported algorithms:
    - "ring": Ring ReduceScatter (MockNcclGroup.cc::genReduceScatterFlowModels)
    """

    def expand_reducescatter(
        self,
        ranks: list[int],
        data_size: int,
        algo: str = "ring",
        job_id: int = 0,
        task_id_start: int = 0,
        context: str = "tp",
    ) -> list[FlowTask]:
        if algo == "ring":
            return self._expand_ring(ranks, data_size, job_id, task_id_start, context)
        raise ValueError(f"ReduceScatterExpander: unsupported algo '{algo}'")

    def _expand_ring(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int,
        task_id_start: int,
        context: str,
    ) -> list[FlowTask]:
        """
        Ring ReduceScatter implementation.

        Algorithm structure (for n ranks) — identical ring pattern to AllGather:
        - Phase 1: Initial chunk (chunk_id=0), 1 step
          * n flows, each rank sends to next rank, no dependencies
        - Phase 2: Propagation iterations, n-2 steps (chunk_id=1 to n-2)
          * Each step: n flows with diagonal dependency on previous rank's flow

        Total flows: n + n*(n-2) = n*(n-1)
        For 4 ranks: 4 + 8 = 12 flows.

        Reference: MockNcclGroup.cc genReduceScatterFlowModels
        """
        n = len(ranks)
        if n < 2:
            return []

        chunk_size = data_size // n
        chunk_count = n - 1
        tasks: list[FlowTask] = []
        task_id = task_id_start
        ring = _build_ring_topology(ranks)
        comm = _make_comm_type("REDUCESCATTER", context)

        # --- Phase 1: Initial chunk (chunk_id = 0) ---
        task_list: dict[int, int] = {}
        for rank in ranks:
            rank_info = ring[rank]
            tasks.append(FlowTask(
                task_id=task_id,
                job_id=job_id,
                type=TaskType.FLOW,
                src=rank,
                dst=rank_info["next"],
                size_bytes=chunk_size,
                comm_type=comm,
                chunk_id=0,
                num_chunks=chunk_count,
                deps=[],
            ))
            task_list[rank] = task_id
            task_id += 1

        # --- Phase 2: Propagation (n-2 steps) ---
        for step in range(n - 2):
            task_list2: dict[int, int] = {}
            for rank in ranks:
                rank_info = ring[rank]
                prev_rank = rank_info["prev"]
                partner_task_id = task_list[prev_rank]

                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=rank,
                    dst=rank_info["next"],
                    size_bytes=chunk_size,
                    comm_type=comm,
                    chunk_id=1 + step,
                    num_chunks=chunk_count,
                    deps=[partner_task_id],
                ))
                task_list2[rank] = task_id
                task_id += 1
            task_list = task_list2

        return tasks

    def expand_allreduce(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllReduceExpander for AllReduce")

    def expand_allgather(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllGatherExpander for AllGather")

    def expand_alltoall(self, ranks, data_size, job_id=0, task_id_start=0):
        raise NotImplementedError("AlltoAll not yet implemented")


class AlltoAllExpander(CollectiveExpander):
    """
    AlltoAll expander.

    Direct all-to-all communication: every rank sends an equal-sized message
    to every other rank. All flows are independent with no dependencies.

    Reference: MockNcclGroup.cc::genAlltoAllFlowModels
    """

    def expand_alltoall(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int = 0,
        task_id_start: int = 0,
        context: str = "tp",
    ) -> list[FlowTask]:
        """
        Expand AlltoAll into P2P flows.

        Algorithm:
        - N*(N-1) independent flows, each rank sends to every other rank
        - chunk_size = data_size / n
        - No dependencies, chunk_id=0, num_chunks=1

        Args:
            ranks: List of global ranks.
            data_size: Total data size in bytes.
            job_id: Job ID.
            task_id_start: Starting task ID.
            context: Parallelism context ("tp", "dp", "ep").

        Returns:
            List of FlowTask objects.
        """
        n = len(ranks)
        if n < 2:
            return []

        chunk_size = data_size // n
        tasks: list[FlowTask] = []
        task_id = task_id_start
        comm = _make_comm_type("ALLTOALL", context)

        for src in ranks:
            for dst in ranks:
                if src == dst:
                    continue
                tasks.append(FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=src,
                    dst=dst,
                    size_bytes=chunk_size,
                    comm_type=comm,
                    chunk_id=0,
                    num_chunks=1,
                    deps=[],
                ))
                task_id += 1

        return tasks

    def expand_allreduce(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllReduceExpander for AllReduce")

    def expand_allgather(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllGatherExpander for AllGather")

    def expand_reducescatter(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use ReduceScatterExpander for ReduceScatter")

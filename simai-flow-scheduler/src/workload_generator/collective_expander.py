"""
Collective communication to P2P flow expanders.

This module provides the base classes and implementations for expanding
collective communication operations (AllReduce, AllGather, ReduceScatter, AlltoAll)
into point-to-point flow tasks.

Reference: MockNcclGroup.cc in astra-sim-alibabacloud
"""

from __future__ import annotations

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
    ("BROADCAST", "dp"): CommType.DP_BROADCAST,
    # context="ep"
    ("ALLTOALL", "ep"): CommType.EP_ALLTOALL,
    # context="dp_ep" (data-parallel within an expert shard)
    ("ALLREDUCE", "dp_ep"): CommType.DP_EP_ALLREDUCE,
    ("ALLGATHER", "dp_ep"): CommType.DP_EP_ALLGATHER,
    ("REDUCESCATTER", "dp_ep"): CommType.DP_EP_REDUCESCATTER,
    ("ALLTOALL", "dp_ep"): CommType.DP_EP_ALLTOALL,
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
    - AllReduceExpander: ring, tree, double-binary-tree, and halving-doubling
    - AllGatherExpander: ring, tree, double-binary-tree, and halving-doubling
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
            algo: "ring", "tree", "double_binary_tree", or "halving_doubling".
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
            algo: "ring", "tree", "double_binary_tree", or "halving_doubling".
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
            algo: "ring", "tree", "double_binary_tree", or "halving_doubling".
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
    - "tree": Logical binary-tree AllReduce
    - "double_binary_tree": Two shifted binary trees with split chunks
    - "halving_doubling": Recursive-doubling AllReduce (power-of-two ranks)
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
        comm = _make_algorithm_comm_type("ALLREDUCE", context, algo)
        if algo == "tree":
            return _expand_tree_collective(
                ranks=ranks, data_size=data_size, base="ALLREDUCE", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
        if algo == "double_binary_tree":
            return _expand_double_binary_tree_collective(
                ranks=ranks, data_size=data_size, base="ALLREDUCE", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
        if algo == "halving_doubling":
            return _expand_halving_doubling_collective(
                ranks=ranks, data_size=data_size, base="ALLREDUCE", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
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
    - "tree": Logical tree gather followed by tree broadcast
    - "double_binary_tree": Two shifted tree gathers with split chunks
    - "halving_doubling": Recursive-doubling AllGather (power-of-two ranks)
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
        comm = _make_algorithm_comm_type("ALLGATHER", context, algo)
        if algo == "tree":
            return _expand_tree_collective(
                ranks=ranks, data_size=data_size, base="ALLGATHER", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
        if algo == "double_binary_tree":
            return _expand_double_binary_tree_collective(
                ranks=ranks, data_size=data_size, base="ALLGATHER", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
        if algo == "halving_doubling":
            return _expand_halving_doubling_collective(
                ranks=ranks, data_size=data_size, base="ALLGATHER", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
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
    - "tree": Logical tree reduction followed by subtree scatter
    - "double_binary_tree": Two shifted tree reductions with split chunks
    - "halving_doubling": Recursive-halving ReduceScatter (power-of-two ranks)
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
        comm = _make_algorithm_comm_type("REDUCESCATTER", context, algo)
        if algo == "tree":
            return _expand_tree_collective(
                ranks=ranks, data_size=data_size, base="REDUCESCATTER", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
        if algo == "double_binary_tree":
            return _expand_double_binary_tree_collective(
                ranks=ranks, data_size=data_size, base="REDUCESCATTER", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
        if algo == "halving_doubling":
            return _expand_halving_doubling_collective(
                ranks=ranks, data_size=data_size, base="REDUCESCATTER", comm_type=comm,
                job_id=job_id, task_id_start=task_id_start,
            )
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


class BroadcastExpander(CollectiveExpander):
    """Expand a broadcast into direct root-to-peer P2P flows."""

    def expand_broadcast(
        self,
        ranks: list[int],
        data_size: int,
        job_id: int = 0,
        task_id_start: int = 0,
        context: str = "dp",
    ) -> list[FlowTask]:
        if len(ranks) < 2:
            return []

        root = ranks[0]
        comm = _make_comm_type("BROADCAST", context)
        return [
            FlowTask(
                task_id=task_id_start + index,
                job_id=job_id,
                type=TaskType.FLOW,
                src=root,
                dst=dst,
                size_bytes=data_size,
                comm_type=comm,
                chunk_id=0,
                num_chunks=1,
                deps=[],
            )
            for index, dst in enumerate(ranks[1:])
        ]

    def expand_allreduce(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllReduceExpander for AllReduce")

    def expand_allgather(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use AllGatherExpander for AllGather")

    def expand_reducescatter(self, ranks, data_size, algo="ring", job_id=0, task_id_start=0):
        raise NotImplementedError("Use ReduceScatterExpander for ReduceScatter")

    def expand_alltoall(self, ranks, data_size, job_id=0, task_id_start=0):
        raise NotImplementedError("Use AlltoAllExpander for AlltoAll")



def _make_algorithm_comm_type(base: str, context: str, algo: str) -> CommType:
    """Return the existing tree CommType for every non-ring TP algorithm.

    The workload schema distinguishes ring and tree communication for tensor
    parallelism, but intentionally has no separate enum values for double
    binary tree or halving-doubling.  Keeping those algorithms under the
    existing ``*_TREE`` values avoids changing executors and trace consumers.
    """
    if context == "tp" and algo != "ring":
        return {
            "ALLREDUCE": CommType.TP_ALLREDUCE_TREE,
            "ALLGATHER": CommType.TP_ALLGATHER_TREE,
            "REDUCESCATTER": CommType.TP_REDUCESCATTER_TREE,
        }[base]
    return _make_comm_type(base, context)


def _validate_ranks(ranks: list[int], algo: str) -> None:
    """Validate the rank list required by newly added logical algorithms."""
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"{algo}: ranks must be unique")


def _split_bytes(total: int, pieces: int, description: str) -> list[int]:
    """Split bytes exactly and deterministically, rejecting zero-byte flows."""
    if total <= 0:
        raise ValueError(f"{description}: data_size must be positive")
    if pieces <= 0 or total < pieces:
        raise ValueError(
            f"{description}: data_size ({total}) must provide at least one byte "
            f"for each of {pieces} partitions"
        )
    base, remainder = divmod(total, pieces)
    return [base + (1 if index < remainder else 0) for index in range(pieces)]


def _build_binary_tree(
    ranks: list[int],
) -> tuple[int, dict[int, int | None], dict[int, list[int]]]:
    """Build a deterministic complete binary tree from rank-list order.

    Astra-Sim obtains ``up``/``down`` relationships from its channel and node
    topology.  The Python expander receives only a rank list, so rank-list
    order is the logical topology contract.  This also supports non-contiguous
    global rank IDs.
    """
    root = ranks[0]
    parents: dict[int, int | None] = {root: None}
    children: dict[int, list[int]] = {rank: [] for rank in ranks}
    for index, rank in enumerate(ranks[1:], start=1):
        parent = ranks[(index - 1) // 2]
        parents[rank] = parent
        children[parent].append(rank)
    return root, parents, children


def _postorder(root: int, children: dict[int, list[int]]) -> list[int]:
    """Return children before parents, used by tree gather/reduce."""
    result: list[int] = []

    def visit(rank: int) -> None:
        for child in children[rank]:
            visit(child)
        result.append(rank)

    visit(root)
    return result


def _expand_tree_collective(
    *,
    ranks: list[int],
    data_size: int,
    base: str,
    comm_type: CommType,
    job_id: int,
    task_id_start: int,
    chunk_id: int = 0,
    num_chunks: int = 1,
) -> list[FlowTask]:
    """Expand one logical binary-tree chunk into endpoint FLOW tasks.

    This is the FLOW-DAG counterpart of MockNcclGroup's tree ``up`` and
    ``down`` passes.  It deliberately omits C++ channel, packet, and stream
    state while preserving the parent/child communication dependencies.

    ``data_size`` means a full per-rank tensor for AllReduce and
    ReduceScatter, and a full gathered output for AllGather.
    """
    if len(ranks) < 2:
        return []
    _validate_ranks(ranks, f"{base.lower()} tree")

    root, parents, children = _build_binary_tree(ranks)
    order = _postorder(root, children)
    rank_to_index = {rank: index for index, rank in enumerate(ranks)}
    task_id = task_id_start
    tasks: list[FlowTask] = []
    up_ready: dict[int, list[int]] = {rank: [] for rank in ranks}

    if base == "ALLGATHER":
        own_sizes = dict(
            zip(ranks, _split_bytes(data_size, len(ranks), "tree allgather"))
        )

        def up_size(rank: int) -> int:
            return subtree_sizes[rank]
    else:
        own_sizes = {}
        up_size = lambda _rank: data_size

    subtree_sizes: dict[int, int] = {}
    if base == "ALLGATHER":
        for rank in order:
            subtree_sizes[rank] = own_sizes[rank] + sum(
                subtree_sizes[child] for child in children[rank]
            )

    # Upward gather/reduce.  A sender waits until all flows from its children
    # have completed, matching the C++ nodeprevs dependency construction.
    for rank in order:
        if rank == root:
            continue
        parent = parents[rank]
        assert parent is not None
        task = FlowTask(
            task_id=task_id,
            job_id=job_id,
            type=TaskType.FLOW,
            src=rank,
            dst=parent,
            size_bytes=up_size(rank),
            comm_type=comm_type,
            chunk_id=chunk_id,
            num_chunks=num_chunks,
            deps=list(up_ready[rank]),
        )
        tasks.append(task)
        up_ready[parent].append(task_id)
        task_id += 1

    # Downward broadcast/scatter.  Root can start only after all reductions or
    # gathered subtrees have arrived; every other node waits for its parent.
    down_ready: dict[int, list[int]] = {root: list(up_ready[root])}
    output_sizes = (
        _split_bytes(data_size, len(ranks), "tree reducescatter")
        if base == "REDUCESCATTER"
        else []
    )
    queue = [root]
    while queue:
        parent = queue.pop(0)
        for child in children[parent]:
            if base == "REDUCESCATTER":
                down_size = sum(
                    output_sizes[rank_to_index[rank]]
                    for rank in _subtree_ranks(child, children)
                )
            else:
                down_size = data_size
            task = FlowTask(
                task_id=task_id,
                job_id=job_id,
                type=TaskType.FLOW,
                src=parent,
                dst=child,
                size_bytes=down_size,
                comm_type=comm_type,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                deps=list(down_ready[parent]),
            )
            tasks.append(task)
            down_ready[child] = [task_id]
            task_id += 1
            queue.append(child)
    return tasks


def _subtree_ranks(root: int, children: dict[int, list[int]]) -> list[int]:
    """Return the ranks in a subtree in deterministic preorder."""
    result = [root]
    for child in children[root]:
        result.extend(_subtree_ranks(child, children))
    return result


def _expand_halving_doubling_collective(
    *,
    ranks: list[int],
    data_size: int,
    base: str,
    comm_type: CommType,
    job_id: int,
    task_id_start: int,
) -> list[FlowTask]:
    """Build the hypercube FLOW DAG used by Halving-Doubling.

    This follows HalvingDoubling.cc's ``rank_offset`` and message-size
    evolution, at FLOW granularity.  Like the C++ implementation, it requires
    a power-of-two participant count.
    """
    n = len(ranks)
    if n < 2:
        return []
    _validate_ranks(ranks, f"{base.lower()} halving_doubling")
    if n & (n - 1):
        raise ValueError("halving_doubling requires a power-of-two rank count")
    if data_size <= 0 or data_size % n:
        raise ValueError(
            f"halving_doubling {base.lower()}: data_size must be positive and divisible by rank count"
        )

    stages = n.bit_length() - 1
    task_id = task_id_start
    tasks: list[FlowTask] = []
    previous_by_src: dict[int, int] = {}
    stage_id = 0

    def append_stage(
        step: int,
        size_bytes: int,
        previous_partner_offset: int | None,
    ) -> None:
        nonlocal task_id, previous_by_src, stage_id
        current_by_src: dict[int, int] = {}
        for index, src in enumerate(ranks):
            partner = ranks[index ^ (1 << step)]
            if previous_partner_offset is None:
                deps = []
            else:
                previous_partner = ranks[index ^ previous_partner_offset]
                deps = [previous_by_src[previous_partner]]
            tasks.append(
                FlowTask(
                    task_id=task_id,
                    job_id=job_id,
                    type=TaskType.FLOW,
                    src=src,
                    dst=partner,
                    size_bytes=size_bytes,
                    comm_type=comm_type,
                    chunk_id=stage_id,
                    num_chunks=(2 * stages if base == "ALLREDUCE" else stages),
                    deps=deps,
                )
            )
            current_by_src[src] = task_id
            task_id += 1
        previous_by_src = current_by_src
        stage_id += 1

    if base in {"REDUCESCATTER", "ALLREDUCE"}:
        for step in range(stages):
            previous_offset = None if step == 0 else 1 << (step - 1)
            append_stage(step, data_size // (1 << (step + 1)), previous_offset)
    if base == "ALLREDUCE":
        for step in range(stages):
            previous_offset = 1 << (stages - 1) if step == 0 else 1 << (step - 1)
            append_stage(step, data_size // n * (1 << step), previous_offset)
    elif base == "ALLGATHER":
        for step in range(stages):
            previous_offset = None if step == 0 else 1 << (step - 1)
            append_stage(step, data_size // n * (1 << step), previous_offset)
    return tasks


def _expand_double_binary_tree_collective(
    *,
    ranks: list[int],
    data_size: int,
    base: str,
    comm_type: CommType,
    job_id: int,
    task_id_start: int,
) -> list[FlowTask]:
    """Split a collective over two shifted logical binary trees.

    DoubleBinaryTreeTopology alternates root-min and root-max trees.  With no
    physical topology input in this expander, a cyclic rank shift provides the
    corresponding second tree with different root/intermediate roles.
    """
    if len(ranks) < 2:
        return []
    _validate_ranks(ranks, f"{base.lower()} double_binary_tree")
    chunk_sizes = _split_bytes(data_size, 2, f"double_binary_tree {base.lower()}")
    tasks: list[FlowTask] = []
    next_task_id = task_id_start
    for chunk_id, (tree_ranks, chunk_size) in enumerate(
        ((ranks, chunk_sizes[0]), (ranks[1:] + ranks[:1], chunk_sizes[1]))
    ):
        chunk_tasks = _expand_tree_collective(
            ranks=tree_ranks,
            data_size=chunk_size,
            base=base,
            comm_type=comm_type,
            job_id=job_id,
            task_id_start=next_task_id,
            chunk_id=chunk_id,
            num_chunks=2,
        )
        tasks.extend(chunk_tasks)
        next_task_id += len(chunk_tasks)
    return tasks

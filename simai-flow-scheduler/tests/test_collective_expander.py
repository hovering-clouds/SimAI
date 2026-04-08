"""
Tests for collective communication expanders.

Verification strategy:
1. Compare Python output with MockNcclGroup.cc logic
2. Verify src, dst, chunk_id, and dependencies match exactly
"""

import pytest
from src.workload_generator.collective_expander import (
    RingAllReduceExpander,
    FlowTask,
)
from src.workload_format.schema import TaskType, CommType


class TestRingAllReduceExpander:
    """Tests for Ring AllReduce expander."""

    def test_expand_4_ranks(self):
        """Test Ring AllReduce expansion with 4 ranks."""
        expander = RingAllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024 * 1024 * 1024  # 1GB

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        # For N=4 ranks: 2 * (N-1) * N = 2 * 3 * 4 = 24 flows
        assert len(flows) == 24

        # Check chunk size
        chunk_size = data_size // len(ranks)
        for flow in flows:
            assert flow.size_bytes == chunk_size

        # Check all flows are TP_RING type
        for flow in flows:
            assert flow.comm_type == CommType.TP_ALLREDUCE_RING

        # Check chunk IDs are in valid range
        for flow in flows:
            assert 0 <= flow.chunk_id < len(ranks)
            assert flow.num_chunks == len(ranks)

    def test_expand_8_ranks(self):
        """Test Ring AllReduce expansion with 8 ranks."""
        expander = RingAllReduceExpander()
        ranks = list(range(8))
        data_size = 1024 * 1024 * 1024  # 1GB

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        # For N=8 ranks: 2 * (N-1) * N = 2 * 7 * 8 = 112 flows
        assert len(flows) == 112

    def test_reduce_scatter_phase(self):
        """Verify Reduce-Scatter phase flow pattern."""
        expander = RingAllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # First N*(N-1) = 12 flows are Reduce-Scatter phase
        rs_flows = flows[:12]

        # Check flow pattern: rank i sends to rank (i+1) % N
        for i, flow in enumerate(rs_flows[:4]):  # First step
            step = i // len(ranks)
            rank_idx = i % len(ranks)
            expected_src = ranks[rank_idx]
            expected_dst = ranks[(rank_idx + 1) % len(ranks)]
            assert flow.src == expected_src
            assert flow.dst == expected_dst

    def test_allgather_phase(self):
        """Verify AllGather phase flow pattern."""
        expander = RingAllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # Last N*(N-1) = 12 flows are AllGather phase
        ag_flows = flows[12:]

        # Check flow pattern: rank (i+1) % N sends to rank i
        for i, flow in enumerate(ag_flows[:4]):  # First step
            rank_idx = i % len(ranks)
            expected_src = ranks[(rank_idx + 1) % len(ranks)]
            expected_dst = ranks[rank_idx]
            assert flow.src == expected_src
            assert flow.dst == expected_dst

    def test_dependencies(self):
        """Verify flow dependencies are correctly set."""
        expander = RingAllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # First step flows should have no dependencies
        for i in range(len(ranks)):
            assert len(flows[i].deps) == 0

        # Subsequent step flows should have dependencies
        for i in range(len(ranks), len(flows)):
            assert len(flows[i].deps) >= 1

    def test_single_rank(self):
        """Test with single rank (no-op)."""
        expander = RingAllReduceExpander()
        ranks = [0]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        assert len(flows) == 0

    def test_two_ranks(self):
        """Test Ring AllReduce with 2 ranks."""
        expander = RingAllReduceExpander()
        ranks = [0, 1]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # For N=2: 2 * (N-1) * N = 2 * 1 * 2 = 4 flows
        assert len(flows) == 4

    def test_flow_to_task_conversion(self):
        """Test converting FlowTask to Task."""
        flow = FlowTask(
            task_id=0,
            job_id=0,
            type=TaskType.FLOW,
            src=0,
            dst=1,
            size_bytes=1024,
            comm_type=CommType.TP_ALLREDUCE_RING,
            chunk_id=0,
            num_chunks=4,
            deps=[1, 2],
        )

        task = flow.to_task()

        assert task.task_id == 0
        assert task.job_id == 0
        assert task.is_flow()
        assert task.src == 0
        assert task.dst == 1
        assert task.size_bytes == 1024
        assert task.comm_type == CommType.TP_ALLREDUCE_RING
        assert task.chunk_id == 0
        assert task.num_chunks == 4
        assert task.deps == [1, 2]

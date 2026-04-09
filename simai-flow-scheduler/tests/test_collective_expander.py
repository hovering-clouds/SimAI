"""
Tests for collective communication expanders.

Verification strategy:
1. Compare Python output with MockNcclGroup.cc logic
2. Verify src, dst, chunk_id, and dependencies match exactly

C++ reference (MockNcclGroup.cc lines 1047-1310):
- Phase 1 (initial chunk): n flows, chunk_id=0, no parent deps
- Phase 2 (RS iterations): n-2 steps, each n flows, chunk_id=1..n-2
  RS phase total = Phase 1 + Phase 2 = 1 + (n-2) = n-1 steps
- Phase 3 (AG iterations): n-1 steps, each n flows, chunk_id=n-1..2n-3
  AG phase is independent, starts after RS completes
- Total: n + n*(n-2) + n*(n-1) = n * 2*(n-1) flows
- Dependency: each flow depends on task_list[prev_rank] from previous step
- num_chunks = 2*(n-1) (total step count)
"""

import pytest
from src.workload_generator.collective_expander import (
    AllReduceExpander,
    AllGatherExpander,
    ReduceScatterExpander,
    AlltoAllExpander,
    FlowTask,
)
from src.workload_format.schema import TaskType, CommType


class TestAllReduceExpander:
    """Tests for Ring AllReduce expander matching MockNcclGroup.cc logic."""

    def _build_expected_flows_4ranks(self) -> list[dict]:
        """
        Build the expected flow list for 4 ranks matching C++ output exactly.

        Ring: 0->1->2->3->0
        prev: {0:3, 1:0, 2:1, 3:2}
        next: {0:1, 1:2, 2:3, 3:0}
        chunk_count = 2*(4-1) = 6

        Phase 1 (initial, chunk_id=0): 4 flows
          Flow 0: 0->1, deps=[]
          Flow 1: 1->2, deps=[]
          Flow 2: 2->3, deps=[]
          Flow 3: 3->0, deps=[]
          task_list = {0:0, 1:1, 2:2, 3:3}

        Phase 2 (RS, step 0, chunk_id=1): 4 flows
          Flow 4: 0->1, deps=[task_list[prev=3]=3]
          Flow 5: 1->2, deps=[task_list[prev=0]=0]
          Flow 6: 2->3, deps=[task_list[prev=1]=1]
          Flow 7: 3->0, deps=[task_list[prev=2]=2]
          task_list = {0:4, 1:5, 2:6, 3:7}

        Phase 2 (RS, step 1, chunk_id=2): 4 flows
          Flow 8:  0->1, deps=[task_list[prev=3]=7]
          Flow 9:  1->2, deps=[task_list[prev=0]=4]
          Flow 10: 2->3, deps=[task_list[prev=1]=5]
          Flow 11: 3->0, deps=[task_list[prev=2]=6]
          task_list = {0:8, 1:9, 2:10, 3:11}

        Phase 3 (AG, step 0, chunk_id=3): 4 flows
          Flow 12: 0->1, deps=[task_list[prev=3]=11]
          Flow 13: 1->2, deps=[task_list[prev=0]=8]
          Flow 14: 2->3, deps=[task_list[prev=1]=9]
          Flow 15: 3->0, deps=[task_list[prev=2]=10]
          task_list = {0:12, 1:13, 2:14, 3:15}

        Phase 3 (AG, step 1, chunk_id=4): 4 flows
          Flow 16: 0->1, deps=[task_list[prev=3]=15]
          Flow 17: 1->2, deps=[task_list[prev=0]=12]
          Flow 18: 2->3, deps=[task_list[prev=1]=13]
          Flow 19: 3->0, deps=[task_list[prev=2]=14]
          task_list = {0:16, 1:17, 2:18, 3:19}

        Phase 3 (AG, step 2, chunk_id=5): 4 flows
          Flow 20: 0->1, deps=[task_list[prev=3]=19]
          Flow 21: 1->2, deps=[task_list[prev=0]=16]
          Flow 22: 2->3, deps=[task_list[prev=1]=17]
          Flow 23: 3->0, deps=[task_list[prev=2]=18]
        """
        return [
            # Phase 1: initial chunk (chunk_id=0)
            {"task_id": 0, "src": 0, "dst": 1, "chunk_id": 0, "num_chunks": 6, "deps": []},
            {"task_id": 1, "src": 1, "dst": 2, "chunk_id": 0, "num_chunks": 6, "deps": []},
            {"task_id": 2, "src": 2, "dst": 3, "chunk_id": 0, "num_chunks": 6, "deps": []},
            {"task_id": 3, "src": 3, "dst": 0, "chunk_id": 0, "num_chunks": 6, "deps": []},
            # Phase 2: RS step 0 (chunk_id=1)
            {"task_id": 4, "src": 0, "dst": 1, "chunk_id": 1, "num_chunks": 6, "deps": [3]},
            {"task_id": 5, "src": 1, "dst": 2, "chunk_id": 1, "num_chunks": 6, "deps": [0]},
            {"task_id": 6, "src": 2, "dst": 3, "chunk_id": 1, "num_chunks": 6, "deps": [1]},
            {"task_id": 7, "src": 3, "dst": 0, "chunk_id": 1, "num_chunks": 6, "deps": [2]},
            # Phase 2: RS step 1 (chunk_id=2)
            {"task_id": 8, "src": 0, "dst": 1, "chunk_id": 2, "num_chunks": 6, "deps": [7]},
            {"task_id": 9, "src": 1, "dst": 2, "chunk_id": 2, "num_chunks": 6, "deps": [4]},
            {"task_id": 10, "src": 2, "dst": 3, "chunk_id": 2, "num_chunks": 6, "deps": [5]},
            {"task_id": 11, "src": 3, "dst": 0, "chunk_id": 2, "num_chunks": 6, "deps": [6]},
            # Phase 3: AG step 0 (chunk_id=3)
            {"task_id": 12, "src": 0, "dst": 1, "chunk_id": 3, "num_chunks": 6, "deps": [11]},
            {"task_id": 13, "src": 1, "dst": 2, "chunk_id": 3, "num_chunks": 6, "deps": [8]},
            {"task_id": 14, "src": 2, "dst": 3, "chunk_id": 3, "num_chunks": 6, "deps": [9]},
            {"task_id": 15, "src": 3, "dst": 0, "chunk_id": 3, "num_chunks": 6, "deps": [10]},
            # Phase 3: AG step 1 (chunk_id=4)
            {"task_id": 16, "src": 0, "dst": 1, "chunk_id": 4, "num_chunks": 6, "deps": [15]},
            {"task_id": 17, "src": 1, "dst": 2, "chunk_id": 4, "num_chunks": 6, "deps": [12]},
            {"task_id": 18, "src": 2, "dst": 3, "chunk_id": 4, "num_chunks": 6, "deps": [13]},
            {"task_id": 19, "src": 3, "dst": 0, "chunk_id": 4, "num_chunks": 6, "deps": [14]},
            # Phase 3: AG step 2 (chunk_id=5)
            {"task_id": 20, "src": 0, "dst": 1, "chunk_id": 5, "num_chunks": 6, "deps": [19]},
            {"task_id": 21, "src": 1, "dst": 2, "chunk_id": 5, "num_chunks": 6, "deps": [16]},
            {"task_id": 22, "src": 2, "dst": 3, "chunk_id": 5, "num_chunks": 6, "deps": [17]},
            {"task_id": 23, "src": 3, "dst": 0, "chunk_id": 5, "num_chunks": 6, "deps": [18]},
        ]

    def test_expand_4_ranks_count(self):
        """Test Ring AllReduce expansion with 4 ranks - correct flow count."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024 * 1024 * 1024  # 1GB

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        # For N=4: n + n*(n-1) + n*(n-2) = 4 + 12 + 8 = 24 flows
        assert len(flows) == 24

    def test_expand_4_ranks_matches_cpp_exactly(self):
        """Test that Python output matches C++ MockNcclGroup.cc output exactly."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024 * 1024 * 1024
        chunk_size = data_size // len(ranks)

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        expected = self._build_expected_flows_4ranks()

        assert len(flows) == len(expected), f"Expected {len(expected)} flows, got {len(flows)}"

        for i, (flow, exp) in enumerate(zip(flows, expected)):
            assert flow.task_id == exp["task_id"], \
                f"Flow {i}: task_id mismatch, expected {exp['task_id']}, got {flow.task_id}"
            assert flow.src == exp["src"], \
                f"Flow {i}: src mismatch, expected {exp['src']}, got {flow.src}"
            assert flow.dst == exp["dst"], \
                f"Flow {i}: dst mismatch, expected {exp['dst']}, got {flow.dst}"
            assert flow.chunk_id == exp["chunk_id"], \
                f"Flow {i}: chunk_id mismatch, expected {exp['chunk_id']}, got {flow.chunk_id}"
            assert flow.num_chunks == exp["num_chunks"], \
                f"Flow {i}: num_chunks mismatch, expected {exp['num_chunks']}, got {flow.num_chunks}"
            assert flow.size_bytes == chunk_size, \
                f"Flow {i}: size_bytes mismatch, expected {chunk_size}, got {flow.size_bytes}"
            assert flow.deps == exp["deps"], \
                f"Flow {i}: deps mismatch, expected {exp['deps']}, got {flow.deps}"

    def test_expand_8_ranks(self):
        """Test Ring AllReduce expansion with 8 ranks."""
        expander = AllReduceExpander()
        ranks = list(range(8))
        data_size = 1024 * 1024 * 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        # For N=8:
        # Phase 1: 1 step (8 flows)
        # Phase 2 (RS): n-2 = 6 steps (48 flows)
        # Phase 3 (AG): n-1 = 7 steps (56 flows)
        # Total: 8 + 48 + 56 = 112 flows
        assert len(flows) == 112

        chunk_size = data_size // len(ranks)
        for flow in flows:
            assert flow.size_bytes == chunk_size

    def test_initial_chunk_no_dependencies(self):
        """Verify initial chunk flows have no dependencies (matching C++)."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # First n flows are the initial chunk
        for i in range(len(ranks)):
            assert len(flows[i].deps) == 0, \
                f"Initial flow {i} should have no deps, got {flows[i].deps}"

    def test_rs_iterations_have_dependencies(self):
        """Verify RS iteration flows have diagonal dependencies."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # RS iterations start after initial chunk (flow n onwards)
        # Each should have exactly 1 dependency
        for i in range(len(ranks), len(flows)):
            assert len(flows[i].deps) == 1, \
                f"Flow {i} should have exactly 1 dep, got {flows[i].deps}"

    def test_ring_topology_pattern(self):
        """Verify src->dst follows ring pattern: rank i -> rank (i+1) % n."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        n = len(ranks)
        # All flows should follow: rank i sends to (i+1) % n
        # Group flows by step (each step has n flows)
        for step_start in range(0, len(flows), n):
            step_flows = flows[step_start:step_start + n]
            for rank_idx, flow in enumerate(step_flows):
                expected_src = ranks[rank_idx]
                expected_dst = ranks[(rank_idx + 1) % n]
                assert flow.src == expected_src, \
                    f"Flow at step {step_start//n}, rank {rank_idx}: src={flow.src}, expected {expected_src}"
                assert flow.dst == expected_dst, \
                    f"Flow at step {step_start//n}, rank {rank_idx}: dst={flow.dst}, expected {expected_dst}"

    def test_chunk_id_sequence(self):
        """Verify chunk_id increments correctly: 0, 1, 2, ..., 2*(n-1)-1.

        For 4 ranks: chunk_ids are 0 (Phase 1), 1-2 (Phase 2 RS), 3-5 (Phase 3 AG).
        Total = 6 chunks = 2*(4-1).
        """
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        n = len(ranks)
        expected_chunk_count = 2 * (n - 1)  # 6 for 4 ranks

        # Each step (group of n flows) should have the same chunk_id
        # chunk_ids should be: 0, 1, 2, 3, 4, 5
        for step_idx in range(expected_chunk_count):
            step_start = step_idx * n
            step_flows = flows[step_start:step_start + n]
            for flow in step_flows:
                assert flow.chunk_id == step_idx, \
                    f"Flow {flow.task_id}: chunk_id={flow.chunk_id}, expected {step_idx}"
                assert flow.num_chunks == expected_chunk_count, \
                    f"Flow {flow.task_id}: num_chunks={flow.num_chunks}, expected {expected_chunk_count}"

    def test_single_rank(self):
        """Test with single rank (no-op)."""
        expander = AllReduceExpander()
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
        expander = AllReduceExpander()
        ranks = [0, 1]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # For N=2:
        # Phase 1: 1 step (2 flows)
        # Phase 2 (RS): n-2 = 0 steps
        # Phase 3 (AG): n-1 = 1 step (2 flows)
        # Total: 2 + 0 + 2 = 4 flows

        # Structure: initial(2) + RS(0) + AG(2)
        # Initial: 0->1, 1->0
        # AG step 0: 0->1 (dep on 1->0), 1->0 (dep on 0->1)
        assert flows[0].src == 0 and flows[0].dst == 1 and flows[0].deps == []
        assert flows[1].src == 1 and flows[1].dst == 0 and flows[1].deps == []
        assert flows[2].src == 0 and flows[2].dst == 1 and flows[2].deps == [1]
        assert flows[3].src == 1 and flows[3].dst == 0 and flows[3].deps == [0]

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
            num_chunks=6,
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
        assert task.num_chunks == 6
        assert task.deps == [1, 2]

    def test_dependency_chain_integrity(self):
        """Verify that all dependency references are valid (dep task_id exists)."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        valid_task_ids = {f.task_id for f in flows}
        for flow in flows:
            for dep in flow.deps:
                assert dep in valid_task_ids, \
                    f"Flow {flow.task_id} references non-existent dep {dep}"

    def test_dag_no_cycles(self):
        """Verify the dependency graph is a DAG (no cycles)."""
        expander = AllReduceExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allreduce(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # Build adjacency list
        adj = {}
        for flow in flows:
            adj[flow.task_id] = flow.deps

        # Topological sort check
        visited = set()
        in_stack = set()

        def has_cycle(node):
            visited.add(node)
            in_stack.add(node)
            for dep in adj.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in in_stack:
                    return True
            in_stack.remove(node)
            return False

        for task_id in adj:
            if task_id not in visited:
                assert not has_cycle(task_id), "DAG contains cycles"


class TestAllGatherExpander:
    """Tests for Ring AllGather expander matching MockNcclGroup.cc logic.

    C++ reference (MockNcclGroup.cc lines 1417-1682):
    - Phase 1 (initial send): n flows, chunk_id=0, no deps
    - Phase 2 (forwarding): n-2 steps, each n flows, chunk_id=1..n-2
    - Total: n + n*(n-2) = n*(n-1) flows
    - Dependency: each flow depends on task_list[prev_rank] from previous step
    - num_chunks = n-1 (total step count)
    """

    def _build_expected_flows_4ranks(self) -> list[dict]:
        """
        Build expected flow list for 4 ranks matching C++ output exactly.

        Ring: 0->1->2->3->0
        prev: {0:3, 1:0, 2:1, 3:2}
        chunk_count = 4-1 = 3

        Phase 1 (initial send, chunk_id=0): 4 flows
          Flow 0: 0->1, deps=[]
          Flow 1: 1->2, deps=[]
          Flow 2: 2->3, deps=[]
          Flow 3: 3->0, deps=[]
          task_list = {0:0, 1:1, 2:2, 3:3}

        Phase 2 (forwarding, step 1, chunk_id=1): 4 flows
          Flow 4: 0->1, deps=[task_list[prev=3]=3]
          Flow 5: 1->2, deps=[task_list[prev=0]=0]
          Flow 6: 2->3, deps=[task_list[prev=1]=1]
          Flow 7: 3->0, deps=[task_list[prev=2]=2]
          task_list = {0:4, 1:5, 2:6, 3:7}

        Phase 2 (forwarding, step 2, chunk_id=2): 4 flows
          Flow 8:  0->1, deps=[task_list[prev=3]=7]
          Flow 9:  1->2, deps=[task_list[prev=0]=4]
          Flow 10: 2->3, deps=[task_list[prev=1]=5]
          Flow 11: 3->0, deps=[task_list[prev=2]=6]
        """
        return [
            # Phase 1: initial send (chunk_id=0)
            {"task_id": 0, "src": 0, "dst": 1, "chunk_id": 0, "num_chunks": 3, "deps": []},
            {"task_id": 1, "src": 1, "dst": 2, "chunk_id": 0, "num_chunks": 3, "deps": []},
            {"task_id": 2, "src": 2, "dst": 3, "chunk_id": 0, "num_chunks": 3, "deps": []},
            {"task_id": 3, "src": 3, "dst": 0, "chunk_id": 0, "num_chunks": 3, "deps": []},
            # Phase 2: forwarding step 1 (chunk_id=1)
            {"task_id": 4, "src": 0, "dst": 1, "chunk_id": 1, "num_chunks": 3, "deps": [3]},
            {"task_id": 5, "src": 1, "dst": 2, "chunk_id": 1, "num_chunks": 3, "deps": [0]},
            {"task_id": 6, "src": 2, "dst": 3, "chunk_id": 1, "num_chunks": 3, "deps": [1]},
            {"task_id": 7, "src": 3, "dst": 0, "chunk_id": 1, "num_chunks": 3, "deps": [2]},
            # Phase 2: forwarding step 2 (chunk_id=2)
            {"task_id": 8, "src": 0, "dst": 1, "chunk_id": 2, "num_chunks": 3, "deps": [7]},
            {"task_id": 9, "src": 1, "dst": 2, "chunk_id": 2, "num_chunks": 3, "deps": [4]},
            {"task_id": 10, "src": 2, "dst": 3, "chunk_id": 2, "num_chunks": 3, "deps": [5]},
            {"task_id": 11, "src": 3, "dst": 0, "chunk_id": 2, "num_chunks": 3, "deps": [6]},
        ]

    def test_expand_4_ranks_count(self):
        """Test Ring AllGather expansion with 4 ranks - correct flow count."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024 * 1024 * 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        # For N=4: n*(n-1) = 4*3 = 12 flows
        assert len(flows) == 12

    def test_expand_4_ranks_matches_cpp_exactly(self):
        """Test that Python output matches C++ MockNcclGroup.cc output exactly."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024 * 1024 * 1024
        chunk_size = data_size // len(ranks)

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        expected = self._build_expected_flows_4ranks()

        assert len(flows) == len(expected), f"Expected {len(expected)} flows, got {len(flows)}"

        for i, (flow, exp) in enumerate(zip(flows, expected)):
            assert flow.task_id == exp["task_id"], \
                f"Flow {i}: task_id mismatch, expected {exp['task_id']}, got {flow.task_id}"
            assert flow.src == exp["src"], \
                f"Flow {i}: src mismatch, expected {exp['src']}, got {flow.src}"
            assert flow.dst == exp["dst"], \
                f"Flow {i}: dst mismatch, expected {exp['dst']}, got {flow.dst}"
            assert flow.chunk_id == exp["chunk_id"], \
                f"Flow {i}: chunk_id mismatch, expected {exp['chunk_id']}, got {flow.chunk_id}"
            assert flow.num_chunks == exp["num_chunks"], \
                f"Flow {i}: num_chunks mismatch, expected {exp['num_chunks']}, got {flow.num_chunks}"
            assert flow.size_bytes == chunk_size, \
                f"Flow {i}: size_bytes mismatch, expected {chunk_size}, got {flow.size_bytes}"
            assert flow.deps == exp["deps"], \
                f"Flow {i}: deps mismatch, expected {exp['deps']}, got {flow.deps}"
            assert flow.comm_type == CommType.TP_ALLGATHER_RING, \
                f"Flow {i}: comm_type mismatch, expected TP_ALLGATHER_RING, got {flow.comm_type}"

    def test_expand_8_ranks(self):
        """Test Ring AllGather expansion with 8 ranks."""
        expander = AllGatherExpander()
        ranks = list(range(8))
        data_size = 1024 * 1024 * 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            algo="ring",
            job_id=0,
            task_id_start=0,
        )

        # For N=8: n*(n-1) = 8*7 = 56 flows
        assert len(flows) == 56

        chunk_size = data_size // len(ranks)
        for flow in flows:
            assert flow.size_bytes == chunk_size

    def test_initial_send_no_dependencies(self):
        """Verify initial send flows have no dependencies."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # First n flows are the initial send
        for i in range(len(ranks)):
            assert len(flows[i].deps) == 0, \
                f"Initial flow {i} should have no deps, got {flows[i].deps}"

    def test_forwarding_has_dependencies(self):
        """Verify forwarding flows have diagonal dependencies."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # Forwarding flows start after initial send (flow n onwards)
        for i in range(len(ranks), len(flows)):
            assert len(flows[i].deps) == 1, \
                f"Flow {i} should have exactly 1 dep, got {flows[i].deps}"

    def test_two_ranks(self):
        """Test Ring AllGather with 2 ranks."""
        expander = AllGatherExpander()
        ranks = [0, 1]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        # For N=2: n*(n-1) = 2*1 = 2 flows
        # (n-2 = 0 forwarding steps, only initial send)
        assert len(flows) == 2

        assert flows[0].src == 0 and flows[0].dst == 1 and flows[0].deps == []
        assert flows[1].src == 1 and flows[1].dst == 0 and flows[1].deps == []

    def test_single_rank(self):
        """Test with single rank (no-op)."""
        expander = AllGatherExpander()
        ranks = [0]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        assert len(flows) == 0

    def test_chunk_id_sequence(self):
        """Verify chunk_id increments correctly: 0, 1, ..., n-2."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        n = len(ranks)
        expected_chunk_count = n - 1  # 3 for 4 ranks

        for step_idx in range(expected_chunk_count):
            step_start = step_idx * n
            step_flows = flows[step_start:step_start + n]
            for flow in step_flows:
                assert flow.chunk_id == step_idx, \
                    f"Flow {flow.task_id}: chunk_id={flow.chunk_id}, expected {step_idx}"
                assert flow.num_chunks == expected_chunk_count

    def test_dependency_chain_integrity(self):
        """Verify all dependency references are valid."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        valid_task_ids = {f.task_id for f in flows}
        for flow in flows:
            for dep in flow.deps:
                assert dep in valid_task_ids, \
                    f"Flow {flow.task_id} references non-existent dep {dep}"

    def test_dag_no_cycles(self):
        """Verify the dependency graph is a DAG (no cycles)."""
        expander = AllGatherExpander()
        ranks = [0, 1, 2, 3]
        data_size = 1024

        flows = expander.expand_allgather(
            ranks=ranks,
            data_size=data_size,
            job_id=0,
            task_id_start=0,
        )

        adj = {}
        for flow in flows:
            adj[flow.task_id] = flow.deps

        visited = set()
        in_stack = set()

        def has_cycle(node):
            visited.add(node)
            in_stack.add(node)
            for dep in adj.get(node, []):
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in in_stack:
                    return True
            in_stack.remove(node)
            return False

        for task_id in adj:
            if task_id not in visited:
                assert not has_cycle(task_id), "DAG contains cycles"


# ─── ReduceScatter Expander Tests ────────────────────────────────────────────

class TestReduceScatterExpander:
    """Tests for Ring ReduceScatter expander.

    ReduceScatter has the same ring structure as AllGather:
    - Phase 1 (chunk_id=0): n flows, no deps
    - Phase 2 (chunk_id=1..n-2): n-2 steps, diagonal deps
    - Total: n*(n-1) flows
    """

    def _build_expected_flows_4ranks(self) -> list[dict]:
        return [
            {"task_id": 0, "src": 0, "dst": 1, "chunk_id": 0, "deps": []},
            {"task_id": 1, "src": 1, "dst": 2, "chunk_id": 0, "deps": []},
            {"task_id": 2, "src": 2, "dst": 3, "chunk_id": 0, "deps": []},
            {"task_id": 3, "src": 3, "dst": 0, "chunk_id": 0, "deps": []},
            {"task_id": 4, "src": 0, "dst": 1, "chunk_id": 1, "deps": [3]},
            {"task_id": 5, "src": 1, "dst": 2, "chunk_id": 1, "deps": [0]},
            {"task_id": 6, "src": 2, "dst": 3, "chunk_id": 1, "deps": [1]},
            {"task_id": 7, "src": 3, "dst": 0, "chunk_id": 1, "deps": [2]},
            {"task_id": 8, "src": 0, "dst": 1, "chunk_id": 2, "deps": [7]},
            {"task_id": 9, "src": 1, "dst": 2, "chunk_id": 2, "deps": [4]},
            {"task_id": 10, "src": 2, "dst": 3, "chunk_id": 2, "deps": [5]},
            {"task_id": 11, "src": 3, "dst": 0, "chunk_id": 2, "deps": [6]},
        ]

    def test_expand_4_ranks_count(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        assert len(flows) == 12

    def test_expand_4_ranks_exact(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        expected = self._build_expected_flows_4ranks()
        for i, (flow, exp) in enumerate(zip(flows, expected)):
            assert flow.task_id == exp["task_id"]
            assert flow.src == exp["src"]
            assert flow.dst == exp["dst"]
            assert flow.chunk_id == exp["chunk_id"]
            assert flow.deps == exp["deps"]
            assert flow.comm_type == CommType.TP_REDUCESCATTER_RING
            assert flow.size_bytes == 256
            assert flow.num_chunks == 3

    def test_expand_8_ranks(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter(list(range(8)), 1024 * 1024)
        assert len(flows) == 56
        assert all(f.comm_type == CommType.TP_REDUCESCATTER_RING for f in flows)

    def test_initial_no_deps(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        for flow in flows[:4]:
            assert flow.deps == []

    def test_propagation_has_deps(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        for flow in flows[4:]:
            assert len(flow.deps) == 1

    def test_two_ranks(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1], 1024)
        assert len(flows) == 2
        assert all(f.deps == [] for f in flows)

    def test_single_rank(self):
        expander = ReduceScatterExpander()
        assert expander.expand_reducescatter([0], 1024) == []

    def test_ring_topology(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        for flow in flows:
            assert flow.dst == (flow.src + 1) % 4

    def test_chunk_id_sequence(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        assert [f.chunk_id for f in flows] == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    def test_dependency_chain_valid(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        valid_ids = {f.task_id for f in flows}
        for flow in flows:
            for dep in flow.deps:
                assert dep in valid_ids

    def test_dag_no_cycles(self):
        expander = ReduceScatterExpander()
        flows = expander.expand_reducescatter([0, 1, 2, 3], 1024)
        adj = {f.task_id: list(f.deps) for f in flows}
        visited, in_stack = set(), set()

        def has_cycle(node):
            visited.add(node); in_stack.add(node)
            for dep in adj[node]:
                if dep not in visited:
                    if has_cycle(dep): return True
                elif dep in in_stack: return True
            in_stack.discard(node); return False

        for tid in adj:
            if tid not in visited:
                assert not has_cycle(tid), "DAG contains cycles"

    def test_unsupported_algo(self):
        expander = ReduceScatterExpander()
        with pytest.raises(ValueError, match="unsupported algo"):
            expander.expand_reducescatter([0, 1], 1024, algo="tree")


# ─── AlltoAll Expander Tests ─────────────────────────────────────────────────

class TestAlltoAllExpander:
    """Tests for AlltoAll expander. N*(N-1) independent flows."""

    def test_expand_4_ranks_count(self):
        expander = AlltoAllExpander()
        assert len(expander.expand_alltoall([0, 1, 2, 3], 1024)) == 12

    def test_all_pairs_present(self):
        expander = AlltoAllExpander()
        ranks = [0, 1, 2, 3]
        flows = expander.expand_alltoall(ranks, 1024)
        pairs = {(f.src, f.dst) for f in flows}
        for src in ranks:
            for dst in ranks:
                if src != dst:
                    assert (src, dst) in pairs

    def test_no_self_loops(self):
        expander = AlltoAllExpander()
        for f in expander.expand_alltoall([0, 1, 2, 3], 1024):
            assert f.src != f.dst

    def test_no_dependencies(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1, 2, 3], 1024)
        assert all(f.deps == [] for f in flows)

    def test_chunk_fields(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1, 2, 3], 1024)
        assert all(f.chunk_id == 0 for f in flows)
        assert all(f.num_chunks == 1 for f in flows)

    def test_size_bytes(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1, 2, 3], 4096)
        assert all(f.size_bytes == 1024 for f in flows)

    def test_comm_type(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1, 2, 3], 1024)
        assert all(f.comm_type == CommType.TP_ALLTOALL for f in flows)

    def test_8_ranks(self):
        expander = AlltoAllExpander()
        assert len(expander.expand_alltoall(list(range(8)), 1024)) == 56

    def test_two_ranks(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1], 1024)
        assert len(flows) == 2
        assert {(f.src, f.dst) for f in flows} == {(0, 1), (1, 0)}

    def test_single_rank(self):
        expander = AlltoAllExpander()
        assert expander.expand_alltoall([0], 1024) == []

    def test_task_ids_sequential(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1, 2, 3], 1024, task_id_start=100)
        assert [f.task_id for f in flows] == list(range(100, 112))

    def test_task_ids_from_zero(self):
        expander = AlltoAllExpander()
        flows = expander.expand_alltoall([0, 1, 2, 3], 1024)
        assert [f.task_id for f in flows] == list(range(12))

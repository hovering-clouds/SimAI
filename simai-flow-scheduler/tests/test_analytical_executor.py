"""Tests for AnalyticalExecutor."""
import math

import pytest

from src.executor.analytical import AnalyticalExecutor
from src.executor.bandwidth import FairShareAllocator
from src.executor.policy import DefaultSchedulingPolicy
from src.executor.result import ExecutionResult, TaskTiming
from src.executor.runtime import ActiveFlow
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    Network,
    P2PWorkload,
    Phase,
    Task,
    TaskType,
)


# ── Helpers ──


def _make_2node_topology(bw_gbps: float = 100.0, latency_us: float = 1.0) -> NetworkTopology:
    """创建 2 节点直连拓扑。"""
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    topo.node_types = {0: "gpu", 1: "gpu"}

    link = Link(src=0, dst=1, bandwidth_gbps=bw_gbps, latency_us=latency_us, error_rate=0.0)
    topo.add_link(link)
    rev = Link(src=1, dst=0, bandwidth_gbps=bw_gbps, latency_us=latency_us, error_rate=0.0)
    topo.add_link(rev)

    return topo


def _make_4node_ring_topology(bw_gbps: float = 100.0, latency_us: float = 1.0) -> NetworkTopology:
    """创建 4 节点环拓扑：0-1, 1-2, 2-3, 3-0。"""
    topo = NetworkTopology()
    topo.total_nodes = 4
    topo.gpu_count = 4
    topo.gpu_nodes = [0, 1, 2, 3]
    topo.switch_nodes = []
    topo.node_types = {i: "gpu" for i in range(4)}

    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for s, d in edges:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=bw_gbps, latency_us=latency_us, error_rate=0.0))
        topo.add_link(Link(src=d, dst=s, bandwidth_gbps=bw_gbps, latency_us=latency_us, error_rate=0.0))

    return topo


def _run_executor(
    workload: P2PWorkload,
    topology: NetworkTopology,
    allocator=None,
) -> ExecutionResult:
    """便捷方法：构建 policy 并运行 executor。"""
    analysis = DefaultAnalyzer(topology).analyze(workload)
    policy = DefaultSchedulingPolicy(analysis, allocator=allocator)
    executor = AnalyticalExecutor(topology, policy)
    return executor.execute(workload)


# ── Test: 单 compute 任务 ──


def test_single_compute():
    """单个 compute 任务：start=0, end=duration。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=1),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=500, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology()
    result = _run_executor(workload, topo)

    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 500
    assert result.total_time_us == 500


# ── Test: compute → flow 链式依赖 ──


def test_compute_then_flow():
    """compute(0→1) → flow(0→1)：flow 在 compute 完成后开始。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=125000000,  # 125 MB = 1 Gbit
                 comm_type=CommType.TP_ALLREDUCE_RING,
                 deps=[0], phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=1.0)
    result = _run_executor(workload, topo)

    # compute: 0→100
    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100

    # flow: start >= 100
    flow = result.per_task[1]
    assert flow.start_time_us == 100
    # 传输时间 = 125000000 * 8 / (100 * 1e3) = 10000 us，传播延迟 = 1 us
    # end = 100 + 1 + 10000 = 10101
    assert flow.end_time_us == 10101


# ── Test: 独占带宽无竞争 ──


def test_exclusive_bandwidth():
    """两条不冲突的 flow：各走不同链路，互不干扰。"""
    # 4 节点：0→1 和 2→3 走不同链路，无竞争
    topo = _make_4node_ring_topology(bw_gbps=100.0, latency_us=0.5)

    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=4),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=125000000,  # 1 Gbit
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=2, dst=3, size_bytes=125000000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )
    result = _run_executor(workload, topo)

    # 两条流独占带宽，传输时间 = 1 Gbit / 100 Gbps = 10 us
    # 传播延迟 = 0.5 us（单跳）
    for tid in [0, 1]:
        flow = result.per_task[tid]
        assert flow.start_time_us == 0
        # transmission = 125000000 * 8 / (100 * 1e3) = 10000 us
        # propagation = int(0.5) = 0
        expected_end = int(0.5) + int(125000000 * 8 / (100.0 * 1e3))
        assert flow.end_time_us == expected_end


# ── Test: 两条流竞争同一链路 ──


def test_contention_fair_share():
    """两条流共享同一链路：各得一半带宽，完成时间约为独占的 2 倍。"""
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=0.0)

    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=125000000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=125000000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )
    result = _run_executor(workload, topo)

    # 独占时 transmission = 10000 us，竞争时 bw=50 Gbps，transmission = 20000 us
    for tid in [0, 1]:
        flow = result.per_task[tid]
        assert flow.start_time_us == 0
        # 125000000 * 8 / (50 * 1e3) = 20000 us
        assert flow.end_time_us == 20000


# ── Test: compute_cursor 推进 ──


def test_sequential_computes():
    """同一节点上的多个 compute 任务严格串行执行。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD, layer_id=0),
            Task(task_id=1, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=200, phase=Phase.FORWARD, layer_id=1),
            Task(task_id=2, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=150, phase=Phase.FORWARD, layer_id=2),
        ],
    )
    topo = _make_2node_topology()
    result = _run_executor(workload, topo)

    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100
    assert result.per_task[1].start_time_us == 100
    assert result.per_task[1].end_time_us == 300
    assert result.per_task[2].start_time_us == 300
    assert result.per_task[2].end_time_us == 450


# ── Test: DAG 依赖链 compute → flow → compute ──


def test_compute_flow_compute_chain():
    """compute(A) → flow(A→B) → compute(B)：第二个 compute 在 flow 完成后开始。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD, layer_id=0),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=12500000,  # 0.1 Gbit
                 comm_type=CommType.TP_ALLREDUCE_RING,
                 deps=[0], phase=Phase.FORWARD, layer_id=0),
            Task(task_id=2, job_id=0, type=TaskType.COMPUTE,
                 node=1, duration_us=50, phase=Phase.FORWARD, layer_id=0,
                 deps=[1]),
        ],
    )
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=1.0)
    result = _run_executor(workload, topo)

    # compute_0: 0 → 100
    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100

    # flow: start=100, transmission = 12500000*8/(100*1e3) = 1000 us, latency = 1 us
    # end = 100 + 1 + 1000 = 1101
    assert result.per_task[1].start_time_us == 100
    assert result.per_task[1].end_time_us == 1101

    # compute_1: start=1101, duration=50, end=1151
    assert result.per_task[2].start_time_us == 1101
    assert result.per_task[2].end_time_us == 1151


# ── Test: 多节点 compute 各自独立 ──


def test_parallel_computes_different_nodes():
    """不同节点上的 compute 任务并行执行。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.COMPUTE,
                 node=1, duration_us=200, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology()
    result = _run_executor(workload, topo)

    # 两个 compute 同时从 time=0 开始
    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100
    assert result.per_task[1].start_time_us == 0
    assert result.per_task[1].end_time_us == 200
    assert result.total_time_us == 200


# ── Test: ExecutionResult 统计 ──


def test_execution_result_stats():
    """验证 ExecutionResult 的 total_time 和 makespan 计算。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.COMPUTE,
                 node=1, duration_us=300, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology()
    result = _run_executor(workload, topo)

    assert result.total_time_us == 300
    assert result.makespan_us == 300  # max_end(300) - min_start(0)
    assert result.job_iteration_times[0] == 300


# ── Test: FairShareAllocator 单元测试 ──


def test_fair_share_allocator_no_contention():
    """单条流独占链路：获得全部带宽。"""
    topo = _make_2node_topology(bw_gbps=100.0)
    allocator = FairShareAllocator()

    flow = ActiveFlow(
        task_id=0, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
        path=[0, 1], start_time=0, last_update_time=0,
    )
    from src.static_analysis.passes.routing_hints import RoutingHints
    hints = RoutingHints(topology=topo)

    result = allocator.allocate([flow], topo, hints, current_time=0)
    assert result[0] == 100.0


def test_fair_share_allocator_contention():
    """两条流共享链路：各得一半。"""
    topo = _make_2node_topology(bw_gbps=100.0)
    allocator = FairShareAllocator()

    flow1 = ActiveFlow(
        task_id=0, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
        path=[0, 1], start_time=0, last_update_time=0,
    )
    flow2 = ActiveFlow(
        task_id=1, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
        path=[0, 1], start_time=0, last_update_time=0,
    )
    from src.static_analysis.passes.routing_hints import RoutingHints
    hints = RoutingHints(topology=topo)

    result = allocator.allocate([flow1, flow2], topo, hints, current_time=0)
    assert result[0] == 50.0
    assert result[1] == 50.0


# ── Test: 带 flow 完成后带宽重新分配 ──


def test_bandwidth_reallocation_on_completion():
    """第一条流完成后，第二条流获得全部带宽。"""
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=0.0)

    # flow_0 较小 (10 us @ 50 Gbps)，flow_1 较大
    # 两条流同时开始，各得 50 Gbps
    # flow_0 完成后，flow_1 获得 100 Gbps，remaining_bytes 按新的 100 Gbps 计算
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=62500000,  # 0.5 Gbit, 10 us @ 50 Gbps
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=625000000,  # 5 Gbit
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )
    result = _run_executor(workload, topo)

    # flow_0: start=0, bw=50 Gbps
    #   transmission = 62500000 * 8 / (50 * 1e3) = 10000 us
    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 10000

    # flow_1: start=0
    #   0~10000 us: bw=50 Gbps, transmitted = 10000 * 50 * 1e3 / 8 = 62500000 bytes
    #   remaining after 10000 us = 625000000 - 62500000 = 562500000 bytes
    #   10000 us onwards: bw=100 Gbps
    #   transmission = 562500000 * 8 / (100 * 1e3) = 45000 us
    #   end = 10000 + 45000 = 55000 us
    assert result.per_task[1].start_time_us == 0
    assert result.per_task[1].end_time_us == 55000

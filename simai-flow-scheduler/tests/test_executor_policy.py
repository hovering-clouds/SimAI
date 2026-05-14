"""Tests for SchedulingPolicy and DefaultSchedulingPolicy."""
import pytest

from src.executor.analytical import AnalyticalExecutor
from src.executor.bandwidth import BandwidthAllocator, FairShareAllocator
from src.executor.policy import DefaultSchedulingPolicy, SchedulingPolicy
from src.executor.runtime import ActiveFlow
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.passes.routing_hints import RoutingHints, compute_routing_hints
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Phase,
    Task,
    TaskType,
)


# ── Helpers ──


def _make_2node_topology(bw_gbps=100.0, latency_us=1.0):
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    topo.node_types = {0: "gpu", 1: "gpu"}
    for s, d in [(0, 1), (1, 0)]:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=bw_gbps, latency_us=latency_us, error_rate=0.0))
    return topo


# ── Test: Default policy emits all ready tasks ──


def test_default_emits_all_ready():
    """默认策略的 emit_ready_tasks 返回所有 ready task 的 ID。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=1000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=2000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology()
    analysis = DefaultAnalyzer(topo).analyze(workload)
    policy = DefaultSchedulingPolicy(analysis)
    policy.initialize(workload, topo)

    ready = [t for t in workload.tasks if t.is_flow()]
    emitted = policy.emit_ready_tasks(0, ready)
    assert set(emitted) == {0, 1}


# ── Test: Default policy path equals RoutingHints.get_path ──


def test_default_path_matches_routing_hints():
    """默认策略 get_flow_path 返回与 RoutingHints 相同的路径。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=1000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology()
    analysis = DefaultAnalyzer(topo).analyze(workload)
    hints = analysis.routing_hints
    policy = DefaultSchedulingPolicy(analysis)
    policy.initialize(workload, topo)

    task = workload.tasks[0]
    assert policy.get_flow_path(task) == hints.get_path(0, 1)
    assert policy.get_flow_path(task) == [0, 1]


# ── Test: Default policy bandwidth equals FairShareAllocator ──


def test_default_bandwidth_equals_fair_share():
    """默认策略 allocate_bandwidth 返回与 FairShareAllocator 相同的结果。"""
    topo = _make_2node_topology(bw_gbps=100.0)
    empty_wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2), tasks=[])
    analysis = DefaultAnalyzer(topo).analyze(empty_wl)
    policy = DefaultSchedulingPolicy(analysis)

    flow1 = ActiveFlow(task_id=0, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
                       path=[0, 1], start_time=0, last_update_time=0)
    flow2 = ActiveFlow(task_id=1, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
                       path=[0, 1], start_time=0, last_update_time=0)

    # Without topology in policy, allocate_bandwidth needs topology
    # So we need to initialize the policy first
    policy.initialize(
        P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2), tasks=[]),
        topo,
    )

    result = policy.allocate_bandwidth(0, [flow1, flow2])
    assert result[0] == 50.0
    assert result[1] == 50.0


# ── Test: Custom policy can hold a ready flow ──


class HoldFirstFlowPolicy(SchedulingPolicy):
    """自定义策略：第一轮 drain 只放行 compute，延迟 flow 到后续轮次。"""

    def __init__(self, routing_hints):
        self.routing_hints = routing_hints
        self._topology = None
        self._emit_call_count = 0

    def initialize(self, workload, topology):
        self._topology = topology

    def emit_ready_tasks(self, current_time, ready_tasks):
        self._emit_call_count += 1
        if self._emit_call_count <= 2:
            # 初始 drain (t=0) 的前两次调用：只放行 compute，延迟 flow
            compute_ids = [t.task_id for t in ready_tasks if t.is_compute()]
            if compute_ids:
                return compute_ids
            return []
        # 后续 drain (compute_done 触发)：放行所有
        return [t.task_id for t in ready_tasks]

    def get_flow_path(self, task):
        return self.routing_hints.get_path(task.src, task.dst)

    def allocate_bandwidth(self, current_time, active_flows):
        return FairShareAllocator().allocate(active_flows, self._topology, self.routing_hints, current_time)

    def on_task_emitted(self, current_time, task):
        pass

    def on_task_completed(self, current_time, task):
        pass


def test_custom_policy_can_hold_flow():
    """自定义策略可以延迟 flow 准入。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=125000000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=1.0)
    hints = compute_routing_hints(topo, workload)

    policy = HoldFirstFlowPolicy(hints)
    executor = AnalyticalExecutor(topo, policy)
    result = executor.execute(workload)

    # compute_0 应该正常完成
    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100

    # flow_1 在 t=0 被策略延迟，在 compute_done 后 (t=100) 的 drain 中被放行
    assert 1 in result.per_task
    assert result.per_task[1].start_time_us == 100


# ── Test: Custom policy can return a precomputed path ──


class PrecomputedPathPolicy(SchedulingPolicy):
    """自定义策略：返回预计算的路径。"""

    def __init__(self, path_map):
        self.path_map = path_map
        self._topology = None

    def initialize(self, workload, topology):
        self._topology = topology

    def emit_ready_tasks(self, current_time, ready_tasks):
        return [t.task_id for t in ready_tasks]

    def get_flow_path(self, task):
        return self.path_map[(task.src, task.dst)]

    def allocate_bandwidth(self, current_time, active_flows):
        hints = RoutingHints(topology=self._topology)
        return FairShareAllocator().allocate(active_flows, self._topology, hints, current_time)

    def on_task_emitted(self, current_time, task):
        pass

    def on_task_completed(self, current_time, task):
        pass


def test_custom_policy_precomputed_path():
    """自定义策略可以返回预计算的路径。"""
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=1.0)

    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=125000000,
                 comm_type=CommType.TP_ALLREDUCE_RING, phase=Phase.FORWARD),
        ],
    )

    policy = PrecomputedPathPolicy({(0, 1): [0, 1]})
    executor = AnalyticalExecutor(topo, policy)
    result = executor.execute(workload)

    assert result.per_task[0].start_time_us == 0
    # transmission = 125000000 * 8 / (100 * 1e3) = 10000 us, propagation = 1 us
    assert result.per_task[0].end_time_us == 10001


# ── Test: Existing executor timing test produces identical result ──


def test_identical_timing_with_new_api():
    """使用新的 policy API 产生的任务计时与预期一致（compute → flow → compute 链）。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD, layer_id=0),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=12500000,
                 comm_type=CommType.TP_ALLREDUCE_RING,
                 deps=[0], phase=Phase.FORWARD, layer_id=0),
            Task(task_id=2, job_id=0, type=TaskType.COMPUTE,
                 node=1, duration_us=50, phase=Phase.FORWARD, layer_id=0,
                 deps=[1]),
        ],
    )
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=1.0)
    analysis = DefaultAnalyzer(topo).analyze(workload)

    policy = DefaultSchedulingPolicy(analysis)
    executor = AnalyticalExecutor(topo, policy)
    result = executor.execute(workload)

    assert result.per_task[0].start_time_us == 0
    assert result.per_task[0].end_time_us == 100
    assert result.per_task[1].start_time_us == 100
    assert result.per_task[1].end_time_us == 1101
    assert result.per_task[2].start_time_us == 1101
    assert result.per_task[2].end_time_us == 1151


# ── Test: Custom policy on_task_emitted / on_task_completed callbacks ──


class TrackingPolicy(SchedulingPolicy):
    """跟踪所有通知回调的策略。"""

    def __init__(self, routing_hints):
        self.routing_hints = routing_hints
        self._topology = None
        self.emitted: list[int] = []
        self.completed: list[int] = []

    def initialize(self, workload, topology):
        self._topology = topology

    def emit_ready_tasks(self, current_time, ready_tasks):
        return [t.task_id for t in ready_tasks]

    def get_flow_path(self, task):
        return self.routing_hints.get_path(task.src, task.dst)

    def allocate_bandwidth(self, current_time, active_flows):
        return FairShareAllocator().allocate(active_flows, self._topology, self.routing_hints, current_time)

    def on_task_emitted(self, current_time, task):
        self.emitted.append(task.task_id)

    def on_task_completed(self, current_time, task):
        self.completed.append(task.task_id)


def test_tracking_policy_callbacks():
    """自定义策略的 on_task_emitted / on_task_completed 回调被正确调用。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD),
            Task(task_id=1, job_id=0, type=TaskType.FLOW,
                 src=0, dst=1, size_bytes=12500000,
                 comm_type=CommType.TP_ALLREDUCE_RING,
                 deps=[0], phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology(bw_gbps=100.0, latency_us=1.0)
    hints = compute_routing_hints(topo, workload)

    policy = TrackingPolicy(hints)
    executor = AnalyticalExecutor(topo, policy)
    executor.execute(workload)

    # on_task_emitted: compute_0 at t=0, flow_1 after compute_done at t=100
    assert policy.emitted == [0, 1]
    # on_task_completed: compute_0 at t=100, flow_1 at t=1101
    assert policy.completed == [0, 1]


# ── Test: Deadlock detection ──


class NeverAdmitPolicy(SchedulingPolicy):
    """从不放行任何 task 的策略，用于测试死锁检测。"""

    def __init__(self):
        self._topology = None

    def initialize(self, workload, topology):
        self._topology = topology

    def emit_ready_tasks(self, current_time, ready_tasks):
        return []

    def get_flow_path(self, task):
        return [task.src, task.dst]

    def allocate_bandwidth(self, current_time, active_flows):
        return {}

    def on_task_emitted(self, current_time, task):
        pass

    def on_task_completed(self, current_time, task):
        pass


def test_deadlock_detection():
    """当 policy 从不放行 task 时，应抛出 RuntimeError 死锁错误。"""
    workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=1),
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, phase=Phase.FORWARD),
        ],
    )
    topo = _make_2node_topology()
    executor = AnalyticalExecutor(topo, NeverAdmitPolicy())

    with pytest.raises(RuntimeError, match="Deadlock detected"):
        executor.execute(workload)

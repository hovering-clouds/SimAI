"""Tests for PuppeteerSchedulingPolicy."""
import pytest

from src.executor.puppeteer_policy import PuppeteerSchedulingPolicy
from src.executor.runtime import ActiveFlow
from src.static_analysis.passes.puppeteer_coordination import ResourceDependencyTable
from src.static_analysis.passes.puppeteer_routing import RouteTable
from src.static_analysis.passes.puppeteer_tte import TTEInfo
from src.static_analysis.passes.task_serializer import ExecutionPlan
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Phase,
    Task,
    TaskType,
)


# --- Helpers ---


def _make_2node_topology(bw_gbps=100.0, latency_us=1.0):
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    for s, d in [(0, 1), (1, 0)]:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=bw_gbps, latency_us=latency_us, error_rate=0.0))
    return topo


def _make_flow(task_id, src, dst, size_bytes, deps=None):
    return Task(
        task_id=task_id, job_id=0, type=TaskType.FLOW,
        src=src, dst=dst, size_bytes=size_bytes,
        comm_type=CommType.TP_ALLREDUCE_RING, deps=deps or [],
    )


def _make_compute(task_id, duration_us, node=0, deps=None):
    return Task(
        task_id=task_id, job_id=0, type=TaskType.COMPUTE,
        node=node, duration_us=duration_us, deps=deps or [],
    )


# ============================================================
# Tests for PuppeteerSchedulingPolicy
# ============================================================


class TestPuppeteerSchedulingPolicy:
    def test_emit_flow_immediate_no_coordination(self):
        """Flow without coordination group is emitted immediately."""
        route_table = RouteTable(paths={0: [0, 1]})
        tte_info = {0: TTEInfo(task_id=0, tte_us=float("inf"), priority_score=0.0, priority_class="background")}
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        emitted = policy.emit_ready_tasks(0, [f0])
        assert emitted == [0]

    def test_emit_compute_in_order(self):
        """Compute tasks are emitted in the order specified by ExecutionPlan."""
        route_table = RouteTable()
        tte_info = {}
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan(compute_order={0: [0, 1]})

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=0)
        # c0 should be emitted first
        emitted = policy.emit_ready_tasks(0, [c0, c1])
        assert emitted == [0]

        # After completing c0, c1 should be next
        policy.on_task_completed(100, c0)
        emitted = policy.emit_ready_tasks(100, [c1])
        assert emitted == [1]

    def test_coordination_delays_flow(self):
        """Flow in coordination group waits for all members to be ready."""
        route_table = RouteTable(paths={0: [0, 1], 1: [0, 1]})
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=0.0, priority_score=0.0, priority_class="critical"),
            1: TTEInfo(task_id=1, tte_us=0.0, priority_score=0.0, priority_class="critical"),
        }
        resource_dep = ResourceDependencyTable(
            peers={0: {1}, 1: {0}},
            groups={"g0": {0, 1}},
        )
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)

        topo = _make_2node_topology()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2),
                         tasks=[_make_flow(0, 0, 1, 1000), _make_flow(1, 0, 1, 1000)])
        policy.initialize(wl, topo)

        # Only flow 0 is ready, flow 1 is not in ready pool yet
        emitted = policy.emit_ready_tasks(0, [wl.tasks[0]])
        assert emitted == []  # should wait

        # Both ready now
        emitted = policy.emit_ready_tasks(0, [wl.tasks[0], wl.tasks[1]])
        assert set(emitted) == {0, 1}

    def test_coordination_completed_flow_released(self):
        """If a group member is already completed, remaining member can proceed."""
        route_table = RouteTable(paths={0: [0, 1], 1: [0, 1]})
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=0.0, priority_score=0.0, priority_class="critical"),
            1: TTEInfo(task_id=1, tte_us=0.0, priority_score=0.0, priority_class="critical"),
        }
        resource_dep = ResourceDependencyTable(
            peers={0: {1}, 1: {0}},
            groups={"g0": {0, 1}},
        )
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        topo = _make_2node_topology()
        wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2),
                         tasks=[_make_flow(0, 0, 1, 1000), _make_flow(1, 0, 1, 1000)])
        policy.initialize(wl, topo)

        # Flow 0 is completed, flow 1 is ready
        policy.on_task_completed(100, wl.tasks[0])
        emitted = policy.emit_ready_tasks(100, [wl.tasks[1]])
        assert emitted == [1]

    def test_get_flow_path(self):
        """get_flow_path returns the precomputed path."""
        route_table = RouteTable(paths={0: [0, 1]})
        tte_info = {0: TTEInfo(task_id=0, tte_us=float("inf"), priority_score=0.0, priority_class="background")}
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        assert policy.get_flow_path(f0) == [0, 1]

    def test_get_flow_path_missing(self):
        """Missing route table entry raises KeyError."""
        route_table = RouteTable()  # empty
        tte_info = {}
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        with pytest.raises(KeyError, match="No route"):
            policy.get_flow_path(f0)

    def test_allocate_bandwidth_weighted(self):
        """TTE-aware weighted allocation gives more to critical flows."""
        route_table = RouteTable(paths={0: [0, 1], 1: [0, 1]})
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=0.0, priority_score=0.0, priority_class="critical"),
            1: TTEInfo(task_id=1, tte_us=1000.0, priority_score=0.001, priority_class="elastic"),
        }
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan, allocator_mode="weighted")
        topo = _make_2node_topology(bw_gbps=100.0)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)), topo)

        f0 = ActiveFlow(task_id=0, src=0, dst=1, size_bytes=10000, remaining_bytes=10000,
                        path=[0, 1], start_time=0, last_update_time=0)
        f1 = ActiveFlow(task_id=1, src=0, dst=1, size_bytes=10000, remaining_bytes=10000,
                        path=[0, 1], start_time=0, last_update_time=0)

        alloc = policy.allocate_bandwidth(0, [f0, f1])
        # Critical flow (TTE=0) should get more bandwidth than elastic (TTE=1000)
        assert alloc[0] > alloc[1]

    def test_allocate_bandwidth_strict_priority(self):
        """Strict priority allocates bandwidth to critical before background."""
        route_table = RouteTable(paths={0: [0, 1], 1: [0, 1]})
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=0.0, priority_score=0.0, priority_class="critical"),
            1: TTEInfo(task_id=1, tte_us=float("inf"), priority_score=0.0, priority_class="background"),
        }
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan, allocator_mode="strict_priority")
        topo = _make_2node_topology(bw_gbps=100.0)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)), topo)

        f0 = ActiveFlow(task_id=0, src=0, dst=1, size_bytes=10000, remaining_bytes=10000,
                        path=[0, 1], start_time=0, last_update_time=0)
        f1 = ActiveFlow(task_id=1, src=0, dst=1, size_bytes=10000, remaining_bytes=10000,
                        path=[0, 1], start_time=0, last_update_time=0)

        alloc = policy.allocate_bandwidth(0, [f0, f1])
        # Critical gets full bandwidth (link capacity)
        assert alloc[0] > 0
        assert alloc[0] >= alloc[1]

    def test_on_task_completed_advances_cursor(self):
        """Completing a compute task advances the per-node cursor."""
        route_table = RouteTable()
        tte_info = {}
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan(compute_order={0: [0, 1]})

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        c0 = _make_compute(0, 100, node=0)

        assert policy.compute_cursor[0] == 0
        policy.on_task_completed(100, c0)
        assert policy.compute_cursor[0] == 1

    def test_compute_ordering_respected(self):
        """Policy respects compute ordering, only emitting next in sequence."""
        route_table = RouteTable()
        tte_info = {}
        resource_dep = ResourceDependencyTable()
        plan = ExecutionPlan(compute_order={0: [0, 1]})

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        c0 = _make_compute(0, 100, node=0)
        c1 = _make_compute(1, 200, node=0)

        # Only c0 (first in order) should be emitted
        emitted = policy.emit_ready_tasks(0, [c1, c0])
        assert emitted == [0]

    def test_emitted_flows_no_coordination(self):
        """All flows without coordination groups are emitted immediately."""
        route_table = RouteTable(paths={0: [0, 1], 1: [0, 1]})
        tte_info = {
            0: TTEInfo(task_id=0, tte_us=float("inf"), priority_score=0.0, priority_class="background"),
            1: TTEInfo(task_id=1, tte_us=float("inf"), priority_score=0.0, priority_class="background"),
        }
        resource_dep = ResourceDependencyTable()  # no groups
        plan = ExecutionPlan()

        policy = PuppeteerSchedulingPolicy(route_table, tte_info, resource_dep, plan)
        policy.initialize(P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2)),
                          _make_2node_topology())

        f0 = _make_flow(0, src=0, dst=1, size_bytes=1000)
        f1 = _make_flow(1, src=0, dst=1, size_bytes=2000)
        emitted = policy.emit_ready_tasks(0, [f0, f1])
        assert set(emitted) == {0, 1}

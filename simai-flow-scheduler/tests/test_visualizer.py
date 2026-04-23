"""Tests for ChromeTraceVisualizer (base, verbose, compact)."""
import json
import tempfile
from pathlib import Path

import pytest

from src.executor.analytical import AnalyticalExecutor
from src.executor.result import ExecutionResult, TaskTiming
from src.executor.visualizer import (
    ChromeTraceCompact,
    ChromeTraceVerbose,
    ChromeTraceVisualizer,
)
from src.static_analysis.routing_hints import compute_routing_hints
from src.static_analysis.task_serializer import ExecutionPlan
from src.static_analysis.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Meta,
    P2PWorkload,
    Phase,
    Task,
    TaskType,
)


def _make_2node_topology(bw_gbps=100.0, latency_us=1.0):
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    topo.node_types = {0: "gpu", 1: "gpu"}
    for src, dst in [(0, 1), (1, 0)]:
        topo.add_link(Link(src=src, dst=dst, bandwidth_gbps=bw_gbps,
                           latency_us=latency_us, error_rate=0.0))
    return topo


def _run_executor(workload, topo, compute_order=None):
    if compute_order is None:
        compute_order = {}
        for t in workload.tasks:
            if t.is_compute() and t.node is not None:
                compute_order.setdefault(t.node, []).append(t.task_id)
    hints = compute_routing_hints(topo, workload)
    plan = ExecutionPlan(compute_order=compute_order)
    executor = AnalyticalExecutor(topo, hints)
    return executor.execute(workload, plan)


def _simple_workload():
    """1 compute + 1 flow + 1 compute chain."""
    return P2PWorkload(
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
                 node=1, duration_us=50,
                 deps=[1], phase=Phase.FORWARD, layer_id=0),
        ],
    )


# ── 基类测试 ──

class TestChromeTraceVisualizer:
    """ChromeTraceVisualizer 不能直接实例化（abstract）。"""

    def test_cannot_instantiate_base(self):
        workload = _simple_workload()
        with pytest.raises(TypeError):
            ChromeTraceVisualizer(workload)


# ── Verbose 模式测试 ──

class TestChromeTraceVerbose:

    def test_generates_valid_json(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        # 应该能序列化为 JSON
        json_str = json.dumps(events)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)

    def test_event_count(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        # 3 tasks → 3 X events
        x_events = [e for e in events if e.get("ph") == "X"]
        assert len(x_events) == 3

    def test_metadata_events(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        meta = [e for e in events if e.get("ph") == "M"]
        # 应该有 process_name 和 thread_name 事件
        process_names = [e for e in meta if e["name"] == "process_name"]
        thread_names = [e for e in meta if e["name"] == "thread_name"]
        assert len(process_names) >= 1
        assert len(thread_names) >= 2  # 至少有 Node 0 (Compute) 和 Node 1 (Comm)

    def test_compute_event_fields(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        compute_events = [e for e in events if e.get("cat") == "compute"]
        assert len(compute_events) == 2

        evt = compute_events[0]
        assert evt["ph"] == "X"
        assert evt["name"] == "fwd L0"
        assert "task_id" in evt["args"]
        assert "deps" in evt["args"]
        assert "dep_descriptions" in evt["args"]

    def test_flow_event_fields(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        flow_events = [e for e in events if e.get("cat") == "flow"]
        assert len(flow_events) == 1

        evt = flow_events[0]
        assert evt["name"] == "tp_ar 0\u21921"
        assert evt["args"]["src"] == 0
        assert evt["args"]["dst"] == 1
        assert evt["args"]["size_bytes"] == 12500000

    def test_tid_encoding(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        x_events = [e for e in events if e.get("ph") == "X"]
        for evt in x_events:
            if evt["cat"] == "compute":
                assert evt["tid"] % 2 == 0  # 偶数
            else:
                assert evt["tid"] % 2 == 1  # 奇数

    def test_export_to_file(self):
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            viz.export(result, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, list)
            assert len(data) > 0
        finally:
            Path(path).unlink()


# ── Compact 模式测试 ──

class TestChromeTraceCompact:

    def test_merges_collective_flows(self):
        """两条同类型的 flow 应该被合并为一个事件。"""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=3),
            tasks=[
                Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                     node=0, duration_us=100, phase=Phase.FORWARD, layer_id=0),
                Task(task_id=1, job_id=0, type=TaskType.FLOW,
                     src=0, dst=1, size_bytes=5000,
                     comm_type=CommType.TP_ALLREDUCE_RING,
                     deps=[0], phase=Phase.FORWARD, layer_id=0),
                Task(task_id=2, job_id=0, type=TaskType.FLOW,
                     src=0, dst=2, size_bytes=5000,
                     comm_type=CommType.TP_ALLREDUCE_RING,
                     deps=[0], phase=Phase.FORWARD, layer_id=0),
            ],
        )
        topo = NetworkTopology()
        topo.total_nodes = 3
        topo.gpu_count = 3
        topo.gpu_nodes = [0, 1, 2]
        topo.switch_nodes = []
        topo.node_types = {0: "gpu", 1: "gpu", 2: "gpu"}
        for src, dst in [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)]:
            topo.add_link(Link(src=src, dst=dst, bandwidth_gbps=100.0,
                               latency_us=0.0, error_rate=0.0))

        result = _run_executor(workload, topo)
        viz = ChromeTraceCompact(workload)
        events = viz.to_events(result)

        # flow X 事件：node 0 的 Comm 行应该只有 1 个合并事件（两条 flow 合并）
        flow_events = [e for e in events
                       if e.get("cat") == "flow" and e.get("ph") == "X"]
        node0_flows = [e for e in flow_events if e["tid"] == 1]  # tid=1 = Node 0 Comm
        assert len(node0_flows) == 1
        assert node0_flows[0]["args"]["num_flows"] == 2
        assert node0_flows[0]["args"]["total_bytes"] == 10000

    def test_flow_arrows_present(self):
        """Compact 模式应该生成 flow event 箭头。"""
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceCompact(workload)
        events = viz.to_events(result)

        arrows_s = [e for e in events if e.get("ph") == "s"]
        arrows_f = [e for e in events if e.get("ph") == "f"]
        # compute → flow 和 flow → compute 两条箭头
        assert len(arrows_s) >= 1
        assert len(arrows_f) >= 1
        # 每个 s 应该有对应的 f
        s_ids = {e["id"] for e in arrows_s}
        f_ids = {e["id"] for e in arrows_f}
        assert s_ids == f_ids

    def test_no_arrows_in_verbose(self):
        """Verbose 模式不应该生成 flow arrows。"""
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)
        viz = ChromeTraceVerbose(workload)
        events = viz.to_events(result)

        arrows = [e for e in events if e.get("ph") in ("s", "f")]
        assert len(arrows) == 0

    def test_deps_info_in_args(self):
        """两种模式下事件的 args 应包含 deps 信息。"""
        workload = _simple_workload()
        topo = _make_2node_topology()
        result = _run_executor(workload, topo)

        for VizClass in [ChromeTraceVerbose, ChromeTraceCompact]:
            viz = VizClass(workload)
            events = viz.to_events(result)
            x_events = [e for e in events if e.get("ph") == "X"]
            for evt in x_events:
                assert "deps" in evt["args"]

    def test_multi_job_grouping(self):
        """多 Job 场景下 pid 分组正确。"""
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=2, num_nodes=2),
            tasks=[
                Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                     node=0, duration_us=100, phase=Phase.FORWARD),
                Task(task_id=1, job_id=1, type=TaskType.COMPUTE,
                     node=1, duration_us=200, phase=Phase.FORWARD),
            ],
        )
        topo = _make_2node_topology()
        result = _run_executor(workload, topo, compute_order={0: [0], 1: [1]})

        for VizClass in [ChromeTraceVerbose, ChromeTraceCompact]:
            viz = VizClass(workload)
            events = viz.to_events(result)
            x_events = [e for e in events if e.get("ph") == "X"]
            pids = {e["pid"] for e in x_events}
            assert pids == {0, 1}

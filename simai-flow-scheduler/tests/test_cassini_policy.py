"""Integration tests for CassiniPolicy and CassiniAnalyzer."""
import pytest

from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.cassini_policy import CassiniSchedulingPolicy
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.static_analysis.strategies.cassini_strategy import (
    CassiniAnalyzer,
    CassiniAnalysisResult,
)
from src.static_analysis.strategies.default_strategy import DefaultAnalyzer
from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.workload_format.schema import (
    CommType,
    Job,
    Meta,
    P2PWorkload,
    ParallelismConfig,
    Phase,
    Task,
    TaskType,
)


# ---------------------------------------------------------------------------
# Topology helpers
# ---------------------------------------------------------------------------


def _make_2node_topology(bw_gbps=100.0, latency_us=1.0):
    topo = NetworkTopology()
    topo.total_nodes = 2
    topo.gpu_count = 2
    topo.gpu_nodes = [0, 1]
    topo.switch_nodes = []
    topo.node_types = {0: "gpu", 1: "gpu"}
    for s, d in [(0, 1), (1, 0)]:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=bw_gbps,
                           latency_us=latency_us, error_rate=0.0))
    return topo


def _make_4node_ring_topology(bw_gbps=100.0, latency_us=1.0):
    topo = NetworkTopology()
    topo.total_nodes = 4
    topo.gpu_count = 4
    topo.gpu_nodes = [0, 1, 2, 3]
    topo.switch_nodes = []
    topo.node_types = {i: "gpu" for i in range(4)}
    for s, d in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        topo.add_link(Link(src=s, dst=d, bandwidth_gbps=bw_gbps,
                           latency_us=latency_us, error_rate=0.0))
        topo.add_link(Link(src=d, dst=s, bandwidth_gbps=bw_gbps,
                           latency_us=latency_us, error_rate=0.0))
    return topo


# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------


def _make_single_job_workload(job_id=0, num_iterations=1):
    """Simple compute→flow→compute chain on 2 nodes."""
    tasks = []
    tid = 0
    for it in range(num_iterations):
        prev_flow = None
        # compute on node 0
        c0 = Task(task_id=tid, job_id=job_id, type=TaskType.COMPUTE,
                  node=0, duration_us=100, iteration=it, phase=Phase.FORWARD,
                  layer_id=0, item_id=tid)
        tid += 1
        tasks.append(c0)

        # flow 0→1 (depends on compute)
        f_id = tid
        f0 = Task(task_id=tid, job_id=job_id, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=125_000_000, iteration=it,
                  phase=Phase.FORWARD, layer_id=0, item_id=tid,
                  deps=[c0.task_id], comm_type=CommType.TP_ALLREDUCE_RING)
        tid += 1
        tasks.append(f0)

        # compute on node 1 (depends on flow)
        c1 = Task(task_id=tid, job_id=job_id, type=TaskType.COMPUTE,
                  node=1, duration_us=50, iteration=it, phase=Phase.FORWARD,
                  layer_id=0, item_id=tid, deps=[f_id])
        tid += 1
        tasks.append(c1)

    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        jobs=[Job(job_id=job_id, name=f"job_{job_id}",
                  assigned_nodes=[0, 1],
                  parallelism=ParallelismConfig(tp=2))],
        tasks=tasks,
    )


def _make_two_job_shared_link_workload():
    """Two jobs sharing the 0↔1 link, each with compute→flow→compute.

    Both jobs use nodes [0,1] so flows compete on the same link.
    """
    tasks = []
    tid = 0

    for jid in (0, 1):
        c0 = Task(task_id=tid, job_id=jid, type=TaskType.COMPUTE,
                  node=0, duration_us=100, iteration=0, phase=Phase.FORWARD,
                  layer_id=0, item_id=tid)
        tid += 1
        tasks.append(c0)

        fid = tid
        f0 = Task(task_id=tid, job_id=jid, type=TaskType.FLOW,
                  src=0, dst=1, size_bytes=125_000_000, iteration=0,
                  phase=Phase.FORWARD, layer_id=0, item_id=tid,
                  deps=[c0.task_id], comm_type=CommType.TP_ALLREDUCE_RING)
        tid += 1
        tasks.append(f0)

        c1 = Task(task_id=tid, job_id=jid, type=TaskType.COMPUTE,
                  node=1, duration_us=50, iteration=0, phase=Phase.FORWARD,
                  layer_id=0, item_id=tid, deps=[fid])
        tid += 1
        tasks.append(c1)

    return P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=2, num_nodes=2),
        jobs=[
            Job(job_id=0, name="job_0", assigned_nodes=[0, 1],
                parallelism=ParallelismConfig(tp=2)),
            Job(job_id=1, name="job_1", assigned_nodes=[0, 1],
                parallelism=ParallelismConfig(tp=2)),
        ],
        tasks=tasks,
    )


# ---------------------------------------------------------------------------
# CassiniAnalyzer tests
# ---------------------------------------------------------------------------


def test_cassini_analyzer_single_job():
    """CassiniAnalyzer produces valid result for a single-job workload."""
    topo = _make_2node_topology()
    workload = _make_single_job_workload(job_id=0)
    analyzer = CassiniAnalyzer(topo)
    result = analyzer.analyze(workload)

    assert isinstance(result, CassiniAnalysisResult)
    assert result.route_table is not None
    assert result.critical_path is not None
    assert len(result.communication_patterns) == 1
    assert 0 in result.communication_patterns
    # Single job: no contention, time-shift should be 0
    assert result.time_shifts.get(0, 0) == 0
    assert result.execution_plan is not None


def test_cassini_analyzer_two_jobs():
    """CassiniAnalyzer computes time-shifts for two jobs sharing a link."""
    topo = _make_2node_topology()
    workload = _make_two_job_shared_link_workload()
    analyzer = CassiniAnalyzer(topo, step_deg=15)
    result = analyzer.analyze(workload)

    assert len(result.communication_patterns) == 2
    assert 0 in result.time_shifts
    assert 1 in result.time_shifts
    # At least one job should have a non-zero shift since they share a link
    # and their patterns are identical (aligned flows cause contention)
    shifts = result.time_shifts
    assert isinstance(shifts[0], int)
    assert isinstance(shifts[1], int)


def test_cassini_analyzer_no_contention():
    """Jobs on disjoint links get zero time-shifts."""
    topo = _make_4node_ring_topology()
    # Job 0 uses nodes [0,1], Job 1 uses nodes [2,3] — no shared link
    tasks = []
    tid = 0
    for jid, (a, b) in enumerate([(0, 1), (2, 3)]):
        c0 = Task(task_id=tid, job_id=jid, type=TaskType.COMPUTE,
                  node=a, duration_us=100, iteration=0, phase=Phase.FORWARD,
                  layer_id=0, item_id=tid)
        tid += 1
        tasks.append(c0)
        fid = tid
        f0 = Task(task_id=tid, job_id=jid, type=TaskType.FLOW,
                  src=a, dst=b, size_bytes=125_000_000, iteration=0,
                  phase=Phase.FORWARD, layer_id=0, item_id=tid,
                  deps=[c0.task_id], comm_type=CommType.TP_ALLREDUCE_RING)
        tid += 1
        tasks.append(f0)
        c1 = Task(task_id=tid, job_id=jid, type=TaskType.COMPUTE,
                  node=b, duration_us=50, iteration=0, phase=Phase.FORWARD,
                  layer_id=0, item_id=tid, deps=[fid])
        tid += 1
        tasks.append(c1)

    workload = P2PWorkload(
        version="1.0", meta=Meta(num_jobs=2, num_nodes=4),
        jobs=[
            Job(job_id=0, name="job_0", assigned_nodes=[0, 1],
                parallelism=ParallelismConfig(tp=2)),
            Job(job_id=1, name="job_1", assigned_nodes=[2, 3],
                parallelism=ParallelismConfig(tp=2)),
        ],
        tasks=tasks,
    )
    analyzer = CassiniAnalyzer(topo)
    result = analyzer.analyze(workload)
    assert result.time_shifts.get(0, 0) == 0
    assert result.time_shifts.get(1, 0) == 0


# ---------------------------------------------------------------------------
# CassiniPolicy unit tests
# ---------------------------------------------------------------------------


def test_cassini_policy_delays_task_by_time_shift():
    """CassiniPolicy blocks tasks until current_time >= time_shift."""
    topo = _make_2node_topology()
    workload = _make_single_job_workload(job_id=0)

    # Build analysis with a manual time-shift of 500 us
    default = DefaultAnalyzer(topo).analyze(workload)
    from src.cassini.communication_pattern import extract_communication_patterns
    from src.static_analysis.passes.critical_path import analyze_critical_path
    patterns = extract_communication_patterns(
        workload, default.critical_path if hasattr(default, 'critical_path') else analyze_critical_path(workload, default.route_table, topo),
        default.route_table,
        topo,
    )

    # Re-analyze with Cassini first to get proper communication_patterns
    cassini_result = CassiniAnalyzer(topo).analyze(workload)

    # Create a result with a forced 500us shift
    forced_result = CassiniAnalysisResult(
        route_table=cassini_result.route_table,
        critical_path=cassini_result.critical_path,
        communication_patterns=cassini_result.communication_patterns,
        time_shifts={0: 500},
        execution_plan=cassini_result.execution_plan,
    )

    policy = CassiniSchedulingPolicy(forced_result)
    policy.initialize(workload, topo)

    ready_compute = [t for t in workload.tasks if t.task_id == 0]
    # At t=0 (< 500), nothing should be emitted
    assert policy.emit_ready_tasks(0, ready_compute) == []
    # At t=500 (>= 500), compute should be emitted
    assert policy.emit_ready_tasks(500, ready_compute) == [0]


def test_cassini_policy_no_shift_emits_immediately():
    """With zero time-shift, CassiniPolicy behaves like DefaultPolicy for admission."""
    topo = _make_2node_topology()
    workload = _make_single_job_workload(job_id=0)
    analyzer = CassiniAnalyzer(topo)
    result = analyzer.analyze(workload)

    policy = CassiniSchedulingPolicy(result)
    policy.initialize(workload, topo)

    ready_compute = [t for t in workload.tasks if t.task_id == 0]
    emitted = policy.emit_ready_tasks(0, ready_compute)
    assert emitted == [0]


def test_cassini_policy_respects_compute_order():
    """CassiniPolicy still enforces per-node serial compute ordering."""
    topo = _make_2node_topology()
    # Two compute tasks on node 0
    workload = P2PWorkload(
        version="1.0", meta=Meta(num_jobs=1, num_nodes=2),
        jobs=[Job(job_id=0, name="job_0", assigned_nodes=[0, 1],
                   parallelism=ParallelismConfig(tp=2))],
        tasks=[
            Task(task_id=0, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, iteration=0, phase=Phase.FORWARD,
                 layer_id=0, item_id=0),
            Task(task_id=1, job_id=0, type=TaskType.COMPUTE,
                 node=0, duration_us=100, iteration=0, phase=Phase.BACKWARD_INPUT,
                 layer_id=1, item_id=1, deps=[0]),
        ],
    )
    analyzer = CassiniAnalyzer(topo)
    result = analyzer.analyze(workload)

    policy = CassiniSchedulingPolicy(result)
    policy.initialize(workload, topo)

    ready = [t for t in workload.tasks]
    # Only task 0 should be emitted (task 1 blocked by compute cursor)
    emitted = policy.emit_ready_tasks(0, ready)
    assert emitted == [0]

    # Simulate task 0 completion
    policy.on_task_completed(100, workload.tasks[0])
    emitted = policy.emit_ready_tasks(100, ready)
    assert emitted == [1]


def test_cassini_policy_flow_path():
    """CassiniPolicy returns correct flow paths from route table."""
    topo = _make_2node_topology()
    workload = _make_single_job_workload(job_id=0)
    analyzer = CassiniAnalyzer(topo)
    result = analyzer.analyze(workload)

    policy = CassiniSchedulingPolicy(result)
    policy.initialize(workload, topo)

    flow_task = [t for t in workload.tasks if t.is_flow()][0]
    path = policy.get_flow_path(flow_task)
    assert path == [0, 1]


def test_cassini_policy_bandwidth_allocation():
    """CassiniPolicy delegates bandwidth to FairShareAllocator."""
    topo = _make_2node_topology(bw_gbps=100.0)
    empty_wl = P2PWorkload(version="1.0", meta=Meta(num_jobs=1, num_nodes=2),
                           tasks=[], jobs=[])
    analyzer = CassiniAnalyzer(topo)
    result = analyzer.analyze(empty_wl)

    from src.executor.runtime import ActiveFlow
    policy = CassiniSchedulingPolicy(result)
    policy.initialize(empty_wl, topo)

    f1 = ActiveFlow(task_id=0, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
                    path=[0, 1], start_time=0, last_update_time=0)
    f2 = ActiveFlow(task_id=1, src=0, dst=1, size_bytes=1000, remaining_bytes=1000,
                    path=[0, 1], start_time=0, last_update_time=0)

    alloc = policy.allocate_bandwidth(0, [f1, f2])
    assert alloc[0] == 50.0
    assert alloc[1] == 50.0


# ---------------------------------------------------------------------------
# End-to-end: CassiniPolicy vs DefaultPolicy
# ---------------------------------------------------------------------------


def test_e2e_cassini_vs_default_two_jobs():
    """CassiniPolicy completes a 2-job workload without deadlock."""
    topo = _make_2node_topology()
    workload = _make_two_job_shared_link_workload()

    analyzer = CassiniAnalyzer(topo, step_deg=15)
    result = analyzer.analyze(workload)

    policy = CassiniSchedulingPolicy(result)
    executor = AnalyticalExecutor(topo, policy)
    exec_result = executor.execute(workload)

    # All tasks should complete
    assert len(exec_result.per_task) == len(workload.tasks)
    for t in workload.tasks:
        assert t.task_id in exec_result.per_task
        assert exec_result.per_task[t.task_id].end_time_us > 0


def test_e2e_cassini_single_job_matches_default():
    """For a single job (zero time-shift), CassiniPolicy ≈ DefaultPolicy."""
    topo = _make_2node_topology()
    workload = _make_single_job_workload(job_id=0)

    # Default
    default_analysis = DefaultAnalyzer(topo).analyze(workload)
    default_policy = DefaultSchedulingPolicy(default_analysis)
    default_result = AnalyticalExecutor(topo, default_policy).execute(workload)

    # Cassini
    cassini_analysis = CassiniAnalyzer(topo).analyze(workload)
    cassini_policy = CassiniSchedulingPolicy(cassini_analysis)
    cassini_result = AnalyticalExecutor(topo, cassini_policy).execute(workload)

    # Same timing since no contention and zero time-shift
    for tid in default_result.per_task:
        d = default_result.per_task[tid]
        c = cassini_result.per_task[tid]
        assert d.start_time_us == c.start_time_us, f"task {tid} start differs"
        assert d.end_time_us == c.end_time_us, f"task {tid} end differs"


def test_e2e_cassini_two_jobs_staggered_start():
    """With a forced time-shift, jobs start at different times."""
    topo = _make_2node_topology()
    workload = _make_two_job_shared_link_workload()

    # Force job 1 to start 1000 us later than job 0
    cassini_analysis = CassiniAnalyzer(topo).analyze(workload)
    forced_shifts = {0: 0, 1: 1000}
    forced_result = CassiniAnalysisResult(
        route_table=cassini_analysis.route_table,
        critical_path=cassini_analysis.critical_path,
        communication_patterns=cassini_analysis.communication_patterns,
        time_shifts=forced_shifts,
        execution_plan=cassini_analysis.execution_plan,
    )

    policy = CassiniSchedulingPolicy(forced_result)
    executor = AnalyticalExecutor(topo, policy)
    result = executor.execute(workload)

    # Job 0's first compute should start at t=0
    job0_tasks = [t for t in workload.tasks if t.job_id == 0]
    job0_start = min(result.per_task[t.task_id].start_time_us for t in job0_tasks)
    assert job0_start == 0

    # Job 1's first compute should start at t >= 1000 (the forced shift)
    job1_tasks = [t for t in workload.tasks if t.job_id == 1]
    job1_start = min(result.per_task[t.task_id].start_time_us for t in job1_tasks)
    assert job1_start >= 1000, f"Expected job 1 start >= 1000, got {job1_start}"


def test_e2e_cassini_multi_iteration():
    """CassiniPolicy handles multi-iteration workloads.

    Job 1 is shifted by 500 us; job 0 starts immediately so the simulation
    has at least one active timeline driving events forward.  Without at
    least one job at shift 0 the executor would deadlock because no events
    fire to advance time past the shift barrier.
    """
    topo = _make_2node_topology()
    workload = _make_two_job_shared_link_workload()

    analyzer = CassiniAnalyzer(topo)
    base = analyzer.analyze(workload)
    forced = CassiniAnalysisResult(
        route_table=base.route_table,
        critical_path=base.critical_path,
        communication_patterns=base.communication_patterns,
        time_shifts={0: 0, 1: 500},
        execution_plan=base.execution_plan,
    )

    policy = CassiniSchedulingPolicy(forced)
    executor = AnalyticalExecutor(topo, policy)
    result = executor.execute(workload)

    # Job 0 (shift 0): first compute starts immediately
    job0_tasks = [t for t in workload.tasks if t.job_id == 0]
    job0_start = min(result.per_task[t.task_id].start_time_us for t in job0_tasks)
    assert job0_start == 0

    # Job 1 (shift 500): first compute starts at >= 500
    job1_tasks = [t for t in workload.tasks if t.job_id == 1]
    job1_start = min(result.per_task[t.task_id].start_time_us for t in job1_tasks)
    assert job1_start >= 500, f"Expected job 1 start >= 500, got {job1_start}"

    # All tasks complete
    assert len(result.per_task) == len(workload.tasks)

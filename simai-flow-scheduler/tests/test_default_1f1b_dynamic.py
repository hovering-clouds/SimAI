from src.static_analysis.passes.topology_loader import Link, NetworkTopology
from src.static_analysis.strategies.default_strategy import OneFOneBAnalyzer
from src.static_analysis.strategies.hermod_strategy import HermodDynamicAnalyzer
from src.static_analysis.passes.hermod_priority import HermodEpMode, HermodScheduleVariant
from src.workload_format.schema import (
    Job, Meta, P2PWorkload, ParallelismConfig, Phase, Task, TaskType,
)


def test_dynamic_1f1b_analyzer_restores_job_stage_mapping():
    """Dynamic mini-workloads need their Job metadata to construct 1F1B."""
    topology = NetworkTopology()
    topology.add_link(Link(0, 1, 100.0, 1.0, 0.0))
    topology.add_link(Link(1, 0, 100.0, 1.0, 0.0))
    job = Job(
        job_id=7,
        name="two-stage",
        assigned_nodes=[0, 1],
        parallelism=ParallelismConfig(tp=1, dp=1, pp=2, ep=1),
    )
    mini_workload = P2PWorkload(
        version="1.0",
        meta=Meta(num_jobs=1, num_nodes=2),
        tasks=[
            Task(1, 7, TaskType.COMPUTE, node=0, duration_us=1,
                 phase=Phase.FORWARD, iteration=0, layer_id=0),
            Task(2, 7, TaskType.COMPUTE, node=1, duration_us=1,
                 phase=Phase.BACKWARD_INPUT, iteration=0, layer_id=0),
        ],
    )

    result = OneFOneBAnalyzer(topology, {7: job}).analyze(mini_workload)

    assert mini_workload.jobs == [job]
    assert result.execution_plan.compute_order == {0: [1], 1: [2]}


def test_dynamic_default_and_hermod_share_the_same_1f1b_plan():
    """Default-vs-Hermod must differ only in bandwidth allocation."""
    topology = NetworkTopology()
    topology.add_link(Link(0, 1, 100.0, 1.0, 0.0))
    topology.add_link(Link(1, 0, 100.0, 1.0, 0.0))
    job = Job(
        job_id=3,
        name="two-stage",
        assigned_nodes=[0, 1],
        parallelism=ParallelismConfig(tp=1, dp=1, pp=2, ep=1),
    )
    tasks = [
        Task(1, 3, TaskType.COMPUTE, node=0, duration_us=1,
             phase=Phase.FORWARD, iteration=0, layer_id=0),
        Task(2, 3, TaskType.COMPUTE, node=1, duration_us=1,
             phase=Phase.BACKWARD_INPUT, iteration=0, layer_id=0),
    ]
    default_workload = P2PWorkload("1.0", Meta(1, 2), tasks=list(tasks))
    hermod_workload = P2PWorkload("1.0", Meta(1, 2), tasks=list(tasks))

    default = OneFOneBAnalyzer(topology, {3: job}).analyze(default_workload)
    hermod = HermodDynamicAnalyzer(
        topology, {3: job}, HermodScheduleVariant.CONVENTIONAL_1F1B,
        HermodEpMode.REJECT, "1f1b",
    ).analyze(hermod_workload)

    assert default.execution_plan.compute_order == hermod.execution_plan.compute_order

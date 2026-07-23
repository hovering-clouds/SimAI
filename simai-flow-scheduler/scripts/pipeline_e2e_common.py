"""Isolated E2E runner shared by the three advanced pipeline smoke scripts.

This module deliberately does not register new modes in Default, Hermod, or
Puppeteer analyzers. Each script selects a strategy-specific workload builder
and serializer directly, then uses the unchanged default policy for execution.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.executor.analytical import AnalyticalExecutor
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.advanced_pipeline_strategies import (
    BidirectionalPipelineAnalyzer,
    InterleavedPipelineAnalyzer,
    ZeroBubblePipelineAnalyzer,
)
from src.workload_format.schema import CommType, Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.bidirectional_pipeline_builder import (
    BidirectionalPipelineWorkloadBuilder,
)
from src.workload_generator.interleaved_pipeline_builder import (
    InterleavedPipelineWorkloadBuilder,
)
from src.workload_generator.zero_bubble_pipeline_builder import (
    ZeroBubblePipelineWorkloadBuilder,
)


DEFAULT_AICB = (
    "inputs/aicb-workload/"
    "A100-gpt_7B_ws4_pp2-world_size4-tp2-pp2-ep1-gbs32-mbs4-"
    "seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
)
DEFAULT_TOPO = (
    "inputs/topologies/"
    "AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
)
MODES = {
    "interleaved_1f1b",
    "zero_bubble",
    "bidirectional",
}


def run_pipeline_e2e_cli(mode: str) -> None:
    """Parse CLI arguments and run one isolated advanced-pipeline E2E."""
    if mode not in MODES:
        raise ValueError(f"Unsupported isolated pipeline mode: {mode!r}")

    parser = argparse.ArgumentParser(
        description=f"{mode} isolated end-to-end simulation",
    )
    parser.add_argument("--aicb", default=DEFAULT_AICB)
    parser.add_argument("--topo", default=DEFAULT_TOPO)
    parser.add_argument("--output", "-o", default=f"outputs/{mode}_e2e")
    if mode == "interleaved_1f1b":
        parser.add_argument(
            "--vpp",
            type=int,
            default=2,
            help="Virtual model chunks per physical pipeline stage.",
        )
        parser.add_argument(
            "--interleave-group-size",
            type=int,
            default=None,
            help="Microbatches per interleaved schedule group; defaults to pp.",
        )
    args = parser.parse_args()
    vpp = getattr(args, "vpp", 2)
    interleave_group_size = getattr(args, "interleave_group_size", None)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Parse AICB: {args.aicb}")
    header, items = AicbParser().parse(args.aicb)
    dp = header.all_gpus // (header.tp * header.pp * header.ep)
    job = Job(
        job_id=0,
        name=Path(args.aicb).stem,
        assigned_nodes=list(range(header.all_gpus)),
        parallelism=ParallelismConfig(
            tp=header.tp,
            dp=dp,
            pp=header.pp,
            ep=header.ep,
        ),
    )

    topology = TopologyLoader().load(args.topo)
    if header.all_gpus > topology.gpu_count:
        raise ValueError(
            f"Workload needs {header.all_gpus} GPUs, topology has {topology.gpu_count}"
        )

    if mode == "interleaved_1f1b":
        analyzer = InterleavedPipelineAnalyzer(
            topology,
            virtual_pipeline_size=vpp,
            interleave_group_size=interleave_group_size,
        )
        builder = InterleavedPipelineWorkloadBuilder(
            vpp,
            analyzer.expansion_task_info,
        )
    elif mode == "zero_bubble":
        analyzer = ZeroBubblePipelineAnalyzer(topology)
        builder = ZeroBubblePipelineWorkloadBuilder(
            analyzer.expansion_task_info,
        )
    else:
        analyzer = BidirectionalPipelineAnalyzer(topology)
        builder = BidirectionalPipelineWorkloadBuilder(
            analyzer.expansion_task_info,
        )

    print(f"[2/4] Build {mode} workload")
    workload = builder.build_from_aicb(header, items, job, comm_algo="ring")
    errors = workload.validate()
    if errors:
        raise ValueError(f"{mode} workload validation failed: {errors}")
    WorkloadWriter().write(workload, output / "workload.json")

    print("[3/4] Run isolated pipeline analyzer")
    analysis = analyzer.analyze(workload)

    print("[4/4] Execute with unchanged DefaultSchedulingPolicy")
    result = AnalyticalExecutor(
        topology,
        DefaultSchedulingPolicy(analysis),
    ).execute(workload)
    result.to_json(output / "execution_result.json")

    metadata: dict[str, dict] = {}
    for task_id, info in analyzer.expansion_task_info.items():
        metadata[str(task_id)] = (
            asdict(info) if is_dataclass(info) else {"value": repr(info)}
        )
    for task_id, info in analyzer.schedule_task_info.items():
        metadata.setdefault(str(task_id), {})["schedule"] = asdict(info)
    (output / "pipeline_task_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    summary = {
        "pipeline": mode,
        "aicb": args.aicb,
        "topology": args.topo,
        "parallelism": {
            "tp": header.tp,
            "dp": dp,
            "pp": header.pp,
            "ep": header.ep,
            "ga": header.ga,
            "vpp": vpp if mode == "interleaved_1f1b" else None,
        },
        "tasks": {
            "total": len(workload.tasks),
            "compute": len(workload.get_compute_tasks()),
            "flow": len(workload.get_flow_tasks()),
            "pp_flow": sum(
                task.comm_type in {CommType.PP_SEND, CommType.PP_RECV}
                for task in workload.get_flow_tasks()
            ),
            "expansion_sidecar": len(analyzer.expansion_task_info),
            "schedule_sidecar": len(analyzer.schedule_task_info),
            "completed": len(result.per_task),
        },
        "makespan_us": result.makespan_us,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))

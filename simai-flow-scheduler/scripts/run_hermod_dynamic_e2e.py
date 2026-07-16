"""Run config-driven multi-job/multi-iteration dynamic Hermod experiments."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.executor.dynamic_executor import DynamicExecutor
from src.executor.hermod_training_expander import HermodTrainingJobExpander
from src.executor.job_policy import FifoJobPolicy
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.hermod_policy import HermodSchedulingPolicy
from src.static_analysis.passes.hermod_priority import (
    HermodEpMode, HermodPriorityAnalysis, HermodScheduleVariant,
)
from src.static_analysis.passes.routing import BfsRouteTable
from src.static_analysis.passes.task_serializer import ExecutionPlan
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import (
    DefaultAnalysisResult, DefaultAnalyzer, OneFOneBAnalyzer,
)
from src.static_analysis.strategies.hermod_strategy import (
    HermodAnalysisResult, HermodDynamicAnalyzer,
)
from src.workload_format.compact_workload import (
    CompactWorkload, JobDAG, JobExpansionInfo, TaskIdAllocator,
)
from src.workload_format.schema import Job, Meta, ParallelismConfig
from src.workload_generator.aicb_parser import AicbParser


DEFAULT_AICB = "inputs/aicb-workload/A100-gpt_13B_ws8_pp2-world_size8-tp4-pp2-ep1-gbs2-mbs1-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
DEFAULT_TOPO = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
CONFIG_KEYS = {
    "workload", "aicb", "topology", "topo", "dp", "pipeline", "variant",
    "jobs", "iterations", "workloads", "modes", "output", "visualize",
    "placement", "gpus_per_server",
}
PIPELINES = {"gpipe", "1f1b"}
MODES = {"default", "hermod"}
VARIANTS = {variant.value for variant in HermodScheduleVariant}
WORKLOAD_KEYS = {"aicb", "dp", "num_jobs", "num_iters"}
PLACEMENTS = {"contiguous", "cyclic_pp_dp"}


def load_config(path: str) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Hermod dynamic config must be a JSON object")
    unknown = set(data) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown Hermod dynamic config keys: {', '.join(sorted(unknown))}")
    for friendly, internal in (("workload", "aicb"), ("topology", "topo")):
        if friendly in data and internal in data:
            raise ValueError(f"Config cannot contain both {friendly!r} and {internal!r}")
        if friendly in data:
            data[internal] = data.pop(friendly)
    for key in ("aicb", "topo", "output"):
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"Config {key!r} must be a string")
    for key in ("dp", "jobs", "iterations"):
        if key in data and (not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] < 1):
            raise ValueError(f"Config {key!r} must be a positive integer")
    if "placement" in data and data["placement"] not in PLACEMENTS:
        raise ValueError(f"Unsupported placement: {data['placement']!r}")
    if "gpus_per_server" in data and (
        not isinstance(data["gpus_per_server"], int)
        or isinstance(data["gpus_per_server"], bool)
        or data["gpus_per_server"] < 1
    ):
        raise ValueError("Config 'gpus_per_server' must be a positive integer")
    if "visualize" in data and not isinstance(data["visualize"], bool):
        raise ValueError("Config 'visualize' must be a boolean")
    if "pipeline" in data and data["pipeline"] not in PIPELINES:
        raise ValueError(f"Unsupported pipeline: {data['pipeline']!r}")
    if "variant" in data and data["variant"] not in VARIANTS:
        raise ValueError(f"Unsupported variant: {data['variant']!r}")
    if "modes" in data and (
        not isinstance(data["modes"], list)
        or not all(isinstance(mode, str) and mode in MODES for mode in data["modes"])
    ):
        raise ValueError("Config modes must be a list containing default and/or hermod")
    if "workloads" in data:
        if any(key in data for key in ("aicb", "dp", "jobs", "iterations")):
            raise ValueError("workloads cannot be combined with legacy aicb/dp/jobs/iterations keys")
        if not isinstance(data["workloads"], list) or not data["workloads"]:
            raise ValueError("Config workloads must be a non-empty list")
        for index, spec in enumerate(data["workloads"]):
            if not isinstance(spec, dict) or set(spec) - WORKLOAD_KEYS:
                raise ValueError(f"Invalid workload spec at index {index}")
            if not isinstance(spec.get("aicb"), str):
                raise ValueError(f"workloads[{index}].aicb must be a string")
            for key in ("dp", "num_jobs", "num_iters"):
                value = spec.get(key)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool) or value < 1
                ):
                    raise ValueError(f"workloads[{index}].{key} must be a positive integer")
    return data


def assigned_nodes_for(
    parallelism: ParallelismConfig,
    placement: str,
    gpus_per_server: int,
) -> list[int]:
    """Map logical [PP][DP][EP][TP] ranks onto physical GPU IDs.

    ``cyclic_pp_dp`` is an experimental contention placement for homogeneous
    servers: every TP group remains local, while both adjacent PP stages and
    DP replicas rotate across servers.  It is not claimed as a paper placement.
    """
    total = parallelism.tp * parallelism.dp * parallelism.pp * parallelism.ep
    if placement == "contiguous":
        return list(range(total))
    if total % gpus_per_server or gpus_per_server % parallelism.tp:
        raise ValueError("cyclic_pp_dp requires whole TP groups on equal-size servers")
    server_count = total // gpus_per_server
    if parallelism.ep != 1:
        raise ValueError("cyclic_pp_dp currently requires ep=1")
    slots_per_server = gpus_per_server // parallelism.tp
    slots_used = [0] * server_count
    nodes: list[int] = []
    for pp_idx in range(parallelism.pp):
        for dp_idx in range(parallelism.dp):
            server = (pp_idx + dp_idx) % server_count
            slot = slots_used[server]
            if slot >= slots_per_server:
                raise ValueError("cyclic_pp_dp cannot balance this PP/DP/server configuration")
            slots_used[server] += 1
            start = server * gpus_per_server + slot * parallelism.tp
            nodes.extend(range(start, start + parallelism.tp))
    return nodes


def build_compact_workload(
    workload_specs: list[dict],
    placement: str = "contiguous",
    gpus_per_server: int = 8,
) -> tuple[CompactWorkload, list[dict], int]:
    dynamic_jobs: list[Job] = []
    expansion_info: dict[int, JobExpansionInfo] = {}
    job_id = 0
    logical_group_id = 0
    resolved_specs: list[dict] = []
    max_required_gpus = 0
    for workload_index, spec in enumerate(workload_specs):
        aicb_path = spec["aicb"]
        header, _ = AicbParser().parse(aicb_path)
        header_dp = header.all_gpus // (header.tp * header.pp * header.ep)
        dp = spec.get("dp", header_dp)
        num_jobs = spec.get("num_jobs", 1)
        num_iters = spec.get("num_iters", 1)
        parallelism = ParallelismConfig(tp=header.tp, dp=dp, pp=header.pp, ep=header.ep)
        assigned_nodes = assigned_nodes_for(parallelism, placement, gpus_per_server)
        max_required_gpus = max(max_required_gpus, len(assigned_nodes))
        resolved_specs.append({
            "aicb": aicb_path, "dp": dp, "num_jobs": num_jobs, "num_iters": num_iters,
            "tp": header.tp, "pp": header.pp, "ep": header.ep, "ga": header.ga,
            "placement": placement,
        })
        for local_job in range(num_jobs):
            previous_iteration_id = None
            for iteration in range(num_iters):
                dynamic_jobs.append(Job(
                    job_id=job_id,
                    name=f"{Path(aicb_path).stem}-w{workload_index}-job{local_job}-iteration{iteration}",
                    assigned_nodes=list(assigned_nodes),
                    parallelism=parallelism,
                ))
                expansion_info[job_id] = JobExpansionInfo(
                    depends_on=[] if previous_iteration_id is None else [previous_iteration_id],
                    job_type="training",
                    trace_src=aicb_path,
                    trace_job_index=iteration,
                    job_group_id=logical_group_id,
                )
                previous_iteration_id = job_id
                job_id += 1
            logical_group_id += 1
    return (
        CompactWorkload(
            version="1.0",
            meta=Meta(num_jobs=len(dynamic_jobs), num_nodes=max_required_gpus),
            jobs=dynamic_jobs,
            job_expansion_info=expansion_info,
        ),
        resolved_specs,
        max_required_gpus,
    )


def run_mode(mode: str, compact: CompactWorkload, topology, args):
    if mode == "default":
        analysis = DefaultAnalysisResult(BfsRouteTable(topology), ExecutionPlan())
        policy = DefaultSchedulingPolicy(analysis)
        if args.pipeline == "1f1b":
            analyzer = OneFOneBAnalyzer(
                topology, {job.job_id: job for job in compact.jobs},
            )
        else:
            analyzer = DefaultAnalyzer(topology)
    else:
        variant = HermodScheduleVariant(args.variant)
        empty = HermodAnalysisResult(
            BfsRouteTable(topology), ExecutionPlan(),
            HermodPriorityAnalysis({}, variant, HermodEpMode.REJECT),
        )
        policy = HermodSchedulingPolicy(empty)
        analyzer = HermodDynamicAnalyzer(
            topology,
            {job.job_id: job for job in compact.jobs},
            variant=variant,
            ep_mode=HermodEpMode.REJECT,
            pipeline_mode=args.pipeline,
        )
    executor = DynamicExecutor(topology=topology, policy=policy, analyzer=analyzer)
    result = executor.execute_dynamic(
        job_dag=JobDAG.from_compact(compact),
        job_expansion_info=compact.job_expansion_info,
        job_policy=FifoJobPolicy(),
        job_expander=HermodTrainingJobExpander(TaskIdAllocator()),
    )
    return executor, result


def main() -> None:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config")
    bootstrap_args, _ = bootstrap.parse_known_args()
    config = load_config(bootstrap_args.config) if bootstrap_args.config else {}

    parser = argparse.ArgumentParser(description="Dynamic multi-job/multi-iteration Hermod §4.1 experiment")
    parser.add_argument("--config", help="JSON experiment configuration; CLI flags override it.")
    parser.add_argument("--aicb", default=None,
                        help="Legacy single-workload input; use config workloads for mixed jobs.")
    parser.add_argument("--topo", default=DEFAULT_TOPO)
    parser.add_argument("--dp", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None, help="Legacy single-workload job count")
    parser.add_argument("--iterations", type=int, default=None, help="Legacy single-workload iteration count")
    parser.add_argument("--pipeline", choices=sorted(PIPELINES), default="1f1b")
    parser.add_argument("--variant", choices=sorted(VARIANTS),
                        default=HermodScheduleVariant.CONVENTIONAL_1F1B.value)
    parser.add_argument("--modes", nargs="+", choices=sorted(MODES), default=["default", "hermod"])
    parser.add_argument("--output", default="outputs/hermod_dynamic_e2e")
    parser.add_argument("--placement", choices=sorted(PLACEMENTS), default="contiguous",
                        help="GPU placement; cyclic_pp_dp deliberately creates PP/DP NIC contention.")
    parser.add_argument("--gpus-per-server", type=int, default=8,
                        help="Server size used only by non-contiguous placement modes.")
    parser.add_argument("--visualize", action="store_true",
                        help="Write <mode>_trace.json Chrome Trace files beside the results.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.aicb is not None and getattr(args, "workloads", None) is not None:
        parser.error("--aicb cannot be combined with config workloads")
    workload_specs = getattr(args, "workloads", None)
    if workload_specs is not None and any(value is not None for value in (args.dp, args.jobs, args.iterations)):
        parser.error("--dp/--jobs/--iterations cannot be combined with config workloads")
    if workload_specs is None:
        workload_specs = [{
            "aicb": args.aicb or DEFAULT_AICB,
            "dp": args.dp,
            "num_jobs": args.jobs or 1,
            "num_iters": args.iterations or 2,
        }]
    compact, resolved_specs, required_gpus = build_compact_workload(
        workload_specs, args.placement, args.gpus_per_server,
    )
    topology = TopologyLoader().load(args.topo)
    if required_gpus > topology.gpu_count:
        raise ValueError(f"Run config needs {required_gpus} GPUs but topology has {topology.gpu_count}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    compact.to_json(output / "compact_workload.json")
    results = {}
    for mode in args.modes:
        # Dynamic execution mutates JobDAG runtime state.  Copy the already
        # parsed compact workload for each mode instead of rebuilding/reparsing
        # every AICB input once per comparison mode.
        mode_compact = copy.deepcopy(compact)
        executor, result = run_mode(mode, mode_compact, topology, args)
        result.to_json(output / f"{mode}_execution_result.json")
        task_meta = {str(task_id): metadata for task_id, metadata in executor._task_meta.items()}
        (output / f"{mode}_task_meta.json").write_text(
            json.dumps(task_meta, indent=2),
            encoding="utf-8",
        )
        if mode == "hermod":
            hermod_metadata = [
                {"task_id": int(task_id), **metadata}
                for task_id, metadata in task_meta.items()
                if metadata.get("coflow_id") is not None
            ]
            (output / "hermod_metadata.json").write_text(
                json.dumps(hermod_metadata, indent=2), encoding="utf-8",
            )
        if args.visualize:
            # Kept in visualize_dynamic.py so standalone and E2E exports have
            # identical labels and Chrome Trace structure.
            from visualize_dynamic import export_trace
            trace_path = output / f"{mode}_trace.json"
            event_count = export_trace(
                str(output / f"{mode}_execution_result.json"),
                str(output / f"{mode}_task_meta.json"),
                str(trace_path),
            )
            print(f"Wrote {event_count:,} Chrome Trace events: {trace_path}")
        results[mode] = result
    summary = {
        "input": {"topology": args.topo, "pipeline": args.pipeline,
                  "variant": args.variant, "ep_mode": "reject",
                  "placement": args.placement, "gpus_per_server": args.gpus_per_server},
        "workloads": resolved_specs,
        "dynamic": {
            "expanded_jobs": len(compact.jobs),
            "max_gpus_per_job": required_gpus,
            "dependency": "each logical job's iterations are sequential; jobs from all workload specs share the topology",
        },
        "makespan_us": {name: result.makespan_us for name, result in results.items()},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Run a PP/DP Hermod §4.1 experiment on an existing AICB input/topology."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.executor.analytical import AnalyticalExecutor
from src.executor.visualizer import ChromeTraceVerbose
from src.executor.policies.default_policy import DefaultSchedulingPolicy
from src.executor.policies.hermod_policy import HermodSchedulingPolicy
from src.executor.policies.puppeteer_policy import PuppeteerSchedulingPolicy
from src.static_analysis.passes.hermod_priority import HermodEpMode, HermodScheduleVariant
from src.static_analysis.passes.topology_loader import TopologyLoader
from src.static_analysis.strategies.default_strategy import DefaultAnalysisResult
from src.static_analysis.strategies.hermod_strategy import HermodAnalyzer
from src.static_analysis.strategies.puppeteer_strategy import PuppeteerAnalyzer
from src.workload_format.schema import Job, ParallelismConfig
from src.workload_format.writer import WorkloadWriter
from src.workload_generator.aicb_parser import AicbParser
from src.workload_generator.hermod_aicb_metadata import HermodAicbMetadataAdapter
from src.workload_generator.workload_builder import WorkloadBuilder


DEFAULT_AICB = "inputs/aicb-workload/A100-gpt_13B_ws8_pp2-world_size8-tp4-pp2-ep1-gbs2-mbs1-seq4096-MOE-False-GEMM-False-flash_attn-True.txt"
DEFAULT_TOPO = "inputs/topologies/AlibabaHPN_16g_8gps_DualToR_DualPlane_200Gbps_A100"
CONFIG_KEYS = {
    "workload", "aicb", "topology", "topo", "dp", "ep_mode", "pipeline",
    "modes", "k_paths", "variant", "output", "visualize",
}
CONFIG_PIPELINES = {"gpipe", "1f1b"}
CONFIG_MODES = {"default", "puppeteer", "hermod"}
CONFIG_VARIANTS = {variant.value for variant in HermodScheduleVariant}


def load_config(path: str) -> dict:
    """Load and validate a JSON experiment configuration."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Hermod config must be a JSON object")
    unknown = set(data) - CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown Hermod config keys: {', '.join(sorted(unknown))}")
    for friendly, internal in (("workload", "aicb"), ("topology", "topo")):
        if friendly in data and internal in data:
            raise ValueError(f"Hermod config cannot contain both {friendly!r} and {internal!r}")
    if "workload" in data:
        data["aicb"] = data.pop("workload")
    if "topology" in data:
        data["topo"] = data.pop("topology")
    for key in ("aicb", "topo", "output"):
        if key in data and not isinstance(data[key], str):
            raise ValueError(f"Hermod config {key!r} must be a string")
    for key in ("dp", "k_paths"):
        if key in data and (not isinstance(data[key], int) or isinstance(data[key], bool) or data[key] < 1):
            raise ValueError(f"Hermod config {key!r} must be a positive integer")
    if "visualize" in data and not isinstance(data["visualize"], bool):
        raise ValueError("Hermod config 'visualize' must be a boolean")
    if "pipeline" in data and data["pipeline"] not in CONFIG_PIPELINES:
        raise ValueError(f"Unsupported Hermod pipeline: {data['pipeline']!r}")
    if "variant" in data and data["variant"] not in CONFIG_VARIANTS:
        raise ValueError(f"Unsupported Hermod variant: {data['variant']!r}")
    if "ep_mode" in data and data["ep_mode"] != HermodEpMode.REJECT.value:
        raise ValueError("Hermod EP is not implemented; ep_mode must be 'reject'")
    if "modes" in data:
        if not isinstance(data["modes"], list) or not all(
            isinstance(mode, str) and mode in CONFIG_MODES for mode in data["modes"]
        ):
            raise ValueError("Hermod config modes must be a list of default/puppeteer/hermod")
    return data


def build_workload(aicb_path: str, dp_override: int | None, ep_mode: HermodEpMode):
    if ep_mode != HermodEpMode.REJECT:
        raise NotImplementedError("Hermod EP experiments are not implemented")
    header, items = AicbParser().parse(aicb_path)
    header_dp = header.all_gpus // (header.tp * header.pp * header.ep)
    dp = dp_override if dp_override is not None else header_dp
    if dp < 1:
        raise ValueError("--dp must be >= 1")
    job = Job(
        job_id=0,
        name=Path(aicb_path).stem,
        assigned_nodes=list(range(header.tp * dp * header.pp * header.ep)),
        parallelism=ParallelismConfig(tp=header.tp, dp=dp, pp=header.pp, ep=header.ep),
    )
    workload = WorkloadBuilder().build_from_aicb(header, items, job, comm_algo="ring")
    records = HermodAicbMetadataAdapter(
        header, items, reject_ep=(ep_mode == HermodEpMode.REJECT),
    ).apply(workload)
    return header, dp, workload, records


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", help="JSON experiment configuration")
    bootstrap_args, _ = bootstrap.parse_known_args()
    config = load_config(bootstrap_args.config) if bootstrap_args.config else {}

    parser = argparse.ArgumentParser(description="Hermod §4.1 PP/DP E2E experiment")
    parser.add_argument("--config", help="JSON experiment configuration; CLI flags override it.")
    parser.add_argument("--aicb", default=DEFAULT_AICB)
    parser.add_argument("--topo", default=DEFAULT_TOPO)
    parser.add_argument("--dp", type=int, default=None,
                        help="Override AICB header DP to synthesize DP collectives (as in Puppeteer experiments).")
    parser.add_argument("--pipeline", choices=["gpipe", "1f1b"], default="1f1b",
                        help="Compute pipeline serializer; add future modes in HermodAnalyzer.")
    parser.add_argument("--ep-mode", choices=[HermodEpMode.REJECT.value], default="reject",
                        help="EP is intentionally unavailable until its separate path is validated.")
    parser.add_argument("--modes", nargs="+", choices=["default", "puppeteer", "hermod"],
                        default=["default", "puppeteer", "hermod"])
    parser.add_argument("--k-paths", type=int, default=4)
    parser.add_argument("--variant", choices=[v.value for v in HermodScheduleVariant],
                        default=HermodScheduleVariant.CONVENTIONAL_1F1B.value)
    parser.add_argument("--output", default="outputs/hermod_e2e")
    parser.add_argument("--visualize", action="store_true",
                        help="Write <mode>_trace.json Chrome Trace files beside the results.")
    parser.set_defaults(**config)
    args = parser.parse_args()

    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    ep_mode = HermodEpMode(args.ep_mode)
    header, dp, workload, records = build_workload(args.aicb, args.dp, ep_mode)
    topology = TopologyLoader().load(args.topo)
    required_gpus = header.tp * dp * header.pp * header.ep
    if required_gpus > topology.gpu_count:
        raise ValueError(f"Run config needs {required_gpus} GPUs but topology has {topology.gpu_count}")

    variant = HermodScheduleVariant(args.variant)
    analysis = HermodAnalyzer(
        topology, variant, ep_mode, pipeline_mode=args.pipeline,
    ).analyze(workload)
    WorkloadWriter().write(workload, out / "workload.json")
    HermodAicbMetadataAdapter.write_sidecar(records, out / "hermod_metadata.json")

    # Same routes and compute order isolate Hermod's allocator effect.
    baseline_analysis = DefaultAnalysisResult(analysis.route_table, analysis.execution_plan)
    results = {}
    if "default" in args.modes:
        results["default"] = AnalyticalExecutor(
            topology, DefaultSchedulingPolicy(baseline_analysis)).execute(workload)
    if "puppeteer" in args.modes:
        puppet = PuppeteerAnalyzer(topology, k_paths=args.k_paths, serializer=args.pipeline).analyze(workload)
        results["puppeteer"] = AnalyticalExecutor(
            topology,
            PuppeteerSchedulingPolicy(
                puppet.route_table, puppet.tte_info, puppet.resource_dependency,
                puppet.execution_plan, allocator_mode="weighted",
            ),
        ).execute(workload)
    if "hermod" in args.modes:
        results["hermod"] = AnalyticalExecutor(
            topology, HermodSchedulingPolicy(analysis)).execute(workload)
    for name, result in results.items():
        result.to_json(out / f"{name}_execution_result.json")
        if args.visualize:
            trace_path = out / f"{name}_trace.json"
            ChromeTraceVerbose(workload).export(result, str(trace_path))
            print(f"Wrote Chrome Trace: {trace_path}")

    coflows = [
        {"coflow_id": info.coflow_id, "task_ids": list(info.task_ids),
         "microbatch_id": info.microbatch_id, "logical_layer_id": info.logical_layer_id,
         "coflow_type": info.coflow_type.value}
        for info in sorted(analysis.priority_analysis.coflows.values(), key=lambda c: c.coflow_id)
    ]
    summary = {
        "input": {"aicb": args.aicb, "topology": args.topo, "variant": args.variant,
                  "pipeline": args.pipeline, "ep_mode": args.ep_mode},
        "parallelism": {"tp": header.tp, "dp": dp,
                        "pp": header.pp, "ep": header.ep, "ga": header.ga},
        "coflows": coflows,
        "coflow_type_counts": {
            name: sum(info.coflow_type.value == name for info in analysis.priority_analysis.coflows.values())
            for name in ("pp", "dp", "ep")
        },
        "makespan_us": {name: result.makespan_us for name, result in results.items()},
    }
    if not summary["coflow_type_counts"]["dp"]:
        summary["validation_note"] = (
            "This supplied AICB workload has no DP coflow (dp=1); it validates the PP "
            "end-to-end path. PP/DP priority is covered by synthetic strict-priority tests."
        )
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""
JobSlicer — converts original trace files into CompactWorkload.

Inference: one batch → one Job (fine-grained, per-batch dependencies)
Training:  one AICB trace → one Job per repeat (entire trace is atomic due to GA gradient sync)
"""

import json

from ..workload_format.schema import Job, Meta, ParallelismConfig
from ..workload_format.compact_workload import (
    CompactWorkload, JobExpansionInfo, SlicerConfig,
)
from .aicb_parser import AicbParser


class InferenceJobSlicer:
    """Read inference trace JSON → one Job per batch."""

    def slice_trace(
        self,
        trace_path: str,
        assigned_nodes: list[int] | None = None,
    ) -> CompactWorkload:
        with open(trace_path) as f:
            trace = json.load(f)

        tp = trace["parallelism"]["tp"]
        ep = trace["parallelism"]["ep"]
        pp = trace["parallelism"]["pp"]
        ws = tp * ep * pp  # world size per replica

        # batch_id (str) → job_id (int)
        batch_to_job: dict[str, int] = {}
        for idx, batch in enumerate(trace["batches"]):
            batch_to_job[batch["batch_id"]] = idx

        num_replicas = max(b["replica_id"] for b in trace["batches"]) + 1
        required = num_replicas * ws
        if assigned_nodes is not None and len(assigned_nodes) < required:
            raise ValueError(
                f"InferenceJobSlicer: assigned_nodes has {len(assigned_nodes)} nodes, "
                f"but trace needs at least {required} ({num_replicas} replicas × {ws} per replica)"
            )
        nodes = assigned_nodes if assigned_nodes is not None else list(range(required))

        jobs = []
        info = {}
        for idx, batch in enumerate(trace["batches"]):
            depends_on = [
                batch_to_job[d] for d in batch.get("depends_on", [])
                if d in batch_to_job
            ]

            replica_id = batch["replica_id"]
            assigned = nodes[replica_id * ws : (replica_id + 1) * ws]

            jobs.append(Job(
                job_id=idx,
                assigned_nodes=assigned,
                parallelism=ParallelismConfig(tp=tp, ep=ep, pp=pp),
            ))
            info[idx] = JobExpansionInfo(
                depends_on=depends_on,
                job_type="inference",
                trace_src=trace_path,
                trace_job_index=idx,
            )

        return CompactWorkload(
            version="1.0",
            meta=Meta(num_jobs=len(jobs), num_nodes=len(nodes)),
            jobs=jobs,
            job_expansion_info=info,
        )


class TrainingJobSlicer:
    """Read AICB trace → one Job per repeat (entire trace is atomic)."""

    def slice_trace(
        self,
        trace_path: str,
        assigned_nodes: list[int] | None = None,
        repeat: int = 1,
    ) -> CompactWorkload:
        header, _ = AicbParser().parse(trace_path)
        required = header.all_gpus
        if assigned_nodes is not None and len(assigned_nodes) != required:
            raise ValueError(
                f"TrainingJobSlicer: assigned_nodes has {len(assigned_nodes)} nodes, "
                f"but trace expects exactly {required} (all_gpus={required})"
            )
        nodes = assigned_nodes if assigned_nodes is not None else list(range(required))

        jobs = []
        info = {}
        for i in range(repeat):
            jobs.append(Job(
                job_id=i,
                assigned_nodes=list(nodes),
                parallelism=ParallelismConfig(
                    tp=header.tp,
                    dp=header.all_gpus // (header.tp * header.ep * header.pp),
                    pp=header.pp,
                    ep=header.ep,
                ),
            ))
            info[i] = JobExpansionInfo(
                depends_on=[i - 1] if i > 0 else [],
                job_type="training",
                trace_src=trace_path,
                trace_job_index=i,
            )

        return CompactWorkload(
            version="1.0",
            meta=Meta(num_jobs=len(jobs), num_nodes=len(nodes)),
            jobs=jobs,
            job_expansion_info=info,
        )

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

            # 每个 job 需要全部节点——expander 内部按 (replica_id * ws) 索引查表
            jobs.append(Job(
                job_id=idx,
                assigned_nodes=list(nodes),
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
            meta=Meta(num_jobs=len(jobs), num_nodes=max(assigned_nodes) + 1),
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
        job_group_id: int = 0,
    ) -> CompactWorkload:
        header, _ = AicbParser().parse(trace_path)
        # New 3D model: total_gpus = tp * dp * pp (ep is sub-division of dp)
        # dp = total_gpus / (tp * pp), ep is recorded separately for the grouper
        world_size = header.tp * header.pp

        if assigned_nodes is not None:
            if len(assigned_nodes) % world_size != 0:
                raise ValueError(
                    f"TrainingJobSlicer: assigned_nodes has {len(assigned_nodes)} nodes, "
                    f"which is not a multiple of world_size={world_size} "
                    f"(tp={header.tp} pp={header.pp})"
                )
            dp = len(assigned_nodes) // world_size
        else:
            dp = header.all_gpus // world_size
            assigned_nodes = list(range(header.all_gpus))

        jobs = []
        info = {}
        for i in range(repeat):
            jobs.append(Job(
                job_id=i,
                assigned_nodes=list(assigned_nodes),
                parallelism=ParallelismConfig(
                    tp=header.tp,
                    dp=dp,
                    pp=header.pp,
                    ep=header.ep,
                ),
            ))
            info[i] = JobExpansionInfo(
                depends_on=[i - 1] if i > 0 else [],
                job_type="training",
                trace_src=trace_path,
                trace_job_index=i,
                job_group_id=job_group_id,
            )

        return CompactWorkload(
            version="1.0",
            meta=Meta(num_jobs=len(jobs), num_nodes=max(assigned_nodes) + 1),
            jobs=jobs,
            job_expansion_info=info,
        )

"""Hermod-only metadata enrichment for validated AICB training workloads.

The generic P2P IR keeps ``iteration`` as an AICB GA-step.  This module is
the sole place that translates a validated GA-step into a Hermod MID.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .aicb_parser import AicbHeader
from ..static_analysis.passes.hermod_priority import classify_coflow_type, HermodCoflowType
from ..workload_format.schema import P2PWorkload


@dataclass(frozen=True)
class HermodMetadataRecord:
    task_id: int
    coflow_id: str
    microbatch_id: int
    logical_layer_id: int
    coflow_type: str
    provenance: str


class HermodAicbMetadataAdapter:
    """Enrich PP/DP flow tasks without changing generic parser semantics."""

    def __init__(self, header: AicbHeader, *, reject_ep: bool = True):
        if header.ga < 1:
            raise ValueError("Hermod AICB adapter requires header.ga >= 1")
        if not reject_ep:
            raise NotImplementedError(
                "Hermod EP metadata is not implemented; EP experiments remain disabled"
            )
        self.header = header
        self.reject_ep = reject_ep

    def apply(self, workload: P2PWorkload) -> list[HermodMetadataRecord]:
        records: list[HermodMetadataRecord] = []
        layer_task_ids = [
            task.layer_id for task in workload.tasks
            if 0 <= task.iteration < self.header.ga
        ]
        if not layer_task_ids:
            raise ValueError("AICB workload contains no GA-layer tasks for Hermod")
        for task in workload.get_flow_tasks():
            coflow_type = classify_coflow_type(task.comm_type)
            if coflow_type is None:
                continue
            if coflow_type == HermodCoflowType.EP and self.reject_ep:
                raise ValueError(
                    f"Hermod EP is disabled: task {task.task_id} is {task.comm_type.value}. "
                    "Use a validated EP path before enabling it."
                )
            if not task.coflow_id:
                raise ValueError(f"Hermod task {task.task_id} has no coflow_id")

            if 0 <= task.iteration < self.header.ga:
                mid = task.iteration
                provenance = "aicb_ga_step"
            elif coflow_type == HermodCoflowType.DP:
                # AICB pre/post DP operations occur outside a GA layer block;
                # §4.1.2 places them after the final microbatch.
                mid = self.header.ga - 1
                provenance = "aicb_dp_post_ga"
            else:
                raise ValueError(
                    f"PP task {task.task_id} has non-GA iteration {task.iteration}"
                )

            # builder.layer_id is a flattened AICB item position.  Persist it
            # in a separate field so the generic IR meaning remains untouched.
            if task.layer_id < 0:
                raise ValueError(
                    f"Hermod task {task.task_id} has no AICB layer position for LID mapping"
                )
            lid = task.layer_id
            task.microbatch_id = mid
            task.logical_layer_id = lid
            records.append(HermodMetadataRecord(
                task_id=task.task_id,
                coflow_id=task.coflow_id,
                microbatch_id=mid,
                logical_layer_id=lid,
                coflow_type=coflow_type.value,
                provenance=provenance,
            ))
        return records

    @staticmethod
    def write_sidecar(records: list[HermodMetadataRecord], path: str | Path) -> None:
        Path(path).write_text(
            json.dumps([asdict(record) for record in records], indent=2),
            encoding="utf-8",
        )

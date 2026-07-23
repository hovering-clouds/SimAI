"""Hermod-only metadata enrichment for validated AICB training workloads.

The generic P2P IR keeps ``iteration`` as an AICB GA-step.  This module is
the sole place that translates a validated GA-step into a Hermod MID.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from ...workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from ...workload_generator.rank_grouper import RankGrouper
from .hermod_priority import classify_coflow_type, HermodCoflowType
from ...workload_format.schema import P2PWorkload, Phase


@dataclass(frozen=True)
class HermodMetadataRecord:
    task_id: int
    coflow_id: str
    microbatch_id: int
    logical_layer_id: int
    coflow_type: str
    provenance: str
    source_operation: str | None = None
    mapping_rule: str = "flat_aicb_position"


class HermodAicbMetadataAdapter:
    """Enrich PP/DP flow tasks without changing generic parser semantics."""

    def __init__(
        self,
        header: AicbHeader,
        aicb_items: list[AicbWorkItem] | None = None,
        *,
        reject_ep: bool = True,
    ):
        if header.ga < 1:
            raise ValueError("Hermod AICB adapter requires header.ga >= 1")
        if not reject_ep:
            raise NotImplementedError(
                "Hermod EP metadata is not implemented; EP experiments remain disabled"
            )
        self.header = header
        self.aicb_items = aicb_items
        self.reject_ep = reject_ep

    def _transformer_lids(self, workload: P2PWorkload) -> tuple[dict[int, int], dict[int, str]]:
        """Recover local Transformer-layer IDs from one AICB GA block.

        AICB training traces name the local model sequence but do not emit a
        numeric layer field.  For GPT-style traces, ``attention_layer`` plus
        its immediately following ``mlp_layer`` form one Transformer layer;
        the leading embedding is an input boundary (LID 0).  This adapter-only
        mapping deliberately leaves the generic IR's ``Task.layer_id`` intact.
        """
        if self.aicb_items is None:
            return {}, {}
        item_by_lid: dict[int, int] = {}
        for task in workload.tasks:
            # PP flows are synthetic and retain item_id=0.  Use regular AICB
            # tasks to establish the layer-position -> source-item mapping.
            if (task.iteration == 0
                    and task.comm_type.value != "pp_send"):
                previous = item_by_lid.setdefault(task.layer_id, task.item_id)
                if previous != task.item_id:
                    raise ValueError(
                        f"AICB layer position {task.layer_id} maps to multiple source items"
                    )
        if not item_by_lid:
            raise ValueError("AICB workload contains no source items for strict Hermod LID mapping")

        lids: dict[int, int] = {}
        operations: dict[int, str] = {}
        next_transformer_lid = 1
        awaiting_mlp = False
        for flat_lid, item_id in sorted(item_by_lid.items()):
            if item_id < 0 or item_id >= len(self.aicb_items):
                raise ValueError(f"AICB item ID {item_id} is outside the supplied source trace")
            name = self.aicb_items[item_id].name
            if name == "embedding_layer":
                if flat_lid != 0 or awaiting_mlp:
                    raise ValueError("Unexpected embedding_layer in a Hermod GA block")
                lids[flat_lid] = 0
            elif name == "attention_layer":
                if awaiting_mlp:
                    raise ValueError("attention_layer is not followed by mlp_layer in AICB GA block")
                lids[flat_lid] = next_transformer_lid
                next_transformer_lid += 1
                awaiting_mlp = True
            elif name == "mlp_layer":
                if not awaiting_mlp:
                    raise ValueError("mlp_layer has no preceding attention_layer in AICB GA block")
                lids[flat_lid] = next_transformer_lid - 1
                awaiting_mlp = False
            else:
                raise ValueError(
                    f"Strict Hermod LID mapping does not recognize AICB operation {name!r}; "
                    "add an explicit model-layer rule before using this trace"
                )
            operations[flat_lid] = name
        if awaiting_mlp:
            raise ValueError("AICB GA block ends with attention_layer without mlp_layer")
        return lids, operations

    @staticmethod
    def _compute_coflow_id(
        task,
        *,
        split_pp_by_layer: bool = False,
        pp_direction: str | None = None,
    ) -> str | None:
        """Synthesise a stable coflow identifier from existing Task fields.

        Used as an opaque grouping key by HermodPriorityAnalysis.  No code
        in the Hermod strategy parses this string.
        """
        ct = classify_coflow_type(task.comm_type)
        if ct is None:
            return None
        suffix = ""
        if split_pp_by_layer and ct == HermodCoflowType.PP:
            # A virtual pipeline has one PP wave per chunk boundary.  Keeping
            # different boundary layers in one coflow would mix LIDs and make
            # the Hermod priority record internally inconsistent.
            suffix = f":l{task.layer_id}"
        if pp_direction is not None and ct == HermodCoflowType.PP:
            suffix += f":p{pp_direction}"
        return f"j{task.job_id}:i{task.iteration}:{task.phase.value}:{ct.value}{suffix}"

    def apply(
        self,
        workload: P2PWorkload,
        *,
        split_pp_by_layer: bool = False,
        split_pp_by_direction: bool = False,
    ) -> dict[int, HermodMetadataRecord]:
        records: dict[int, HermodMetadataRecord] = {}
        layer_task_ids = [
            task.layer_id for task in workload.tasks
            if 0 <= task.iteration < self.header.ga
        ]
        if not layer_task_ids:
            raise ValueError("AICB workload contains no GA-layer tasks for Hermod")
        transformer_lids, operations = self._transformer_lids(workload)
        last_lid = max(transformer_lids.values(), default=None)
        pp_directions = (
            self._pp_directions(workload) if split_pp_by_direction else {}
        )
        for task in workload.get_flow_tasks():
            coflow_id = self._compute_coflow_id(
                task,
                split_pp_by_layer=split_pp_by_layer,
                pp_direction=pp_directions.get(task.task_id),
            )
            if coflow_id is None:
                continue
            coflow_type = classify_coflow_type(task.comm_type)
            if coflow_type == HermodCoflowType.EP and self.reject_ep:
                raise ValueError(
                    f"Hermod EP is disabled: task {task.task_id} is {task.comm_type.value}. "
                    "Use a validated EP path before enabling it."
                )

            if 0 <= task.iteration < self.header.ga:
                mid = task.iteration
                provenance = "aicb_ga_step"
                lid = transformer_lids.get(task.layer_id, task.layer_id)
                operation = operations.get(task.layer_id)
                mapping_rule = ("attention_mlp_transformer_layer"
                                if task.layer_id in transformer_lids else "flat_aicb_position")
            elif coflow_type == HermodCoflowType.DP:
                # AICB pre/post DP operations occur outside a GA layer block;
                # §4.1.2 places them after the final microbatch.
                mid = self.header.ga - 1
                provenance = "aicb_dp_post_ga"
                lid = last_lid + 1 if last_lid is not None else task.layer_id
                operation = None
                mapping_rule = "post_ga_boundary"
            else:
                raise ValueError(
                    f"PP task {task.task_id} has non-GA iteration {task.iteration}"
                )

            if task.layer_id < 0:
                raise ValueError(
                    f"Hermod task {task.task_id} has no AICB layer position for LID mapping"
                )
            records[task.task_id] = HermodMetadataRecord(
                task_id=task.task_id,
                coflow_id=coflow_id,
                microbatch_id=mid,
                logical_layer_id=lid,
                coflow_type=coflow_type.value,
                provenance=provenance,
                source_operation=operation,
                mapping_rule=mapping_rule,
            )
        return records

    @staticmethod
    def _pp_directions(workload: P2PWorkload) -> dict[int, str]:
        """Infer logical pipeline direction from Job stage layout and phase.

        Rank numbers are intentionally ignored: non-contiguous/cyclic
        placements still use the explicit ``assigned_nodes`` order.
        """
        node_to_stage_by_job: dict[int, dict[int, int]] = {}
        for job in workload.jobs:
            grouper = RankGrouper(job.assigned_nodes, job.parallelism)
            stage_size = grouper.dp * grouper.ep * grouper.tp
            node_to_stage_by_job[job.job_id] = {
                node: index // stage_size
                for index, node in enumerate(grouper.nodes)
            }

        result: dict[int, str] = {}
        for task in workload.get_flow_tasks():
            if classify_coflow_type(task.comm_type) != HermodCoflowType.PP:
                continue
            node_to_stage = node_to_stage_by_job.get(task.job_id)
            if node_to_stage is None:
                raise ValueError(
                    f"Hermod bidirectional PP metadata lacks Job {task.job_id}"
                )
            if task.src not in node_to_stage or task.dst not in node_to_stage:
                raise ValueError(
                    f"PP task {task.task_id} endpoint is outside Job {task.job_id} placement"
                )
            src_stage = node_to_stage[task.src]
            dst_stage = node_to_stage[task.dst]
            if abs(src_stage - dst_stage) != 1:
                raise ValueError(
                    f"PP task {task.task_id} is not between adjacent physical stages: "
                    f"{src_stage}->{dst_stage}"
                )
            physical_step = 1 if dst_stage > src_stage else -1
            if task.phase is Phase.FORWARD:
                result[task.task_id] = "down" if physical_step > 0 else "up"
            elif task.phase is Phase.BACKWARD_INPUT:
                result[task.task_id] = "down" if physical_step < 0 else "up"
            else:
                raise ValueError(
                    f"PP task {task.task_id} has unsupported phase {task.phase.value}"
                )
        return result

    @staticmethod
    def write_sidecar(records: dict[int, HermodMetadataRecord] | list[HermodMetadataRecord],
                      path: str | Path) -> None:
        if isinstance(records, dict):
            records = list(records.values())
        Path(path).write_text(
            json.dumps([asdict(record) for record in records], indent=2),
            encoding="utf-8",
        )

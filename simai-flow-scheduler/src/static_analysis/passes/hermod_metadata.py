"""Hermod-only metadata enrichment for validated AICB training workloads.

The generic P2P IR keeps ``iteration`` as an AICB GA-step.  This module is
the sole place that translates a validated GA-step into a Hermod MID.
"""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from ...workload_generator.aicb_parser import AicbHeader, AicbWorkItem
from .hermod_priority import (
    classify_coflow_type,
    HermodCoflowType,
    HermodEpMode,
)
from ...workload_format.schema import P2PWorkload


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
        ep_mode: HermodEpMode = HermodEpMode.REJECT,
    ):
        if header.ga < 1:
            raise ValueError("Hermod AICB adapter requires header.ga >= 1")
        self.header = header
        self.aicb_items = aicb_items
        self.ep_mode = ep_mode

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

        operations = {}
        for flat_lid, item_id in item_by_lid.items():
            if item_id < 0 or item_id >= len(self.aicb_items):
                raise ValueError(f"AICB item ID {item_id} is outside the supplied source trace")
            operations[flat_lid] = self.aicb_items[item_id].name
        names = set(operations.values())
        gpt_names = {"embedding_layer", "attention_layer", "mlp_layer"}
        mixtral_names = {
            "embedding_layer", "attention_column", "attention_row",
            "mlp_moelayer", "final_column",
        }
        if names <= gpt_names:
            lids = self._gpt_lids(operations)
        elif names <= mixtral_names:
            lids = self._mixtral_lids(operations)
        else:
            unknown = sorted(names - gpt_names - mixtral_names)
            raise ValueError(
                "Strict Hermod LID mapping does not recognize AICB operations "
                f"{unknown!r}; add an explicit model-layer rule before using this trace"
            )
        return lids, operations

    @staticmethod
    def _gpt_lids(operations: dict[int, str]) -> dict[int, int]:
        lids: dict[int, int] = {}
        next_transformer_lid = 1
        awaiting_mlp = False
        for flat_lid, name in sorted(operations.items()):
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
        if awaiting_mlp:
            raise ValueError("AICB GA block ends with attention_layer without mlp_layer")
        return lids

    @staticmethod
    def _mixtral_lids(operations: dict[int, str]) -> dict[int, int]:
        """Map one Mixtral GA block to local Transformer/MoE layer IDs.

        AICB emits attention and MoE internals as separate rows.  One
        ``attention_column`` starts a logical layer; its ``attention_row`` and
        all following ``mlp_moelayer`` rows share that LID.  ``final_column``
        is the output boundary of the last local layer.
        """
        lids: dict[int, int] = {}
        current_lid = 0
        saw_attention_row = False
        saw_mlp = False
        for flat_lid, name in sorted(operations.items()):
            if name == "embedding_layer":
                if flat_lid != 0 or current_lid:
                    raise ValueError("Unexpected embedding_layer in a Mixtral GA block")
                lids[flat_lid] = 0
            elif name == "attention_column":
                if current_lid and not (saw_attention_row and saw_mlp):
                    raise ValueError(
                        "Mixtral attention_column starts before the preceding MoE layer is complete"
                    )
                current_lid += 1
                saw_attention_row = False
                saw_mlp = False
                lids[flat_lid] = current_lid
            elif name == "attention_row":
                if current_lid < 1 or saw_attention_row:
                    raise ValueError("Unexpected attention_row in a Mixtral GA block")
                saw_attention_row = True
                lids[flat_lid] = current_lid
            elif name == "mlp_moelayer":
                if current_lid < 1 or not saw_attention_row:
                    raise ValueError("mlp_moelayer has no preceding attention pair")
                saw_mlp = True
                lids[flat_lid] = current_lid
            elif name == "final_column":
                if current_lid < 1 or not (saw_attention_row and saw_mlp):
                    raise ValueError("final_column has no complete preceding Mixtral layer")
                lids[flat_lid] = current_lid
            else:
                raise ValueError(f"Unexpected Mixtral operation {name!r}")
        if current_lid < 1 or not (saw_attention_row and saw_mlp):
            raise ValueError("Mixtral GA block ends with an incomplete Transformer/MoE layer")
        return lids

    @staticmethod
    def _job_layout(workload: P2PWorkload) -> dict[int, tuple[dict[int, int], int]]:
        layouts = {}
        for job in workload.jobs:
            stage_size = job.parallelism.dp * job.parallelism.tp
            layouts[job.job_id] = (
                {node: index // stage_size for index, node in enumerate(job.assigned_nodes)},
                job.parallelism.pp,
            )
        return layouts

    @staticmethod
    def _stage_id(task, layouts: dict[int, tuple[dict[int, int], int]]) -> int:
        layout = layouts.get(task.job_id)
        if layout is None:
            return 0
        node_to_stage, _ = layout
        if task.src not in node_to_stage:
            raise ValueError(
                f"Hermod task {task.task_id} source node {task.src} is not assigned to its job"
            )
        return node_to_stage[task.src]

    def _compute_coflow_id(
        self, task, layouts: dict[int, tuple[dict[int, int], int]],
    ) -> str | None:
        """Synthesise a stable coflow identifier from existing Task fields.

        Used as an opaque grouping key by HermodPriorityAnalysis.  No code
        in the Hermod strategy parses this string.
        """
        ct = classify_coflow_type(task.comm_type)
        if ct is None:
            return None
        prefix = f"j{task.job_id}:i{task.iteration}:{task.phase.value}:{ct.value}"
        if ct == HermodCoflowType.PP:
            layout = layouts.get(task.job_id)
            if layout is None:
                # Synthetic unit workloads without Job metadata retain one PP
                # coflow per phase for backwards compatibility.
                return prefix
            node_to_stage, _ = layout
            if task.src not in node_to_stage or task.dst not in node_to_stage:
                raise ValueError(f"Hermod PP task {task.task_id} has an unassigned endpoint")
            src_stage, dst_stage = node_to_stage[task.src], node_to_stage[task.dst]
            if abs(src_stage - dst_stage) != 1:
                raise ValueError(
                    f"Hermod PP task {task.task_id} does not cross adjacent pipeline stages"
                )
            return f"{prefix}:boundary{min(src_stage, dst_stage)}"
        stage_id = self._stage_id(task, layouts)
        return (
            f"{prefix}:stage{stage_id}:item{task.item_id}:"
            f"{task.comm_type.value}"
        )

    def apply(self, workload: P2PWorkload) -> dict[int, HermodMetadataRecord]:
        records: dict[int, HermodMetadataRecord] = {}
        layer_task_ids = [
            task.layer_id for task in workload.tasks
            if 0 <= task.iteration < self.header.ga
        ]
        if not layer_task_ids:
            raise ValueError("AICB workload contains no GA-layer tasks for Hermod")
        transformer_lids, operations = self._transformer_lids(workload)
        local_layer_count = max(transformer_lids.values(), default=None)
        layouts = self._job_layout(workload)
        for task in workload.get_flow_tasks():
            coflow_id = self._compute_coflow_id(task, layouts)
            if coflow_id is None:
                continue
            coflow_type = classify_coflow_type(task.comm_type)
            if coflow_type == HermodCoflowType.EP and self.ep_mode == HermodEpMode.REJECT:
                raise ValueError(
                    f"Hermod EP is disabled: task {task.task_id} is {task.comm_type.value}. "
                    "Use ep_mode='enable' for a validated EP workload."
                )

            if 0 <= task.iteration < self.header.ga:
                mid = task.iteration
                provenance = "aicb_ga_step"
                local_lid = transformer_lids.get(task.layer_id, task.layer_id)
                stage_id = self._stage_id(task, layouts)
                if coflow_type == HermodCoflowType.PP and local_layer_count is not None:
                    src_stage = stage_id
                    lid = (
                        (src_stage + 1) * local_layer_count
                        if task.phase.value == "forward"
                        else src_stage * local_layer_count + 1
                    )
                elif local_layer_count is not None:
                    lid = stage_id * local_layer_count + local_lid
                else:
                    lid = local_lid
                operation = operations.get(task.layer_id)
                if operation in {"attention_layer", "mlp_layer"}:
                    mapping_rule = "attention_mlp_transformer_layer"
                elif operation in {
                    "attention_column", "attention_row", "mlp_moelayer", "final_column",
                }:
                    mapping_rule = "mixtral_attention_moe_transformer_layer"
                else:
                    mapping_rule = (
                        "model_stage_boundary"
                        if coflow_type == HermodCoflowType.PP else "flat_aicb_position"
                    )
            elif coflow_type == HermodCoflowType.DP:
                # AICB pre/post DP operations occur outside a GA layer block;
                # §4.1.2 places them after the final microbatch.
                mid = self.header.ga - 1
                provenance = "aicb_dp_post_ga"
                job_pp = layouts.get(task.job_id, ({}, 1))[1]
                lid = (
                    local_layer_count * job_pp + 1
                    if local_layer_count is not None else task.layer_id
                )
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
    def write_sidecar(records: dict[int, HermodMetadataRecord] | list[HermodMetadataRecord],
                      path: str | Path) -> None:
        if isinstance(records, dict):
            records = list(records.values())
        Path(path).write_text(
            json.dumps([asdict(record) for record in records], indent=2),
            encoding="utf-8",
        )

"""
Workload Builder - Converts AICB workloads to P2P Workload format.

Two-phase generation approach:
1. Phase 1: Generate all tasks (compute + flow) without cross-phase dependencies
2. Phase 2: Wire dependencies per-node following Forward/Backward/WG ordering

Key design principles:
- AICB files already contain GA-expanded items (outer=GA, inner=layers)
- Compute tasks are per-rank (one per assigned node per phase)
- Dependencies are per-node (each rank's chain is independent)
- Forward goes layer 0→N-1, Backward (IG) goes N-1→0, WG same layer as IG
- Cross-phase deps use receiver-based (dst) flows, not sender-based
"""

from dataclasses import dataclass, field
from typing import Iterator

from ..workload_format.schema import (
    P2PWorkload, Task, Job, Meta, Phase, TaskType
)
from .aicb_parser import AicbHeader, AicbWorkItem, AicbParser
from .rank_grouper import RankGrouper
from .collective_expander import (
    CollectiveExpander, FlowTask,
    AllReduceExpander, AllGatherExpander,
    ReduceScatterExpander, AlltoAllExpander,
)


@dataclass
class FlowGroupResult:
    """Result of expanding a collective communication into P2P flows.

    receiver_index tracks which flows each rank receives (as dst),
    incrementally maintained during flow generation for O(1) per-flow lookup.
    """
    flows: list[FlowTask]
    receiver_index: dict[int, list[int]]  # rank → task_ids where rank is dst

    @staticmethod
    def empty() -> "FlowGroupResult":
        return FlowGroupResult(flows=[], receiver_index={})

    def add_flow(self, flow: FlowTask):
        """Add a flow and update receiver index."""
        self.flows.append(flow)
        if flow.dst not in self.receiver_index:
            self.receiver_index[flow.dst] = []
        self.receiver_index[flow.dst].append(flow.task_id)


@dataclass
class ItemTasks:
    """All tasks for one AICB work item, organized by phase.

    Each *_computes dict maps rank → FlowTask (compute task for that rank).
    Each *_result holds the expanded flows with receiver_index.
    """
    fwd_computes: dict[int, FlowTask] = field(default_factory=dict)
    ig_computes: dict[int, FlowTask] = field(default_factory=dict)
    wg_computes: dict[int, FlowTask] = field(default_factory=dict)
    fwd_result: FlowGroupResult = field(default_factory=FlowGroupResult.empty)
    ig_result: FlowGroupResult = field(default_factory=FlowGroupResult.empty)
    wg_result: FlowGroupResult = field(default_factory=FlowGroupResult.empty)


class WorkloadBuilder:
    """Convert AICB workload to P2P Workload using two-phase generation."""

    # Pre-layer item names (appear before GA loop)
    PRE_LAYER_NAMES = {
        "grad_gather", "grad_param_comm", "grad_param_compute",
        "layernorm", "embedding_grads", "moe_grad_norm1", "moe_grad_norm2",
    }

    # Post-layer item names (appear after GA loop)
    POST_LAYER_NAMES = {
        "embedding_norm", "cross_entropy1", "cross_entropy2", "cross_entropy3",
        "optimizer1", "optimizer2", "optimizer3", "optimizer4",
    }

    def __init__(self):
        self.expanders: dict[str, CollectiveExpander] = {
            "ALLREDUCE": AllReduceExpander(),
            "ALLGATHER": AllGatherExpander(),
            "REDUCESCATTER": ReduceScatterExpander(),
            "ALLTOALL": AlltoAllExpander(),
        }

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        """Convert AICB workload to P2P Workload.

        Two-phase approach:
        1. Generate all tasks (compute + flow) for all items
        2. Wire dependencies following Forward/Backward/WG ordering
        """
        grouper = RankGrouper(job.assigned_nodes, job.parallelism)

        # Validate structure
        num_pre_items = self._count_pre_items(aicb_items)
        num_post_items = self._count_post_items(aicb_items)
        num_layer_items = len(aicb_items) - num_pre_items - num_post_items

        assert num_layer_items >= 0, \
            f"Invalid item count: {len(aicb_items)} total, " \
            f"{num_pre_items} pre, {num_post_items} post"
        if num_layer_items > 0:
            assert num_layer_items % aicb_header.ga == 0, \
                f"Layer items ({num_layer_items}) not divisible by GA ({aicb_header.ga})"
        items_per_ga = num_layer_items // aicb_header.ga if num_layer_items > 0 else 0

        # ===== Phase 1: Generate all tasks (no cross-phase deps) =====
        all_flow_tasks: list[FlowTask] = []
        item_tasks_list: list[ItemTasks] = []
        task_id_counter = 0
        ranks = job.assigned_nodes

        for item_idx, item in enumerate(aicb_items):
            # Calculate iteration and layer_id
            if item_idx < num_pre_items:
                # Pre items: iteration=-1, layer_id=sequential (0,1,2...)
                iteration = -1
                layer_id = item_idx
            elif item_idx >= num_pre_items + num_layer_items:
                # Post items: iteration=ga, layer_id=sequential (0,1,2...)
                iteration = aicb_header.ga
                layer_id = item_idx - (num_pre_items + num_layer_items)
            else:
                # Layer items: iteration=GA step, layer_id=flat index within GA
                iteration = (item_idx - num_pre_items) // items_per_ga
                layer_id = (item_idx - num_pre_items) % items_per_ga

            item_id = item_idx
            item_tasks = ItemTasks()

            # --- Forward phase ---
            item_tasks.fwd_computes, task_id_counter = \
                self._create_compute_tasks_for_phase(
                    ranks, item.forward_compute_time, Phase.FORWARD,
                    layer_id, iteration, item_id, job.job_id, task_id_counter)
            all_flow_tasks.extend(item_tasks.fwd_computes.values())

            if item.forward_comm != "NONE":
                item_tasks.fwd_result, task_id_counter = \
                    self._expand_comm_all_groups(
                        item.forward_comm, item.forward_comm_size,
                        grouper, Phase.FORWARD, layer_id, iteration, item_id,
                        job.job_id, task_id_counter, comm_algo,
                        default_context="tp")
                all_flow_tasks.extend(item_tasks.fwd_result.flows)
                self._wire_compute_to_flows(
                    item_tasks.fwd_computes, item_tasks.fwd_result)

            # --- Backward (IG) phase ---
            item_tasks.ig_computes, task_id_counter = \
                self._create_compute_tasks_for_phase(
                    ranks, item.backward_compute_time, Phase.BACKWARD_INPUT,
                    layer_id, iteration, item_id, job.job_id, task_id_counter)
            all_flow_tasks.extend(item_tasks.ig_computes.values())

            if item.backward_comm != "NONE":
                item_tasks.ig_result, task_id_counter = \
                    self._expand_comm_all_groups(
                        item.backward_comm, item.backward_comm_size,
                        grouper, Phase.BACKWARD_INPUT, layer_id, iteration, item_id,
                        job.job_id, task_id_counter, comm_algo,
                        default_context="tp")
                all_flow_tasks.extend(item_tasks.ig_result.flows)
                self._wire_compute_to_flows(
                    item_tasks.ig_computes, item_tasks.ig_result)

            # --- DP (WG) phase ---
            item_tasks.wg_computes, task_id_counter = \
                self._create_compute_tasks_for_phase(
                    ranks, item.dp_compute_time, Phase.BACKWARD_WEIGHT,
                    layer_id, iteration, item_id, job.job_id, task_id_counter)
            all_flow_tasks.extend(item_tasks.wg_computes.values())

            if item.dp_comm != "NONE":
                item_tasks.wg_result, task_id_counter = \
                    self._expand_comm_all_groups(
                        item.dp_comm, item.dp_comm_size,
                        grouper, Phase.BACKWARD_WEIGHT, layer_id, iteration, item_id,
                        job.job_id, task_id_counter, comm_algo,
                        default_context="dp")
                all_flow_tasks.extend(item_tasks.wg_result.flows)
                self._wire_compute_to_flows(
                    item_tasks.wg_computes, item_tasks.wg_result)

            item_tasks_list.append(item_tasks)

        # ===== Phase 2: Wire dependencies =====
        self._wire_dependencies(
            item_tasks_list, num_layer_items, num_pre_items, items_per_ga)

        # Build P2PWorkload
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(job.assigned_nodes)),
            jobs=[job],
            tasks=[t.to_task() for t in all_flow_tasks],
        )
        return workload

    # ------------------------------------------------------------------
    # Phase 1 helpers: Task generation
    # ------------------------------------------------------------------

    def _create_compute_tasks_for_phase(
        self,
        ranks: list[int],
        compute_time_ns: int,
        phase: Phase,
        layer_id: int,
        iteration: int,
        item_id: int,
        job_id: int,
        task_id_counter: int,
    ) -> tuple[dict[int, FlowTask], int]:
        """Create a compute task for each rank.

        Args:
            compute_time_ns: compute time from AICB file (unit: ns).

        Returns (rank→FlowTask dict, updated task_id_counter).
        """
        duration_us = compute_time_ns // 1000
        tasks: dict[int, FlowTask] = {}
        for rank in ranks:
            task = FlowTask(
                task_id=task_id_counter,
                job_id=job_id,
                type=TaskType.COMPUTE,
                node=rank,
                duration_us=duration_us,
                phase=phase,
                layer_id=layer_id,
                iteration=iteration,
                item_id=item_id,
            )
            tasks[rank] = task
            task_id_counter += 1
        return tasks, task_id_counter

    def _expand_comm_all_groups(
        self,
        comm_type_str: str,
        comm_size: int,
        grouper: RankGrouper,
        phase: Phase,
        layer_id: int,
        iteration: int,
        item_id: int,
        job_id: int,
        task_id_counter: int,
        algo: str = "ring",
        default_context: str = "tp",
    ) -> tuple[FlowGroupResult, int]:
        """Expand communication for all parallel subgroups.

        Args:
            default_context: Parallelism context when comm string has no suffix.
                For forward_comm and backward_comm: "tp" (default per AICB convention).
                For dp_comm: "dp" (the field is inherently DP-scoped).

        Iterates over all subgroups for the comm context (e.g. all TP groups),
        expands each into P2P flows, and aggregates into a FlowGroupResult
        with incrementally maintained receiver_index.
        """
        base_type, context = AicbParser.parse_comm_type(comm_type_str, default_context)
        result = FlowGroupResult.empty()

        for subgroup in self._iter_subgroups(context, grouper):
            if len(subgroup) >= 2:
                flows = self._call_expander(
                    base_type, subgroup, comm_size,
                    job_id, task_id_counter, algo, context)
                for flow in flows:
                    flow.phase = phase
                    flow.layer_id = layer_id
                    flow.iteration = iteration
                    flow.item_id = item_id
                    result.add_flow(flow)
                task_id_counter += len(flows)

        return result, task_id_counter

    def _iter_subgroups(
        self, context: str, grouper: RankGrouper
    ) -> Iterator[list[int]]:
        """Yield all parallel subgroups for the given comm context."""
        if context == "tp":
            for pp_idx in range(grouper.pp):
                for dp_idx in range(grouper.dp):
                    for ep_idx in range(grouper.ep):
                        yield grouper.get_tp_group(pp_idx, dp_idx, ep_idx)
        elif context == "dp":
            for pp_idx in range(grouper.pp):
                for ep_idx in range(grouper.ep):
                    for tp_idx in range(grouper.tp):
                        yield grouper.get_dp_group(pp_idx, ep_idx, tp_idx)
        elif context == "ep":
            for pp_idx in range(grouper.pp):
                for dp_idx in range(grouper.dp):
                    for tp_idx in range(grouper.tp):
                        yield grouper.get_ep_group(pp_idx, dp_idx, tp_idx)
        elif context == "dp_ep":
            for pp_idx in range(grouper.pp):
                for tp_idx in range(grouper.tp):
                    yield grouper.get_dp_ep_group(pp_idx, tp_idx)

    def _call_expander(
        self,
        base_type: str,
        ranks: list[int],
        comm_size: int,
        job_id: int,
        task_id_start: int,
        algo: str,
        context: str,
    ) -> list[FlowTask]:
        """Dispatch to the appropriate collective expander."""
        expander = self.expanders.get(base_type)
        if expander is None:
            raise ValueError(f"Unsupported comm type: {base_type}")

        if base_type == "ALLREDUCE":
            return expander.expand_allreduce(
                ranks, comm_size, algo, job_id, task_id_start, context)
        elif base_type == "ALLGATHER":
            return expander.expand_allgather(
                ranks, comm_size, algo, job_id, task_id_start, context)
        elif base_type == "REDUCESCATTER":
            return expander.expand_reducescatter(
                ranks, comm_size, algo, job_id, task_id_start, context)
        elif base_type == "ALLTOALL":
            return expander.expand_alltoall(
                ranks, comm_size, job_id, task_id_start, context)
        else:
            raise ValueError(f"Unsupported base type: {base_type}")

    def _wire_compute_to_flows(
        self,
        computes: dict[int, FlowTask],
        result: FlowGroupResult,
    ):
        """Connect each rank's compute task to all its src (sender) flows.

        rank R's compute must finish before R can start sending data.
        A rank may have multiple independent src flows (e.g. AlltoAll),
        so we connect to ALL of them.
        """
        for flow in result.flows:
            if flow.src in computes:
                flow.deps.append(computes[flow.src].task_id)

    def _wire_per_node_phase_transition(
        self,
        src_result: FlowGroupResult,
        src_computes: dict[int, FlowTask],
        dst_computes: dict[int, FlowTask],
    ):
        """Wire per-node cross-phase dependencies.

        If src has flows: dst_compute(R) depends on all flows where R is receiver (dst).
        If src has no flows: fall back to compute(R) → compute(R) direct.
        """
        if not src_result.flows:
            # No communication: direct compute(R) → compute(R)
            for rank, dst_compute in dst_computes.items():
                if rank in src_computes:
                    dst_compute.deps.append(src_computes[rank].task_id)
            return

        # Has communication: depend on all receiver flows
        for rank, dst_compute in dst_computes.items():
            received_ids = src_result.receiver_index.get(rank, [])
            dst_compute.deps.extend(received_ids)

    def _wire_dependencies(
        self,
        item_tasks_list: list[ItemTasks],
        num_layer_items: int,
        num_pre_items: int,
        items_per_ga: int,
    ):
        """Wire all dependencies using fork-join pattern.

        GA steps have no data dependency and can execute in parallel.
        Pre/post items are barriers before/after all GA steps.

          Forward:
            pre_0.fwd → ... → pre_N.fwd ──┬──→ GA[0].fwd chain
                                           └──→ GA[1].fwd chain  ← parallel
                                                    ...
                                           └──→ GA[K].fwd chain
              GA[0] fwd done ──┬──→ post_0.fwd → ... → post_M.fwd
              GA[K] fwd done ──┘
              Bridge: post_M.fwd → post_M.ig

          Backward (reverse):
            post_M.ig → ... → post_0.ig ──┬──→ GA[0].bkwd chain
                                            └──→ GA[1].bkwd chain  ← parallel
                                                     ...
                                            └──→ GA[K].bkwd chain
              GA[0] bkwd done ──┬──→ pre_N.ig → ... → pre_0.ig
              GA[K] bkwd done ──┘

        Each section internally:
          Forward chain: item[0].fwd → item[1].fwd → ... → item[N].fwd
          Backward chain: item[N].ig → item[N-1].ig → ... → item[0].ig
          IG→WG: item[i].ig → item[i].wg (same layer)
        """
        ga_groups = self._group_items_by_ga(
            item_tasks_list, num_pre_items, num_layer_items, items_per_ga)
        post_start = num_pre_items + num_layer_items
        pre_items = item_tasks_list[:num_pre_items]
        post_items = item_tasks_list[post_start:]

        # ===== Internal wiring for each section =====

        # Pre items: forward chain + backward chain + IG→WG
        self._wire_forward_chain(pre_items)
        self._wire_backward_chain(pre_items)

        # Each GA group: forward chain + backward chain + IG→WG
        for ga_group in ga_groups:
            self._wire_forward_chain(ga_group)
            self._wire_backward_chain(ga_group)

        # Post items: forward chain + bridge + backward chain + IG→WG
        self._wire_forward_chain(post_items)
        if post_items:
            last = post_items[-1]
            self._wire_per_node_phase_transition(
                src_result=last.fwd_result,
                src_computes=last.fwd_computes,
                dst_computes=last.ig_computes)
        self._wire_backward_chain(post_items)

        # ===== Barrier connections between sections =====

        if not ga_groups:
            # No GA groups: connect pre → post directly (both directions)
            if pre_items and post_items:
                # Forward: pre → post
                self._wire_per_node_phase_transition(
                    src_result=pre_items[-1].fwd_result,
                    src_computes=pre_items[-1].fwd_computes,
                    dst_computes=post_items[0].fwd_computes)
                # Backward: post → pre
                self._wire_per_node_phase_transition(
                    src_result=post_items[0].ig_result,
                    src_computes=post_items[0].ig_computes,
                    dst_computes=pre_items[-1].ig_computes)
            return

        # --- Forward barriers ---
        # Fork: pre → all GAs
        if pre_items:
            for ga_group in ga_groups:
                self._wire_per_node_phase_transition(
                    src_result=pre_items[-1].fwd_result,
                    src_computes=pre_items[-1].fwd_computes,
                    dst_computes=ga_group[0].fwd_computes)

        # Join: all GAs → post
        if post_items:
            for ga_group in ga_groups:
                self._wire_per_node_phase_transition(
                    src_result=ga_group[-1].fwd_result,
                    src_computes=ga_group[-1].fwd_computes,
                    dst_computes=post_items[0].fwd_computes)

        # --- Backward barriers ---
        # Fork: post → all GAs
        if post_items:
            for ga_group in ga_groups:
                self._wire_per_node_phase_transition(
                    src_result=post_items[0].ig_result,
                    src_computes=post_items[0].ig_computes,
                    dst_computes=ga_group[-1].ig_computes)

        # Join: all GAs → pre
        if pre_items:
            for ga_group in ga_groups:
                self._wire_per_node_phase_transition(
                    src_result=ga_group[0].ig_result,
                    src_computes=ga_group[0].ig_computes,
                    dst_computes=pre_items[-1].ig_computes)

    def _wire_forward_chain(self, items: list[ItemTasks]):
        """Wire forward chain: item[0].fwd → item[1].fwd → ... → item[N-1].fwd."""
        for i in range(len(items) - 1):
            self._wire_per_node_phase_transition(
                src_result=items[i].fwd_result,
                src_computes=items[i].fwd_computes,
                dst_computes=items[i + 1].fwd_computes)

    def _wire_backward_chain(self, items: list[ItemTasks]):
        """Wire backward chain + IG→WG: item[N-1].ig → ... → item[0].ig, each ig→wg."""
        for i in range(len(items) - 1, -1, -1):
            # IG→WG same layer
            self._wire_per_node_phase_transition(
                src_result=items[i].ig_result,
                src_computes=items[i].ig_computes,
                dst_computes=items[i].wg_computes)
            # IG→previous layer IG (reverse)
            if i > 0:
                self._wire_per_node_phase_transition(
                    src_result=items[i].ig_result,
                    src_computes=items[i].ig_computes,
                    dst_computes=items[i - 1].ig_computes)

    def _group_items_by_ga(
        self,
        item_tasks_list: list[ItemTasks],
        num_pre_items: int,
        num_layer_items: int,
        items_per_ga: int,
    ) -> list[list[ItemTasks]]:
        """Group layer items by GA step.

        Items are in outer=GA, inner=layers order:
          [pre...][GA0_items, GA1_items, ...][post...]
        """
        if num_layer_items == 0:
            return []

        num_ga_steps = num_layer_items // items_per_ga
        ga_groups = []
        for g in range(num_ga_steps):
            start = num_pre_items + g * items_per_ga
            ga_group = item_tasks_list[start:start + items_per_ga]
            ga_groups.append(ga_group)
        return ga_groups

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _count_pre_items(self, items: list[AicbWorkItem]) -> int:
        """Count pre-layer items (grad_gather, etc.) at the start."""
        count = 0
        for item in items:
            if item.name in self.PRE_LAYER_NAMES:
                count += 1
            else:
                break
        return count

    def _count_post_items(self, items: list[AicbWorkItem]) -> int:
        """Count post-layer items (embedding_norm, optimizer, etc.) at the end."""
        count = 0
        for item in reversed(items):
            if item.name in self.POST_LAYER_NAMES:
                count += 1
            else:
                break
        return count

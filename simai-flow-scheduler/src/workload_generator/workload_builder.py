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

from collections.abc import Iterator
from dataclasses import dataclass, field

from ..workload_format.schema import Job, Meta, P2PWorkload, Phase, TaskType
from .aicb_parser import AicbHeader, AicbParser, AicbWorkItem
from .collective_expander import (
    AllGatherExpander,
    AllReduceExpander,
    AlltoAllExpander,
    BroadcastExpander,
    CollectiveExpander,
    FlowTask,
    ReduceScatterExpander,
)
from .rank_grouper import RankGrouper
from .zero_semantics import (
    ZeroItemKind,
    classify_zero_item,
    is_zero_post_item,
    is_zero_pre_item,
    is_zero_workload,
)


@dataclass
class FlowGroupResult:
    """Result of expanding a collective communication into P2P flows.

    receiver_index tracks data-ready receiver flows. completion_index tracks
    event completion for each rank; the two differ for broadcast roots.
    """
    flows: list[FlowTask]
    receiver_index: dict[int, list[int]]  # rank 鈫?task_ids where rank is dst
    completion_index: dict[int, list[int]]

    @staticmethod
    def empty() -> "FlowGroupResult":
        return FlowGroupResult(flows=[], receiver_index={}, completion_index={})

    def add_flow(self, flow: FlowTask):
        """Add a flow and update default receiver/completion indices."""
        self.flows.append(flow)
        self.receiver_index.setdefault(flow.dst, []).append(flow.task_id)
        self.completion_index.setdefault(flow.dst, []).append(flow.task_id)

    def add_completion(self, rank: int, task_ids: list[int]):
        """Record non-receiver completion work, used by broadcast roots."""
        self.completion_index.setdefault(rank, []).extend(task_ids)


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


@dataclass
class PPFlowResult:
    """PP flow generation result, indexed for wiring.

    forward_flows[(ga_idx, pp_boundary)][sender_rank] = forward PP flow
    backward_flows[(ga_idx, pp_boundary)][sender_rank] = backward PP flow
    """
    forward_flows: dict[tuple[int, int], dict[int, FlowTask]]
    backward_flows: dict[tuple[int, int], dict[int, FlowTask]]
    all_flows: list[FlowTask]


@dataclass
class ZeroBuildState:
    """Per-GA dependency state for ZeRO/FSDP-style synchronization."""

    bucket_wg_computes: dict[int, list[FlowTask]] = field(default_factory=dict)


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
            "BROADCAST": BroadcastExpander(),
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
        if is_zero_workload(aicb_items):
            return self._build_zero_from_aicb(aicb_header, aicb_items, job, comm_algo)

        return self._build_generic_from_aicb(aicb_header, aicb_items, job, comm_algo)

    def _build_generic_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        """Original generic AICB expansion path."""
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

        # ===== Phase 1.5 + 2.5: PP flows (if pp > 1) =====
        if grouper.pp > 1 and aicb_header.pp_comm_size > 0:
            ga_groups = self._group_items_by_ga(
                item_tasks_list, num_pre_items, num_layer_items, items_per_ga)
            pp_result, task_id_counter = self._generate_pp_flows(
                grouper, aicb_header, ga_groups, items_per_ga,
                job.job_id, task_id_counter)
            all_flow_tasks.extend(pp_result.all_flows)
            self._wire_pp_dependencies(
                pp_result, ga_groups, grouper)

        # Build P2PWorkload
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(job.assigned_nodes)),
            jobs=[job],
            tasks=[t.to_task() for t in all_flow_tasks],
        )
        return workload

    def _build_zero_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        """Build a P2P workload for DeepSpeed ZeRO/FSDP-like SimAI rows.

        This path keeps the public IR unchanged and only changes dependency
        direction for ZeRO parameter all-gather and gradient reduce-scatter.
        """
        grouper = RankGrouper(job.assigned_nodes, job.parallelism)
        num_pre_items = self._count_zero_pre_items(aicb_items)
        num_post_items = self._count_zero_post_items(aicb_items)
        ga_item_indices = self._split_zero_ga_item_indices(
            aicb_items, num_pre_items, num_post_items, aicb_header.ga)
        ga_item_locations = {
            item_idx: (ga_index, layer_id)
            for ga_index, group in enumerate(ga_item_indices)
            for layer_id, item_idx in enumerate(group)
        }

        all_flow_tasks: list[FlowTask] = []
        item_tasks_list: list[ItemTasks] = []
        task_id_counter = 0
        ranks = job.assigned_nodes

        for item_idx, item in enumerate(aicb_items):
            if item_idx < num_pre_items:
                iteration = -1
                layer_id = item_idx
            elif item_idx >= len(aicb_items) - num_post_items:
                iteration = aicb_header.ga
                layer_id = item_idx - (len(aicb_items) - num_post_items)
            else:
                iteration, layer_id = ga_item_locations.get(item_idx, (0, 0))

            item_tasks, task_id_counter = self._create_zero_item_tasks(
                item=item,
                ranks=ranks,
                grouper=grouper,
                layer_id=layer_id,
                iteration=iteration,
                item_id=item_idx,
                job_id=job.job_id,
                task_id_counter=task_id_counter,
                comm_algo=comm_algo,
            )
            all_flow_tasks.extend(item_tasks.fwd_computes.values())
            all_flow_tasks.extend(item_tasks.ig_computes.values())
            all_flow_tasks.extend(item_tasks.wg_computes.values())
            all_flow_tasks.extend(item_tasks.fwd_result.flows)
            all_flow_tasks.extend(item_tasks.ig_result.flows)
            all_flow_tasks.extend(item_tasks.wg_result.flows)
            item_tasks_list.append(item_tasks)

        pre_items = item_tasks_list[:num_pre_items]
        post_items = item_tasks_list[len(aicb_items) - num_post_items:]
        ga_groups = [
            [item_tasks_list[item_idx] for item_idx in group]
            for group in ga_item_indices
        ]
        self._wire_zero_dependencies(pre_items, ga_groups, post_items)

        if grouper.pp > 1 and aicb_header.pp_comm_size > 0:
            items_per_ga = max((len(group) for group in ga_groups), default=0)
            pp_result, task_id_counter = self._generate_pp_flows(
                grouper, aicb_header, ga_groups, items_per_ga,
                job.job_id, task_id_counter)
            all_flow_tasks.extend(pp_result.all_flows)
            self._wire_zero_pp_dependencies(pp_result, ga_groups, grouper)

        return P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(job.assigned_nodes)),
            jobs=[job],
            tasks=[t.to_task() for t in all_flow_tasks],
        )

    def _create_zero_item_tasks(
        self,
        item: AicbWorkItem,
        ranks: list[int],
        grouper: RankGrouper,
        layer_id: int,
        iteration: int,
        item_id: int,
        job_id: int,
        task_id_counter: int,
        comm_algo: str,
    ) -> tuple[ItemTasks, int]:
        """Create tasks for one ZeRO row without generic dp_comm ordering."""
        kind = classify_zero_item(item.name)
        item_tasks = ItemTasks()

        if kind is ZeroItemKind.FWD_PARAM_ALLGATHER:
            if item.dp_comm != "NONE":
                item_tasks.fwd_result, task_id_counter = self._expand_comm_all_groups(
                    item.dp_comm, item.dp_comm_size, grouper, Phase.FORWARD,
                    layer_id, iteration, item_id, job_id, task_id_counter,
                    comm_algo, default_context="dp")
            return item_tasks, task_id_counter

        if kind is ZeroItemKind.BWD_PARAM_ALLGATHER:
            if item.dp_comm != "NONE":
                item_tasks.ig_result, task_id_counter = self._expand_comm_all_groups(
                    item.dp_comm, item.dp_comm_size, grouper, Phase.BACKWARD_INPUT,
                    layer_id, iteration, item_id, job_id, task_id_counter,
                    comm_algo, default_context="dp")
            return item_tasks, task_id_counter

        if kind in {
            ZeroItemKind.GRAD_REDUCESCATTER,
            ZeroItemKind.STEP_GRAD_REDUCESCATTER,
            ZeroItemKind.GRAD_SYNC,
        }:
            if item.dp_comm != "NONE":
                item_tasks.wg_result, task_id_counter = self._expand_comm_all_groups(
                    item.dp_comm, item.dp_comm_size, grouper, Phase.BACKWARD_WEIGHT,
                    layer_id, iteration, item_id, job_id, task_id_counter,
                    comm_algo, default_context="dp")
            return item_tasks, task_id_counter

        if kind in {ZeroItemKind.STEP, ZeroItemKind.INIT} and item.dp_comm != "NONE":
            phase = Phase.OPTIMIZER if kind is ZeroItemKind.STEP else Phase.FORWARD
            item_tasks.wg_result, task_id_counter = self._expand_comm_all_groups(
                item.dp_comm, item.dp_comm_size, grouper, phase,
                layer_id, iteration, item_id, job_id, task_id_counter,
                comm_algo, default_context="dp")
            return item_tasks, task_id_counter

        if item.forward_compute_time > 0:
            item_tasks.fwd_computes, task_id_counter = \
                self._create_compute_tasks_for_phase(
                    ranks, item.forward_compute_time, Phase.FORWARD,
                    layer_id, iteration, item_id, job_id, task_id_counter)
        if item.forward_comm != "NONE":
            item_tasks.fwd_result, task_id_counter = self._expand_comm_all_groups(
                item.forward_comm, item.forward_comm_size, grouper, Phase.FORWARD,
                layer_id, iteration, item_id, job_id, task_id_counter,
                comm_algo, default_context="tp")
            self._wire_compute_to_flows(item_tasks.fwd_computes, item_tasks.fwd_result)

        if item.backward_compute_time > 0:
            phase = (
                Phase.BACKWARD_WEIGHT
                if kind is ZeroItemKind.BWD_WEIGHT_COMPUTE
                else Phase.BACKWARD_INPUT
            )
            target = "wg_computes" if phase is Phase.BACKWARD_WEIGHT else "ig_computes"
            computes, task_id_counter = self._create_compute_tasks_for_phase(
                ranks, item.backward_compute_time, phase,
                layer_id, iteration, item_id, job_id, task_id_counter)
            setattr(item_tasks, target, computes)
        if item.backward_comm != "NONE":
            item_tasks.ig_result, task_id_counter = self._expand_comm_all_groups(
                item.backward_comm, item.backward_comm_size, grouper,
                Phase.BACKWARD_INPUT, layer_id, iteration, item_id, job_id,
                task_id_counter, comm_algo, default_context="tp")
            self._wire_compute_to_flows(item_tasks.ig_computes, item_tasks.ig_result)

        if item.dp_compute_time > 0:
            item_tasks.wg_computes, task_id_counter = \
                self._create_compute_tasks_for_phase(
                    ranks, item.dp_compute_time, Phase.BACKWARD_WEIGHT,
                    layer_id, iteration, item_id, job_id, task_id_counter)
        if item.dp_comm != "NONE":
            item_tasks.wg_result, task_id_counter = self._expand_comm_all_groups(
                item.dp_comm, item.dp_comm_size, grouper, Phase.BACKWARD_WEIGHT,
                layer_id, iteration, item_id, job_id, task_id_counter,
                comm_algo, default_context="dp")
            self._wire_compute_to_flows(item_tasks.wg_computes, item_tasks.wg_result)

        return item_tasks, task_id_counter

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

        for subgroup_index, subgroup in enumerate(self._iter_subgroups(context, grouper)):
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
                if base_type == "BROADCAST" and flows:
                    # The root has no incoming flow, but its broadcast event is
                    # complete only after it has sent to every peer.
                    result.add_completion(subgroup[0], [flow.task_id for flow in flows])
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
        elif base_type == "BROADCAST":
            return expander.expand_broadcast(
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
              Per-GA bridge:  each GA[-1].fwd → GA[-1].ig
              Post bridge:    post[-1].fwd → post[-1].ig

          Backward (reverse):
            each GA chain: GA[-1].ig → ... → GA[0].ig  (per GA, independent)
              GA[0] bkwd done ──┬──→ pre_N.ig → ... → pre_0.ig
              GA[K] bkwd done ──┘

        Each section internally:
          Forward chain: item[0].fwd → item[1].fwd → ... → item[N].fwd
          Backward chain: item[N].ig → item[N-1].ig → ... → item[0].ig
          IG→WG: item[i].ig → item[i].wg (same layer)

        Note: Backward Fork (post[0].ig → GA[-1].ig) is intentionally
        omitted — post items' backward is all compute=0/comm=NONE and
        does not gate GA backward. The post→GA ordering in GPipe is
        enforced by compute_order, not DAG.
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

        # Post items: forward chain + backward chain + IG→WG
        self._wire_forward_chain(post_items)
        self._wire_backward_chain(post_items)

        # Per-GA bridge: each GA group's backward depends on its own forward.
        # This ensures GA's ig has a data dependency on its fwd, which is
        # the correct per-microbatch backprop dependency. For 1F1B scheduling,
        # this is essential — each GA's backward can start as soon as its
        # forward completes, rather than waiting for a global barrier.
        #
        # Also keeps the post items bridge (post[-1].fwd → post[-1].ig)
        # for the optimizer/cross-entropy backward (which is all NONE/0).
        for ga_group in ga_groups:
            last_item = ga_group[-1]
            self._wire_per_node_phase_transition(
                src_result=last_item.fwd_result,
                src_computes=last_item.fwd_computes,
                dst_computes=last_item.ig_computes)

        if post_items:
            last_post = post_items[-1]
            self._wire_per_node_phase_transition(
                src_result=last_post.fwd_result,
                src_computes=last_post.fwd_computes,
                dst_computes=last_post.ig_computes)

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

    def _wire_zero_dependencies(
        self,
        pre_items: list[ItemTasks],
        ga_groups: list[list[ItemTasks]],
        post_items: list[ItemTasks],
    ):
        """Wire ZeRO/FSDP all-gather-before-compute dependencies."""
        initial_fwd_deps = self._wire_zero_event_sequence(pre_items, {})
        ga_forward_outputs: list[dict[int, list[int]]] = []
        ga_backward_outputs: list[dict[int, list[int]]] = []
        for ga_group in ga_groups:
            fwd_output = self._wire_zero_forward_group(ga_group, initial_fwd_deps)
            ga_forward_outputs.append(fwd_output)
            bwd_output = self._wire_zero_backward_group(ga_group, fwd_output)
            ga_backward_outputs.append(bwd_output)

        if post_items:
            self._wire_zero_event_sequence(
                post_items, self._merge_rank_deps(ga_backward_outputs))

    def _wire_zero_forward_group(
        self,
        ga_group: list[ItemTasks],
        initial_deps: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        current_deps = {rank: deps[:] for rank, deps in initial_deps.items()}

        for item_tasks in ga_group:
            if item_tasks.fwd_result.flows and not item_tasks.fwd_computes:
                self._wire_rank_deps_to_flows(current_deps, item_tasks.fwd_result)
                current_deps = self._result_output_by_rank(item_tasks.fwd_result)
                continue
            if not item_tasks.fwd_computes:
                continue

            self._add_rank_deps(item_tasks.fwd_computes, current_deps)
            current_deps = self._phase_output_by_rank(
                item_tasks.fwd_computes, item_tasks.fwd_result)

        return current_deps

    def _wire_zero_backward_group(
        self,
        ga_group: list[ItemTasks],
        initial_deps: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        state = ZeroBuildState()
        current_deps = {rank: deps[:] for rank, deps in initial_deps.items()}
        latest_ig_output: dict[int, list[int]] = {}
        terminal_deps: dict[int, list[int]] = {}

        for item_tasks in ga_group:
            if item_tasks.ig_result.flows and not item_tasks.ig_computes:
                self._wire_rank_deps_to_flows(current_deps, item_tasks.ig_result)
                current_deps = self._result_output_by_rank(item_tasks.ig_result)
                continue

            if item_tasks.ig_computes:
                self._add_rank_deps(item_tasks.ig_computes, current_deps)
                latest_ig_output = self._phase_output_by_rank(
                    item_tasks.ig_computes, item_tasks.ig_result)
                current_deps = latest_ig_output

            if item_tasks.wg_computes:
                wg_deps = latest_ig_output or current_deps
                self._add_rank_deps(item_tasks.wg_computes, wg_deps)
                for rank, compute in item_tasks.wg_computes.items():
                    state.bucket_wg_computes.setdefault(rank, []).append(compute)

            if item_tasks.wg_result.flows and not item_tasks.wg_computes:
                if state.bucket_wg_computes:
                    self._wire_bucket_wg_to_flows(
                        state.bucket_wg_computes, item_tasks.wg_result)
                else:
                    # Parameter-level AICB omits standalone compute rows for
                    # norm-like 1D parameters. Their gradient bucket still
                    # belongs after the most recent backward computation.
                    self._wire_rank_deps_to_flows(current_deps, item_tasks.wg_result)
                terminal_deps = self._merge_rank_deps(
                    [terminal_deps, self._result_output_by_rank(item_tasks.wg_result)])
                state.bucket_wg_computes.clear()

        pending_wg_deps = {
            rank: [compute.task_id for compute in computes]
            for rank, computes in state.bucket_wg_computes.items()
        }
        return self._merge_rank_deps([current_deps, terminal_deps, pending_wg_deps])

    def _wire_zero_event_sequence(
        self,
        items: list[ItemTasks],
        initial_deps: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        """Wire ordered init/step/post events and return their completions."""
        current_deps = {rank: deps[:] for rank, deps in initial_deps.items()}
        for item_tasks in items:
            comm_result = self._zero_comm_only_result(item_tasks)
            if comm_result is not None:
                self._wire_rank_deps_to_flows(current_deps, comm_result)
                current_deps = self._result_output_by_rank(comm_result)
                continue

            if item_tasks.fwd_computes:
                self._add_rank_deps(item_tasks.fwd_computes, current_deps)
                current_deps = self._phase_output_by_rank(
                    item_tasks.fwd_computes, item_tasks.fwd_result)
            if item_tasks.ig_computes:
                self._add_rank_deps(item_tasks.ig_computes, current_deps)
                self._wire_per_node_phase_transition(
                    item_tasks.ig_result, item_tasks.ig_computes,
                    item_tasks.wg_computes)
        return current_deps

    @staticmethod
    def _zero_comm_only_result(item_tasks: ItemTasks) -> FlowGroupResult | None:
        if item_tasks.fwd_computes or item_tasks.ig_computes or item_tasks.wg_computes:
            return None
        for result in (
            item_tasks.fwd_result, item_tasks.ig_result, item_tasks.wg_result,
        ):
            if result.flows:
                return result
        return None

    @staticmethod
    def _merge_rank_deps(
        outputs: list[dict[int, list[int]]],
    ) -> dict[int, list[int]]:
        merged: dict[int, list[int]] = {}
        for output in outputs:
            for rank, deps in output.items():
                merged.setdefault(rank, []).extend(dep for dep in deps if dep not in merged[rank])
        return merged

    @staticmethod
    def _wire_rank_deps_to_flows(
        deps_by_rank: dict[int, list[int]],
        result: FlowGroupResult,
    ):
        for flow in result.flows:
            for dep in deps_by_rank.get(flow.src, []):
                if dep not in flow.deps:
                    flow.deps.append(dep)

    @staticmethod
    def _wire_bucket_wg_to_flows(
        wg_by_rank: dict[int, list[FlowTask]],
        result: FlowGroupResult,
    ):
        for flow in result.flows:
            for compute in wg_by_rank.get(flow.src, []):
                if compute.task_id not in flow.deps:
                    flow.deps.append(compute.task_id)

    @staticmethod
    def _result_output_by_rank(result: FlowGroupResult) -> dict[int, list[int]]:
        return {rank: deps[:] for rank, deps in result.completion_index.items()}

    def _phase_output_by_rank(
        self,
        computes: dict[int, FlowTask],
        result: FlowGroupResult,
    ) -> dict[int, list[int]]:
        if result.flows:
            return {rank: deps[:] for rank, deps in result.completion_index.items()}
        return {rank: [task.task_id] for rank, task in computes.items()}

    def _add_rank_deps(
        self,
        computes: dict[int, FlowTask],
        deps_by_rank: dict[int, list[int]],
    ):
        for rank, compute in computes.items():
            compute.deps.extend(deps_by_rank.get(rank, []))

    def _add_receiver_deps(
        self,
        computes: dict[int, FlowTask],
        result: FlowGroupResult,
    ):
        for rank, compute in computes.items():
            compute.deps.extend(result.receiver_index.get(rank, []))

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

    def _count_zero_pre_items(self, items: list[AicbWorkItem]) -> int:
        count = 0
        for item in items:
            if is_zero_pre_item(item.name):
                count += 1
            else:
                break
        return count

    def _count_zero_post_items(self, items: list[AicbWorkItem]) -> int:
        count = 0
        for item in reversed(items):
            if item.name in self.POST_LAYER_NAMES or is_zero_post_item(item.name):
                count += 1
            else:
                break

        return count

    def _split_zero_ga_item_indices(
        self,
        items: list[AicbWorkItem],
        num_pre_items: int,
        num_post_items: int,
        ga: int,
    ) -> list[list[int]]:
        """Split ZeRO layer rows by explicit GA markers or legacy uniform size."""
        body_end = len(items) - num_post_items
        body_indices = list(range(num_pre_items, body_end))
        boundary_indices = [
            item_idx for item_idx in body_indices
            if classify_zero_item(items[item_idx].name) is ZeroItemKind.GA_BOUNDARY
        ]
        if boundary_indices:
            groups: list[list[int]] = [[]]
            for item_idx in body_indices:
                if item_idx in boundary_indices:
                    groups.append([])
                else:
                    groups[-1].append(item_idx)
            if len(groups) != ga or any(not group for group in groups):
                raise ValueError(
                    "Invalid ZeRO GA boundary layout: "
                    f"groups={len(groups)}, ga={ga}, boundaries={boundary_indices}"
                )
            return groups

        if ga < 1 or len(body_indices) % ga != 0:
            raise ValueError(
                "ZeRO workload has non-uniform GA items but no "
                "zero{stage}_ga_boundary metadata; regenerate the AICB workload."
            )
        items_per_ga = len(body_indices) // ga
        return [
            body_indices[index * items_per_ga:(index + 1) * items_per_ga]
            for index in range(ga)
        ]

    # ------------------------------------------------------------------
    # PP flow generation and wiring
    # ------------------------------------------------------------------

    def _generate_pp_flows(
        self,
        grouper: RankGrouper,
        aicb_header: AicbHeader,
        ga_groups: list[list[ItemTasks]],
        items_per_ga: int,
        job_id: int,
        task_id_counter: int,
    ) -> tuple["PPFlowResult", int]:
        """Generate PP flow tasks for all GA steps and PP stage boundaries.

        For each (ga_step, pp_boundary, dp, ep, tp):
          - Forward PP flow: stage k → stage k+1 (activation)
          - Backward PP flow: stage k+1 → stage k (gradient)
        """
        from ..workload_format.schema import CommType
        forward_pp: dict[tuple[int, int], dict[int, FlowTask]] = {}
        backward_pp: dict[tuple[int, int], dict[int, FlowTask]] = {}
        all_flows: list[FlowTask] = []

        for ga_idx in range(len(ga_groups)):
            for pp_boundary in range(grouper.pp - 1):
                fwd_flows: dict[int, FlowTask] = {}
                bwd_flows: dict[int, FlowTask] = {}

                for dp_idx in range(grouper.dp):
                    for ep_idx in range(grouper.ep):
                        for tp_idx in range(grouper.tp):
                            src_rank = grouper.get_pp_rank(
                                pp_boundary, dp_idx, ep_idx, tp_idx)
                            dst_rank = grouper.get_pp_rank(
                                pp_boundary + 1, dp_idx, ep_idx, tp_idx)

                            fwd_flow = FlowTask(
                                task_id=task_id_counter,
                                job_id=job_id,
                                type=TaskType.FLOW,
                                src=src_rank,
                                dst=dst_rank,
                                size_bytes=aicb_header.pp_comm_size,
                                comm_type=CommType.PP_SEND,
                                phase=Phase.FORWARD,
                                layer_id=items_per_ga - 1,
                                iteration=ga_idx,
                            )
                            fwd_flows[src_rank] = fwd_flow
                            all_flows.append(fwd_flow)
                            task_id_counter += 1

                            bwd_flow = FlowTask(
                                task_id=task_id_counter,
                                job_id=job_id,
                                type=TaskType.FLOW,
                                src=dst_rank,
                                dst=src_rank,
                                size_bytes=aicb_header.pp_comm_size,
                                comm_type=CommType.PP_SEND,
                                phase=Phase.BACKWARD_INPUT,
                                layer_id=0,
                                iteration=ga_idx,
                            )
                            bwd_flows[dst_rank] = bwd_flow
                            all_flows.append(bwd_flow)
                            task_id_counter += 1

                forward_pp[(ga_idx, pp_boundary)] = fwd_flows
                backward_pp[(ga_idx, pp_boundary)] = bwd_flows

        return PPFlowResult(
            forward_flows=forward_pp,
            backward_flows=backward_pp,
            all_flows=all_flows,
        ), task_id_counter

    def _wire_pp_dependencies(
        self,
        pp_result: "PPFlowResult",
        ga_groups: list[list[ItemTasks]],
        grouper: RankGrouper,
    ):
        """Wire PP flow sender/receiver dependencies.

        Forward PP:
          sender dep: last layer's fwd output → PP flow (receiver-based)
          receiver dep: PP flow → first layer's fwd compute

        Backward PP:
          sender dep: first layer's ig output → PP flow (receiver-based)
          receiver dep: PP flow → last layer's ig compute
        """
        for ga_idx, ga_group in enumerate(ga_groups):
            last_item = ga_group[-1]
            first_item = ga_group[0]

            for pp_boundary in range(grouper.pp - 1):
                fwd_flows = pp_result.forward_flows[(ga_idx, pp_boundary)]
                bwd_flows = pp_result.backward_flows[(ga_idx, pp_boundary)]

                for dp_idx in range(grouper.dp):
                    for ep_idx in range(grouper.ep):
                        for tp_idx in range(grouper.tp):
                            src_rank = grouper.get_pp_rank(
                                pp_boundary, dp_idx, ep_idx, tp_idx)
                            dst_rank = grouper.get_pp_rank(
                                pp_boundary + 1, dp_idx, ep_idx, tp_idx)

                            # Forward PP: src_rank sends activation after L{vpp-1}.fwd
                            fwd_pp = fwd_flows[src_rank]
                            self._wire_to_flow_sender(
                                fwd_pp, last_item.fwd_computes,
                                last_item.fwd_result, src_rank)
                            # dst_rank's L0.fwd waits for the PP flow
                            first_item.fwd_computes[dst_rank].deps.append(fwd_pp.task_id)

                            # Backward PP: dst_rank sends gradient after L0.ig
                            bwd_pp = bwd_flows[dst_rank]
                            self._wire_to_flow_sender(
                                bwd_pp, first_item.ig_computes,
                                first_item.ig_result, dst_rank)
                            # src_rank's L{vpp-1}.ig waits for the PP flow
                            last_item.ig_computes[src_rank].deps.append(bwd_pp.task_id)

    def _wire_to_flow_sender(
        self,
        flow: FlowTask,
        src_computes: dict[int, FlowTask],
        src_result: FlowGroupResult,
        sender_rank: int,
    ):
        """Wire source phase output to a PP flow's sender side.

        Receiver-based: if source has comm flows, depend on flows where
        sender_rank is receiver (dst); otherwise depend on compute directly.

        In real training with Megatron-style tensor parallelism 
        (without Sharded Activations), the activation/gradient sent 
        across PP stages is the POST-ALLREDUCE result — each TP rank 
        produces a partial output, and only after ALLREDUCE does every 
        rank hold the complete tensor. Therefore PP_SEND/PP_BACKWARD 
        must wait for TP ALLREDUCE to finish.
        """
        if src_result.flows:
            completion_ids = src_result.completion_index.get(sender_rank, [])
            flow.deps.extend(completion_ids)
        else:
            if sender_rank in src_computes:
                flow.deps.append(src_computes[sender_rank].task_id)

    def _wire_zero_pp_dependencies(
        self,
        pp_result: "PPFlowResult",
        ga_groups: list[list[ItemTasks]],
        grouper: RankGrouper,
    ):
        """Wire PP dependencies for ZeRO groups with explicit comm-only rows."""
        for ga_idx, ga_group in enumerate(ga_groups):
            fwd_items = [item for item in ga_group if item.fwd_computes]
            ig_items = [item for item in ga_group if item.ig_computes]
            if not fwd_items or not ig_items:
                continue

            first_fwd_item = fwd_items[0]
            last_fwd_item = fwd_items[-1]
            last_model_ig_item = ig_items[0]
            first_model_ig_item = ig_items[-1]

            for pp_boundary in range(grouper.pp - 1):
                fwd_flows = pp_result.forward_flows[(ga_idx, pp_boundary)]
                bwd_flows = pp_result.backward_flows[(ga_idx, pp_boundary)]

                for dp_idx in range(grouper.dp):
                    for ep_idx in range(grouper.ep):
                        for tp_idx in range(grouper.tp):
                            src_rank = grouper.get_pp_rank(
                                pp_boundary, dp_idx, ep_idx, tp_idx)
                            dst_rank = grouper.get_pp_rank(
                                pp_boundary + 1, dp_idx, ep_idx, tp_idx)

                            fwd_pp = fwd_flows[src_rank]
                            self._wire_to_flow_sender(
                                fwd_pp, last_fwd_item.fwd_computes,
                                last_fwd_item.fwd_result, src_rank)
                            if dst_rank in first_fwd_item.fwd_computes:
                                first_fwd_item.fwd_computes[dst_rank].deps.append(
                                    fwd_pp.task_id)

                            bwd_pp = bwd_flows[dst_rank]
                            self._wire_to_flow_sender(
                                bwd_pp, first_model_ig_item.ig_computes,
                                first_model_ig_item.ig_result, dst_rank)
                            if src_rank in last_model_ig_item.ig_computes:
                                last_model_ig_item.ig_computes[src_rank].deps.append(
                                    bwd_pp.task_id)

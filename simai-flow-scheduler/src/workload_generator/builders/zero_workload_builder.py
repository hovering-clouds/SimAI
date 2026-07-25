"""ZeRO/FSDP-specific workload expansion from AICB rows.

This strategy-specific builder extends :class:`WorkloadBuilder` with DeepSpeed
ZeRO stage 1-3 and FSDP DAG semantics.  The common workload schema and the
generic :class:`WorkloadBuilder` remain unchanged.

Three groupings of AICB rows are handled:

* Comm-only rows (no compute) — parameter all-gather, gradient
  reduce-scatter, step synchronisation collectives.
* Compute rows — forward, backward-input, and backward-weight-gradient
  computations (the latter two are split by the ZeRO generator).
* Init / step / post rows — broadcast-model, overflow check, gradient norm.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ...workload_format.schema import Job, Meta, P2PWorkload, Phase
from ..aicb_parser import AicbHeader, AicbWorkItem
from ..collective_expander import FlowTask
from ..rank_grouper import RankGrouper
from ..workload_builder import FlowGroupResult, ItemTasks, WorkloadBuilder
from .zero_semantics import (
    ZeroItemKind,
    classify_zero_item,
    is_zero_post_item,
    is_zero_pre_item,
    is_zero_workload,
)


@dataclass
class ZeroBuildState:
    """Per-GA dependency state for ZeRO/FSDP-style synchronisation."""

    bucket_wg_computes: dict[int, list[FlowTask]] = field(default_factory=dict)


class ZeroWorkloadBuilder(WorkloadBuilder):
    """Build a P2P workload for DeepSpeed ZeRO/FSDP-formatted AICB rows.

    The public IR is unchanged; only dependency direction for ZeRO-parameter
    all-gather and gradient reduce-scatter differs from the generic path.
    """

    def build_from_aicb(
        self,
        aicb_header: AicbHeader,
        aicb_items: list[AicbWorkItem],
        job: Job,
        comm_algo: str = "ring",
    ) -> P2PWorkload:
        """Convert a ZeRO/FSDP AICB workload into a P2PWorkload.

        Raises :class:`ValueError` if *aicb_items* are not ZeRO rows.
        """
        if not is_zero_workload(aicb_items):
            raise ValueError(
                "ZeroWorkloadBuilder requires a DeepSpeed ZeRO/FSDP "
                "AICB row format; use WorkloadBuilder for generic workloads."
            )

        grouper = RankGrouper(job.assigned_nodes, job.parallelism)
        num_pre_items = self._count_zero_pre_items(aicb_items)
        num_post_items = self._count_zero_post_items(aicb_items)
        ga_item_indices = self._split_zero_ga_item_indices(
            aicb_items, num_pre_items, num_post_items, aicb_header.ga,
        )
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
                job.job_id, task_id_counter,
            )
            all_flow_tasks.extend(pp_result.all_flows)
            self._wire_zero_pp_dependencies(pp_result, ga_groups, grouper)

        return P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(job.assigned_nodes)),
            jobs=[job],
            tasks=[t.to_task() for t in all_flow_tasks],
        )

    # ------------------------------------------------------------------
    # Item-level task creation
    # ------------------------------------------------------------------

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
        """Create tasks for one ZeRO row without generic *dp_comm* ordering.

        Each ZeRO row kind (all-gather, reduce-scatter, compute, …) expands
        into a different subset of forward / input-gradient / weight-gradient
        compute and flow tasks.
        """
        kind = classify_zero_item(item.name)
        item_tasks = ItemTasks()

        # --- Comm-only rows (no compute) ---------------------------------
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

        # --- Rows with compute (forward / backward) ----------------------

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
    # Dependency wiring
    # ------------------------------------------------------------------

    def _wire_zero_dependencies(
        self,
        pre_items: list[ItemTasks],
        ga_groups: list[list[ItemTasks]],
        post_items: list[ItemTasks],
    ):
        """Wire ZeRO/FSDP all-gather-before-compute dependencies.

        Pre- and post-item sequences are wired as ordered events; each GA
        group independently wires its forward-backward chain.
        """
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
        """Wire one GA group's forward chain.

        Returns the completion state (task IDs per rank) at the group's end.
        """
        current_deps = {rank: deps[:] for rank, deps in initial_deps.items()}

        for item_tasks in ga_group:
            # Comm-only row (e.g. fwd param allgather)
            if item_tasks.fwd_result.flows and not item_tasks.fwd_computes:
                self._wire_rank_deps_to_flows(current_deps, item_tasks.fwd_result)
                current_deps = self._result_output_by_rank(item_tasks.fwd_result)
                continue
            if not item_tasks.fwd_computes:
                continue

            # Compute row with optional fwd comm
            self._add_rank_deps(item_tasks.fwd_computes, current_deps)
            current_deps = self._phase_output_by_rank(
                item_tasks.fwd_computes, item_tasks.fwd_result)

        return current_deps

    def _wire_zero_backward_group(
        self,
        ga_group: list[ItemTasks],
        initial_deps: dict[int, list[int]],
    ) -> dict[int, list[int]]:
        """Wire one GA group's backward chain (IG + WG with bucketing).

        ZeRO gradient reduce-scatter rows are comm-only and must wait for
        every weight-gradient compute in the same bucket.
        """
        state = ZeroBuildState()
        current_deps = {rank: deps[:] for rank, deps in initial_deps.items()}
        latest_ig_output: dict[int, list[int]] = {}
        terminal_deps: dict[int, list[int]] = {}

        for item_tasks in ga_group:
            # Comm-only backward row (e.g. bwd param allgather)
            if item_tasks.ig_result.flows and not item_tasks.ig_computes:
                self._wire_rank_deps_to_flows(current_deps, item_tasks.ig_result)
                current_deps = self._result_output_by_rank(item_tasks.ig_result)
                continue

            # Input-gradient compute
            if item_tasks.ig_computes:
                self._add_rank_deps(item_tasks.ig_computes, current_deps)
                latest_ig_output = self._phase_output_by_rank(
                    item_tasks.ig_computes, item_tasks.ig_result)
                current_deps = latest_ig_output

            # Weight-gradient compute — bucket until the reduce-scatter
            if item_tasks.wg_computes:
                wg_deps = latest_ig_output or current_deps
                self._add_rank_deps(item_tasks.wg_computes, wg_deps)
                for rank, compute in item_tasks.wg_computes.items():
                    state.bucket_wg_computes.setdefault(rank, []).append(compute)

            # Comm-only reduce-scatter — the bucket's terminal
            if item_tasks.wg_result.flows and not item_tasks.wg_computes:
                if state.bucket_wg_computes:
                    self._wire_bucket_wg_to_flows(
                        state.bucket_wg_computes, item_tasks.wg_result)
                else:
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
        """Wire ordered init / step / post events.

        Returns the completion state at the end of the sequence.
        """
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

    # ------------------------------------------------------------------
    # PP flow wiring for ZeRO groups
    # ------------------------------------------------------------------

    def _wire_zero_pp_dependencies(
        self,
        pp_result,  # PPFlowResult
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

    # ------------------------------------------------------------------
    # Per-rank dependency helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_rank_deps(
        outputs: list[dict[int, list[int]]],
    ) -> dict[int, list[int]]:
        """Merge dependency dicts: each rank collects all unique task IDs."""
        merged: dict[int, list[int]] = {}
        for output in outputs:
            for rank, deps in output.items():
                existing = merged.setdefault(rank, [])
                for dep in deps:
                    if dep not in existing:
                        existing.append(dep)
        return merged

    @staticmethod
    def _wire_rank_deps_to_flows(
        deps_by_rank: dict[int, list[int]],
        result: FlowGroupResult,
    ):
        """Make every flow in *result* depend on the sender's current deps."""
        for flow in result.flows:
            for dep in deps_by_rank.get(flow.src, []):
                if dep not in flow.deps:
                    flow.deps.append(dep)

    @staticmethod
    def _wire_bucket_wg_to_flows(
        wg_by_rank: dict[int, list[FlowTask]],
        result: FlowGroupResult,
    ):
        """Make every flow depend on all bucketed weight-gradient computes."""
        for flow in result.flows:
            for compute in wg_by_rank.get(flow.src, []):
                if compute.task_id not in flow.deps:
                    flow.deps.append(compute.task_id)

    @staticmethod
    def _result_output_by_rank(
        result: FlowGroupResult,
    ) -> dict[int, list[int]]:
        """Return the completion task IDs per rank as a dep dict."""
        return {rank: deps[:] for rank, deps in result.completion_index.items()}

    def _phase_output_by_rank(
        self,
        computes: dict[int, FlowTask],
        result: FlowGroupResult,
    ) -> dict[int, list[int]]:
        """Return terminal task IDs for a phase (flows > compute when both exist)."""
        if result.flows:
            return {rank: deps[:] for rank, deps in result.completion_index.items()}
        return {rank: [task.task_id] for rank, task in computes.items()}

    @staticmethod
    def _add_rank_deps(
        computes: dict[int, FlowTask],
        deps_by_rank: dict[int, list[int]],
    ):
        """Add per-rank dependencies to each rank's compute task."""
        for rank, compute in computes.items():
            compute.deps.extend(deps_by_rank.get(rank, []))

    @staticmethod
    def _add_receiver_deps(
        computes: dict[int, FlowTask],
        result: FlowGroupResult,
    ):
        """Add receiver-flow dependencies to each rank's compute task."""
        for rank, compute in computes.items():
            compute.deps.extend(result.receiver_index.get(rank, []))

    @staticmethod
    def _zero_comm_only_result(
        item_tasks: ItemTasks,
    ) -> FlowGroupResult | None:
        """If *item_tasks* has only communication (no compute), return it."""
        if item_tasks.fwd_computes or item_tasks.ig_computes or item_tasks.wg_computes:
            return None
        for result in (
            item_tasks.fwd_result, item_tasks.ig_result, item_tasks.wg_result,
        ):
            if result.flows:
                return result
        return None

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _count_zero_pre_items(self, items: list[AicbWorkItem]) -> int:
        """Count consecutive init rows at the start."""
        count = 0
        for item in items:
            if is_zero_pre_item(item.name):
                count += 1
            else:
                break
        return count

    def _count_zero_post_items(self, items: list[AicbWorkItem]) -> int:
        """Count consecutive post rows (step, optimizer, …) at the end."""
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
        """Split ZeRO layer rows by explicit GA markers or legacy uniform size.

        Returns a list of *ga* groups, each a list of item indices.
        """
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
                    f"groups={len(groups)}, ga={ga}, "
                    f"boundaries={boundary_indices}"
                )
            return groups

        if ga < 1 or len(body_indices) % ga != 0:
            raise ValueError(
                "ZeRO workload has non-uniform GA items but no "
                "zero{stage}_ga_boundary metadata; "
                "regenerate the AICB workload."
            )
        items_per_ga = len(body_indices) // ga
        return [
            body_indices[index * items_per_ga:(index + 1) * items_per_ga]
            for index in range(ga)
        ]

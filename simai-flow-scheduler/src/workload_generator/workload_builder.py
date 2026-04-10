"""
Workload Builder - Converts AICB workloads to P2P Workload format.

This module orchestrates the conversion from AICB training workload format
to P2P Workload IR by:
1. Using RankGrouper to derive rank groups for communication operations
2. Iterating through AICB work items and generating compute + flow tasks
3. Building task dependency DAG (forward → backward → DP phases)
4. Managing global task_id incrementing

Key insight: AICB files already contain GA-expanded items.
If ga=24 and vpp=80, there are 1920 layer items in the file.
The builder does NOT need to loop over GA — it just assigns iteration IDs.
"""

from typing import Optional
from ..workload_format.schema import (
    P2PWorkload, Task, Job, Meta, Phase, TaskType, CommType
)
from .aicb_parser import AicbHeader, AicbWorkItem, AicbParser
from .rank_grouper import RankGrouper
from .collective_expander import (
    CollectiveExpander, FlowTask,
    AllReduceExpander, AllGatherExpander,
    ReduceScatterExpander, AlltoAllExpander,
)


class WorkloadBuilder:
    """
    Convert AICB workload to P2P Workload.

    Key insight: AICB file already contains GA-expanded items.
    If ga=24 and vpp=80, there are 1920 layer items in the file.

    Usage:
        parser = AicbParser()
        header, items = parser.parse("workload.txt")
        job = Job(job_id=0, assigned_nodes=list(range(8)),
                  parallelism=ParallelismConfig(tp=2, dp=2, pp=1, ep=1))
        builder = WorkloadBuilder()
        workload = builder.build_from_aicb(header, items, job)
    """

    # Pre-layer item names (appear before GA loop)
    PRE_LAYER_NAMES = {
        "grad_gather", "grad_param_comm", "grad_param_compute",
        "embedding_grads", "moe_grad_norm1", "moe_grad_norm2",
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
        """
        Convert AICB workload to P2P Workload.

        Algorithm:
        1. Create RankGrouper from job.assigned_nodes + parallelism
        2. Validate structure (pre/layer/post items, GA count)
        3. Iterate through all aicb_items, generating tasks
        4. Build dependency chains within and across items
        5. Return complete P2PWorkload
        """
        # Step 1: Create RankGrouper
        grouper = RankGrouper(job.assigned_nodes, job.parallelism)

        # Step 2: Validate structure
        num_pre_items = self._count_pre_items(aicb_items)
        num_post_items = self._count_post_items(aicb_items)
        num_layer_items = len(aicb_items) - num_pre_items - num_post_items

        assert num_layer_items >= 0, \
            f"Invalid item count: {len(aicb_items)} total, " \
            f"{num_pre_items} pre, {num_post_items} post"
        assert num_layer_items % aicb_header.ga == 0, \
            f"Layer items ({num_layer_items}) not divisible by GA ({aicb_header.ga})"
        assert num_layer_items // aicb_header.ga == aicb_header.vpp, \
            f"Expected vpp={aicb_header.vpp}, got {num_layer_items // aicb_header.ga}"

        # Step 3: Initialize task collection
        all_tasks: list[FlowTask] = []
        task_id_counter = 0
        prev_last_task_id: Optional[int] = None

        # Step 4: Iterate through all items
        for item_idx, item in enumerate(aicb_items):
            # Calculate iteration and phase context
            if item_idx < num_pre_items:
                iteration = 0
                is_pre_layer = True
                is_post_layer = False
            elif item_idx >= num_pre_items + num_layer_items:
                iteration = aicb_header.ga
                is_pre_layer = False
                is_post_layer = True
            else:
                iteration = (item_idx - num_pre_items) // aicb_header.vpp
                is_pre_layer = False
                is_post_layer = False

            layer_id = item_idx

            # Track first task ID of this item for cross-item dependency
            first_task_of_item: Optional[int] = None

            # --- Forward phase ---
            if item.forward_compute_time > 0:
                compute_task = self._create_compute_task(
                    task_id=task_id_counter,
                    job_id=job.job_id,
                    duration_us=item.forward_compute_time,
                    node=job.assigned_nodes[0],
                    phase=Phase.FORWARD,
                    layer_id=layer_id,
                    iteration=iteration,
                )
                if first_task_of_item is None:
                    first_task_of_item = task_id_counter
                all_tasks.append(compute_task)
                task_id_counter += 1
                last_compute_id = compute_task.task_id

                if item.forward_comm != "NONE":
                    base_type, context = AicbParser.parse_comm_type(item.forward_comm)
                    flows = self._expand_comm(
                        base_type, context, item.forward_comm_size,
                        grouper, pp_idx=0, dp_idx=0, ep_idx=0, tp_idx=0,
                        phase=Phase.FORWARD, job_id=job.job_id,
                        task_id_start=task_id_counter, algo=comm_algo,
                    )
                    # First flow depends on compute task
                    if flows:
                        flows[0].deps.append(last_compute_id)
                    all_tasks.extend(flows)
                    task_id_counter += len(flows)

            # --- Backward phase ---
            if item.backward_compute_time > 0:
                compute_task = self._create_compute_task(
                    task_id=task_id_counter,
                    job_id=job.job_id,
                    duration_us=item.backward_compute_time,
                    node=job.assigned_nodes[0],
                    phase=Phase.BACKWARD_INPUT,
                    layer_id=layer_id,
                    iteration=iteration,
                )
                if first_task_of_item is None:
                    first_task_of_item = task_id_counter
                all_tasks.append(compute_task)
                task_id_counter += 1

                if item.backward_comm != "NONE":
                    base_type, context = AicbParser.parse_comm_type(item.backward_comm)
                    flows = self._expand_comm(
                        base_type, context, item.backward_comm_size,
                        grouper, pp_idx=0, dp_idx=0, ep_idx=0, tp_idx=0,
                        phase=Phase.BACKWARD_INPUT, job_id=job.job_id,
                        task_id_start=task_id_counter, algo=comm_algo,
                    )
                    if flows:
                        flows[0].deps.append(compute_task.task_id)
                    all_tasks.extend(flows)
                    task_id_counter += len(flows)

            # --- DP phase ---
            if item.dp_compute_time > 0:
                compute_task = self._create_compute_task(
                    task_id=task_id_counter,
                    job_id=job.job_id,
                    duration_us=item.dp_compute_time,
                    node=job.assigned_nodes[0],
                    phase=Phase.BACKWARD_WEIGHT,
                    layer_id=layer_id,
                    iteration=iteration,
                )
                if first_task_of_item is None:
                    first_task_of_item = task_id_counter
                all_tasks.append(compute_task)
                task_id_counter += 1

                if item.dp_comm != "NONE":
                    base_type, context = AicbParser.parse_comm_type(item.dp_comm)
                    flows = self._expand_comm(
                        base_type, context, item.dp_comm_size,
                        grouper, pp_idx=0, dp_idx=0, ep_idx=0, tp_idx=0,
                        phase=Phase.BACKWARD_WEIGHT, job_id=job.job_id,
                        task_id_start=task_id_counter, algo=comm_algo,
                    )
                    if flows:
                        flows[0].deps.append(compute_task.task_id)
                    all_tasks.extend(flows)
                    task_id_counter += len(flows)

            # Cross-item dependency: previous item's last task → current item's first task
            if prev_last_task_id is not None and first_task_of_item is not None:
                # Find the first task of current item and add dependency
                for task in all_tasks:
                    if task.task_id == first_task_of_item:
                        task.deps.append(prev_last_task_id)
                        break

            # Update prev_last_task_id
            if all_tasks:
                prev_last_task_id = all_tasks[-1].task_id

        # Step 5: Build P2PWorkload
        workload = P2PWorkload(
            version="1.0",
            meta=Meta(num_jobs=1, num_nodes=len(job.assigned_nodes)),
            jobs=[job],
            tasks=[t.to_task() for t in all_tasks],
        )
        return workload

    def _expand_comm(
        self,
        base_type: str,
        context: str,
        comm_size: int,
        grouper: RankGrouper,
        pp_idx: int, dp_idx: int, ep_idx: int, tp_idx: int,
        phase: Phase,
        job_id: int,
        task_id_start: int,
        algo: str = "ring",
    ) -> list[FlowTask]:
        """
        Expand a communication operation into P2P flows.

        Context mapping to RankGrouper methods:
        - "tp" → get_tp_group(pp_idx, dp_idx, ep_idx)
        - "dp" → get_dp_group(pp_idx, ep_idx, tp_idx)
        - "ep" → get_ep_group(pp_idx, dp_idx, tp_idx)
        - "dp_ep" → get_dp_ep_group(pp_idx, tp_idx)
        """
        # Select rank group based on context
        if context == "tp":
            ranks = grouper.get_tp_group(pp_idx, dp_idx, ep_idx)
        elif context == "dp":
            ranks = grouper.get_dp_group(pp_idx, ep_idx, tp_idx)
        elif context == "ep":
            ranks = grouper.get_ep_group(pp_idx, dp_idx, tp_idx)
        elif context == "dp_ep":
            ranks = grouper.get_dp_ep_group(pp_idx, tp_idx)
        else:
            raise ValueError(f"Unknown context: {context}")

        # Call appropriate expander
        expander = self.expanders.get(base_type)
        if expander is None:
            raise ValueError(f"Unsupported comm type: {base_type}")

        if base_type == "ALLREDUCE":
            flows = expander.expand_allreduce(
                ranks, comm_size, algo, job_id, task_id_start
            )
        elif base_type == "ALLGATHER":
            flows = expander.expand_allgather(
                ranks, comm_size, algo, job_id, task_id_start
            )
        elif base_type == "REDUCESCATTER":
            flows = expander.expand_reducescatter(
                ranks, comm_size, algo, job_id, task_id_start
            )
        elif base_type == "ALLTOALL":
            flows = expander.expand_alltoall(
                ranks, comm_size, job_id, task_id_start
            )
        else:
            raise ValueError(f"Unsupported base type: {base_type}")

        # Set phase and iteration on all flows
        for flow in flows:
            flow.phase = phase

        return flows

    def _create_compute_task(
        self,
        task_id: int,
        job_id: int,
        duration_us: int,
        node: int,
        phase: Phase,
        layer_id: int,
        iteration: int,
    ) -> FlowTask:
        """Create a compute task."""
        return FlowTask(
            task_id=task_id,
            job_id=job_id,
            type=TaskType.COMPUTE,
            node=node,
            duration_us=duration_us,
            phase=phase,
            layer_id=layer_id,
            iteration=iteration,
        )

    def _count_pre_items(self, items: list[AicbWorkItem]) -> int:
        """Count pre-layer items (grad_gather, etc.) at the start."""
        count = 0
        for item in items:
            if item.name in self.PRE_LAYER_NAMES:
                count += 1
            else:
                break  # Stop at first non-pre-layer item
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

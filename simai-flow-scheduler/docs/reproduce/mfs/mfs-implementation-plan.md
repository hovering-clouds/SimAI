# MFS Inference Replay Implementation Plan

> **For agentic workers:** This plan describes how to reproduce the main ideas
> of `39-MFS.pdf` in `simai-flow-scheduler` under the current project split:
> Vidur performs request/batch scheduling and emits a batch-level trace;
> `simai-flow-scheduler` replays that trace with per-flow and per-compute-task
> network simulation. Implement this plan phase by phase. Keep each phase
> independently testable.

**Goal:** Add an inference replay policy that approximates MFS's multi-stage
flow scheduling for TTFT-sensitive LLM serving.

**Architecture:** Keep request and batch scheduling in Vidur. In
`simai-flow-scheduler`, consume the fixed Vidur trace, expand it to a P2P task
DAG, build MFS-specific sidecar metadata, and run a new policy/allocator that
prioritizes early-stage blocking communication while deferring P2D traffic until
it becomes urgent.

**Tech Stack:** Python 3.10+, dataclasses, pytest, existing
`P2PWorkload`, `AnalyticalExecutor`, `SchedulingPolicy`, `BandwidthAllocator`,
and Vidur trace replay path.

---

## 1. Background

### 1.1 Current Project Split

The current inference simulation stack has two layers:

1. Vidur simulates request-level and batch-level behavior.
   It decides request admission, batch formation, replica placement, prefill and
   decode scheduling, and coarse execution timing.

2. `simai-flow-scheduler` replays Vidur's batch trace.
   It expands each batch into compute tasks and communication flows, then uses
   a fine-grained analytical executor to simulate per-flow contention.

This split is important. MFS contains both request-level control-plane logic and
flow-level network scheduling logic. The replay layer should not take over
batch formation or request admission. Instead, it should reproduce the network
side of MFS as faithfully as possible on top of the trace that Vidur already
produced.

### 1.2 Relevant Existing Modules

Use these existing modules as the foundation:

- `src/workload_generator/inference_trace_expander.py`
  Expands Vidur inference traces into `P2PWorkload`. It already handles
  prefill/decode batches, PP stages, TP/EP collectives, PP sends, and
  prefill-to-decode KV transfer flows.

- `src/workload_format/schema.py`
  Defines `Task`, `P2PWorkload`, `Phase`, and `CommType`. Existing fields
  `phase`, `layer_id`, `comm_type`, and `deps` are enough for Phase 1 MFS
  metadata. Avoid schema changes in Phase 1.

- `src/static_analysis/strategies/default_strategy.py`
  Builds the default route table and compute execution order.

- `src/static_analysis/passes/puppeteer_tte.py`
  Existing example of a sidecar static-analysis pass that computes flow
  priority metadata without changing the workload schema.

- `src/executor/analytical.py`
  Event-driven executor. It delegates task admission, path lookup, and bandwidth
  allocation to a `SchedulingPolicy`.

- `src/executor/policies/base_policy.py`
  Policy interface. New MFS policy should implement this interface instead of
  modifying the executor.

- `src/executor/bandwidth_allocators/tte_aware_allocator.py`
  Existing strict-priority and weighted allocation pattern. Use it as a style
  reference, but implement MFS-specific queue logic in a new allocator.

- `vidur-alibabacloud/vidur/trace_recorder.py`
  Vidur trace recorder. It already contains `record_batch_stage()`, which is
  useful for PP/per-stage trace capture. Later phases can extend request-level
  trace fields here.

### 1.3 Important Current Limitation

`simai-flow-scheduler` currently models `KV_CACHE_TRANSFER` as the transfer from
prefill replica/stage to decode replica/stage. In the MFS paper, this maps most
closely to Stage 3, "Prefill-to-Decode (P2D) transfer".

The paper's Stage 1, "KV-cache reuse", means fetching reusable KV blocks before
or during prefill. That is not currently represented as a separate flow type in
the replay trace. Do not pretend Phase 1 covers true Stage 1 reuse traffic.
Phase 1 covers:

- Stage 2: collective communication during prefill/decode compute
- Stage 3: P2D KV transfer from prefill to decode
- PP send flows, treated as early-stage blocking communication

Future work can add true remote KV reuse flows if Vidur emits them.

---

## 2. MFS Strategy Reference

This section summarizes the MFS mechanisms that matter for implementation.

### 2.1 Multi-Stage Flow Abstraction

MFS views one TTFT-sensitive prefill request as a sequence of layers. Each layer
can involve multiple communication stages:

- Stage 1: remote KV-cache reuse
- Stage 2: collective communication, such as EP all-to-all or TP collective
- Stage 3: P2D transfer to decode workers

Flows from different stages overlap and contend for shared network links. MFS
does not optimize individual flow completion time. It tries to maximize TTFT SLO
attainment by coordinating all stages of a request.

### 2.2 Defer-and-Promote

The core MFS principle is "Defer-and-Promote".

Flows that do not need to finish immediately start in lower priority queues.
They are promoted only when they become urgent. This prevents loose-deadline
traffic, especially P2D traffic, from consuming bandwidth too early.

The key tradeoff:

- Early-stage communication that blocks the next layer should run promptly.
- P2D traffic should not finish unnecessarily early because early completion
  does not improve TTFT if decode cannot use the data yet or if other stages are
  more critical.

### 2.3 RMLQ

MFS implements Defer-and-Promote using a Reverse Multi-Level Queue (RMLQ).
Traditional MLFQ demotes long-running tasks. RMLQ starts flows low and promotes
them as urgency increases.

In this replay implementation, RMLQ can be approximated by assigning each active
flow to a priority queue and allocating bandwidth from high to low priority.
Within the same queue, use fair sharing.

### 2.4 RLI for Implicit-Deadline Early Flows

Early-stage flows do not have exact flow-level deadlines. MFS approximates their
urgency with Relative Layer Index:

```text
RLI(flow, t) = L_target(flow) - L_current(t)
```

Lower RLI means the flow is needed sooner. A flow with `RLI = 0` is blocking the
current layer and should have high priority.

In `simai-flow-scheduler`, `L_target` can be approximated from the flow's
`layer_id` or the earliest downstream compute layer. `L_current` can be tracked
per replica/stage/node group as compute tasks complete.

### 2.5 MLU for Explicit-Deadline P2D Flows

P2D flows have explicit request-level TTFT deadlines. MFS uses Minimal Link
Utilization (MLU) to decide when to promote them:

```text
required_bw = remaining_bits / remaining_time
MLU = required_bw / bottleneck_link_bw
```

If MLU is low, defer the P2D flow. If MLU crosses configured thresholds, promote
it so it can still finish just in time.

Phase 1 should use a heuristic because current traces do not include request
deadlines. Phase 2 should implement MLU once Vidur emits deadline fields.

### 2.6 RED, Feasibility, and Pruning

The paper also includes robust inter-request scheduling:

- RED (Robust Effective Deadline) ranks batches while avoiding piggyback effects
  from a few tight-deadline requests.
- Feasibility checks estimate whether a batch can still meet its deadline.
- Selective pruning demotes infeasible requests into a scavenger queue.

These are request/batch-level controls. In the current architecture, the true
versions belong in Vidur. `simai-flow-scheduler` may consume RED ranks or
scavenger labels from trace metadata, but should not reshape batches in replay.

---

## 3. Implementation Scope

### 3.1 Phase 1: MFS-Lite in Replay Layer

Implement without Vidur trace changes.

Covered mechanisms:

- MFS sidecar metadata
- RLI-style priority for early-stage blocking flows
- RMLQ-style strict priority queues
- P2D defer-and-promote heuristic
- Inference E2E script comparing default policy and MFS policy

Not covered:

- true request deadlines
- real MLU
- RED
- request pruning
- true Stage 1 KV-cache reuse

### 3.2 Phase 2: Deadline-Aware MFS

Add small Vidur trace extensions and implement deadline-aware P2D promotion.

Covered mechanisms:

- request arrival time
- request TTFT budget/deadline
- P2D-to-request deadline mapping
- MLU threshold based promotion
- P2D earliness and deadline miss reporting

### 3.3 Phase 3: RED and Soft Scavenger Approximation

Keep true request scheduling in Vidur. Add replay-side support for metadata
that Vidur can emit.

Covered mechanisms:

- consume batch/request priority rank from trace
- same-RLI tie-breaking by Vidur-provided rank
- consume scavenger labels
- demote scavenger flows in RMLQ
- optional feasibility analysis report

---

## 4. File Plan

### 4.1 New Files

- `src/static_analysis/passes/mfs_context.py`
  Build sidecar metadata that maps task IDs to MFS concepts such as batch ID,
  request IDs, stage ID, MFS stage, target layer, and whether a flow is P2D.

- `src/static_analysis/passes/mfs_rli.py`
  Compute static and runtime-friendly RLI metadata for early-stage flows.

- `src/static_analysis/strategies/mfs_strategy.py`
  Analyzer that wraps default route and compute-order analysis and adds MFS
  context/RLI metadata.

- `src/executor/bandwidth_allocators/mfs_allocator.py`
  RMLQ-style allocator. Allocate capacity by queue priority, fair sharing within
  each queue.

- `src/executor/policies/mfs_policy.py`
  Scheduling policy that preserves compute ordering, uses default routing, and
  delegates MFS bandwidth allocation to `MfsAllocator`.

- `scripts/run_mfs_inference_e2e.py`
  Reproduce inference E2E flow with both default and MFS policies and report
  TTFT, P2D earliness, collective completion, and makespan.

- `tests/test_mfs_context.py`
  Unit tests for trace/batch/task metadata extraction.

- `tests/test_mfs_allocator.py`
  Unit tests for priority queue allocation and P2D promotion.

- `tests/test_mfs_policy.py`
  Executor-level tests for contention behavior.

### 4.2 Existing Files to Modify

- `src/static_analysis/passes/__init__.py`
  Export MFS analysis helpers if the package uses explicit exports.

- `src/static_analysis/strategies/__init__.py`
  Export `MfsAnalyzer` if needed by scripts/tests.

- `src/executor/bandwidth_allocators/__init__.py`
  Export `MfsAllocator` if current package style requires it.

- `src/executor/policies/__init__.py`
  Export `MfsSchedulingPolicy` if current package style requires it.

- `src/workload_generator/inference_trace_expander.py`
  Phase 1 should avoid behavioral changes. If needed, only improve
  `batch_task_map` metadata so MFS context can recover `stage_id` and
  `request_ids` reliably.

- `vidur-alibabacloud/vidur/trace_recorder.py`
  Phase 2 only. Add request timing/deadline fields.

---

## 5. Phase 1 Detailed Tasks: MFS-Lite

### Task 1: Build MFS Context Metadata

**Files:**

- Create: `simai-flow-scheduler/src/static_analysis/passes/mfs_context.py`
- Test: `simai-flow-scheduler/tests/test_mfs_context.py`

**Design:**

Create dataclasses:

```python
from dataclasses import dataclass, field
from enum import Enum


class MfsStage(str, Enum):
    EARLY = "early"
    P2D = "p2d"
    BACKGROUND = "background"


@dataclass
class MfsTaskInfo:
    task_id: int
    batch_id: str | None
    request_ids: tuple[int, ...]
    stage_id: int
    mfs_stage: MfsStage
    target_layer: int
    comm_role: str


@dataclass
class MfsContext:
    task_info: dict[int, MfsTaskInfo] = field(default_factory=dict)
    batch_to_tasks: dict[str, tuple[int, ...]] = field(default_factory=dict)
    request_to_tasks: dict[int, tuple[int, ...]] = field(default_factory=dict)
```

Build function:

```python
def build_mfs_context(workload: P2PWorkload, batch_task_map: dict) -> MfsContext:
    ...
```

Classification rules:

- `CommType.KV_CACHE_TRANSFER` -> `MfsStage.P2D`, `comm_role="p2d_transfer"`
- `CommType.PP_SEND` -> `MfsStage.EARLY`, `comm_role="pp_send"`
- `TP_*`, `EP_ALLTOALL`, `DP_*` collective flows -> `MfsStage.EARLY`,
  `comm_role="collective"`
- compute tasks -> `MfsStage.BACKGROUND`, `comm_role="compute"`
- unknown flow tasks -> `MfsStage.BACKGROUND`, `comm_role="unknown"`

Use `batch_task_map` to recover `batch_id`, `request_ids`, and `stage_id`.
If `batch_task_map` does not include `stage_id`, default to `0`.

**Tests:**

- A synthetic trace from `tests/test_inference_trace_expander.py` should produce
  P2D metadata for `kv_*` tasks.
- Collective flows inside a prefill batch should be classified as early flows.
- `request_to_tasks` should include all tasks belonging to that request's
  batch and P2D transfer.

**Acceptance:**

- `pytest tests/test_mfs_context.py -v` passes.
- No existing tests fail.

### Task 2: Compute RLI Metadata

**Files:**

- Create: `simai-flow-scheduler/src/static_analysis/passes/mfs_rli.py`
- Test: `simai-flow-scheduler/tests/test_mfs_context.py` or
  `simai-flow-scheduler/tests/test_mfs_rli.py`

**Design:**

Create dataclass:

```python
@dataclass
class RliInfo:
    task_id: int
    target_layer: int
    base_rli: int
```

Provide:

```python
def compute_static_rli(
    workload: P2PWorkload,
    context: MfsContext,
    current_layer_by_stage: dict[tuple[int, int], int] | None = None,
) -> dict[int, RliInfo]:
    ...
```

For Phase 1, compute `base_rli` from static layer information:

- early-stage flow: `max(task.layer_id - current_layer, 0)`
- P2D flow: do not use RLI for priority; use a sentinel such as large integer
- background/unknown: large integer

`current_layer_by_stage` key should be `(job_id, stage_id)`.
If missing, current layer defaults to `0`.

**Acceptance:**

- Flow at layer 0 gets RLI 0 when current layer is 0.
- Flow at layer 2 gets RLI 2 when current layer is 0.
- P2D flow does not outrank RLI 0 collective in Phase 1.

### Task 3: Add MFS Analyzer

**Files:**

- Create: `simai-flow-scheduler/src/static_analysis/strategies/mfs_strategy.py`
- Modify: `simai-flow-scheduler/src/static_analysis/strategies/__init__.py`
- Test: `simai-flow-scheduler/tests/test_mfs_context.py`

**Design:**

Mirror `DefaultAnalyzer`:

```python
@dataclass
class MfsAnalysisResult:
    route_table: RouteTable
    execution_plan: ExecutionPlan
    mfs_context: MfsContext
    rli_info: dict[int, RliInfo]


class MfsAnalyzer:
    def __init__(self, topology: NetworkTopology):
        self.topology = topology

    def analyze(self, workload: P2PWorkload, batch_task_map: dict) -> MfsAnalysisResult:
        route_table = BfsStrategy().compute_routes(workload, self.topology)
        execution_plan = CppReferenceSerializer().serialize(workload)
        context = build_mfs_context(workload, batch_task_map)
        rli_info = compute_static_rli(workload, context)
        return MfsAnalysisResult(route_table, execution_plan, context, rli_info)
```

**Acceptance:**

- Analyzer returns default route table and execution plan plus MFS metadata.
- Existing default analyzer remains unchanged.

### Task 4: Implement RMLQ Allocator

**Files:**

- Create: `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py`
- Modify: `simai-flow-scheduler/src/executor/bandwidth_allocators/__init__.py`
- Test: `simai-flow-scheduler/tests/test_mfs_allocator.py`

**Design:**

Create config:

```python
@dataclass
class MfsAllocatorConfig:
    num_queues: int = 4
    p2d_initial_queue: int = 0
    early_rli0_queue: int = 2
    early_default_queue: int = 1
    urgent_p2d_queue: int = 3
    p2d_promotion_delay_us: int = 1000
```

Allocator behavior:

1. Classify each active flow into a queue.
2. For each link, allocate capacity from high queue to low queue.
3. Within one queue on one link, fair-share remaining capacity.
4. Flow allocation is the minimum allocation across links in its path.

Phase 1 queue rules:

- P2D starts in `p2d_initial_queue`.
- P2D moves to `urgent_p2d_queue` if
  `current_time - flow.start_time >= p2d_promotion_delay_us`.
- Early flow with RLI 0 uses `early_rli0_queue`.
- Early flow with RLI > 0 uses `early_default_queue`.
- Background uses queue 0.

Keep this allocator independent of `AnalyticalExecutor`.

**Tests:**

- Two flows on same link, queues 2 and 0: queue 2 gets full link, queue 0 gets
  zero while queue 2 has demand.
- Two flows in same queue on same link: each gets half bandwidth.
- P2D starts at low queue and promotes after configured delay.
- Multi-hop flow allocation uses path bottleneck.

**Acceptance:**

- `pytest tests/test_mfs_allocator.py -v` passes.

### Task 5: Implement MFS Scheduling Policy

**Files:**

- Create: `simai-flow-scheduler/src/executor/policies/mfs_policy.py`
- Modify: `simai-flow-scheduler/src/executor/policies/__init__.py`
- Test: `simai-flow-scheduler/tests/test_mfs_policy.py`

**Design:**

The policy should resemble `DefaultSchedulingPolicy`:

- Preserve compute ordering via `ExecutionPlan`.
- Use `route_table.get_path(task)` for routing.
- Use `MfsAllocator` for bandwidth allocation.
- Track completed compute layers per stage if needed for dynamic RLI updates.

Constructor:

```python
class MfsSchedulingPolicy(SchedulingPolicy):
    def __init__(
        self,
        analysis: MfsAnalysisResult,
        allocator_config: MfsAllocatorConfig | None = None,
    ):
        ...
```

Admission:

- Compute tasks: same behavior as default policy; only emit next compute on each
  node according to `execution_plan`.
- Flow tasks: emit when ready. Phase 1 should rely on allocator priority rather
  than holding ready flows indefinitely.

Completion:

- On compute completion, advance compute cursor.
- Optionally update `(job_id, stage_id) -> current_layer` using context metadata.

**Tests:**

- A workload with one collective and one P2D flow sharing a link should complete
  collective first under MFS when P2D is not promoted.
- With a very small `p2d_promotion_delay_us`, P2D should be promoted and receive
  high priority after the delay.
- Compute ordering should match default behavior.

**Acceptance:**

- `pytest tests/test_mfs_policy.py -v` passes.
- `pytest tests/test_executor_policy.py -v` still passes.

### Task 6: Add MFS Inference E2E Script

**Files:**

- Create: `simai-flow-scheduler/scripts/run_mfs_inference_e2e.py`

**Design:**

Start from `scripts/run_inference_e2e.py`.

The new script should:

1. Load the same inference trace.
2. Expand it with `InferenceTraceExpander`.
3. Run default policy.
4. Run MFS policy.
5. Write both execution results.
6. Write a comparison report.

Recommended output directory:

```text
outputs/mfs_inference_e2e/
```

Recommended report fields:

- `policy`
- `makespan_us`
- per-request `ttft_us`
- P2D flow completion times
- collective flow completion times
- optional `p2d_earliness_us` when deadline is unavailable should be omitted or
  marked `null`

**Acceptance:**

- Script runs on the repository's existing sample trace.
- Default and MFS outputs are written separately.
- The report makes it clear which policy produced each metric.

---

## 6. Phase 2 Detailed Tasks: Deadline-Aware P2D Promotion

### Task 7: Extend Vidur Request Trace Metadata

**Files:**

- Modify: `vidur-alibabacloud/vidur/trace_recorder.py`
- Test location depends on existing Vidur test layout. If no direct tests exist,
  add a focused unit test around `TraceRecorder.record_request()`.

**Design:**

Extend request entries from:

```json
{
  "num_prefill_tokens": 1024,
  "num_decode_tokens": 64
}
```

to:

```json
{
  "num_prefill_tokens": 1024,
  "num_decode_tokens": 64,
  "arrival_time_us": 0,
  "ttft_slo_us": 2000000,
  "deadline_us": 2000000
}
```

Implementation notes:

- `arrival_time_us` should come from request arrival metrics if available.
- `ttft_slo_us` can come from config if Vidur has an explicit TTFT SLO. If not,
  add an optional trace-recorder config field with a default.
- `deadline_us = arrival_time_us + ttft_slo_us`.
- Keep backward compatibility: `simai-flow-scheduler` must handle traces without
  these fields.

### Task 8: Parse Deadlines in MFS Context

**Files:**

- Modify: `simai-flow-scheduler/src/static_analysis/passes/mfs_context.py`
- Test: `simai-flow-scheduler/tests/test_mfs_context.py`

**Design:**

Add:

```python
@dataclass
class MfsRequestInfo:
    request_id: int
    arrival_time_us: int | None
    ttft_slo_us: int | None
    deadline_us: int | None
```

Add `request_info: dict[int, MfsRequestInfo]` to `MfsContext`.

`build_mfs_context()` should accept optional raw trace:

```python
def build_mfs_context(
    workload: P2PWorkload,
    batch_task_map: dict,
    trace: dict | None = None,
) -> MfsContext:
    ...
```

If no trace or no deadline fields exist, request info fields should be `None`.

### Task 9: Implement MLU Promotion

**Files:**

- Modify: `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py`
- Test: `simai-flow-scheduler/tests/test_mfs_allocator.py`

**Design:**

Add config:

```python
@dataclass
class MfsAllocatorConfig:
    ...
    p2d_mlu_thresholds: tuple[float, ...] = (0.5, 0.75, 0.9)
    enable_deadline_promotion: bool = False
```

If `enable_deadline_promotion` is true and a P2D flow has request deadline:

1. Compute remaining time: `deadline_us - current_time`.
2. Compute remaining bits from `ActiveFlow.remaining_bytes`.
3. Estimate bottleneck bandwidth on flow path.
4. Compute `mlu = required_bw / bottleneck_bw`.
5. Map MLU to queue:
   - below first threshold: low P2D queue
   - threshold 1: intermediate queue
   - threshold 2 or above: urgent P2D queue
   - `remaining_time <= 0`: urgent P2D queue

If a P2D flow maps to multiple request IDs, use the earliest available deadline.

**Acceptance:**

- P2D with loose deadline remains low priority.
- P2D with tight deadline is promoted.
- Missing deadline falls back to Phase 1 delay-based heuristic.

### Task 10: Add Deadline Metrics

**Files:**

- Modify: `simai-flow-scheduler/scripts/run_mfs_inference_e2e.py`

**Design:**

When deadlines exist, report:

- per-request deadline
- per-request TTFT
- deadline met boolean
- deadline miss amount
- P2D earliness:

```text
earliness_us = deadline_us - p2d_completion_us
```

Positive earliness means P2D completed before deadline. Negative means deadline
miss.

---

## 7. Phase 3 Detailed Tasks: RED and Scavenger Approximation

### Task 11: Consume Vidur-Provided Priority Metadata

**Files:**

- Modify: `simai-flow-scheduler/src/static_analysis/passes/mfs_context.py`
- Modify: `simai-flow-scheduler/src/executor/bandwidth_allocators/mfs_allocator.py`
- Test: `simai-flow-scheduler/tests/test_mfs_context.py`
- Test: `simai-flow-scheduler/tests/test_mfs_allocator.py`

**Trace fields to support:**

At batch level:

```json
{
  "mfs_red_rank": 3,
  "mfs_priority_class": "main"
}
```

At request level:

```json
{
  "mfs_priority_class": "main"
}
```

Allowed priority classes:

- `main`
- `scavenger`

Replay semantics:

- Same queue and same RLI: lower `mfs_red_rank` wins tie-breaking if allocator
  implements deterministic ordering.
- `scavenger` flows are demoted to queue 0 unless they are hard-deadline P2D
  already past deadline.

### Task 12: Optional Feasibility Analysis Report

**Files:**

- Create: `simai-flow-scheduler/src/static_analysis/passes/mfs_feasibility.py`
- Test: `simai-flow-scheduler/tests/test_mfs_feasibility.py`

**Design:**

This is an analysis/reporting pass, not an admission-control mechanism.

Inputs:

- workload
- route table
- topology
- MFS context
- request deadlines when available

Output:

```python
@dataclass
class MfsFeasibilityRecord:
    request_id: int
    bottleneck_link: tuple[int, int] | None
    total_bytes_on_bottleneck: int
    estimated_comm_time_us: int
    deadline_us: int | None
    feasible: bool | None
```

Use this to understand overload, not to delete tasks from replay.

---

## 8. Testing Strategy

### 8.1 Unit Tests

Add small synthetic workloads rather than relying only on large traces.

Required scenarios:

- One collective flow and one P2D flow share a single bottleneck link.
- RLI 0 collective outranks RLI 2 early flow.
- P2D starts low and promotes after delay in Phase 1.
- P2D promotes by MLU in Phase 2.
- Missing deadline fields preserve Phase 1 behavior.
- Scavenger class demotes flows in Phase 3.

### 8.2 Integration Tests

Use existing inference trace tests and scripts:

```bash
cd simai-flow-scheduler
pytest tests/test_inference_trace_expander.py -v
pytest tests/test_executor_policy.py -v
pytest tests/test_mfs_context.py tests/test_mfs_allocator.py tests/test_mfs_policy.py -v
```

Run E2E:

```bash
cd simai-flow-scheduler
uv run python scripts/run_mfs_inference_e2e.py
```

If `uv` is unavailable in the environment, use:

```bash
cd simai-flow-scheduler
python scripts/run_mfs_inference_e2e.py
```

### 8.3 Expected Behavioral Checks

For Phase 1:

- MFS should reduce or preserve non-overlapped collective completion time under
  synthetic contention.
- MFS may increase P2D completion time when P2D is loose. That is acceptable.
- MFS should not deadlock with ready P2D flows.

For Phase 2:

- P2D should not miss deadlines in simple feasible synthetic workloads.
- Loose P2D should complete closer to deadline than under always-high-priority
  scheduling.

For Phase 3:

- Replay should consume priority metadata but should not reshape the trace.
- Scavenger demotion should be visible in flow timing under contention.

---

## 9. Non-Goals and Guardrails

Do not implement these in `simai-flow-scheduler` unless a later design changes
the project boundary:

- request admission control
- batch formation changes
- RED-driven batch construction
- deleting or splitting requests from an already recorded batch
- real switch DSCP/priority queue enforcement
- packet-level scheduling
- real NCCL or Mooncake integration

Do not modify `AnalyticalExecutor` for Phase 1 unless the policy interface is
insufficient. The current executor already delegates the right decisions to
policy and allocator.

Prefer sidecar metadata over workload schema changes in Phase 1. Schema changes
should be reserved for fields that multiple components truly need to persist.

---

## 10. Suggested Development Order

1. Implement `MfsContext`.
2. Implement static RLI metadata.
3. Add `MfsAnalyzer`.
4. Implement `MfsAllocator`.
5. Implement `MfsSchedulingPolicy`.
6. Add synthetic policy/allocator tests.
7. Add `run_mfs_inference_e2e.py`.
8. Extend Vidur trace with deadline fields.
9. Add MLU promotion.
10. Add RED/scavenger metadata consumption.
11. Add feasibility analysis report.

Commit after each independently passing phase. Suggested commit boundaries:

- `feat(mfs): add replay metadata analysis`
- `feat(mfs): add rmlq allocator`
- `feat(mfs): add inference scheduling policy`
- `feat(mfs): add inference e2e comparison script`
- `feat(mfs): support deadline-aware p2d promotion`
- `feat(mfs): consume red and scavenger metadata`

---

## 11. Completion Criteria

Phase 1 is complete when:

- MFS policy can run on existing inference traces without Vidur changes.
- Synthetic tests show collective/PP early-stage flows outrank deferred P2D.
- Default inference E2E still runs.
- MFS inference E2E produces a comparison report.

Phase 2 is complete when:

- Vidur trace includes backward-compatible request deadline metadata.
- MFS allocator uses MLU for P2D promotion when deadlines are available.
- Reports include TTFT deadline attainment and P2D earliness.

Phase 3 is complete when:

- Replay can consume Vidur-provided RED rank and scavenger labels.
- Scavenger flows are demoted without changing trace structure.
- Feasibility report identifies bottleneck request/load contributors without
  performing request admission in replay.


# Puppeteer Phase 3: Segment-Level Resource Coordination Plan

> **For agentic workers:** This phase depends on Phase 1, Phase 1 Extend, and
> the route/rate planning pieces of Phase 2. Use the existing policy-owned task
> admission model. Do not reintroduce executor-side scheduling assumptions.

## 1. Goal

Phase 3 implements a closer reproduction of Puppeteer's resource-dependency
mechanism by rewriting planned flow tasks into smaller flow segments.

This replaces the Phase 2 "co-start whole-flow group" approximation with
segment-level synchronization. The key idea is:

> Resource dependencies should synchronize planned rate-state boundaries, not
> force whole flows to start together.

## 2. Why This Phase Exists

Phase 2 considered `coordination_groups` at the flow level:

```text
if flow A and flow B share a link and overlap, wait until both are ready,
then emit both together
```

This is too coarse for the original Puppeteer idea.

Example:

```text
Flow A planned window: [1, 3]
Flow B planned window: [2, 4]
```

The planned resource states are:

```text
[1, 2): A alone
[2, 3): A and B share the link
[3, 4): B alone
```

If A waits until B is ready at time 2, then A's planned solo segment `[1, 2)` is
lost. If both flows are emitted together, the plan no longer represents the
partial overlap. Whole-flow co-start therefore violates the structure of the
offline plan.

Puppeteer's handshake mechanism is better understood as synchronizing transitions
between planned resource states:

- entering a shared-rate segment
- leaving a shared-rate segment
- changing from solo rate to shared rate
- changing from shared rate back to solo rate

Phase 3 models this by splitting each planned flow into explicit segment flow
tasks. Existing `emit_ready_tasks` coordination can then operate on segments
instead of whole flows.

## 3. Non-Goals

Do not implement:

- real packet-level handshake traffic
- NIC or switch enforcement
- source-routing protocol details
- queue occupancy simulation
- fully dynamic replanning after runtime delays

Do not modify `AnalyticalExecutor` unless a missing hook is truly generic and
cannot be expressed through segmented workload tasks plus existing policy hooks.

## 4. Design Decision: Rewrite Workload

This phase intentionally rewrites the workload.

Input:

```text
original P2PWorkload
route table
planned timing windows
planned rate schedule
resource synchronization groups
```

Output:

```text
segmented P2PWorkload
segment metadata table
segment-level coordination groups
```

The segmented workload replaces each original flow task with one or more flow
segment tasks. This keeps later execution compatible with the Phase 1/2 executor
and policy model:

- executor still sees normal `TaskType.FLOW` tasks
- bandwidth allocator still sees active flows
- policy admission still uses `emit_ready_tasks`
- resource coordination groups operate on segment task ids

This avoids a separate segment adapter inside executor.

## 5. Core Semantics

### 5.1 Original Flow Replacement

For each original flow task `F`:

1. remove `F` from the executable task list
2. create segment flow tasks `F_seg_0 ... F_seg_n`
3. attach original `F.deps` to `F_seg_0`
4. chain segments with explicit dependencies:

```text
F_seg_0 -> F_seg_1 -> ... -> F_seg_n
```

5. redirect original dependents of `F` to depend on `F_seg_n`

This preserves the semantic meaning of the original flow: downstream tasks can
start only after all bytes of the original flow have been transmitted.

### 5.2 Segment Size

Each segment represents a fixed number of bytes:

```text
segment_bytes = planned_rate_gbps * segment_duration_us * 1e9 / 8 / 1e6
```

Use integer bytes and correct rounding on the final segment so that:

```text
sum(segment.size_bytes for segment in flow_segments) == original_flow.size_bytes
```

The final segment absorbs rounding error.

### 5.3 Segment Path

All segments of one original flow use the same planned path in the first version.

This matches the common case in Puppeteer where the route is assigned per flow
and rates change over time.

Future extensions may support path changes at segment boundaries, but Phase 3
should not require that.

### 5.4 Segment Rate

Each segment should have a planned rate.

Because existing `Task` does not have a rate field, keep segment rate in an
external metadata table rather than adding workload schema fields immediately.

Recommended metadata:

```python
@dataclass
class FlowSegmentInfo:
    segment_task_id: int
    original_flow_task_id: int
    segment_index: int
    planned_start_us: int
    planned_end_us: int
    planned_rate_gbps: float
    planned_path: list[int]
    bytes: int
```

The policy can use this table for:

- segment route lookup
- rate enforcement
- debug output
- experiment reporting

## 6. Segment Boundary Construction

### 6.1 Inputs

The segmentation pass needs:

- original `P2PWorkload`
- `RouteTable`
- planned flow start/end times
- planned rates per flow over time, or enough information to derive them
- shared-link overlap information

Planned timing should come from the offline Puppeteer route/rate planner, not
from actual runtime execution.

### 6.2 Boundaries

For each original flow, collect segment boundaries from:

1. original planned start time
2. original planned end time
3. start time of any overlapping peer on a shared link
4. end time of any overlapping peer on a shared link
5. planned rate transition times for this flow

Sort and deduplicate boundaries.

For flow `A: [1, 3]` overlapping with `B: [2, 4]`, A's boundaries are:

```text
1, 2, 3
```

B's boundaries are:

```text
2, 3, 4
```

### 6.3 Segment Creation

Create one segment for every adjacent boundary pair with non-zero duration:

```text
[b0, b1), [b1, b2), ...
```

Skip zero-duration segments.

If a planned rate is zero for an interval, do not create a data segment for that
interval. Waiting should be modeled by dependencies or policy admission, not by
zero-byte active network flows.

## 7. Segment-Level Resource Dependencies

### 7.1 What to Synchronize

Synchronize segments that represent the same planned shared resource state.

Two segments should be coordinated when:

1. their original flows share at least one physical link
2. their planned segment windows overlap exactly or substantially after boundary
   splitting
3. they are planned to use compatible shared rates on the shared link
4. they are not already ordered by data dependencies

For the example:

```text
A0: [1, 2] solo
A1: [2, 3] shared
B1: [2, 3] shared
B2: [3, 4] solo
```

Create a coordination group for:

```text
{A1, B1}
```

Do not coordinate:

```text
A0 with B1
B2 with A1
```

### 7.2 How Policy Uses Groups

The policy sees segment tasks in `ready_pool`.

Rule:

```text
if a segment belongs to a coordination group:
  emit it only when all required group members are ready
else:
  emit it normally
```

This works because original flow ordering has been rewritten as explicit segment
dependencies:

```text
A0 -> A1
B1 -> B2
```

If B is delayed, A can still run A0. A1 waits for B1 only at the planned shared
segment boundary.

### 7.3 Why This Matches Puppeteer Better

The handshake no longer means "whole flows start together".

It means:

```text
all participants in the next planned shared resource state have reached that
state before any of them enters it
```

This matches Puppeteer's motivation: preserve the offline route/rate plan under
compute jitter without forcing unrelated solo portions to wait.

## 8. Rate Enforcement

### 8.1 Segment-Aware Allocator

Add or extend a bandwidth allocator that can enforce planned segment rates:

```python
class PlannedRateAllocator(BandwidthAllocator):
    def __init__(self, segment_info: dict[int, FlowSegmentInfo]):
        self.segment_info = segment_info

    def allocate(self, active_flows, topology, routing_hints, current_time):
        return {
            flow.task_id: self.segment_info[flow.task_id].planned_rate_gbps
            for flow in active_flows
        }
```

The allocator must validate that link capacity is not exceeded:

```text
for each link:
  sum(planned_rate of active segments using link) <= link.bandwidth_gbps + eps
```

If violated, raise a clear error. This indicates the offline plan is invalid.

### 8.2 Interaction With Segment Size

If segment size is computed from planned rate and planned duration, then under
planned execution the segment finishes at its planned end boundary.

If runtime jitter delays segment start, the segment still transmits the same
bytes at the same planned rate. The whole later schedule shifts through
dependencies and coordination.

This is the desired approximation for Puppeteer's static plan with runtime
synchronization.

## 9. Data Structures

Recommended new structures:

```python
@dataclass
class FlowSegmentInfo:
    segment_task_id: int
    original_flow_task_id: int
    segment_index: int
    planned_start_us: int
    planned_end_us: int
    planned_rate_gbps: float
    planned_path: list[int]
    bytes: int


@dataclass
class SegmentedWorkloadResult:
    workload: P2PWorkload
    segment_info: dict[int, FlowSegmentInfo]
    original_to_segments: dict[int, list[int]]
    segment_to_original: dict[int, int]
    coordination_groups: dict[str, set[int]]
```

These can live in a new static-analysis pass module or in a small
Puppeteer-specific planner module.

## 10. File Mapping

| File | Action | Notes |
|------|--------|-------|
| `src/static_analysis/passes/puppeteer_segmentation.py` | Create | split planned flow tasks into segment tasks |
| `src/executor/puppeteer_policy.py` | Extend | coordinate segment task groups |
| `src/executor/puppeteer_bandwidth.py` | Create/extend | enforce planned segment rates |
| `src/static_analysis/strategies/puppeteer_strategy.py` | Extend | run segmentation after route/rate planning |
| `tests/test_puppeteer_segmentation.py` | Create | unit tests for flow splitting and dependency rewiring |
| `tests/test_puppeteer_segment_policy.py` | Create | policy tests for segment-level coordination |
| `scripts/run_puppeteer_reproduce.py` | Extend | add segment-coordination mode |

## 11. Development Tasks

### Task 1: Define segment data structures

- [ ] Add `FlowSegmentInfo`
- [ ] Add `SegmentedWorkloadResult`
- [ ] Keep metadata outside `workload_format/schema.py`
- [ ] Add basic construction tests

### Task 2: Implement flow dependency rewiring

- [ ] Build dependents map from original workload
- [ ] Replace each flow with segment tasks
- [ ] Attach original deps to first segment
- [ ] Chain segment tasks
- [ ] Redirect original dependents to final segment
- [ ] Preserve compute tasks unchanged

### Task 3: Implement boundary extraction

- [ ] Collect planned start/end boundaries per flow
- [ ] Add peer overlap boundaries for shared links
- [ ] Add rate transition boundaries
- [ ] Sort and deduplicate boundaries
- [ ] Skip zero-duration intervals

### Task 4: Compute segment bytes

- [ ] Compute bytes from planned rate and duration
- [ ] Round intermediate segment sizes deterministically
- [ ] Assign remaining bytes to final segment
- [ ] Assert segment byte sum equals original flow size

### Task 5: Build segment coordination groups

- [ ] Identify segments sharing links in the same planned resource state
- [ ] Exclude segments already ordered by data dependency
- [ ] Create group ids deterministically
- [ ] Verify partial-overlap example creates `{A1, B1}` only

### Task 6: Implement planned-rate allocator

- [ ] Return planned rate for each active segment
- [ ] Validate per-link aggregate rate against capacity
- [ ] Raise clear error on invalid plan
- [ ] Add tests for valid and invalid capacity cases

### Task 7: Integrate with Puppeteer policy

- [ ] Use segment route table in `get_flow_path`
- [ ] Use segment coordination groups in `emit_ready_tasks`
- [ ] Use planned-rate allocator in `allocate_bandwidth`
- [ ] Keep default/Puppeteer non-segment modes available for comparison

### Task 8: Add end-to-end experiment mode

- [ ] Add `segment-full` or equivalent mode
- [ ] Report number of original flows, segment tasks, and coordination groups
- [ ] Report any rate-capacity validation failures
- [ ] Compare against Phase 2 `route-tte` mode

## 12. Tests

Required checks:

```bash
pytest tests/test_puppeteer_segmentation.py -v
pytest tests/test_puppeteer_segment_policy.py -v
pytest tests/test_puppeteer_policy.py -v
pytest tests/test_analytical_executor.py -v
pytest tests/ -v
```

Important scenarios:

1. Partial overlap:

```text
A: [1, 3]
B: [2, 4]
```

Expected segments:

```text
A0 [1,2], A1 [2,3], B1 [2,3], B2 [3,4]
```

Expected coordination group:

```text
{A1, B1}
```

2. No overlap:

```text
A: [1,2]
B: [2,3]
```

Expected: no shared coordination group.

3. Dependency-ordered overlap:

If `B` depends on `A`, do not create a resource coordination group between their
segments.

4. Byte preservation:

For every original flow:

```text
sum(segment bytes) == original size_bytes
```

5. Downstream preservation:

If a compute task depended on original flow `F`, it now depends on `F`'s final
segment.

6. Runtime delay:

If one segment in a coordination group becomes ready late, earlier solo segments
can still run, and only the shared segment waits.

## 13. Acceptance Criteria

Phase 3 is complete when:

- original flow tasks can be rewritten into segment flow tasks
- original flow dependencies are preserved through first/final segment rewiring
- partial overlaps are represented as segment-level coordination groups
- planned segment rates are enforced by a planned-rate allocator
- executor remains unchanged except for genuinely generic hooks already present
- experiments can compare non-segment Puppeteer mode and segment-level
  coordination mode
- docs and experiment output clearly state that this models Puppeteer resource
  dependencies at segment/rate-boundary granularity

## 14. Handoff Notes

Do not model Puppeteer's resource dependencies as whole-flow co-start barriers in
this phase.

Use this mental model instead:

```text
flow-level plan -> segment-level executable workload
segment boundary -> planned rate-state transition
coordination group -> participants in the same shared rate state
```

This phase deliberately rewrites workload tasks. That is acceptable because the
segmented workload is the executable form of a specific Puppeteer plan, not the
canonical workload IR for every policy.


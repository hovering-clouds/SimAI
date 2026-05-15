# Puppeteer Phase 2: Reproduction Plan

> **For agentic workers:** This phase depends on
> [Phase 1](puppeteer-phase1-plan.md) and
> [Phase 1 Extend](puppeteer-phase1-extend-plan.md). Do not start Phase 2 until
> the executor supports `SchedulingPolicy`, default behavior is proven unchanged,
> and task admission is fully policy-owned. Phase 2 should add Puppeteer-like
> analysis passes and scheduling policy implementations without rewriting the
> core event loop.

## 1. Goal

Phase 2 implements a practical reproduction of the main scheduling ideas in
`Puppeteer: A Network Planner for AI Training Workloads`.

The goal is not a bit-for-bit reproduction of the paper's simulator. The goal is
to implement the main strategy elements in `simai-flow-scheduler`:

1. offline flow-aware greedy routing
2. TTE-aware bandwidth allocation
3. resource-dependency-based runtime coordination

The implementation should allow controlled comparison against the existing
default fair-share behavior.

## 2. Background

Puppeteer treats AI training traffic as predictable. It uses the workload DAG,
physical topology, and collective expansion to build a network plan before the
training iteration runs.

The paper's plan controls:

- route: which path each flow uses
- rate: which bandwidth each flow receives over time
- coordination: which peer flows should synchronize before sending

Important interpretation for this project:

- Path selection is an offline planning result, not runtime dynamic routing.
- The executor should query a precomputed per-flow route table.
- Runtime policy controls admission and rate, not switch routing tables.

See [puppeteer-strategy-notes.md](puppeteer-strategy-notes.md) for a
strategy-level summary.

## 3. Dependencies on Phase 1 and Phase 1 Extend

Phase 2 assumes these Phase 1 / Phase 1 Extend interfaces exist:

- `SchedulingPolicy`
- `DefaultSchedulingPolicy`
- `AnalyticalExecutor` delegates:
  - ready task emission
  - flow path lookup
  - bandwidth allocation
  - task emitted/completed notifications
- `AnalyticalExecutor.execute(workload)` does not require `ExecutionPlan`
- default compute ordering is implemented inside `DefaultSchedulingPolicy`
- executor has no hard-coded per-node compute cursor logic

Expected policy hooks:

```python
emit_ready_tasks(current_time, ready_tasks) -> list[int]
get_flow_path(task) -> list[int]
allocate_bandwidth(current_time, active_flows) -> dict[int, float]
on_task_emitted(current_time, task) -> None
on_task_completed(current_time, task) -> None
```

If the exact names differ after Phase 1, preserve the same responsibilities.
Do not rely on a generic executor-owned `SimulationState`; Puppeteer-specific
tables and bookkeeping should live inside the Puppeteer policy.

## 4. Target Architecture

### 4.1 Static Analysis as Composable Passes

Puppeteer-specific knowledge should live in static analysis passes and policy
objects, not in `analytical.py`.

Recommended layout:

```text
src/static_analysis/
  passes/
    critical_path.py              # existing pass or re-export
    routing_hints.py              # existing shortest path pass or re-export
    puppeteer_tte.py              # new
    puppeteer_routing.py          # new
    puppeteer_coordination.py     # new
  strategies/
    puppeteer_strategy.py         # builds PuppeteerSchedulingPolicy

src/executor/
  policy.py                       # Phase 1 general policy interface
  bandwidth.py                    # add Puppeteer-aware allocator here
  puppeteer_policy.py             # optional, or place under policy.py if small
```

### 4.2 Policy Composition

`PuppeteerAnalysisStrategy` should assemble the required passes, then construct a
`PuppeteerSchedulingPolicy`.

Pseudo-flow:

```text
PuppeteerAnalysisStrategy.build_policy(workload, topology, execution_plan)
  -> build initial route table using default shortest paths
  -> run optimistic timing / initial TTE analysis
  -> run offline greedy route planning
  -> optionally recompute optimistic timing / TTE with greedy routes
  -> run resource dependency analysis
  -> create PuppeteerSchedulingPolicy(
         route_table,
         tte_table,
         coordination_table,
         bandwidth_allocator,
     )
```

The executor receives only the resulting policy. It should not receive a generic
`ExecutionHints` object and should not inspect Puppeteer-specific fields.

Bootstrap note:

- TTE needs flow duration estimates, which need paths.
- Greedy routing needs optimistic flow timing, which may use TTE/critical-path
  timing.
- Resolve this circularity with a bootstrap pass: use shortest paths for the
  first optimistic timing, produce greedy routes, then recompute TTE once with
  the greedy route table if precision is needed.

## 5. Data Structures

These structures are internal to Puppeteer analysis/policy code. They should not
be added to `workload_format/schema.py` in the first implementation.

### 5.1 RouteTable

```python
@dataclass
class RouteTable:
    paths: dict[int, list[int]]  # flow task_id -> node path
```

Rules:

- Every flow task should have one path.
- Paths are computed before execution.
- `PuppeteerSchedulingPolicy.get_flow_path` only looks up this table.

### 5.2 TTEInfo

```python
@dataclass
class TTEInfo:
    task_id: int
    tte_us: float
    priority_score: float
    priority_class: str  # "critical" | "elastic" | "background"
```

Recommended initial mapping:

- `tte_us == 0`: `critical`
- `0 < tte_us <= small_threshold`: `elastic`
- larger TTE or no direct compute exposure: `background`

The exact threshold should be configurable. A safe first version can use:

```python
small_threshold_us = 1000
```

### 5.3 ResourceDependencyTable

```python
@dataclass
class ResourceDependencyTable:
    peers: dict[int, set[int]]          # flow task_id -> peer flow ids
    groups: dict[str, set[int]]         # group id -> flow ids
```

Interpretation:

- peers are flows that may share planned resources and should coordinate
- groups represent co-start barriers used by `emit_ready_tasks`
- these are policy-level resource dependencies, not workload data dependencies

## 6. Pass 1: TTE Analysis

### 6.1 Purpose

Compute Time-to-Exposed (TTE) for flow tasks. TTE measures how much a flow can be
delayed before it exposes compute stall.

Paper formula:

```text
TTE(flow) = Start(child) - Finish(flow)
```

`child` is a dependent task of the flow. `Start(child)` is determined by the
latest-finishing parent of that child in an optimistic schedule.

### 6.2 Inputs

- `P2PWorkload`
- topology
- route table or routing hints for estimating flow duration
- `ExecutionPlan` if compute ordering must be included in timing

### 6.3 Algorithm

1. Build an optimistic timing pass over the workload DAG.
2. Estimate compute duration from `duration_us`.
3. Estimate flow duration at line rate using the planned path.
4. Include implicit compute-order edges from `ExecutionPlan`.
5. For each flow task:
   - inspect dependent child tasks
   - for each child, compute `child_start - flow_finish`
   - use the minimum non-negative value as the flow TTE
6. Convert TTE to priority info.

### 6.4 Notes

Existing `critical_path.py` already computes ASAP/ALAP and slack, but it may not
fully include compute-order implicit edges unless they are represented in the DAG
or handled by the pass. The TTE pass must explicitly account for `ExecutionPlan`
ordering, otherwise priorities can be wrong.

Recommended implementation order:

1. compute TTE with default shortest paths
2. generate greedy route table
3. recompute TTE with greedy routes
4. use recomputed TTE for bandwidth allocation and coordination

For flows with no downstream child:

- assign `tte_us = inf`
- classify as `background`

For flows with multiple children:

- use the most urgent child, i.e. minimum TTE

## 7. Pass 2: Offline Greedy Routing

### 7.1 Purpose

Produce a precomputed route table using a Puppeteer-like least-active-link
heuristic.

This is an offline planning pass. It should not be implemented as runtime
dynamic routing.

### 7.2 Inputs

- `P2PWorkload`
- `NetworkTopology`
- optimistic timing info from TTE or critical-path pass

### 7.3 Algorithm

1. Sort flow tasks by optimistic start time.
2. Maintain planned active intervals for links.
3. When a flow is considered:
   - find candidate paths from `src` to `dst`
   - score each path by expected link activity in the flow's planned interval
   - choose the least-active path
   - record path in `RouteTable`
   - update planned link activity for the selected path

### 7.4 Candidate Paths

Initial implementation options:

1. Use BFS shortest path as the only candidate.
   - easiest, but gives no real routing improvement
2. Generate up to `k` simple shortest paths.
   - recommended first useful implementation
3. Add Clos-aware routing using topology layer metadata.
   - closer to the paper, but depends on topology classification quality

Recommended Phase 2 first version: implement `k_shortest_paths` with a small
default such as `k=4`, falling back to BFS when fewer paths exist.

### 7.5 Path Scoring

Simple first scoring:

```text
score(path) = max active_flow_count(link, interval) over links in path
```

Tie-breakers:

1. lower max active count
2. lower sum active count
3. shorter path length
4. deterministic lexical order of node ids

This keeps routing deterministic for tests.

## 8. Pass 3: Resource Dependency Analysis

### 8.1 Purpose

Build a lightweight approximation of Puppeteer's runtime coordination mechanism.

The purpose is to prevent flows that are planned to share resources from starting
too far apart when runtime compute variation changes their arrival times.

### 8.2 Inputs

- `P2PWorkload`
- `RouteTable`
- optimistic flow start windows
- optional TTE table

### 8.3 Initial Approximation

Create co-start groups for flows that satisfy all conditions:

1. their planned paths share at least one physical link
2. their optimistic active intervals overlap
3. they are not already ordered by workload DAG dependencies
4. at least one flow is critical or near-critical

The fourth condition keeps the first implementation from over-synchronizing
background traffic.

### 8.4 Output

`ResourceDependencyTable` with group membership.

The table is consumed by `PuppeteerSchedulingPolicy.emit_ready_tasks`.

## 9. Puppeteer Scheduling Policy

### 9.1 Responsibilities

`PuppeteerSchedulingPolicy` should:

- hold `RouteTable`
- hold `TTEInfo` per flow
- hold `ResourceDependencyTable`
- implement resource-aware emission
- implement route lookup
- delegate rate control to a Puppeteer-aware allocator

### 9.2 emit_ready_tasks

Default rule:

- compute tasks are emitted immediately
- flow tasks without coordination group are emitted immediately
- flow tasks in a group wait until all required group members are ready
- once a group is ready, emit all ready members at the same timestamp

Important:

- A flow delayed by coordination remains in executor `ready_pool`.
- It is not active and should not consume bandwidth.
- This differs from assigning zero bandwidth to an active flow.

Deadlock prevention:

- If a group references tasks that can never become ready due to DAG order, the
  policy can release currently ready members after detecting that non-ready peers
  are already completed or causally impossible.
- First implementation may instead keep group construction conservative to avoid
  impossible groups.

### 9.3 get_flow_path

Return `route_table.paths[task_id]`.

If no path exists:

- raise a clear error with task id, src, dst, and policy name
- do not fall back silently, because missing route planning indicates a bug

### 9.4 allocate_bandwidth

Use TTE-aware allocation. See next section.

## 10. TTE-Aware Bandwidth Allocation

### 10.1 Location

Add allocator implementation in:

```text
src/executor/bandwidth.py
```

or a new file:

```text
src/executor/puppeteer_bandwidth.py
```

Keep `FairShareAllocator` unchanged.

### 10.2 Initial Policy

First version can use strict priority:

1. group active flows by shared link
2. for each link, identify highest-priority flows using that link
3. allocate bandwidth to critical flows before elastic/background flows
4. a flow's final bandwidth is the minimum allocation over its path

Priority order:

```text
critical < elastic < background
```

where lower TTE means higher priority.

### 10.3 Avoiding Permanent Starvation

Pure strict priority can starve background flows in synthetic tests. Add a minimum
share option:

```python
min_background_share = 0.05
```

If enabled, background flows receive at least a small fraction of bottleneck
capacity when they are active. This is a simulation knob and should be documented
in experiment output.

### 10.4 Simpler Alternative

Weighted fair sharing is easier and more stable:

```text
weight = 1 / max(tte_us, epsilon)
```

Recommended implementation sequence:

1. implement weighted allocator first
2. add strict-priority mode behind a config flag

This allows tests to validate monotonic priority behavior without immediately
creating starvation edge cases.

## 11. Experiment Entrypoints

Add a script or extend an existing script to compare policies:

```text
scripts/run_puppeteer_reproduce.py
```

Suggested modes:

| Mode | Routing | Bandwidth | Coordination |
|------|---------|-----------|--------------|
| `default` | BFS shortest path | fair share | none |
| `route-only` | Puppeteer route table | fair share | none |
| `tte-only` | BFS shortest path | TTE-aware | none |
| `route-tte` | Puppeteer route table | TTE-aware | none |
| `full` | Puppeteer route table | TTE-aware | co-start coordination |

The script should output:

- total makespan
- per-job iteration time
- average/median flow completion time
- exposed communication time if available
- number of coordination groups
- number of flows delayed by coordination

## 12. File Mapping

| File | Action | Notes |
|------|--------|-------|
| `src/static_analysis/passes/puppeteer_tte.py` | Create | TTE and priority analysis |
| `src/static_analysis/passes/puppeteer_routing.py` | Create | offline greedy route table |
| `src/static_analysis/passes/puppeteer_coordination.py` | Create | resource dependency groups |
| `src/static_analysis/strategies/puppeteer_strategy.py` | Create | builds policy from passes |
| `src/executor/puppeteer_policy.py` | Create | `PuppeteerSchedulingPolicy` |
| `src/executor/bandwidth.py` or `puppeteer_bandwidth.py` | Extend/Create | TTE-aware allocator |
| `scripts/run_puppeteer_reproduce.py` | Create | comparison runner |
| `tests/test_puppeteer_tte.py` | Create | TTE pass tests |
| `tests/test_puppeteer_routing.py` | Create | route table tests |
| `tests/test_puppeteer_coordination.py` | Create | resource dependency tests |
| `tests/test_puppeteer_policy.py` | Create | policy behavior tests |

## 13. Development Tasks

### Task 1: Define Puppeteer data structures

- [ ] Add `RouteTable`
- [ ] Add `TTEInfo`
- [ ] Add `ResourceDependencyTable`
- [ ] Keep structures internal to Puppeteer modules
- [ ] Do not modify workload schema

### Task 2: Implement TTE pass

- [ ] Build optimistic timing with explicit compute-order edges
- [ ] Estimate flow finish time using planned or shortest path
- [ ] Compute per-flow TTE from downstream children
- [ ] Classify priority
- [ ] Add tests for linear chain, diamond dependency, PP-vs-DP-style conflict

### Task 3: Implement offline greedy routing pass

- [ ] Generate candidate paths
- [ ] Sort flows by optimistic start time
- [ ] Score candidate paths by planned link activity
- [ ] Produce deterministic route table
- [ ] Add tests with topology where two candidate paths exist

### Task 4: Implement resource dependency pass

- [ ] Detect shared-link overlapping flow pairs
- [ ] Filter pairs already ordered by DAG dependency
- [ ] Build conservative co-start groups
- [ ] Add tests for overlapping peers and already-dependent flows

### Task 5: Implement TTE-aware allocator

- [ ] Add weighted allocation mode
- [ ] Add optional strict-priority mode
- [ ] Ensure allocated bandwidth never exceeds per-link capacity
- [ ] Ensure no active non-zero-size flow receives missing allocation
- [ ] Add tests for critical flow receiving more bandwidth than background flow

### Task 6: Implement PuppeteerSchedulingPolicy

- [ ] Use coordination table in `emit_ready_tasks`
- [ ] Use route table in `get_flow_path`
- [ ] Use TTE-aware allocator in `allocate_bandwidth`
- [ ] Maintain internal bookkeeping in notification hooks
- [ ] Add policy tests independent of full executor when possible

### Task 7: Add end-to-end comparison

- [ ] Add comparison script
- [ ] Run default and Puppeteer modes on a small workload
- [ ] Report makespan and key metrics
- [ ] Keep outputs under `outputs/` or another ignored path

## 14. Tests

Required checks:

```bash
pytest tests/test_analytical_executor.py -v
pytest tests/test_puppeteer_tte.py -v
pytest tests/test_puppeteer_routing.py -v
pytest tests/test_puppeteer_coordination.py -v
pytest tests/test_puppeteer_policy.py -v
pytest tests/ -v
```

Important test scenarios:

1. A critical flow and a background flow share a link; critical flow receives
   more bandwidth.
2. Two flows have alternative paths; greedy routing spreads them when possible.
3. Coordinated peer flows do not become active until all members are ready.
4. Default mode still matches Phase 1 behavior.
5. Missing route table entry fails loudly.

## 15. Acceptance Criteria

Phase 2 is complete when:

- Puppeteer policy runs through `AnalyticalExecutor` without executor-specific
  Puppeteer branches
- route table is fully precomputed before execution
- TTE-aware allocation changes bandwidth distribution compared with fair share
- coordination can delay ready flows before they become active
- default policy remains available as baseline
- comparison script can run at least one small workload in all modes
- tests document and verify the approximations used

## 16. Known Approximations

This reproduction intentionally simplifies several parts of the paper:

- Path enforcement is modeled as precomputed route lookup, not switch-level
  source routing.
- Resource dependency is modeled as co-start admission control, not packet-level
  handshake traffic.
- TTE is based on optimistic timing and may not reflect congestion-adjusted slack.
- Greedy routing uses generic graph candidate paths unless Clos metadata is added.
- Queue occupancy is not explicitly modeled; zero-queue is approximated by keeping
  aggregate allocated rate within link capacity.

These approximations should be visible in docs and experiment output.

## 17. Handoff Notes

Keep Phase 2 changes isolated:

- If a change is needed in `analytical.py`, first verify whether Phase 1 policy
  hooks are insufficient.
- If a new field seems necessary in `Task`, first try keeping it inside a
  Puppeteer pass or policy object.
- If an algorithm requires a new generic pass, place it under
  `static_analysis/passes/` rather than inside the policy class.

The intended shape is:

```text
analysis passes produce private Puppeteer tables
  -> strategy builds PuppeteerSchedulingPolicy
  -> executor runs unchanged event loop through policy hooks
```

# Puppeteer Phase 1: Framework Refactor Plan

> **For agentic workers:** This phase is a behavior-preserving refactor. Do not add
> Puppeteer-specific algorithms, fields, or experiment scripts in this phase. The goal
> is to make the existing executor and analysis code strategy-friendly while keeping
> current functionality and outputs unchanged.

## 1. Goal

Phase 1 prepares `simai-flow-scheduler` for future scheduling-policy research.

The current `AnalyticalExecutor` already implements a working discrete-event
simulator, but it directly owns several policy decisions:

- when dependency-satisfied tasks are emitted
- how flow paths are queried
- how active flows receive bandwidth

This phase extracts those decisions behind general strategy interfaces. The executor
should remain responsible for the event queue and DAG progress only.

## 2. Non-Goals

Do not implement any Puppeteer behavior in Phase 1:

- no TTE calculation
- no greedy least-active route planning
- no resource-dependency barrier logic
- no Puppeteer-specific names in production code
- no new workload schema fields for scheduling hints

The default behavior after refactor must match the current behavior:

- shortest-path routing through existing `RoutingHints`
- all ready tasks emitted immediately
- fair-share bandwidth allocation through existing `FairShareAllocator`

## 3. Current Baseline

Relevant modules before the refactor:

| File | Current Role |
|------|--------------|
| `src/executor/analytical.py` | Event loop, dependency release, path lookup, active flow management |
| `src/executor/bandwidth.py` | `BandwidthAllocator` and `FairShareAllocator` |
| `src/static_analysis/routing_hints.py` | BFS shortest-path cache and link load stats |
| `src/static_analysis/analyzer.py` | Runs existing analysis modules in fixed order |
| `src/static_analysis/task_serializer.py` | Builds `ExecutionPlan.compute_order` |

The current executor constructor is:

```python
AnalyticalExecutor(
    topology: NetworkTopology,
    routing_hints: RoutingHints,
    allocator: BandwidthAllocator | None = None,
)
```

Phase 1 intentionally replaces this API with a policy-based API. Existing tests
and scripts should be migrated during the refactor instead of preserving both
constructor styles.

## 4. Target Architecture

### 4.1 Executor Boundary

`analytical.py` should own only the simulation mechanics:

1. maintain event queue
2. maintain dependency counters and dependents
3. maintain per-node compute cursor
4. maintain ready-but-not-emitted task pool
5. create and update `ActiveFlow`
6. record task start/end times
7. call policy hooks for decisions

It should not know why one task is delayed, why a flow has a path, or why a flow
receives a specific bandwidth.

### 4.2 Scheduling Policy Interface

Add a general policy module:

```text
src/executor/
  policy.py
```

Recommended interface:

```python
class SchedulingPolicy(ABC):
    def initialize(
        self,
        workload: P2PWorkload,
        topology: NetworkTopology,
        execution_plan: ExecutionPlan,
    ) -> None:
        ...

    def emit_ready_tasks(
        self,
        current_time: int,
        ready_tasks: list[Task],
    ) -> list[int]:
        ...

    def get_flow_path(
        self,
        task: Task,
    ) -> list[int]:
        ...

    def allocate_bandwidth(
        self,
        current_time: int,
        active_flows: list[ActiveFlow],
    ) -> dict[int, float]:
        ...

    def on_task_emitted(
        self,
        current_time: int,
        task: Task,
    ) -> None:
        ...

    def on_task_completed(
        self,
        current_time: int,
        task: Task,
    ) -> None:
        ...
```

Notes:

- `emit_ready_tasks` is an admission hook. It decides which dependency-satisfied
  tasks may start now.
- `get_flow_path` is a read-only route lookup. It must not be interpreted as
  runtime dynamic routing. Default policy returns existing shortest paths.
- `allocate_bandwidth` is the rate-control hook.
- `on_task_emitted` and `on_task_completed` are notification hooks for policy
  bookkeeping.

### 4.3 Policy State Boundary

Do not introduce a heavyweight `SimulationState` object in Phase 1.

Reasoning:

- `emit_ready_tasks` already receives the currently ready tasks.
- `allocate_bandwidth` already receives the currently active flows.
- `get_flow_path` only needs the flow task.
- coordination policies can maintain their own completed/emitted sets through
  notification hooks.
- route tables and priority tables should live inside concrete policies, not in
  a generic executor-owned hints object.

The executor should expose events and the minimum required arguments, not its
internal state. If a future policy genuinely needs more context, add a narrow
read-only context object for that specific need after discussion.

Guidelines:

- Policy implementations may keep private state initialized from
  `initialize(...)`.
- Policy implementations may update private state in `on_task_emitted` and
  `on_task_completed`.
- The executor remains the single owner of event queue, dependency counters,
  ready pool, active-flow map, and task timing records.
- Do not expose methods that let policies directly mutate executor internals.
- Avoid circular imports between `policy.py` and `analytical.py`. If
  `ActiveFlow` remains defined in `analytical.py`, use `typing.TYPE_CHECKING`,
  string annotations, or move shared runtime dataclasses to a small module such
  as `src/executor/runtime.py`.

### 4.4 DefaultSchedulingPolicy

Implement a default policy that preserves current behavior:

```python
class DefaultSchedulingPolicy(SchedulingPolicy):
    def __init__(
        self,
        routing_hints: RoutingHints,
        allocator: BandwidthAllocator | None = None,
    ):
        self.routing_hints = routing_hints
        self.allocator = allocator or FairShareAllocator()
```

Behavior:

- `initialize`: store `topology`, `workload`, and `execution_plan` if needed by
  the allocator or later policy bookkeeping
- `emit_ready_tasks`: return all ready tasks
- `get_flow_path`: call `routing_hints.get_path(task.src, task.dst)`
- `allocate_bandwidth`: delegate to `FairShareAllocator` or injected allocator
- notification hooks: no-op

### 4.5 New Public API

Use one executor construction style after Phase 1:

```python
policy = DefaultSchedulingPolicy(
    routing_hints=routing_hints,
    allocator=FairShareAllocator(),
)
executor = AnalyticalExecutor(
    topology=topology,
    policy=policy,
)
```

For custom strategies:

```python
policy = CustomSchedulingPolicy(...)
executor = AnalyticalExecutor(
    topology=topology,
    policy=policy,
)
```

All existing scripts and tests that construct `AnalyticalExecutor` should be
migrated to this policy-based API during Phase 1.

## 5. Ready Pool Refactor

Current code immediately pushes `flow_ready` or `compute_ready` when dependencies
are satisfied. Phase 1 should introduce a common ready pool:

1. when a task becomes eligible, add task id to `ready_pool`
2. call `_drain_ready_pool(current_time)`
3. convert ready ids to `Task` objects and pass them to `policy.emit_ready_tasks`
4. emit only returned task ids
5. leave non-emitted ids in `ready_pool`

For compute tasks, eligibility also requires that the task is the current cursor
entry for its node in `ExecutionPlan.compute_order`.

Pseudo-flow:

```text
task dependency count reaches zero
  -> if flow: add to ready_pool
  -> if compute and compute_cursor allows it: add to ready_pool
  -> drain_ready_pool(now)

drain_ready_pool(now)
  -> ready_tasks = [task_map[tid] for tid in ready_pool]
  -> emitted = policy.emit_ready_tasks(now, ready_tasks)
  -> for each emitted task:
       remove from ready_pool
       notify policy.on_task_emitted
       if compute: schedule compute_done
       if flow: create ActiveFlow using policy.get_flow_path(task)
```

Deadlock handling:

- If the event queue is empty but `ready_pool` is not empty, raise a clear error.
- The error should include the pending task ids and the policy class name.
- Default policy should never leave tasks pending.

## 6. BandwidthAllocator Boundary

Keep `BandwidthAllocator` as a reusable sub-strategy. Do not delete it.

Phase 1 should make it usable through `SchedulingPolicy.allocate_bandwidth`.

Expected relationship:

```text
AnalyticalExecutor
  -> SchedulingPolicy.allocate_bandwidth(...)
       -> BandwidthAllocator.allocate(...)
```

This keeps future policies free to either reuse `BandwidthAllocator` or implement
rate control directly.

## 7. Static Analysis Refactor

Phase 1 may reorganize static analysis into composable passes, but it must keep
old import paths working.

Recommended target layout:

```text
src/static_analysis/
  passes/
    critical_path.py
    routing_hints.py
    contention_analysis.py
    traffic_matrix.py
    topology_loader.py
    node_view.py
    workload_summary.py
  strategies/
    default_strategy.py
  analyzer.py
```

Compatibility requirement:

- Existing modules such as `src/static_analysis/critical_path.py` should remain
  importable.
- Use thin wrappers or re-exports if files are moved.
- Existing tests should not need import changes in Phase 1 unless the project
  maintainers explicitly approve a breaking cleanup.

`DefaultAnalysisStrategy` should simply run the existing analysis passes in the
same order as current `WorkloadAnalyzer.analyze`.

## 8. File Mapping

| File | Action | Notes |
|------|--------|-------|
| `src/executor/policy.py` | Create | `SchedulingPolicy`, `DefaultSchedulingPolicy` |
| `src/executor/analytical.py` | Refactor | delegate admission, path lookup, and bandwidth allocation |
| `src/executor/runtime.py` | Optional create | shared `Event` and `ActiveFlow` if needed to avoid circular imports |
| `src/executor/bandwidth.py` | Keep | minor signature adjustments only if necessary |
| `src/executor/__init__.py` | Update | export new policy classes |
| `src/static_analysis/passes/` | Optional create | keep old imports working |
| `src/static_analysis/strategies/default_strategy.py` | Optional create | wraps current analyzer behavior |
| `tests/test_analytical_executor.py` | Update/add | prove default behavior unchanged |
| `tests/test_executor_policy.py` | Create | focused tests for policy hooks |

## 9. Development Tasks

### Task 1: Add policy abstractions

- [ ] Create `src/executor/policy.py`
- [ ] Define `SchedulingPolicy`
- [ ] Implement `DefaultSchedulingPolicy`
- [ ] Export public classes in `src/executor/__init__.py`
- [ ] Add unit tests for default policy methods

### Task 2: Refactor executor admission path

- [ ] Add `ready_pool` to `AnalyticalExecutor.execute`
- [ ] Add `_mark_task_ready` helper
- [ ] Add `_drain_ready_pool` helper
- [ ] Route all newly eligible tasks through `_drain_ready_pool`
- [ ] Preserve compute cursor behavior
- [ ] Add deadlock error for non-empty ready pool with empty event queue

### Task 3: Refactor flow start path

- [ ] Replace direct `routing_hints.get_path` calls with `policy.get_flow_path`
- [ ] Store selected path in `current_paths`
- [ ] Create `ActiveFlow` with the returned path
- [ ] Notify `policy.on_task_emitted` for compute and flow tasks

### Task 4: Refactor bandwidth allocation path

- [ ] Replace direct allocator calls with `policy.allocate_bandwidth`
- [ ] Keep remaining-byte update logic in executor
- [ ] Keep lazy deletion/version logic unchanged
- [ ] Notify `policy.on_task_completed` when compute or flow finishes

### Task 5: Migrate public API

- [ ] Replace executor constructor with `AnalyticalExecutor(topology, policy)`
- [ ] Remove `routing_hints` and `allocator` from executor constructor
- [ ] Update all scripts to construct `DefaultSchedulingPolicy`
- [ ] Update all tests to construct `DefaultSchedulingPolicy`
- [ ] Do not keep a parallel backward-compatible constructor path

### Task 6: Static analysis pass organization

- [ ] Decide whether to move files or only add strategy wrappers
- [ ] If moving files, keep re-export wrappers at old paths
- [ ] Add `DefaultAnalysisStrategy` that reproduces `WorkloadAnalyzer.analyze`
- [ ] Keep `WorkloadAnalyzer` behavior unchanged

## 10. Tests

Required checks:

```bash
pytest tests/test_analytical_executor.py -v
pytest tests/test_routing_hints.py -v
pytest tests/test_critical_path.py -v
pytest tests/test_contention_analysis.py -v
pytest tests/ -v
```

Add focused tests:

1. Default policy emits all ready tasks.
2. Default policy path equals existing `RoutingHints.get_path`.
3. Default policy bandwidth equals `FairShareAllocator`.
4. A custom policy can hold a ready flow and release it later.
5. A custom policy can return a precomputed path.
6. Existing executor timing tests produce identical results.

## 11. Acceptance Criteria

Phase 1 is complete only if:

- all existing tests pass
- existing scripts use the new policy-based executor API
- default execution produces the same task timings as before the refactor
- executor does not contain paper-specific policy logic
- no code references Puppeteer-specific terms outside documentation
- ready admission, path lookup, and bandwidth allocation are delegated through
  policy interfaces

## 12. Handoff Notes

Future Phase 2 agents should be able to implement Puppeteer without modifying the
core event loop. If Phase 2 requires changing `analytical.py`, first check whether
the missing behavior is a general hook that belongs in Phase 1.

Keep the rule simple:

> Executor advances time and dependencies. Policies decide admission, paths, and rates.

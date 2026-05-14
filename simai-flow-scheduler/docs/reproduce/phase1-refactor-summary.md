# Phase 1 Refactor Summary

> Behavior-preserving refactoring of the `simai-flow-scheduler` execution and
> static-analysis architecture, preparing for Puppeteer strategy implementation
> in Phase 2.

## Motivation

The original `AnalyticalExecutor` directly owned several policy decisions:
task admission, flow path selection, and bandwidth allocation. This made it
impossible to plug in custom scheduling strategies without modifying the core
event loop.

Phase 1 extracts those decisions behind general interfaces, creating a clean
boundary between the simulation engine and the scheduling policy.

---

## 1. Policy Interface (Executor Layer)

### New file: `src/executor/policy.py`

Defines `SchedulingPolicy` (ABC) with 6 hooks:

| Hook | Purpose |
|------|---------|
| `initialize(workload, topology)` | One-time setup before simulation starts |
| `emit_ready_tasks(current_time, ready_tasks)` | Admission control — choose which DAG-ready tasks may start |
| `get_flow_path(task)` | Route lookup for a flow task |
| `allocate_bandwidth(current_time, active_flows)` | Rate control for active flows |
| `on_task_emitted(current_time, task)` | Notification when a task passes admission |
| `on_task_completed(current_time, task)` | Notification when a task finishes |

**`DefaultSchedulingPolicy`** implements the default behavior:
- All flows admitted immediately; compute tasks admitted in C++ reference order (per-node serial execution)
- Shortest-path routing via `RoutingHints`
- Fair-share bandwidth via `FairShareAllocator`
- No longer accepts a custom allocator — always uses `FairShareAllocator`

### Changed: `src/executor/analytical.py`

Original executor constructor:
```python
AnalyticalExecutor(topology, routing_hints, allocator=None)
```

After refactor:
```python
AnalyticalExecutor(topology, policy)
```

The executor now owns only the simulation mechanics:
- Event queue (heapq)
- Dependency counters and dependents
- Ready pool (DAG-ready tasks awaiting admission)
- Active flow management
- Task timing records

All scheduling decisions are delegated to `policy`.

### Ready Pool Pattern

When a task's dependencies are satisfied, it enters the `ready_pool`.
`_drain_ready_pool()` calls `policy.emit_ready_tasks()` to decide which
ready tasks may actually start. Non-admitted tasks remain in the pool and
are reconsidered on subsequent drains (triggered by task completions).

Deadlock detection: if the event queue empties while the ready pool is
non-empty, a `RuntimeError` is raised with the pending task IDs and policy
name.

---

## 2. Static Analysis Reorganization

### Before

```
src/static_analysis/
  analyzer.py              # WorkloadAnalyzer + WorkloadAnalysisResult
  task_serializer.py       # TaskSerializer + OrderingStrategy + CppReferenceOrdering
  routing_hints.py
  critical_path.py
  ...
```

### After

```
src/static_analysis/
  passes/                  # Individual analysis modules
    task_serializer.py     # ExecutionPlan, TaskSerializer(ABC), CppReferenceSerializer
    routing_hints.py
    critical_path.py
    contention_analysis.py
    traffic_matrix.py
    topology_loader.py
    node_view.py
    workload_summary.py
    __init__.py
  strategies/              # Composable analysis workflows
    default_strategy.py    # DefaultAnalyzer / DefaultAnalysisResult (minimal)
    example_strategy.py    # ExampleAnalyzer / ExampleAnalysisResult (full reference)
    __init__.py
  __init__.py
```

### Key changes

**`TaskSerializer` refactored from strategy-object pattern to subclass pattern:**

```python
# Before
class TaskSerializer:
    def __init__(self, ordering_strategy: OrderingStrategy):
        self.ordering_strategy = ordering_strategy
    def serialize(self, workload) -> ExecutionPlan:
        return self.ordering_strategy.order(workload)

# After
class TaskSerializer(ABC):
    @abstractmethod
    def serialize(self, workload) -> ExecutionPlan: ...

class CppReferenceSerializer(TaskSerializer):
    def serialize(self, workload) -> ExecutionPlan: ...
```

**`DefaultAnalyzer` split into minimal + full reference:**

- `DefaultAnalyzer` / `DefaultAnalysisResult` — minimal, only computes `routing_hints` + `execution_plan`. Used by `DefaultSchedulingPolicy`. Fast.
- `ExampleAnalyzer` / `ExampleAnalysisResult` — runs all 6 passes (routing hints, critical path, contention groups, node views, traffic matrix, summary). Serves as a reference template for custom strategies.

**`analyzer.py` removed** — no backward-compat wrappers kept. All imports updated to point to `strategies/default_strategy.py` or `strategies/example_strategy.py`.

**`CppReferenceSerializer` runs inside `DefaultAnalyzer.analyze()`** — the resulting `ExecutionPlan` is stored directly in `DefaultAnalysisResult.execution_plan`, eliminating redundant serialization calls in scripts.

---

## 3. Data Flow

```
Input: P2PWorkload + NetworkTopology
            │
            ▼
DefaultAnalyzer.analyze(workload)
  ├── compute_routing_hints()    → routing_hints
  └── CppReferenceSerializer()   → execution_plan
            │
            ▼
DefaultAnalysisResult
  ├── routing_hints
  └── execution_plan
            │
            ▼
DefaultSchedulingPolicy(analysis)
  ├── self.routing_hints = analysis.routing_hints
  └── self.compute_order = analysis.execution_plan.compute_order
            │
            ▼
AnalyticalExecutor(topology, policy)
  └── policy.emit_ready_tasks()   → admission
  └── policy.get_flow_path()      → routing
  └── policy.allocate_bandwidth() → rate control
```

For custom strategies (Phase 2):

```
CustomAnalyzer.analyze(workload)   → CustomAnalysisResult
CustomSchedulingPolicy(result)     → implements policy hooks
AnalyticalExecutor(topology, policy)
```

---

## 4. File Manifest

| File | Status | Role |
|------|--------|------|
| `src/executor/policy.py` | Created | `SchedulingPolicy` ABC + `DefaultSchedulingPolicy` |
| `src/executor/analytical.py` | Refactored | Event loop only; delegates to policy |
| `src/executor/bandwidth.py` | Kept | `BandwidthAllocator` ABC + `FairShareAllocator` |
| `src/executor/runtime.py` | Kept | `ActiveFlow` dataclass |
| `src/executor/result.py` | Kept | `ExecutionResult` dataclass |
| `src/static_analysis/passes/task_serializer.py` | Moved (was `src/static_analysis/`) | `ExecutionPlan`, `TaskSerializer`(ABC), `CppReferenceSerializer` |
| `src/static_analysis/strategies/default_strategy.py` | Created | Minimal `DefaultAnalyzer` / `DefaultAnalysisResult` |
| `src/static_analysis/strategies/example_strategy.py` | Created | Full `ExampleAnalyzer` / `ExampleAnalysisResult` (reference) |
| `src/static_analysis/__init__.py` | Updated | Exports all public symbols |
| `tests/test_executor_policy.py` | Created | Policy hook unit tests |
| `tests/test_analyzer.py` | Updated | Tests for `ExampleAnalyzer` |
| `tests/test_task_serializer.py` | Updated | Tests for `TaskSerializer` / `CppReferenceSerializer` |

---

## 5. Deleted Files

| File | Replacement |
|------|-------------|
| `src/static_analysis/analyzer.py` | `strategies/default_strategy.py` + `strategies/example_strategy.py` |
| `src/static_analysis/task_serializer.py` | `passes/task_serializer.py` |

---

## 6. Script Changes

All three e2e scripts (`run_e2e.py`, `run_inference_e2e.py`, `run_mixed_e2e.py`):

- Removed redundant `CppReferenceSerializer().serialize()` step (now handled inside `DefaultAnalyzer.analyze()`)
- Updated `DefaultSchedulingPolicy` construction to pass `analysis=analysis` instead of `routing_hints=analysis.routing_hints`
- Replaced `critical_path` prints with `execution_plan.compute_order` info

---

## 7. Test Coverage

461 tests across all modules pass. Key test files:

| File | Tests | Focus |
|------|-------|-------|
| `tests/test_analytical_executor.py` | 10 | Executor event loop with policy delegation |
| `tests/test_executor_policy.py` | 8 | Policy hooks, custom policies, deadlock detection |
| `tests/test_analyzer.py` | 11 | Full `ExampleAnalyzer` pipeline |
| `tests/test_task_serializer.py` | 23 | `ExecutionPlan`, `CppReferenceSerializer`, `TaskSerializer` |
| `tests/test_visualizer.py` | 13 | Chrome trace output |
| `tests/test_workload_builder.py` | 36 | Workload DAG construction |

---

## 8. Ready for Phase 2

The architecture now supports custom strategies without touching the core
event loop:

1. **Analyzer** — Implement a custom analyzer (or reuse `ExampleAnalyzer`)
   to produce strategy-specific analysis results
2. **Policy** — Implement `SchedulingPolicy` hooks using those results
3. **Wire together** — `AnalyticalExecutor(topology, CustomPolicy(result))`

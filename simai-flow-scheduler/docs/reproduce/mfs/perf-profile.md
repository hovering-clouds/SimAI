# MFS Policy Performance Profile

Recorded: 2026-05-25
Workload: inference_trace_stage1_pp2.json (874,432 tasks: 249,856 compute + 624,576 flow)
Topology: AlibabaHPN_32g_8gps_DualToR_DualPlane_200Gbps_A100_stage1 (174 nodes)
Policies: Default vs MFS

## Summary

| Policy | Execution Time | Makespan | vs Default |
|--------|---------------|----------|------------|
| Default | ~191s | 5.663s | 1.0x |
| MFS | ~373s | 5.521s | 1.95x |

## Default Policy Profile (top 15 by cumtime)

| ncalls | tottime | cumtime | function |
|--------|---------|---------|----------|
| 1 | 8.5s | 191.5s | execute |
| 1,214,991 | 16.2s | 133.0s | _reallocate_bandwidth |
| 20,047,552 | 3.3s | 108.9s | _handle_flow_completion |
| 1,214,991 | 1.0s | 62.9s | allocate_bandwidth (default) |
| 1,214,991 | 39.3s | 61.9s | fair_share_allocator.allocate |
| 874,432 | 0.5s | 42.1s | _release_dependents |
| 870,528 | 0.3s | 41.5s | _mark_task_ready |
| 1,744,961 | 1.3s | 41.1s | _drain_ready_pool |
| 20,297,408 | 19.8s | 38.6s | heappop |
| 197,050,847 | 22.1s | 22.1s | __lt__ (heap compare) |
| 182,610,128 | 16.4s | 16.4s | dict.get |
| 20,297,408 | 4.3s | 7.7s | heappush |

## MFS Policy Profile (top 15 by cumtime)

| ncalls | tottime | cumtime | function |
|--------|---------|---------|----------|
| 1 | 8.7s | 372.8s | execute |
| 1,215,071 | 23.6s | 250.8s | _reallocate_bandwidth |
| 22,039,925 | 3.7s | 192.9s | _handle_flow_completion |
| 1,215,071 | 0.6s | 155.7s | allocate_bandwidth (mfs) |
| 1,215,071 | 25.2s | 155.1s | mfs_allocator.allocate |
| 1,320,138 | 68.8s | 100.3s | _allocate_fair_share |
| 22,289,781 | 51.8s | 101.4s | heappop |
| 512,169,195 | 58.4s | 58.4s | __lt__ (heap compare) |
| 22,487,815 | 6.9s | 16.6s | _queue_for |
| 11,060,407 | 4.9s | 7.7s | _compute_rli |
| 22,289,781 | 13.5s | 36.3s | push_event |
| 22,039,925 | 16.6s | 33.5s | _compute_propagation_delay |
| 334,827,984 | 31.5s | 31.5s | dict.get |
| 1,092,315 | 0.6s | 1.0s | _compute_red |
| 1,092,315 | 0.5s | 0.7s | _is_infeasible |

## MFS-Specific Breakdown

Total MFS overhead over default: ~182s.

### allocator.allocate — 155s (41.6% of MFS time)

| Phase | Cost | Detail |
|-------|------|--------|
| Phase 1: queue classification | ~25s | _queue_for (17s) + _compute_rli (8s) |
| Phase 2: RED + feasibility | ~2s | _compute_red (1s) + _is_infeasible (1s) |
| Phase 3: build link_rem | ~19s | get_link for each unique link per cycle (74M calls) |
| Phase 4: _allocate_fair_share | ~101s | 3-pass link iteration × 1.3M calls |
| Other (loop/alloc overhead) | ~8s | |

### _allocate_fair_share — 101s (27% of MFS time)

Three passes over each flow's path links:
1. Count flows sharing each link (link_counts)
2. Compute per-flow bandwidth as min(rem / count) across links
3. Deduct allocated bandwidth from link_rem

Each call: ~18 flows × ~3 hop links × 3 passes = 162 link iterations.
1.3M calls × 162 iterations = ~210M link traversals at ~0.5μs each.

### Event processing overhead — ~170s

MFS has ~2M more events than default (22M vs 20M) due to EARLY/P2D flow state changes causing more reallocations. Extra operations:
- heappop: +2M calls, +32s
- __lt__: +315M comparisons, +36s
- push_event: +2M calls, +13s (mostly propagation delay compute)

## System

- Platform: macOS (Darwin)
- Python: 3.x (via `uv run`)
- Model: deepseek-671B (tp=2, ep=4, pp=2)
- Hardware: Apple Silicon

"""Chrome Trace visualizers for ExecutionResult.

Outputs Chrome Trace JSON files viewable in chrome://tracing.
Three classes sharing a common base:
- ChromeTraceVisualizer: base class with shared logic
- ChromeTraceCompact: merges collective flows into single events + flow arrows
- ChromeTraceVerbose: shows every individual P2P flow
"""
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional

from ..workload_format.schema import P2PWorkload, Task, Phase, CommType, TaskType
from .result import ExecutionResult, TaskTiming


# ── 缩写映射 ──

PHASE_ABBREV = {
    Phase.FORWARD: "fwd",
    Phase.BACKWARD_INPUT: "bwd_i",
    Phase.BACKWARD_WEIGHT: "bwd_w",
    Phase.OPTIMIZER: "opt",
}

COMM_TYPE_ABBREV = {
    CommType.TP_ALLREDUCE_RING: "tp_ar",
    CommType.TP_ALLREDUCE_TREE: "tp_at",
    CommType.TP_ALLGATHER_RING: "tp_ag",
    CommType.TP_ALLGATHER_TREE: "tp_at",
    CommType.TP_REDUCESCATTER_RING: "tp_rs",
    CommType.TP_REDUCESCATTER_TREE: "tp_rt",
    CommType.TP_ALLTOALL: "tp_a2a",
    CommType.DP_ALLREDUCE: "dp_ar",
    CommType.DP_ALLGATHER: "dp_ag",
    CommType.DP_REDUCESCATTER: "dp_rs",
    CommType.DP_ALLTOALL: "dp_a2a",
    CommType.EP_ALLTOALL: "ep_a2a",
    CommType.PP_SEND: "pp_snd",
    CommType.PP_RECV: "pp_rcv",
}


def _abbrev_comm(comm_type: CommType) -> str:
    return COMM_TYPE_ABBREV.get(comm_type, str(comm_type.value))


def _abbrev_phase(phase: Phase) -> str:
    return PHASE_ABBREV.get(phase, str(phase.value))


def _task_description(task: Task, tag: str = "") -> str:
    """生成 task 人类可读描述，tag 追加到 phase/comm 后面。"""
    if task.is_compute():
        phase = _abbrev_phase(task.phase)
        suffix = f" {tag}" if tag else ""
        return f"{phase}{suffix} L{task.layer_id}"
    else:
        comm = _abbrev_comm(task.comm_type)
        return f"{comm} {task.src}->{task.dst}"


# ── 基类 ──

class ChromeTraceVisualizer(ABC):
    """Chrome Trace 可视化基类。

    负责公共逻辑：JSON 文件写入、metadata 事件生成、行名称管理、
    缩写规则、deps 信息。
    子类通过实现 _build_events() 定义不同的展示粒度。
    """

    def __init__(self, workload: P2PWorkload):
        self.workload = workload
        self._task_map: dict[int, Task] = {t.task_id: t for t in workload.tasks}
        # 推断 GA steps 数量（compute 任务的最大 iteration 值）
        compute_iters = [t.iteration for t in workload.tasks if t.is_compute() and t.iteration >= 0]
        self._ga = max(compute_iters) if compute_iters else 0

    def export(self, result: ExecutionResult, path: str) -> None:
        """导出为 Chrome Trace JSON 文件。"""
        events = self.to_events(result)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

    def to_events(self, result: ExecutionResult) -> list[dict]:
        """转换为 Chrome Trace 事件列表。"""
        events = []
        events.extend(self._build_metadata_events(result))
        events.extend(self._build_events(result))
        return events

    @abstractmethod
    def _build_events(self, result: ExecutionResult) -> list[dict]:
        """子类实现：生成具体的任务事件。"""
        pass

    # ── 公共辅助方法 ──

    def _tid_for_compute(self, node_id: int) -> int:
        return node_id * 2

    def _tid_for_comm(self, node_id: int) -> int:
        return node_id * 2 + 1

    def _iteration_tag(self, iteration: int) -> str:
        if iteration < 0:
            return "pre"
        if self._ga > 0 and iteration >= self._ga:
            return "post"
        return f"GA{iteration}"

    def _build_metadata_events(self, result: ExecutionResult) -> list[dict]:
        """生成行名称和分组名称的 metadata 事件。"""
        events = []
        seen_pids: set[int] = set()
        seen_tids: set[tuple[int, int]] = set()

        for timing in result.per_task.values():
            task = self._task_map.get(timing.task_id)
            if task is None:
                continue
            pid = task.job_id
            if task.is_compute():
                tid = self._tid_for_compute(timing.node)
            else:
                tid = self._tid_for_comm(task.src if task.src is not None else 0)

            if pid not in seen_pids:
                seen_pids.add(pid)
                events.append({
                    "name": "process_name", "ph": "M",
                    "pid": pid, "tid": 0,
                    "args": {"name": f"Job {pid}"},
                })

            key = (pid, tid)
            if key not in seen_tids:
                seen_tids.add(key)
                node_id = timing.node if task.is_compute() else (task.src or 0)
                track = "Compute" if task.is_compute() else "Comm"
                events.append({
                    "name": "thread_name", "ph": "M",
                    "pid": pid, "tid": tid,
                    "args": {"name": f"Node {node_id} ({track})"},
                })

        return events

    def _build_compute_events(self, result: ExecutionResult) -> list[dict]:
        """生成所有 compute 任务的 X 事件（两种模式通用）。"""
        events = []
        for timing in result.per_task.values():
            task = self._task_map.get(timing.task_id)
            if task is None or not task.is_compute():
                continue
            # Skip zero-duration compute tasks (pre/post sync points) — they
            # stack vertically and clutter the track without showing timing.
            if timing.end_time_us == timing.start_time_us:
                continue
            events.append(self._make_compute_event(task, timing))
        return events

    def _make_compute_event(self, task: Task, timing: TaskTiming) -> dict:
        tag = self._iteration_tag(task.iteration)
        name = _task_description(task, tag)
        return {
            "name": name,
            "cat": "compute",
            "ph": "X",
            "ts": timing.start_time_us,
            "dur": timing.end_time_us - timing.start_time_us,
            "pid": task.job_id,
            "tid": self._tid_for_compute(timing.node),
            "args": {
                "task_id": task.task_id,
                "phase": task.phase.value,
                "iteration": task.iteration,
                "layer_id": task.layer_id,
                "duration_us": task.duration_us,
                "deps": task.deps,
                "dep_descriptions": [
                    _task_description(
                        self._task_map[d],
                        self._iteration_tag(self._task_map[d].iteration) if self._task_map[d].is_compute() else "",
                    )
                    for d in task.deps if d in self._task_map
                ],
            },
        }

    def _make_flow_event(self, task: Task, timing: TaskTiming) -> dict:
        name = _task_description(task, "")
        return {
            "name": name,
            "cat": "flow",
            "ph": "X",
            "ts": timing.start_time_us,
            "dur": timing.end_time_us - timing.start_time_us,
            "pid": task.job_id,
            "tid": self._tid_for_comm(task.src if task.src is not None else 0),
            "args": {
                "task_id": task.task_id,
                "src": task.src,
                "dst": task.dst,
                "size_bytes": task.size_bytes,
                "comm_type": task.comm_type.value,
                "phase": task.phase.value,
                "iteration": task.iteration,
                "deps": task.deps,
                "dep_descriptions": [
                    _task_description(
                        self._task_map[d],
                        self._iteration_tag(self._task_map[d].iteration) if self._task_map[d].is_compute() else "",
                    )
                    for d in task.deps if d in self._task_map
                ],
            },
        }

    @staticmethod
    def _resolve_overlaps(events: list[dict]) -> None:
        """Resolve overlapping X events on the same (pid, tid) for Chrome Trace.

        Chrome Trace drops overlapping X events on the same track.  Ring AllReduce
        pipelines naturally produce small (1-2 us) overlaps between adjacent flows
        on some nodes: a node can simultaneously receive L4's last chunk from one
        predecessor and L3's first chunk from another predecessor on different
        physical links.  This is correct simulation behavior, but the same track
        ends up with partially overlapping slices.
        Trim the earlier event's tail by the overlap amount to prevent Chrome Trace
        from silently dropping events.
        """
        by_track: dict[tuple, list[dict]] = defaultdict(list)
        for event in events:
            if event.get("ph") == "X":
                by_track[(event["pid"], event["tid"])].append(event)

        for track_events in by_track.values():
            track_events.sort(key=lambda e: (e["ts"], -(e["ts"] + e["dur"])))
            for i in range(1, len(track_events)):
                prev = track_events[i - 1]
                curr = track_events[i]
                prev_end = prev["ts"] + prev["dur"]
                if curr["ts"] >= prev_end:
                    continue
                curr_end = curr["ts"] + curr["dur"]
                if curr_end <= prev_end:
                    pass  # proper nesting or same start — no action needed
                else:
                    prev["dur"] = curr["ts"] - prev["ts"]


# ── Verbose 模式 ──

class ChromeTraceVerbose(ChromeTraceVisualizer):
    """详细模式：每条 P2P flow 单独显示。"""

    def _build_events(self, result: ExecutionResult) -> list[dict]:
        events = []
        events.extend(self._build_compute_events(result))
        flow_events = self._build_flow_events(result)
        self._resolve_overlaps(flow_events)
        events.extend(flow_events)
        return events

    def _build_flow_events(self, result: ExecutionResult) -> list[dict]:
        events = []
        for timing in result.per_task.values():
            task = self._task_map.get(timing.task_id)
            if task is None or not task.is_flow():
                continue
            # Skip zero-duration flows: they render as stacked arrows with
            # no visual information, cluttering the track.
            if timing.end_time_us == timing.start_time_us:
                continue
            events.append(self._make_flow_event(task, timing))
        return events


# ── Flow Detail 模式 ──

class ChromeTraceFlowDetail(ChromeTraceVisualizer):
    """指定时间窗口内的 P2P flow 详细视图（带 lane 分配）。

    仅展示通信任务，保留原始时间戳以便与 Verbose trace 对齐比较。
    重叠流通过贪心区间着色分配到不同子轨道，Chrome Trace 并排渲染。
    适用于在大规模 workload 中放大观察某个集合通信操作的流级细节。
    """

    def __init__(
        self,
        workload: P2PWorkload,
        time_start_us: int,
        time_end_us: int,
    ):
        super().__init__(workload)
        self.time_start = time_start_us
        self.time_end = time_end_us

    def _build_metadata_events(self, result: ExecutionResult) -> list[dict]:
        _, metadata = self._compute_laned_flows(result)
        return metadata

    def _build_events(self, result: ExecutionResult) -> list[dict]:
        flow_events, _ = self._compute_laned_flows(result)
        return flow_events

    def _compute_laned_flows(
        self, result: ExecutionResult
    ) -> tuple[list[dict], list[dict]]:
        """Compute lane-allocated flow events and metadata for the time window.

        Flows whose [start, end] is completely contained in [time_start, time_end]
        are included.  Greedy interval partitioning assigns each flow to the first
        available lane on its (job, src_node) track.

        Returns:
            (flow_events, metadata_events)
        """
        groups: dict[tuple[int, int], list[tuple[Task, TaskTiming]]] = defaultdict(list)

        for timing in result.per_task.values():
            task = self._task_map.get(timing.task_id)
            if task is None or not task.is_flow():
                continue
            if timing.end_time_us == timing.start_time_us:
                continue
            if not (timing.start_time_us >= self.time_start
                    and timing.end_time_us <= self.time_end):
                continue
            src = task.src if task.src is not None else 0
            groups[(task.job_id, src)].append((task, timing))

        events: list[dict] = []
        metadata: list[dict] = []
        seen_pids: set[int] = set()
        tid_counter = 0

        for (job_id, src_node), items in sorted(groups.items()):
            items.sort(key=lambda x: x[1].start_time_us)

            # Greedy lane allocation
            lanes: list[int] = []       # end time of last event per lane
            lane_tids: list[int] = []   # tid assigned to each lane
            node_lanes: list[tuple[int, int]] = []  # (tid, lane_idx) for metadata

            for task, timing in items:
                end = timing.end_time_us
                assigned = -1

                for i, lane_end in enumerate(lanes):
                    if timing.start_time_us >= lane_end:
                        lanes[i] = end
                        assigned = i
                        break

                if assigned == -1:
                    lanes.append(end)
                    lane_tids.append(tid_counter)
                    tid_counter += 1
                    assigned = len(lane_tids) - 1
                    node_lanes.append((lane_tids[assigned], assigned))

                event = self._make_flow_event(task, timing)
                event["tid"] = lane_tids[assigned]
                events.append(event)

            # Metadata
            if job_id not in seen_pids:
                seen_pids.add(job_id)
                metadata.append({
                    "name": "process_name", "ph": "M",
                    "pid": job_id, "tid": 0,
                    "args": {"name": f"Job {job_id}"},
                })

            multi = len(node_lanes) > 1
            for tid, idx in node_lanes:
                suffix = f"/{idx + 1}" if multi else ""
                metadata.append({
                    "name": "thread_name", "ph": "M",
                    "pid": job_id, "tid": tid,
                    "args": {"name": f"Node {src_node} (Comm{suffix})"},
                })

        return events, metadata


# ── Compact 模式 ──

# 合并键：(job_id, node_id, iteration, phase, layer_id, comm_type, item_id)
MergeKey = tuple[int, int, int, str, int, str, int]

# 合并结果：{merge_id: {start, end, task_ids, total_bytes, ...}}
MergedFlow = dict


class ChromeTraceCompact(ChromeTraceVisualizer):
    """紧凑模式：合并同一集合通信的所有 flow，附带 Flow Event 箭头。"""

    def __init__(self, workload: P2PWorkload, show_arrows: bool = False):
        super().__init__(workload)
        self.show_arrows = show_arrows

    def _build_events(self, result: ExecutionResult) -> list[dict]:
        events = []
        events.extend(self._build_compute_events(result))
        merged = self._merge_collective_flows(result)
        merged_events = self._build_merged_flow_events(result, merged)
        self._resolve_overlaps(merged_events)
        events.extend(merged_events)
        if self.show_arrows:
            events.extend(self._build_flow_arrows(result, merged))
        return events

    def _merge_collective_flows(self, result: ExecutionResult) -> dict[MergeKey, MergedFlow]:
        """按 (job_id, node_id, iteration, phase, layer_id, comm_type, item_id) 分组 flow。

        只合并每个节点作为 receiver（dst）的 flow，与依赖链保持一致：
        依赖链仅等待 receiver flow 完成，merge 也只看 receiver flow，
        避免发送方向的尾部 flow 与下一个集合通信的接收方向产生重叠。
        """
        groups: dict[MergeKey, list[tuple[Task, TaskTiming]]] = defaultdict(list)

        for timing in result.per_task.values():
            task = self._task_map.get(timing.task_id)
            if task is None or not task.is_flow():
                continue
            # 只合并 receiver（dst）方向的 flow
            if task.dst is None:
                continue
            key: MergeKey = (
                task.job_id, task.dst, task.iteration,
                task.phase.value, task.layer_id,
                task.comm_type.value, task.item_id,
            )
            groups[key].append((task, timing))

        merged: dict[MergeKey, MergedFlow] = {}
        for key, items in groups.items():
            task_ids = [t.task_id for t, _ in items]
            starts = [tim.start_time_us for _, tim in items]
            ends = [tim.end_time_us for _, tim in items]
            total_bytes = sum(t.size_bytes or 0 for t, _ in items)
            task0 = items[0][0]  # 取第一个 task 作为代表

            merged[key] = {
                "task_ids": task_ids,
                "start_time_us": min(starts),
                "end_time_us": max(ends),
                "total_bytes": total_bytes,
                "num_flows": len(items),
                "comm_type": task0.comm_type.value,
                "phase": task0.phase.value,
                "iteration": task0.iteration,
                "layer_id": task0.layer_id,
                "job_id": task0.job_id,
                "node_id": key[1],
            }

        return merged

    def _build_merged_flow_events(
        self, result: ExecutionResult, merged: dict[MergeKey, MergedFlow]
    ) -> list[dict]:
        """为每个合并组生成一个 X 事件。"""
        events = []
        for key, m in merged.items():
            if m["end_time_us"] == m["start_time_us"]:
                continue
            comm = COMM_TYPE_ABBREV.get(
                CommType(m["comm_type"]), m["comm_type"]
            )
            phase = PHASE_ABBREV.get(Phase(m["phase"]), m["phase"])
            tag = self._iteration_tag(m["iteration"])
            name = f"{comm} {phase} {tag} L{m['layer_id']}"

            # 收集该合并事件涉及的所有 deps
            all_deps: set[int] = set()
            dep_descriptions: list[str] = []
            for tid in m["task_ids"]:
                task = self._task_map.get(tid)
                if task:
                    for d in task.deps:
                        if d not in all_deps:
                            all_deps.add(d)
                            if d in self._task_map:
                                dep_descriptions.append(
                                    _task_description(
                                        self._task_map[d],
                                        self._iteration_tag(self._task_map[d].iteration) if self._task_map[d].is_compute() else "",
                                    )
                                )

            events.append({
                "name": name,
                "cat": "flow",
                "ph": "X",
                "ts": m["start_time_us"],
                "dur": m["end_time_us"] - m["start_time_us"],
                "pid": m["job_id"],
                "tid": self._tid_for_comm(m["node_id"]),
                "args": {
                    "merged_from": m["task_ids"],
                    "num_flows": m["num_flows"],
                    "total_bytes": m["total_bytes"],
                    "comm_type": m["comm_type"],
                    "phase": m["phase"],
                    "iteration": m["iteration"],
                    "layer_id": m["layer_id"],
                    "deps": sorted(all_deps),
                    "dep_descriptions": dep_descriptions,
                },
            })
        return events

    def _build_flow_arrows(
        self, result: ExecutionResult, merged: dict[MergeKey, MergedFlow]
    ) -> list[dict]:
        """生成 compute <-> 合并 flow 之间的 Flow Event 箭头。"""
        timing_map = result.per_task

        # 构建 task_id -> merged_key 的反向映射
        task_to_merged: dict[int, MergeKey] = {}
        # 记录可见的 merged key（duration > 0）
        visible_merged: set[MergeKey] = set()
        for key, m in merged.items():
            if m["end_time_us"] > m["start_time_us"]:
                visible_merged.add(key)
            for tid in m["task_ids"]:
                task_to_merged[tid] = key

        # 记录可见的 compute task_id（duration > 0）
        visible_computes: set[int] = set()
        for timing in result.per_task.values():
            if timing.end_time_us > timing.start_time_us:
                task = self._task_map.get(timing.task_id)
                if task and task.is_compute():
                    visible_computes.add(timing.task_id)

        events: list[dict] = []
        arrow_counter = 0

        # 遍历 merged events，为它们的 deps 画箭头
        for key, m in merged.items():
            if key not in visible_merged:
                continue
            m_pid = m["job_id"]
            m_tid = self._tid_for_comm(m["node_id"])
            m_start = m["start_time_us"]

            # 收集该合并组的所有上游 deps（去重）
            upstream_deps: set[int] = set()
            for tid in m["task_ids"]:
                task = self._task_map.get(tid)
                if task:
                    for d in task.deps:
                        upstream_deps.add(d)

            # 为每个上游 dep 画箭头
            for dep_id in upstream_deps:
                dep_task = self._task_map.get(dep_id)
                dep_timing = timing_map.get(dep_id)
                if dep_task is None or dep_timing is None:
                    continue
                # 跳过零时长的 compute dep（不在 trace 中显示）
                if dep_task.is_compute() and dep_id not in visible_computes:
                    continue
                # 跳过属于零时长 merged event 的 flow dep
                if dep_task.is_flow():
                    dep_mkey = task_to_merged.get(dep_id)
                    if dep_mkey is None or dep_mkey not in visible_merged:
                        continue

                arrow_id = arrow_counter
                arrow_counter += 1

                # dep 源事件的位置
                if dep_task.is_compute():
                    src_tid = self._tid_for_compute(dep_timing.node)
                else:
                    src_tid = self._tid_for_comm(dep_task.src or 0)

                # 发送端（dep 完成）
                events.append({
                    "name": "dep", "cat": "dep", "ph": "s",
                    "id": arrow_id,
                    "ts": dep_timing.end_time_us,
                    "pid": dep_task.job_id, "tid": src_tid,
                })
                # 接收端（merged event 开始）
                events.append({
                    "name": "dep", "cat": "dep", "ph": "f",
                    "id": arrow_id,
                    "ts": m_start,
                    "pid": m_pid, "tid": m_tid,
                })

        # 遍历 compute events，为依赖 merged flow 的 compute 画箭头
        for timing in result.per_task.values():
            if timing.task_id not in visible_computes:
                continue
            task = self._task_map.get(timing.task_id)
            if task is None:
                continue

            for dep_id in task.deps:
                dep_key = task_to_merged.get(dep_id)
                if dep_key is None or dep_key not in visible_merged:
                    continue
                dep_merged = merged[dep_key]

                arrow_id = arrow_counter
                arrow_counter += 1

                # 发送端（merged event 完成）
                events.append({
                    "name": "dep", "cat": "dep", "ph": "s",
                    "id": arrow_id,
                    "ts": dep_merged["end_time_us"],
                    "pid": dep_merged["job_id"],
                    "tid": self._tid_for_comm(dep_merged["node_id"]),
                })
                # 接收端（compute 开始）
                events.append({
                    "name": "dep", "cat": "dep", "ph": "f",
                    "id": arrow_id,
                    "ts": timing.start_time_us,
                    "pid": task.job_id,
                    "tid": self._tid_for_compute(timing.node),
                })

        return events

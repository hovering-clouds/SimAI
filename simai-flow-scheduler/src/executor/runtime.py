"""Shared runtime data structures for executor and policy modules."""
from dataclasses import dataclass


@dataclass
class ActiveFlow:
    """当前正在传输的流。"""
    task_id: int
    src: int
    dst: int
    size_bytes: int
    remaining_bytes: int
    path: list[int]
    start_time: int
    last_update_time: int
    current_bw_gbps: float = 0.0
    estimated_end_time: int = 0
    version: int = 0

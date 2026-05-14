"""
Traffic matrix - node-to-node traffic volume statistics.

Computes aggregate traffic between node pairs, identifies top senders/receivers.

Useful for:
- Understanding traffic hotspots
- Identifying communication patterns
- Load balancing decisions
"""

from dataclasses import dataclass, field

from ...workload_format.schema import P2PWorkload


@dataclass
class TrafficMatrix:
    traffic: dict[tuple[int, int], int] = field(default_factory=dict)
    top_senders: list[tuple[int, int]] = field(default_factory=list)
    top_receivers: list[tuple[int, int]] = field(default_factory=list)

    def get_traffic(self, src: int, dst: int) -> int:
        return self.traffic.get((src, dst), 0)

    def get_total_traffic(self) -> int:
        return sum(self.traffic.values())


def compute_traffic_matrix(workload: P2PWorkload) -> TrafficMatrix:
    tm = TrafficMatrix()

    for task in workload.tasks:
        if not task.is_flow():
            continue
        if task.src is None or task.dst is None:
            continue

        link_id = (task.src, task.dst)
        tm.traffic[link_id] = tm.traffic.get(link_id, 0) + (task.size_bytes or 0)

    sender_bytes: dict[int, int] = {}
    receiver_bytes: dict[int, int] = {}

    for (src, dst), bytes_count in tm.traffic.items():
        sender_bytes[src] = sender_bytes.get(src, 0) + bytes_count
        receiver_bytes[dst] = receiver_bytes.get(dst, 0) + bytes_count

    tm.top_senders = sorted(sender_bytes.items(), key=lambda x: x[1], reverse=True)
    tm.top_receivers = sorted(
        receiver_bytes.items(), key=lambda x: x[1], reverse=True
    )

    return tm

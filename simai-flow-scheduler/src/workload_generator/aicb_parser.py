"""
AICB Workload Format Parser.

Parses AICB training workload files (.txt) into structured data.
Supports HYBRID_TRANSFORMER_FWD_IN_BCKWD and MICRO formats.

File layout (HYBRID_TRANSFORMER_FWD_IN_BCKWD):
  Line 1: Header with parallelism configuration (space-separated key:value pairs)
  Line 2: Item count
  Line 3+: Items (whitespace-separated, 12 fields per line)

File layout (MICRO):
  Line 1: "MICRO"
  Line 2: Item count
  Line 3+: Items (same 12-field format)

Each item line has 12 whitespace-separated fields:
  [0] name                    - Layer/operation name
  [1] placeholder             - Usually -1
  [2] forward_compute_time    - Forward pass compute time
  [3] forward_comm            - Forward comm type (NONE, ALLREDUCE, ALLGATHER, ...)
  [4] forward_comm_size       - Forward comm data size in bytes
  [5] backward_compute_time   - Backward pass compute time
  [6] backward_comm           - Backward comm type
  [7] backward_comm_size      - Backward comm data size in bytes
  [8] dp_compute_time         - DP/weight-gradient compute time
  [9] dp_comm                 - DP comm type
  [10] dp_comm_size           - DP comm data size in bytes
  [11] process_time           - Process time (default 100)

Reference: astra-sim-alibabacloud/astra-sim/workload/Workload.cc
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class AicbHeader:
    """Parsed AICB workload header (parallelism configuration)."""

    tp: int
    ep: int
    pp: int
    vpp: int  # = model total layers
    ga: int  # gradient accumulation steps
    all_gpus: int  # world size
    pp_comm_size: int  # PP stage activation size (0 if absent)


@dataclass
class AicbWorkItem:
    """A single workload item from an AICB file.

    Key insight: AICB files already contain the full GA expansion.
    The file has num_pre_items + ga * vpp items total.
    Builder does NOT need to loop over GA — it just needs to assign iteration IDs.

    Each item represents one layer/operation with compute and communication
    info for three phases: forward, backward, and dp (weight gradient).

    Reference: astra-sim-alibabacloud/astra-sim/workload/Workload.cc lines 1189-1409
    """

    name: str
    forward_compute_time: int
    forward_comm: str  # "NONE", "ALLREDUCE", "ALLGATHER_DP_EP", etc.
    forward_comm_size: int
    backward_compute_time: int
    backward_comm: str
    backward_comm_size: int
    dp_compute_time: int
    dp_comm: str
    dp_comm_size: int
    process_time: int  # field[11], usually constant, not used by builder


class AicbParser:
    """Parser for AICB workload files (.txt)."""

    def parse(self, file_path: str) -> tuple[AicbHeader, list[AicbWorkItem]]:
        """Parse an AICB workload file into header + work items.

        Supports HYBRID_TRANSFORMER_FWD_IN_BCKWD format.
        """
        lines = Path(file_path).read_text().strip().split("\n")
        header = self._parse_header(lines[0])
        count = int(lines[1])
        items = self._parse_items(lines, start=2, count=count)
        return header, items

    def parse_micro(self, file_path: str) -> list[AicbWorkItem]:
        """Parse a MICRO format benchmark file.

        MICRO files have no parallelism header — just "MICRO" as line 1.
        """
        lines = Path(file_path).read_text().strip().split("\n")
        count = int(lines[1])
        return self._parse_items(lines, start=2, count=count)

    @staticmethod
    def parse_comm_type(comm_str: str, default_context: str = "tp") -> tuple[str, str]:
        """Parse a comm type string into (base_type, parallelism_context).

        Suffix mapping per astra-sim Workload.cc lines 1329-1369:
            no suffix  → default_context  (typically tp for forward/backward fields)
            _DP        → dp
            _EP        → ep
            _DP_EP     → dp_ep

        Examples:
            "ALLGATHER_DP_EP" → ("ALLGATHER", "dp_ep")
            "ALLREDUCE"        → ("ALLREDUCE", "tp")  # default
            "ALLTOALL"         → ("ALLTOALL", "tp")   # no suffix → TP
            "ALLTOALL_EP"      → ("ALLTOALL", "ep")   # _EP → EP
            "NONE"             → ("NONE", "")
        """
        if not comm_str or comm_str == "NONE":
            return ("NONE", "")

        # Check suffixes longest-first to avoid partial matches
        if comm_str.endswith("_DP_EP"):
            return (comm_str[:-6], "dp_ep")
        if comm_str.endswith("_DP"):
            return (comm_str[:-3], "dp")
        if comm_str.endswith("_EP"):
            return (comm_str[:-3], "ep")

        # No suffix → default context (tp for fwd/bwd, dp for dp_comm field)
        return (comm_str, default_context)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_header(self, header_line: str) -> AicbHeader:
        """Parse the header line of an AICB workload file.

        The header is space-separated. Example:
          HYBRID_TRANSFORMER_FWD_IN_BCKWD model_parallel_NPU_group: 2 ep: 16
            pp: 12 vpp: 8 ga: 24 all_gpus: 9216
            checkpoints: 0 checkpoint_initiates: 0 pp_comm 50331648

        Note: pp_comm may appear without a colon.
        """
        tokens = header_line.split()

        tp = ep = pp = vpp = ga = all_gpus = 0
        pp_comm_size = 0

        i = 1  # skip the parallelism-policy token at index 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "model_parallel_NPU_group:":
                tp = int(tokens[i + 1])
                i += 2
            elif tok == "ep:":
                ep = int(tokens[i + 1])
                i += 2
            elif tok == "pp:":
                pp = int(tokens[i + 1])
                i += 2
            elif tok == "vpp:":
                vpp = int(tokens[i + 1])
                i += 2
            elif tok == "ga:":
                ga = int(tokens[i + 1])
                i += 2
            elif tok == "all_gpus:":
                all_gpus = int(tokens[i + 1])
                i += 2
            elif tok in ("pp_comm", "pp_comm:"):
                pp_comm_size = int(tokens[i + 1])
                i += 2
            elif tok in ("checkpoints:", "checkpoint_initiates:"):
                count = int(tokens[i + 1])
                i += 2 + count  # skip count value + layer IDs
            else:
                i += 1

        return AicbHeader(
            tp=tp,
            ep=ep,
            pp=pp,
            vpp=vpp,
            ga=ga,
            all_gpus=all_gpus,
            pp_comm_size=pp_comm_size,
        )

    def _parse_items(self, lines: list[str], start: int, count: int) -> list[AicbWorkItem]:
        """Parse item lines starting at *start*, up to *count* items."""
        items: list[AicbWorkItem] = []
        for i in range(start, min(start + count, len(lines))):
            item = self._parse_item_line(lines[i])
            if item is not None:
                items.append(item)
        return items

    @staticmethod
    def _parse_item_line(line: str) -> AicbWorkItem | None:
        """Parse a single whitespace-separated item line.

        Returns None if the line has fewer than 12 fields.
        Field 11 (process_time / wg_update_time) is usually constant and not
        used by the builder, but we parse it for completeness.
        """
        fields = line.split()
        if len(fields) < 12:
            return None

        return AicbWorkItem(
            name=fields[0],
            forward_compute_time=int(fields[2]),
            forward_comm=fields[3],
            forward_comm_size=int(fields[4]),
            backward_compute_time=int(fields[5]),
            backward_comm=fields[6],
            backward_comm_size=int(fields[7]),
            dp_compute_time=int(fields[8]),
            dp_comm=fields[9],
            dp_comm_size=int(fields[10]),
            process_time=int(fields[11]),
        )

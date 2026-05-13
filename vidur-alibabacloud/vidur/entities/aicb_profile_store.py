"""
AICB Profile Store - load and query per-layer AICB CSV profiling data.

Loads all matching CSV files from a directory at initialization time,
then provides nearest-match lookup by (phase, batch_size, seq_length).

Reference: simai-flow-scheduler/src/workload_generator/inference_profile.py
"""

import csv
import os
import re
from pathlib import Path
from typing import Dict, Optional


class AicbProfileStore:
    """
    Load AICB per-layer CSV profiling data, query by (phase, bs, seq) with nearest-match.

    CSV format (tab-separated): layer_id, layer_name, comp_time (ns), comm_size (bytes)
    Filename format: vidur-{model}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs{bs}-seq{seq}-{phase}.csv
    """

    def __init__(self, dir_path: str, tp: int, ep: int, pp: int):
        self._profiles: Dict[str, dict] = {}
        # key: "{phase}_bs{bs}_seq{seq}"
        # value: {layer_id: {layer_name: {"comp_time": ns, "comm_size": bytes}}}
        self._tp = tp
        self._ep = ep
        self._pp = pp

        loaded = self._load_directory(dir_path)
        print(f"[AicbProfileStore] Loaded {loaded} profiles from {dir_path} "
              f"(tp={tp}, ep={ep}, pp={pp})")
        if loaded > 0:
            print(f"[AicbProfileStore] Keys: {self.list_profiles()}")

    def _load_directory(self, dir_path: str) -> int:
        path = Path(dir_path)
        if not path.is_dir():
            print(f"[AicbProfileStore] Warning: not a directory: {dir_path}")
            return 0

        loaded = 0
        for file_path in sorted(path.glob("*.csv")):
            parsed = self._parse_filename(file_path.stem)
            if parsed is None:
                continue

            key, file_tp, file_ep, file_pp = parsed

            # Filter by parallelism parameters (exact match)
            if file_tp != self._tp or file_ep != self._ep or file_pp != self._pp:
                continue

            try:
                data = self._load_csv(str(file_path))
                if data:
                    self._profiles[key] = data
                    loaded += 1
            except Exception as e:
                print(f"[AicbProfileStore] Warning: skipping {file_path.name}: {e}")

        return loaded

    @staticmethod
    def _parse_filename(stem: str) -> Optional[tuple]:
        """
        Parse Vidur-format filename.
        Format: vidur-{model}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs{bs}-seq{seq}-{phase}
        Returns: (key, tp, ep, pp) or None
        """
        pattern = (
            r"^vidur-.+-world_size\d+-tp(\d+)-pp(\d+)-ep(\d+)"
            r"-bs(\d+)-seq(\d+)-(prefill|decode)$"
        )
        match = re.match(pattern, stem)
        if not match:
            return None

        tp, pp, ep, bs, seq, phase = match.groups()
        key = f"{phase}_bs{bs}_seq{seq}"
        return (key, int(tp), int(ep), int(pp))

    @staticmethod
    def _load_csv(csv_path: str) -> dict:
        """
        Parse a tab-separated CSV file.
        Returns: {layer_id: {layer_name: {"comp_time": ns, "comm_size": bytes}}}
        """
        data: Dict[int, Dict[str, Dict[str, float]]] = {}

        with open(csv_path, newline="") as f:
            # Detect delimiter
            first_line = f.readline()
            f.seek(0)
            delimiter = "\t" if "\t" in first_line else ","

            reader = csv.DictReader(f, delimiter=delimiter)
            if reader.fieldnames is None:
                return {}

            # Normalize headers
            reader.fieldnames = [h.strip() for h in reader.fieldnames]

            for row in reader:
                row = {k.strip(): v for k, v in row.items()}

                required = ("layer_id", "layer_name", "comp_time", "comm_size")
                if any(c not in row for c in required):
                    continue

                layer_id = int(row["layer_id"])
                layer_name = row["layer_name"].strip()

                if layer_id not in data:
                    data[layer_id] = {}
                data[layer_id][layer_name] = {
                    "comp_time": float(row["comp_time"]),
                    "comm_size": float(row["comm_size"]),
                }

        return data

    def get_profile(self, phase: str, bs: int, seq: int) -> dict:
        """
        Find the best matching profile for (phase, bs, seq).

        Selection priority:
          1. Exact match on phase + bs + seq
          2. Same phase + bs, nearest seq
          3. Same phase, nearest bs, any seq

        Raises KeyError if no profiles loaded for the given phase.
        """
        candidates = self._parse_keys_for_phase(phase)
        if not candidates:
            raise KeyError(
                f"No profiles loaded for phase '{phase}'. "
                f"Available: {self.list_profiles()}"
            )

        # 1. Exact match
        exact_key = f"{phase}_bs{bs}_seq{seq}"
        if exact_key in candidates:
            return self._profiles[exact_key]

        # 2. Same phase + bs, nearest seq
        same_bs = [(k, b, s) for k, b, s in candidates if b == bs]
        if same_bs:
            best = min(same_bs, key=lambda x: abs(x[2] - seq))
            print(f"[AicbProfileStore] Nearest match for {phase} bs={bs} seq={seq}: "
                  f"{best[0]} (delta_seq={abs(best[2] - seq)})")
            return self._profiles[best[0]]

        # 3. Same phase, nearest bs
        best = min(candidates, key=lambda x: abs(x[1] - bs))
        print(f"[AicbProfileStore] Nearest match for {phase} bs={bs} seq={seq}: "
              f"{best[0]} (delta_bs={abs(best[1] - bs)})")
        return self._profiles[best[0]]

    def _parse_keys_for_phase(self, phase: str) -> list:
        """Parse loaded keys into (key, bs, seq) tuples for a given phase."""
        result = []
        prefix = f"{phase}_bs"
        for key in self._profiles:
            if not key.startswith(prefix):
                continue
            try:
                rest = key[len(prefix):]
                bs_str, seq_str = rest.split("_seq")
                result.append((key, int(bs_str), int(seq_str)))
            except (ValueError, IndexError):
                continue
        return result

    def list_profiles(self) -> list:
        """Return list of all loaded profile keys."""
        return list(self._profiles.keys())

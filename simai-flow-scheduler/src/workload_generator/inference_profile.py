"""
Inference profile store - load and query per-layer AICB CSV profiling data for inference.

CSV format (tab-separated):
    layer_id    layer_name    comp_time    comm_size

where comp_time is in nanoseconds and comm_size is in bytes.

Usage:
    store = InferenceProfileStore(tp=2, ep=1, pp=1)
    store.load_directory("results/workload/")
    profiles = store.get_profile_for_batch("prefill", bs=4, seq=4096)
"""

import csv
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerProfile:
    """Per-layer profiling data from AICB CSV."""
    layer_id: int
    layer_name: str        # "attention", "mlp", "moe"
    comp_time_us: int      # compute time in microseconds
    comm_size_bytes: int   # communication size in bytes


class InferenceProfileStore:
    """
    Load AICB per-layer CSV profiling data, query by key.

    Keys follow naming convention: "{phase}_bs{batch_size}_seq{seq_length}"
    e.g. "prefill_bs4_seq4096", "decode_bs4_seq1".

    Optionally filter loaded profiles by parallelism parameters (tp/ep/pp).
    """

    def __init__(
        self,
        tp: Optional[int] = None,
        ep: Optional[int] = None,
        pp: Optional[int] = None,
    ):
        """
        Initialize profile store with optional parallelism filters.

        Args:
            tp: Tensor parallelism degree (filter profiles by tp).
            ep: Expert parallelism degree (filter profiles by ep).
            pp: Pipeline parallelism degree (filter profiles by pp).
        """
        self._profiles: dict[str, list[LayerProfile]] = {}
        self._tp = tp
        self._ep = ep
        self._pp = pp

    def load_directory(self, dir_path: str) -> int:
        """
        Load all CSV/TSV files from a directory, parsing keys from filenames.

        Expected filename format: {phase}_bs{bs}_seq{seq}_tp{tp}_ep{ep}_pp{pp}.csv
        Example: prefill_bs4_seq4096_tp2_ep1_pp1.csv

        Files not matching the format or parallelism filters are skipped.

        Args:
            dir_path: Directory containing profile CSV files.

        Returns:
            Number of profiles loaded.
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"Not a directory: {dir_path}")

        loaded = 0
        for file_path in path.glob("*.csv"):
            parsed = self._parse_filename(file_path.stem)
            if parsed is None:
                continue  # Skip files that don't match format

            key, tp, ep, pp = parsed

            # Filter by parallelism parameters
            if self._tp is not None and tp != self._tp:
                continue
            if self._ep is not None and ep != self._ep:
                continue
            if self._pp is not None and pp != self._pp:
                continue

            try:
                self.load(str(file_path), key)
                loaded += 1
            except (ValueError, KeyError) as e:
                # Skip invalid files but continue loading others
                print(f"Warning: skipping {file_path.name}: {e}")

        return loaded

    def _parse_filename(self, stem: str) -> Optional[tuple[str, int, int, int]]:
        """
        Parse profile key and parallelism params from filename.

        Supports two formats:
        1. New: {phase}_bs{bs}_seq{seq}_tp{tp}_ep{ep}_pp{pp}
        2. Vidur: vidur-{model}-world_size{ws}-tp{tp}-pp{pp}-ep{ep}-bs{bs}-seq{seq}-{phase}

        Returns:
            (key, tp, ep, pp) where key is "{phase}_bs{bs}_seq{seq}",
            or None if format doesn't match.
        """
        # Try new format first: prefill_bs4_seq4096_tp2_ep1_pp1
        pattern = r"^(prefill|decode)_bs(\d+)_seq(\d+)_tp(\d+)_ep(\d+)_pp(\d+)$"
        match = re.match(pattern, stem)
        if match:
            phase, bs, seq, tp, ep, pp = match.groups()
            key = f"{phase}_bs{bs}_seq{seq}"
            return (key, int(tp), int(ep), int(pp))

        # Try Vidur format: vidur-DeepSeek-671B-world_size1-tp1-pp1-ep1-bs1-seq512-prefill
        pattern = r"^vidur-.+-world_size\d+-tp(\d+)-pp(\d+)-ep(\d+)-bs(\d+)-seq(\d+)-(prefill|decode)$"
        match = re.match(pattern, stem)
        if match:
            tp, pp, ep, bs, seq, phase = match.groups()
            key = f"{phase}_bs{bs}_seq{seq}"
            return (key, int(tp), int(ep), int(pp))

        return None

    def load(self, csv_path: str, key: str) -> None:
        """
        Load a tab-separated CSV file and store profiles under ``key``.

        Args:
            csv_path: Path to the CSV/TSV file.
            key: Identifier for this profile (e.g. "prefill_bs4_seq4096").

        Raises:
            FileNotFoundError: If csv_path does not exist.
            ValueError: If required columns are missing.
            KeyError: If key is already loaded.
        """
        if key in self._profiles:
            raise KeyError(f"Profile key '{key}' already loaded")

        profiles = []
        with open(csv_path, newline='') as f:
            # Try TSV first (Vidur default), fall back to comma-separated CSV
            content = f.read()
            if not content.strip():
                raise ValueError(f"Empty file: {csv_path}")

        # Re-open to parse
        with open(csv_path, newline='') as f:
            delimiter = '\t'
            # Detect delimiter: if first line has tabs use TSV, else comma
            first_line = f.readline()
            f.seek(0)
            if '\t' not in first_line:
                delimiter = ','

            reader = csv.DictReader(f, delimiter=delimiter)

            if reader.fieldnames is None:
                raise ValueError(f"Cannot read headers from: {csv_path}")

            # Normalize header names (strip whitespace)
            reader.fieldnames = [h.strip() for h in reader.fieldnames]

            for row in reader:
                # Normalize keys too
                row = {k.strip(): v for k, v in row.items()}

                if any(c not in row for c in ('layer_id', 'layer_name', 'comp_time', 'comm_size')):
                    missing = [c for c in ('layer_id', 'layer_name', 'comp_time', 'comm_size') if c not in row]
                    raise ValueError(f"Missing columns {missing} in {csv_path}")

                profiles.append(LayerProfile(
                    layer_id=int(row['layer_id']),
                    layer_name=row['layer_name'].strip(),
                    comp_time_us=int(float(row['comp_time']) / 1000),  # ns → us
                    comm_size_bytes=int(float(row['comm_size'])),
                ))

        if not profiles:
            raise ValueError(f"No profiling data found in: {csv_path}")

        self._profiles[key] = profiles

    def get_profile(self, key: str) -> list[LayerProfile]:
        """
        Return profiles for a given key.

        Args:
            key: Profile identifier (e.g. "prefill_bs4_seq4096").

        Raises:
            KeyError: If key is not found.
        """
        if key not in self._profiles:
            raise KeyError(
                f"Profile key '{key}' not found. "
                f"Available keys: {list(self._profiles.keys())}"
            )
        return self._profiles[key]

    def list_profiles(self) -> list[str]:
        """Return list of all loaded profile keys."""
        return list(self._profiles.keys())

    def get_profile_for_batch(
        self, phase: str, bs: int, seq: int,
    ) -> list[LayerProfile]:
        """Find the best matching profile for a batch.

        Searches loaded profiles for the closest match by (phase, bs, seq).
        Key format: "{phase}_bs{batch_size}_seq{seq_length}".

        Selection priority:
        1. Exact match on phase + bs + seq
        2. Same phase + bs, nearest seq (lower first)
        3. Same phase, nearest bs, any seq

        Args:
            phase: "prefill" or "decode".
            bs: Batch size (number of requests).
            seq: Total tokens for prefill, or KV cache seq_len for decode.

        Raises:
            KeyError: If no profile matches the given phase.
        """
        candidates = self._parse_keys_for_phase(phase)
        if not candidates:
            raise KeyError(
                f"No profiles loaded for phase '{phase}'. "
                f"Available: {self.list_profiles()}"
            )

        # Exact match
        exact_key = f"{phase}_bs{bs}_seq{seq}"
        if exact_key in candidates:
            return self._profiles[exact_key]

        # Same phase + bs, nearest seq
        same_bs = [(k, b, s) for k, b, s in candidates if b == bs]
        if same_bs:
            best = min(same_bs, key=lambda x: abs(x[2] - seq))
            return self._profiles[best[0]]

        # Same phase, nearest bs
        best = min(candidates, key=lambda x: abs(x[1] - bs))
        return self._profiles[best[0]]

    def _parse_keys_for_phase(self, phase: str) -> list[tuple[str, int, int]]:
        """Parse loaded keys into (key, bs, seq) tuples for a given phase."""
        result = []
        prefix = f"{phase}_bs"
        for key in self._profiles:
            if not key.startswith(prefix):
                continue
            try:
                rest = key[len(prefix):]  # e.g. "1_seq512"
                bs_str, seq_str = rest.split("_seq")
                result.append((key, int(bs_str), int(seq_str)))
            except (ValueError, IndexError):
                continue
        return result

    def get_layers_by_name(self, key: str, layer_name: str) -> list[LayerProfile]:
        """
        Return profiles for a specific layer type within a profile.

        Args:
            key: Profile identifier.
            layer_name: Layer type filter (e.g. "attention", "mlp", "moe").

        Returns:
            List of LayerProfile matching the given layer_name.
        """
        all_profiles = self.get_profile(key)
        return [p for p in all_profiles if p.layer_name == layer_name]

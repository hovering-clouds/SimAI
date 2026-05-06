"""
Inference profile store - load and query per-layer AICB CSV profiling data for inference.

CSV format (tab-separated):
    layer_id    layer_name    comp_time    comm_size

where comp_time is in nanoseconds and comm_size is in bytes.

Usage:
    store = InferenceProfileStore()
    store.load("results/workload/vidur-DeepSeek-671B-...-bs4-seq4096-prefill.csv",
               "prefill_bs4_seq4096")
    profiles = store.get_profile("prefill_bs4_seq4096")
"""

import csv
from dataclasses import dataclass, field


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
    """

    def __init__(self):
        self._profiles: dict[str, list[LayerProfile]] = {}

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

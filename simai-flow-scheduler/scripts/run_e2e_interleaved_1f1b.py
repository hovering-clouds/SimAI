"""Run an isolated Interleaved 1F1B workload/serializer E2E.

Usage:
    python scripts/run_e2e_interleaved_1f1b.py
    python scripts/run_e2e_interleaved_1f1b.py --vpp 2 --aicb <path>
"""

from utils.pipeline_e2e_common import run_pipeline_e2e_cli


if __name__ == "__main__":
    run_pipeline_e2e_cli("interleaved_1f1b")

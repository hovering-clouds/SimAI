"""Run an isolated Zero Bubble workload/serializer E2E.

Usage:
    python scripts/run_e2e_zero_bubble.py
    python scripts/run_e2e_zero_bubble.py --aicb <path> --topo <path>
"""

from pipeline_e2e_common import run_pipeline_e2e_cli


if __name__ == "__main__":
    run_pipeline_e2e_cli("zero_bubble")

"""Run the isolated basic-Chimera workload/serializer E2E.

Usage:
    python scripts/run_e2e_bidirectional.py
    python scripts/run_e2e_bidirectional.py --aicb <path> --topo <path>

The ``bidirectional`` mode name is retained for compatibility.
"""

from pipeline_e2e_common import run_pipeline_e2e_cli


if __name__ == "__main__":
    run_pipeline_e2e_cli("bidirectional")

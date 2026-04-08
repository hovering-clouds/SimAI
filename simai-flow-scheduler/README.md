# SimAI Flow Scheduler

Fine-grained P2P flow scheduling simulator for AI training clusters.

## Overview

This project extends SimAI with support for fine-grained point-to-point (P2P) flow scheduling, enabling research into bandwidth allocation and priority scheduling for AI training workloads.

### Key Features

- **Fine-grained workload representation**: Expands collective communications into P2P flows with dependencies
- **Multi-job support**: Multiple training jobs sharing network topology
- **Pluggable scheduling policies**: Priority, bandwidth allocation, WFQ, etc.
- **Dual execution backends**: Analytical (fast) and NS-3 (high-fidelity)

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 1: Input (Topology + Job descriptions)       │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2: Workload Generator (Collective→P2P)       │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3: P2P Workload JSON (Intermediate Rep.)     │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  Layer 4: Executor (Scheduler + Analytical/NS-3)    │
└─────────────────────────────────────────────────────┘
```

## Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with plotting dependencies
pip install -e ".[dev,plot]"
```

## Quick Start

```python
from simai_flow_scheduler import (
    RingAllReduceExpander,
    P2PWorkload,
    WorkloadWriter,
)

# Create expander
expander = RingAllReduceExpander()

# Expand a collective operation
ranks = [0, 1, 2, 3]
data_size = 1024 * 1024 * 1024  # 1GB
flows = expander.expand_allreduce(ranks, data_size, job_id=0)

# Create workload
workload = P2PWorkload(
    version="1.0",
    meta={"num_jobs": 1, "num_nodes": 4},
    jobs=[{"job_id": 0, "assigned_nodes": [0, 1, 2, 3]}],
    tasks=[f.to_task() for f in flows],
)

# Write to file
writer = WorkloadWriter()
writer.write(workload, "output/workload.json")
```

## Project Structure

```
simai-flow-scheduler/
├── src/
│   ├── workload_format/      # JSON Schema, validator, reader/writer
│   └── workload_generator/   # Collective→P2P expanders
├── tests/                    # Unit tests
├── inputs/
│   ├── topologies/           # Network topology JSON files
│   └── jobs/                 # Job configuration JSON files
├── examples/
│   ├── single_job/           # Single job examples
│   └── multi_job/            # Multi-job examples
├── docs/                     # Documentation
└── specs/                    # Design specifications
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
# Format code
black src/ tests/

# Lint code
ruff check src/ tests/
```

### Type Checking

```bash
mypy src/
```

## License

Apache 2.0

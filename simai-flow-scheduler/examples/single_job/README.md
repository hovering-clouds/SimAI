# Single Job Example

This example demonstrates how to convert an AICB training workload to a P2P Workload.

## Files

- `sample_workload.txt` - Sample AICB workload (2 layers, TP=2, 4 GPUs)
- `generate_workload.py` - Python script to generate P2P Workload
- `single_job_workload.json` - Generated P2P Workload (output)

## Usage

```bash
cd simai-flow-scheduler
python examples/single_job/generate_workload.py
```

## What it does

1. **Parse** the AICB workload file (`sample_workload.txt`)
2. **Define** a job with parallelism configuration:
   - TP=2, DP=2, PP=1, EP=1 (4 GPUs total)
3. **Build** P2P Workload using `WorkloadBuilder`:
   - Expands ALLGATHER/REDUCESCATTER to per-rank P2P flows
   - Creates compute tasks for each rank
   - Builds dependency chains (forward → backward)
4. **Validate** the output workload structure
5. **Write** to JSON file

## Output Structure

The generated `single_job_workload.json` contains:
- ~40 tasks (24 compute + 16 flow) for 2 layers
- Per-rank compute tasks (4 ranks × 2 phases × 2 layers = 16+, plus dependencies)
- Ring AllGather and ReduceScatter flow expansions
- Proper DAG dependencies

## Customization

To use your own AICB workload:
1. Replace `sample_workload.txt` with your `.txt` file
2. Update the parallelism config in `generate_workload.py`
3. Run the script again

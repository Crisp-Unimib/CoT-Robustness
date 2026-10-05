# Leonardo HPC vLLM Scripts

Scripts for running vLLM inference on Leonardo HPC (CINECA) with Singularity containers.

## Quick Start

```bash
# Copy scripts to Leonardo
scp leonardo_scripts/* leonardo:/path/to/workdir/

# Submit 2-node data-parallel job
sbatch run_vllm_singularity.sbatch

# Check status
squeue --me

# After job starts, test from your local machine
python generate_math/generate_rollouts.py -p vLLM -m "Qwen/Qwen3-30B-A3B-Thinking-2507" ...
```

---

## Configuration Options

### Single Node (Recommended for simplicity)

**Hardware**: 1 node × 4× A100 64GB (TP=4)

| Setting | Value | Notes |
|---------|-------|-------|
| `--tensor-parallel-size` | 4 | All 4 GPUs for one model |
| `--gpu-memory-utilization` | 0.85 | Safe for CUDA graphs |
| `--max-num-seqs` | 256 | High concurrency |
| `--max-num-batched-tokens` | 32768 | Large batches |
| `--enable-prefix-caching` | Yes | 96%+ hit rate |
| `--enable-chunked-prefill` | Yes | Better long-context |
| `--disable-custom-all-reduce` | Yes | Required on A100s |

**Performance**: ~4,300 tok/s peak, 100 concurrent requests

### 2-Node Data Parallel (Current)

**Hardware**: 2 nodes × 4× A100 64GB (DP=2, TP=4 per node)

| Setting | Value | Notes |
|---------|-------|-------|
| `--data-parallel-size` | 2 | 1 model replica per node |
| `--data-parallel-size-local` | 1 | Force 1 replica per node |
| `--tensor-parallel-size` | 4 | 4 GPUs per replica |
| `--data-parallel-backend` | ray | Ray handles coordination |

**Performance**: ~4,000 tok/s combined (Ray overhead reduces per-engine throughput)

---

## Performance History

| Date | Config | Peak Throughput | Notes |
|------|--------|-----------------|-------|
| Initial | 1 node, default | ~1,900 tok/s | Baseline |
| Optimized | 1 node, tuned | **~4,300 tok/s** | Best single-node |
| 2-node DP | 2 nodes, Ray | ~4,000 tok/s | Ray overhead negates gains |

### Key Findings

- **Single node is optimal** for this workload due to Ray inter-node overhead
- Prefix caching achieves 96-97% hit rate on rollout workloads
- KV cache usage stays below 30% even at high concurrency
- CUDA graphs provide significant speedup (don't use `--enforce-eager`)

---

## Lessons Learned

### What Worked ✅

1. **Direct vLLM execution** - No Ray overhead on single node
2. **High concurrency** - `--max-num-seqs 256` handles 100+ concurrent requests
3. **Large batch tokens** - `--max-num-batched-tokens 32768` improves throughput
4. **Prefix caching** - Essential for rollout workloads with shared prompts
5. **CUDA graphs** - Keep enabled for best performance

### What Didn't Work ❌

1. **2-node DP with Ray** - Inter-node overhead negates throughput gains
2. **`--gpu-memory-utilization > 0.85`** - OOM during CUDA graph capture
3. **Custom all-reduce** - Crashes with "CUDA error 455" on A100s
4. **`--num-scheduler-steps`** - Parameter doesn't exist in vLLM 0.12

### vLLM 0.12 Notes

- V0 engine was **removed** in vLLM 0.11+
- `--data-parallel-size` now works with V1 + Ray
- `--data-parallel-size-local` controls per-node replica count
- Setting `VLLM_USE_V1=0` has **no effect** (V0 code is gone)

---

## Troubleshooting

### CUDA Error 455 (custom_all_reduce)
```
Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:455 'invalid argument'
```
**Fix**: Add `--disable-custom-all-reduce`

### OOM During CUDA Graph Capture
```
torch.AcceleratorError: CUDA error: out of memory (during capture_model)
```
**Fix**: Reduce `--gpu-memory-utilization` to 0.85

### Ray DP: "Not enough resources to allocate N placement groups"
**Fix**: Add `--data-parallel-size-local 1` to distribute replicas across nodes

### Ray DP: Worker node can't find launcher script
**Fix**: Script must be in shared filesystem (`$SLURM_SUBMIT_DIR`), not `/tmp`

---

## Future Work

- [ ] Test 2 independent vLLM servers with client-side load balancing (may be faster than Ray DP)
- [ ] Explore speculative decoding for MoE model
- [ ] Try higher memory utilization with `--enforce-eager` fallback

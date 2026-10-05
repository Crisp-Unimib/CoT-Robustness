#!/bin/bash
#SBATCH --job-name=moe_tune
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=<your-account>
#SBATCH --output=moe_tune_%j.out
#SBATCH --error=moe_tune_%j.err

# MoE Kernel Tuning for Qwen3-Next on Leonardo
# This generates optimized Triton kernel configs for your GPUs

set -e  # Exit on any error

module purge
module load gcc/12.2.0
module load cuda/12.2

cd $SLURM_SUBMIT_DIR

export HF_HOME=$WORK/$USER/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

# --- MODEL RESOLUTION ---
MODEL_REPO="Qwen/Qwen3-Next-80B-A3B-Thinking"
CACHE_MODEL_NAME="models--${MODEL_REPO//\//--}"
MODEL_PATH=$(find $HF_HUB_CACHE/$CACHE_MODEL_NAME/snapshots -maxdepth 1 -mindepth 1 -type d | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find model snapshot in $HF_HUB_CACHE/$CACHE_MODEL_NAME"
    echo "Please ensure the model is downloaded and cached."
    exit 1
fi

echo "Using resolved model path: $MODEL_PATH"

# Output directory for tuned configs
SAVE_DIR="$WORK/$USER/moe_configs"
mkdir -p $SAVE_DIR

CONTAINER_IMAGE="$SLURM_SUBMIT_DIR/vllm-openai_latest.sif"

# Use local vLLM repo (must be cloned in submission directory)
VLLM_REPO_DIR="$SLURM_SUBMIT_DIR/vllm"

if [ ! -d "$VLLM_REPO_DIR" ]; then
    echo "Error: Local vLLM repository not found at $VLLM_REPO_DIR"
    echo "Please run: git clone --depth 1 https://github.com/vllm-project/vllm.git vllm"
    echo "inside the submission directory before submitting this job."
    exit 1
fi

# Copy ONLY the benchmark script to /tmp to avoid importing local vllm folder
BENCHMARK_SCRIPT="/tmp/benchmark_moe_$SLURM_JOB_ID.py"
cp "$VLLM_REPO_DIR/benchmarks/kernels/benchmark_moe.py" "$BENCHMARK_SCRIPT"

# Build bind paths (all variables now defined)
BIND_PATHS="$HF_HOME:$HF_HOME,/tmp:/tmp,$SAVE_DIR:$SAVE_DIR,$MODEL_PATH:$MODEL_PATH"

echo "=============================================="
echo "MoE Kernel Tuning Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Save directory: $SAVE_DIR"
echo "=============================================="

singularity exec --nv \
    --bind $BIND_PATHS \
    --env HF_HOME=$HF_HOME \
    --env HF_HUB_CACHE=$HF_HUB_CACHE \
    --env HF_HUB_OFFLINE=1 \
    --env TRITON_CACHE_DIR=/tmp/triton_cache_$USER \
    --env TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER \
    --env TORCH_HOME=/tmp/torch_cache_$USER \
    --env XDG_CACHE_HOME=/tmp/cache_$USER \
    --pwd /tmp \
    $CONTAINER_IMAGE \
    python3 $BENCHMARK_SCRIPT \
    --model $MODEL_PATH \
    --tp-size 4 \
    --dtype auto \
    --tune \
    --save-dir $SAVE_DIR \
    --trust-remote-code

# Cleanup the copied script
rm -f "$BENCHMARK_SCRIPT"

echo "=============================================="
echo "Tuning complete!"
echo "Config files saved to: $SAVE_DIR"
echo ""
echo "To use the tuned configs, add this to launch_vllm_single.sh:"
echo "  export VLLM_TUNED_CONFIG_FOLDER=$SAVE_DIR"
echo "=============================================="

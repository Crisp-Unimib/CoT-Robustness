#!/bin/bash
# Single-node vLLM launcher for Leonardo HPC with Singularity
# Optimal TP=4 configuration (no Ray, no DP)

echo "[DEBUG $(date)] launch_vllm_single.sh starting on $(hostname)"
echo "[DEBUG] WORK=$WORK, USER=$USER"

export HF_HOME=$WORK/$USER/cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1

# Use tuned MoE kernel configurations from benchmark
export VLLM_TUNED_CONFIG_FOLDER=$WORK/$USER/moe_configs

# Redirect compilation caches to /tmp (node-local) to avoid multi-node corruption
export VLLM_CACHE_ROOT=/tmp/vllm_cache_$USER
export TORCH_HOME=/tmp/torch_cache_$USER
export XDG_CACHE_HOME=/tmp/cache_$USER
export TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_$USER
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
mkdir -p $VLLM_CACHE_ROOT $TORCH_HOME $XDG_CACHE_HOME $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR 2>/dev/null || true

# --- MODEL CONFIGURATION ---
MODEL_REPO="Qwen/Qwen3-Next-80B-A3B-Thinking"
CACHE_MODEL_NAME="models--${MODEL_REPO//\//--}"
MODEL_PATH=$(find $HF_HUB_CACHE/$CACHE_MODEL_NAME/snapshots -maxdepth 1 -mindepth 1 -type d | head -n 1)

if [ -z "$MODEL_PATH" ]; then
    echo "Error: Could not find model snapshot in $HF_HUB_CACHE/$CACHE_MODEL_NAME"
    MODEL_PATH=$MODEL_REPO
fi

echo "Using model: $MODEL_PATH"

BIND_PATHS="$HF_HOME:$HF_HOME,/tmp:/tmp,$VLLM_TUNED_CONFIG_FOLDER:$VLLM_TUNED_CONFIG_FOLDER"
if [[ -d "$MODEL_PATH" ]]; then
    BIND_PATHS="$BIND_PATHS,$MODEL_PATH:$MODEL_PATH"
fi

CONTAINER_IMAGE="$SLURM_SUBMIT_DIR/vllm-openai_latest.sif"

echo "=============================================="
echo "Starting vLLM API server (TP=4, Single Node)"
echo "=============================================="
echo "Optimization settings:"
echo "  - GPU memory utilization: 0.85"
echo "  - Max sequences: 128"
echo "  - Max batched tokens: 16384"
echo "  - MTP Speculative decoding: enabled (2 tokens)"
echo "  - Chunked prefill: disabled (required for MTP)"
echo "=============================================="

# SINGLE-NODE TP=4 CONFIGURATION with MTP speculative decoding
singularity exec --nv --bind $BIND_PATHS \
    --env PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF \
    --env HF_HOME=$HF_HOME \
    --env HF_HUB_CACHE=$HF_HUB_CACHE \
    --env HF_HUB_OFFLINE=$HF_HUB_OFFLINE \
    --env VLLM_CACHE_ROOT=$VLLM_CACHE_ROOT \
    --env VLLM_TUNED_CONFIG_FOLDER=$VLLM_TUNED_CONFIG_FOLDER \
    --env TORCH_HOME=$TORCH_HOME \
    --env XDG_CACHE_HOME=$XDG_CACHE_HOME \
    --env TORCHINDUCTOR_CACHE_DIR=$TORCHINDUCTOR_CACHE_DIR \
    --env TRITON_CACHE_DIR=$TRITON_CACHE_DIR \
    $CONTAINER_IMAGE \
    python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --served-model-name $MODEL_REPO \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --dtype bfloat16 \
    --max-model-len 40960 \
    --gpu-memory-utilization 0.85 \
    --speculative-config '{"method": "qwen3_next_mtp", "num_speculative_tokens": 3}' \
    --no-enable-chunked-prefill \
    --max-num-seqs 128 \
    --max-num-batched-tokens 40960 \
    --disable-custom-all-reduce \
    --trust-remote-code \
    --compilation_config.cudagraph_mode=PIECEWISE

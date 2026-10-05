#!/bin/bash
#SBATCH --job-name=vllm-ind
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --time=05:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=<your-account>
#SBATCH --output=vllm_%j.out
#SBATCH --error=vllm_%j.err

# Independent vLLM Servers (No Ray DP overhead)
# Each node runs its own vLLM instance for true Nx throughput
# Uses SLURM_NNODES (set automatically by SLURM based on --nodes above)

# Load necessary modules
module purge
module load gcc/12.2.0
module load cuda/12.2

cd $SLURM_SUBMIT_DIR

# --- MEMORY MANAGEMENT ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- NODE IDENTIFICATION ---
NODELIST=$(scontrol show hostnames $SLURM_JOB_NODELIST)
NODELIST_ARRAY=($NODELIST)

echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_NNODES"
for i in $(seq 0 $((SLURM_NNODES - 1))); do
    echo "Node $i: ${NODELIST_ARRAY[$i]}"
done
echo "=============================================="
echo ""
echo "=============================================="
echo "SSH TUNNEL COMMAND (copy and paste):"
SSH_CMD="ssh"
for i in $(seq 0 $((SLURM_NNODES - 1))); do
    PORT=$((8000 + i))
    SSH_CMD="$SSH_CMD -L $PORT:${NODELIST_ARRAY[$i]}:8000"
done
SSH_CMD="$SSH_CMD $USER@login.leonardo.cineca.it"
echo "$SSH_CMD"
echo "=============================================="
echo ""

# Create launcher script
LAUNCHER_SCRIPT="$SLURM_SUBMIT_DIR/node_launcher_$SLURM_JOB_ID.sh"

cat > "$LAUNCHER_SCRIPT" << 'LAUNCHER_EOF'
#!/bin/bash

NODE_RANK=$SLURM_PROCID
SLURM_SUBMIT_DIR=$1
JOB_ID=$2

# Redirect output to node-specific files
exec > "$SLURM_SUBMIT_DIR/vllm_node${NODE_RANK}_${JOB_ID}.out" 2>&1

echo "[$HOSTNAME] Starting vLLM server (node rank $NODE_RANK)"
echo "Output file: vllm_node${NODE_RANK}_${JOB_ID}.out"

# Run the single-node launcher
bash "$SLURM_SUBMIT_DIR/launch_vllm_single.sh"
LAUNCHER_EOF

chmod +x "$LAUNCHER_SCRIPT"

# Launch on all nodes
echo "Starting independent vLLM servers on $SLURM_NNODES nodes..."
srun --nodes=$SLURM_NNODES --ntasks=$SLURM_NNODES bash "$LAUNCHER_SCRIPT" "$SLURM_SUBMIT_DIR" "$SLURM_JOB_ID"

# Cleanup
rm -f "$LAUNCHER_SCRIPT"

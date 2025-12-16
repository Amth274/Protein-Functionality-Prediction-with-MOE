#!/bin/bash
#SBATCH --job-name=protein_moe_training
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err

# Job configuration for HPC protein functionality training
# Modify the SBATCH parameters above according to your cluster configuration

echo "Starting Protein Functionality MoE Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"

# Load necessary modules (modify according to your HPC environment)
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0

# Activate virtual environment
source venv/bin/activate

# Set environment variables for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12355
export WORLD_SIZE=$SLURM_NTASKS
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Set NCCL environment variables for better performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_DISABLE=0

# Create necessary directories
mkdir -p logs outputs data/processed

echo "Environment setup complete"
echo "CUDA devices: $CUDA_VISIBLE_DEVICES"
echo "World size: $WORLD_SIZE"
echo "Master address: $MASTER_ADDR"

# Run the training script
srun python scripts/train_hpc.py \
    --config configs/moe_config.yaml \
    --output-dir outputs/protein_moe_$(date +%Y%m%d_%H%M%S)

echo "Training completed at: $(date)"
echo "Job finished"
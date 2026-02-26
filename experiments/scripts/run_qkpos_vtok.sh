#!/bin/bash
#SBATCH --job-name=add_tf_qkpos
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_%j.out

module load proxy/default 2>/dev/null || true

cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF

echo "Node: $(hostname)"
echo "Cores: $(nproc)"
echo "Variant: QK=pos, V=tok, 2 layers, no MLP"

python -u train_qkpos_vtok.py

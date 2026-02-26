#!/bin/bash
# Submit d5 experiments (tok_dim=2 for smaller models)

cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF

for variant in d5_f4 d5_f3 d5_f2 d5_f1 d5_f3_hd2 d5_f4_hd2 d4s_f4 d4s_f3 d6_f3_ref; do
    sbatch --job-name="d5_${variant}" \
           --output="slurm_d5_${variant}_%j.out" \
           --partition=cpu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=24:00:00 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_d5.py --variant ${variant}"
    echo "Submitted d5_${variant}"
done

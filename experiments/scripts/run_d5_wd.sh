#!/bin/bash
# Submit d5 experiments with weight decay and LR scheduling

cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF

for variant in d5_f3_wd01 d5_f3_wd1 d5_f3_cos d5_f3_wd_cos d5_f4_wd01 d5_f4_wd_cos d6_f3_wd01 d6_f3_wd1 d5_f3_lr3 d5_f4_lr3; do
    sbatch --job-name="d5w_${variant}" \
           --output="slurm_d5w_${variant}_%j.out" \
           --partition=cpu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=24:00:00 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_d5_wd.py --variant ${variant}"
    echo "Submitted d5w_${variant}"
done

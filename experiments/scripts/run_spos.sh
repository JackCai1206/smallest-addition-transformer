#!/bin/bash
# Submit shared position encoding experiments

cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF

for variant in d6_f3_sxy d5_f4_sxy d5_f3_sxy d5_f2_sxy d6_f3_sxyz d5_f3_sxyz d5_f4_sxyz d5_f3_sxy_hd2; do
    sbatch --job-name="sp_${variant}" \
           --output="slurm_sp_${variant}_%j.out" \
           --partition=cpu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=24:00:00 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_d5_spos.py --variant ${variant}"
    echo "Submitted sp_${variant}"
done

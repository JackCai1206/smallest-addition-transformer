#!/bin/bash
# Submit v2 experiments with unbuffered Python output

VARIANTS=(baseline_curr d4_basic d4_shareA d4_leader d4_grok d4_rms d4_shareA3 d4_shareA_fpos d4_full d4_r3 d6_nb_fh3 d6_shrink)

for v in "${VARIANTS[@]}"; do
    sbatch --job-name="v2_${v}" \
           --output="slurm_v2_${v}_%j.out" \
           --partition=cpu \
           --time=24:00:00 \
           --mem=8G \
           --cpus-per-task=4 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_v2.py --variant $v"
    echo "Submitted: $v"
done

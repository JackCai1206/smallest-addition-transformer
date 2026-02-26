#!/bin/bash
# Submit all ablation experiments to SLURM
# Each runs on its own node for speed

DIR=/scratch/gpfs/ARORA/zc5794/smallest_addition_TF

for VARIANT in no_ffn shqk shqk_no_ffn d10 no_ffn_1h 1h no_ffn_2L; do
    sbatch --job-name="abl_${VARIANT}" \
           --partition=cpu \
           --time=12:00:00 \
           --mem=8G \
           --cpus-per-task=4 \
           --output="${DIR}/slurm_abl_${VARIANT}_%j.out" \
           --wrap="module load proxy/default 2>/dev/null || true; cd ${DIR}; python -u train_ablation.py --variant ${VARIANT}"
    echo "Submitted: ${VARIANT}"
done

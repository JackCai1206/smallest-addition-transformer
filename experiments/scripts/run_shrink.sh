#!/bin/bash
# Submit incremental shrink experiments to SLURM

VARIANTS=(nb ffn4 ffn3 nb_ffn4 nb_ffn3 fh2 fh3 fpos nb_ffn4_fh2 nb_ffn3_fh2 nb_ffn4_fpos all_shrink)

for v in "${VARIANTS[@]}"; do
    sbatch --job-name="sh_${v}" \
           --output="slurm_sh_${v}_%j.out" \
           --partition=cpu \
           --time=24:00:00 \
           --mem=8G \
           --cpus-per-task=4 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 train_shrink.py --variant $v"
    echo "Submitted: $v"
done

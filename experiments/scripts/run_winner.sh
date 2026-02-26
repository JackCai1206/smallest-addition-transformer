#!/bin/bash
# Submit winner experiments (tied_proj + savings combinations)
# Priority order: most likely to work first

VARIANTS=(
    tp_curr           # 372p - does curriculum help grokking?
    tp_nb             # 342p - just remove biases
    tp_nb_curr        # 342p - remove biases + curriculum
    tp_fpos           # 321p - factored position encoding
    tp_fpos_curr      # 321p - factored pos + curriculum
    tp_nb_f4          # 318p - no bias + FFN4
    tp_nb_f4_curr     # 318p - no bias + FFN4 + curriculum
    tp_nb_f3          # 306p - no bias + FFN3
    tp_nb_f3_curr     # 306p - no bias + FFN3 + curriculum
    tp_nb_fpos        # 291p - no bias + factored pos
    tp_nb_f4_fpos     # 267p - aggressive combo
    tp_nb_f4_fpos_curr  # 267p - aggressive + curriculum
    tp_nb_f3_fpos     # 255p - most aggressive
    tp_nb_f3_fpos_curr  # 255p - most aggressive + curriculum
)

for v in "${VARIANTS[@]}"; do
    sbatch --job-name="win_${v}" \
           --output="slurm_win_${v}_%j.out" \
           --partition=cpu \
           --time=24:00:00 \
           --mem=8G \
           --cpus-per-task=4 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_winner.py --variant $v"
    echo "Submitted: $v ($v)"
done

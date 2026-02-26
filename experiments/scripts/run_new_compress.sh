#!/bin/bash
# Submit new compression experiments on CPU (models are tiny, CPU is fine)
# Round 2: RMSNorm, tied V/Out, spiral pos, all-shared LN

cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF

# Priority 1: Single-technique experiments (most likely to work)
for variant in \
    d6_f2_sxyz_rms_wd01 \
    d6_f2_sxyz_tvo_wd01 \
    d6_f2_sxyz_psp_wd01 \
    d6_f2_sxyz_asln_wd01; do
    sbatch --job-name="spwd_${variant}" \
           --output="slurm_spwd_${variant}_%j.out" \
           --partition=cpu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=12:00:00 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_spos_wd.py --variant ${variant}"
    echo "Submitted spwd_${variant}"
done

# Priority 2: Moderate combos with tied Q/K
for variant in \
    d6_f2_sxyz_rms_tqk_wd01 \
    d6_f2_sxyz_tvo_tqk_wd01 \
    d6_f2_sxyz_psp_tqk_wd01 \
    d6_f2_sxyz_asln_tqk_wd01; do
    sbatch --job-name="spwd_${variant}" \
           --output="slurm_spwd_${variant}_%j.out" \
           --partition=cpu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=12:00:00 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_spos_wd.py --variant ${variant}"
    echo "Submitted spwd_${variant}"
done

# Priority 3: Double spirals and aggressive combos
for variant in \
    d6_f2_sxyz_ps_psp_wd01 \
    d6_f2_sxyz_ps_psp_tqk_wd01 \
    d6_f2_sxyz_ps_tqk_tvo_wd01 \
    d6_f2_sxyz_psp_tqk_tvo_wd01 \
    d6_f2_sxyz_rms_tvo_tqk_wd01 \
    d6_f2_sxyz_rms_ps_tqk_tvo_wd01 \
    d6_f2_sxyz_ps_psp_tqk_tvo_wd01; do
    sbatch --job-name="spwd_${variant}" \
           --output="slurm_spwd_${variant}_%j.out" \
           --partition=cpu \
           --nodes=1 \
           --ntasks=1 \
           --cpus-per-task=4 \
           --mem=8G \
           --time=12:00:00 \
           --wrap="cd /scratch/gpfs/ARORA/zc5794/smallest_addition_TF && python3 -u train_spos_wd.py --variant ${variant}"
    echo "Submitted spwd_${variant}"
done

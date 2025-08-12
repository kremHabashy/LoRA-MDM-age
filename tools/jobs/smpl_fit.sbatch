#!/bin/bash
#SBATCH -J smpl_fit
#SBATCH --partition=CELIASMI
#SBATCH --gres=gpu:celiasmigpu:1
#SBATCH -c 6
#SBATCH --mem=24G
#SBATCH -t 1-00:00:00
#SBATCH -o /u1/khabashy/LoRA-MDM/logs/smpl_fit_%A_%a.out
#SBATCH -e /u1/khabashy/LoRA-MDM/logs/smpl_fit_%A_%a.err
#SBATCH --array=0-9999%2     # <= cap at 2 GPUs total

set -euo pipefail
PROCESSED=/u1/khabashy/LoRA-MDM/data/processed
MODELS=/u1/khabashy/LoRA-MDM/body_models
OUT=/u1/khabashy/LoRA-MDM/data/smpl_fitted_smpl
LIST=/u1/khabashy/LoRA-MDM/lists/able_trials.csv

mkdir -p "$OUT" /u1/khabashy/LoRA-MDM/logs

N=$(wc -l < "$LIST")
ID=${SLURM_ARRAY_TASK_ID}
if [ "$ID" -ge "$N" ]; then
  echo "Done: task $ID >= $N"; exit 0
fi

LINE=$(sed -n "$((ID+1))p" "$LIST")
SUBJ=${LINE%%,*}
TRIAL=${LINE#*,}

echo "Fitting $SUBJ / $TRIAL on $(hostname)"
conda run -n moshpp37 python /u1/khabashy/LoRA-MDM/tools/fit_smpl_markers.py \
  --processed_dir "$PROCESSED" \
  --models_dir "$MODELS" \
  --out_dir "$OUT" \
  --subject "$SUBJ" \
  --trial "$TRIAL" \
  --device cuda \
  --iters 600

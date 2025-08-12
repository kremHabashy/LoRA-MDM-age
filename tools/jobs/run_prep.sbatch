#!/bin/bash
#SBATCH -J vc_prep
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=32G
#SBATCH -o /u1/khabashy/LoRA-MDM/logs/vc_prep_%j.out
#SBATCH -e /u1/khabashy/LoRA-MDM/logs/vc_prep_%j.err

set -euo pipefail

# Make sure log dir exists
mkdir -p /u1/khabashy/LoRA-MDM/logs

# Use the exact Python inside your env
PY=/u1/khabashy/.conda/envs/moshpp37/bin/python

RAW=/u1/khabashy/LoRA-MDM/data/van_criekinge
OUT=/u1/khabashy/LoRA-MDM/data/processed

# (Optional sanity check – shows which Python is used and confirms c3d is importable)
$PY - <<'PYCODE'
import sys
print("Python:", sys.executable)
import c3d
print("c3d import: OK")
PYCODE

$PY /u1/khabashy/LoRA-MDM/tools/dataset_prep.py \
  --data_dir "$RAW" \
  --output_dir "$OUT" \
  --min_valid_ratio 0.5 \
  --gap_fill_max 10 \
  --resample_fps 20 \
  --resample_mode interp

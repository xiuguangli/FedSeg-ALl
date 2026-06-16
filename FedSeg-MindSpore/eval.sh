#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

date_now=$(date +"%Y%m%d_%H%M%S")
ROOT_DIR="data/cityscapes_split_erase19"
CHECKPOINT="${1:-saved.ckpt}"

micromamba run -n fedseg-mindspore python -u segmentation/eval.py \
  --gpu="0" \
  --dataset="cityscapes" \
  --root_dir="${ROOT_DIR}" \
  --num_classes=19 \
  --data="val" \
  --num_workers=4 \
  --model="bisenetv2" \
  --checkpoint="${CHECKPOINT}" \
  --USE_ERASE_DATA=True \
  | tee -a "save/logs/eval_log-${date_now}.txt"

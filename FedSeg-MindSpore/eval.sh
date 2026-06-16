#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

date_now=$(date +"%Y%m%d_%H%M%S")
ROOT_DIR="data/cityscapes_split_erase19"
CHECKPOINT="${1:-saved.ckpt}"
GPU_ID="${GPU_ID-0}"

source "${PROJECT_ROOT}/scripts/mindspore_env.sh"
fedseg_mindspore_prepare_for_gpu_id "${GPU_ID}"

"${FEDSEG_PYTHON[@]}" -u segmentation/eval.py \
  --gpu="${GPU_ID}" \
  --dataset="cityscapes" \
  --root_dir="${ROOT_DIR}" \
  --num_classes=19 \
  --data="val" \
  --num_workers=4 \
  --model="bisenetv2" \
  --checkpoint="${CHECKPOINT}" \
  --USE_ERASE_DATA=True \
  | tee -a "save/logs/eval_log-${date_now}.txt"

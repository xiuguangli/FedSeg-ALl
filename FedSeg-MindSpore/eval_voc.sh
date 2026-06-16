#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

ROOT_DIR="data/voc"
# DEFAULT_CHECKPOINT="save/checkpoints/FedSeg1.ckpt"
DEFAULT_CHECKPOINT="save/checkpoints/fedseg-ms-33.29.ckpt"
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"
GPU_ID="${GPU_ID-0}"

source "${PROJECT_ROOT}/scripts/mindspore_env.sh"
fedseg_mindspore_prepare_for_gpu_id "${GPU_ID}"

CHECKPOINT_ARGS=(--checkpoints "${CHECKPOINT}")

"${FEDSEG_PYTHON[@]}" -u segmentation/eval_voc.py \
  --gpu "${GPU_ID}" \
  --dataset voc \
  --root_dir "${ROOT_DIR}" \
  --num_classes 20 \
  --data val \
  --num_workers 8 \
  --batch_size 24 \
  --model bisenetv2 \
  --profile_runtime True \
  "${CHECKPOINT_ARGS[@]}"

# Example commands:
# bash eval_voc.sh
# GPU_ID="" bash eval_voc.sh  # CPU only
# bash eval_voc.sh save/checkpoints/your_checkpoint.ckpt

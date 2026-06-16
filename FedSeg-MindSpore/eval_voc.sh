#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

ROOT_DIR="data/voc"
# DEFAULT_CHECKPOINT="save/checkpoints/FedSeg1.ckpt"
DEFAULT_CHECKPOINT="save/checkpoints/fedseg-ms-33.29.ckpt"
CHECKPOINT="${1:-${DEFAULT_CHECKPOINT}}"

CHECKPOINT_ARGS=(--checkpoints "${CHECKPOINT}")

# micromamba run -n fedseg-mindspore python -u segmentation/eval_voc.py \
python -u segmentation/eval_voc.py \
  --gpu 0 \
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
# bash eval_voc.sh save/checkpoints/your_checkpoint.ckpt

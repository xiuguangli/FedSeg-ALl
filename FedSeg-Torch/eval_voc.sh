#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

ROOT_DIR="data/voc"
CHECKPOINT="FedSeg1.pth"
# CHECKPOINT="fedseg-31.13.pth"
GPU_ID="${GPU_ID-0}"

source "${PROJECT_ROOT}/scripts/torch_env.sh"
fedseg_torch_prepare_for_gpu_id "${GPU_ID}"

"${FEDSEG_PYTHON[@]}" -u segmentation/eval_voc.py \
  --gpu "${GPU_ID}" \
  --dataset voc \
  --root_dir "${ROOT_DIR}" \
  --num_classes 20 \
  --data val \
  --num_workers 4 \
  --batch_size 1 \
  --model bisenetv2 \
  --checkpoints "${CHECKPOINT}"

# Example commands:
# bash eval_voc.sh
# GPU_ID="" bash eval_voc.sh  # CPU only
# python segmentation/eval_voc.py
# python segmentation/eval_voc.py --checkpoints fedseg-torch.pth
# python segmentation/eval_voc.py --checkpoints save/checkpoints/fedseg-torch.pth

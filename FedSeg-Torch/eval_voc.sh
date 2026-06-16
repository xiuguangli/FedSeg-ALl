#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="data/voc"
CHECKPOINT="FedSeg1.pth"
# CHECKPOINT="fedseg-31.13.pth"

python -u segmentation/eval_voc.py \
  --gpu 0 \
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
# python segmentation/eval_voc.py
# python segmentation/eval_voc.py --checkpoints fedseg-torch.pth
# python segmentation/eval_voc.py --checkpoints save/checkpoints/fedseg-torch.pth

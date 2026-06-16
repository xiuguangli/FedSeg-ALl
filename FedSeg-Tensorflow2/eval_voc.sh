#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

ROOT_DIR="${ROOT_DIR:-data/voc}"
DATASET="${DATASET:-voc}"
NUM_CLS="${NUM_CLS:-20}"
GPU_ID="${GPU_ID:-0}"
# CHECKPOINT="${CHECKPOINT:-FedSeg.weights.h5}"
CHECKPOINT="${CHECKPOINT:-fedseg-tf.weights.h5}"
EVAL_BS="${EVAL_BS:-8}"
EVAL_BUCKETS="${EVAL_BUCKETS:-}"
FAST_NHWC="${FAST_NHWC:-True}"
EVAL_PREBATCH="${EVAL_PREBATCH:-False}"
EVAL_TAIL_PAD_BATCH="${EVAL_TAIL_PAD_BATCH:-False}"
EVAL_TF_METRIC="${EVAL_TF_METRIC:-False}"
EVAL_TFDATA_BATCH="${EVAL_TFDATA_BATCH:-True}"
NUM_WORKERS="${NUM_WORKERS:-4}"
PROFILE_RUNTIME="${PROFILE_RUNTIME:-False}"

python -u segmentation/eval_voc.py \
  --gpu "${GPU_ID}" \
  --dataset "${DATASET}" \
  --root "./" \
  --root_dir "${ROOT_DIR}" \
  --USE_ERASE_DATA=True \
  --num_classes "${NUM_CLS}" \
  --data "val" \
  --num_workers "${NUM_WORKERS}" \
  --eval_bs "${EVAL_BS}" \
  --eval_buckets "${EVAL_BUCKETS}" \
  --fast_nhwc "${FAST_NHWC}" \
  --eval_prebatch "${EVAL_PREBATCH}" \
  --eval_tail_pad_batch "${EVAL_TAIL_PAD_BATCH}" \
  --eval_tf_metric "${EVAL_TF_METRIC}" \
  --eval_tfdata_batch "${EVAL_TFDATA_BATCH}" \
  --profile_runtime "${PROFILE_RUNTIME}" \
  --model "bisenetv2" \
  --checkpoints "${CHECKPOINT}"

# Example commands:
# bash eval_voc.sh
# EVAL_BS=4 bash eval_voc.sh
# EVAL_BS=8 bash eval_voc.sh
# FAST_NHWC=True bash eval_voc.sh
# PROFILE_RUNTIME=True bash eval_voc.sh
# EVAL_TAIL_PAD_BATCH=True bash eval_voc.sh
# EVAL_TF_METRIC=True bash eval_voc.sh
# EVAL_TFDATA_BATCH=True bash eval_voc.sh
# EVAL_BUCKETS=384x512,512x512 bash eval_voc.sh
# python segmentation/eval_voc.py --checkpoints FedSeg.weights.h5
# python segmentation/eval_voc.py --checkpoints save/checkpoints/FedSeg.weights.h5

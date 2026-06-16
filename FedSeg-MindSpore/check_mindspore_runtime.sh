#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID-0}"

source "${PROJECT_ROOT}/scripts/mindspore_env.sh"
fedseg_mindspore_prepare_for_gpu_id "${GPU_ID}"

"${FEDSEG_PYTHON[@]}" -u segmentation/check_mindspore_runtime.py --gpu "${GPU_ID}"

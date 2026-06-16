#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "${PROJECT_ROOT}"

GPU_ID="${GPU_ID-0}"

source "${PROJECT_ROOT}/scripts/torch_env.sh"
fedseg_torch_prepare_for_gpu_id "${GPU_ID}"

"${FEDSEG_PYTHON[@]}" -u segmentation/check_torch_runtime.py --gpu "${GPU_ID}"

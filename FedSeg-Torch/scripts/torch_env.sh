#!/usr/bin/env bash

# Shared launcher helper for the PyTorch backend.
#
# On Linux, pyproject.toml pins CUDA wheels from the PyTorch cu128 index.
# This helper makes the shell scripts consistently use the local uv
# environment instead of whatever python happens to be first on PATH.

fedseg_torch_set_python() {
  if command -v uv >/dev/null 2>&1 && [[ -f "pyproject.toml" ]]; then
    FEDSEG_PYTHON=(uv run python)
  else
    FEDSEG_PYTHON=(python)
  fi
}

fedseg_torch_is_cpu_request() {
  local gpu_id="${1-}"
  [[ -z "${gpu_id}" || "${gpu_id}" == "cpu" || "${gpu_id}" == "CPU" || "${gpu_id}" == "-1" ]]
}

fedseg_torch_prepare_for_gpu_id() {
  local gpu_id="${1-0}"
  fedseg_torch_set_python

  if fedseg_torch_is_cpu_request "${gpu_id}"; then
    export CUDA_VISIBLE_DEVICES="-1"
  fi
}

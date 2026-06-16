#!/usr/bin/env bash

# Shared runtime setup for the TensorFlow backend.
#
# Python packages are pinned by uv.lock. TensorFlow GPU wheels install CUDA
# runtime libraries via the tensorflow[and-cuda] extra; this helper makes those
# libraries visible before TensorFlow is imported.

fedseg_tensorflow_prepend_ld_path() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

fedseg_tensorflow_prepend_path() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  case ":${PATH:-}:" in
    *":${dir}:"*) ;;
    *) export PATH="${dir}${PATH:+:${PATH}}" ;;
  esac
}

fedseg_tensorflow_set_python() {
  if command -v uv >/dev/null 2>&1 && [[ -f "pyproject.toml" ]]; then
    FEDSEG_PYTHON=(uv run python)
  else
    FEDSEG_PYTHON=(python)
  fi
}

fedseg_tensorflow_is_cpu_request() {
  local gpu_id="${1-}"
  [[ -z "${gpu_id}" || "${gpu_id}" == "cpu" || "${gpu_id}" == "CPU" || "${gpu_id}" == "-1" ]]
}

fedseg_tensorflow_prepare_pip_cuda_paths() {
  local line
  while IFS= read -r line; do
    [[ -n "${line}" ]] || continue
    case "${line}" in
      LD_LIBRARY_PATH=*) fedseg_tensorflow_prepend_ld_path "${line#LD_LIBRARY_PATH=}" ;;
      PATH=*) fedseg_tensorflow_prepend_path "${line#PATH=}" ;;
    esac
  done < <("${FEDSEG_PYTHON[@]}" - <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

seen = set()
for sys_path in sys.path:
    nvidia_root = Path(sys_path) / "nvidia"
    if not nvidia_root.is_dir():
        continue
    for lib_dir in sorted(nvidia_root.glob("*/lib")):
        path = str(lib_dir)
        if path not in seen:
            seen.add(path)
            print("LD_LIBRARY_PATH=" + path)
    for bin_dir in sorted(nvidia_root.glob("*/bin")):
        path = str(bin_dir)
        if path not in seen:
            seen.add(path)
            print("PATH=" + path)
PY
)
}

fedseg_tensorflow_prepare_for_gpu_id() {
  local gpu_id="${1-0}"
  fedseg_tensorflow_set_python

  if fedseg_tensorflow_is_cpu_request "${gpu_id}"; then
    export CUDA_VISIBLE_DEVICES="-1"
  else
    export CUDA_VISIBLE_DEVICES="${gpu_id}"
    fedseg_tensorflow_prepare_pip_cuda_paths
  fi
}

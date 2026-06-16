#!/usr/bin/env bash

# Shared runtime setup for the MindSpore backend.
#
# Python packages are pinned by uv.lock, but MindSpore GPU still needs NVIDIA
# driver, CUDA and cuDNN shared libraries to be visible before Python starts.
# This helper does not hard-code a single workstation path: it prefers an
# explicit FEDSEG_CUDA_HOME, then probes common CUDA 11.x locations.

fedseg_mindspore_prepend_ld_path() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  case ":${LD_LIBRARY_PATH:-}:" in
    *":${dir}:"*) ;;
    *) export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
}

fedseg_mindspore_prepend_path() {
  local dir="$1"
  [[ -d "${dir}" ]] || return 0
  case ":${PATH:-}:" in
    *":${dir}:"*) ;;
    *) export PATH="${dir}${PATH:+:${PATH}}" ;;
  esac
}

fedseg_mindspore_has_glob() {
  compgen -G "$1" >/dev/null
}

fedseg_mindspore_has_cuda11_runtime() {
  local cuda_root="$1"
  local lib_dir
  local has_cuda=0
  local has_cudnn=0

  [[ -d "${cuda_root}" ]] || return 1

  for lib_dir in "${cuda_root}/targets/x86_64-linux/lib" "${cuda_root}/lib64" "${cuda_root}/lib"; do
    [[ -d "${lib_dir}" ]] || continue
    if fedseg_mindspore_has_glob "${lib_dir}/libcudart.so.11*" &&
       fedseg_mindspore_has_glob "${lib_dir}/libcublas.so.11*"; then
      has_cuda=1
    fi
    if fedseg_mindspore_has_glob "${lib_dir}/libcudnn.so.8*"; then
      has_cudnn=1
    fi
  done

  [[ "${has_cuda}" == "1" && "${has_cudnn}" == "1" ]]
}

fedseg_mindspore_prepare_runtime() {
  if [[ "${FEDSEG_SKIP_CUDA_SETUP:-0}" == "1" ]]; then
    return 0
  fi

  local candidate
  local selected_cuda_home=""
  local candidates=(
    "${FEDSEG_CUDA_HOME:-}"
    "${CUDA_HOME:-}"
    "/usr/local/cuda-11.6"
    "/opt/cuda-11.6"
    "${HOME:-}/local/cuda/cuda-11.6"
    "${HOME:-}/cuda-11.6"
    "/usr/local/cuda"
    "/opt/cuda"
  )

  for candidate in "${candidates[@]}"; do
    [[ -n "${candidate}" ]] || continue
    if fedseg_mindspore_has_cuda11_runtime "${candidate}"; then
      selected_cuda_home="${candidate}"
      break
    fi
  done

  if [[ -n "${selected_cuda_home}" ]]; then
    export CUDA_HOME="${selected_cuda_home}"
    fedseg_mindspore_prepend_path "${CUDA_HOME}/bin"
    fedseg_mindspore_prepend_ld_path "${CUDA_HOME}/targets/x86_64-linux/lib"
    fedseg_mindspore_prepend_ld_path "${CUDA_HOME}/lib64"
    fedseg_mindspore_prepend_ld_path "${CUDA_HOME}/lib"
  fi

  # libcuda.so is provided by the NVIDIA driver, not the CUDA toolkit.
  fedseg_mindspore_prepend_ld_path "/usr/lib/x86_64-linux-gnu"
  fedseg_mindspore_prepend_ld_path "/lib/x86_64-linux-gnu"
}

fedseg_mindspore_set_python() {
  if command -v uv >/dev/null 2>&1 && [[ -f "pyproject.toml" ]]; then
    FEDSEG_PYTHON=(uv run python)
  else
    FEDSEG_PYTHON=(python)
  fi
}

fedseg_mindspore_is_cpu_request() {
  local gpu_id="${1-}"
  [[ -z "${gpu_id}" || "${gpu_id}" == "cpu" || "${gpu_id}" == "CPU" || "${gpu_id}" == "-1" ]]
}

fedseg_mindspore_prepare_for_gpu_id() {
  local gpu_id="${1-0}"
  if fedseg_mindspore_is_cpu_request "${gpu_id}"; then
    # Avoid MindSpore probing an incompatible CUDA_HOME during explicit CPU runs.
    unset CUDA_HOME
  else
    fedseg_mindspore_prepare_runtime
  fi
  fedseg_mindspore_set_python
}

import os
import textwrap

import mindspore as ms


def _is_cpu_request(gpu_value):
    if gpu_value is None:
        return True
    value = str(gpu_value).strip().lower()
    return value in {"", "cpu", "none", "-1"}


def _gpu_error_message(gpu_value, exc):
    return textwrap.dedent(
        """
        MindSpore GPU initialization failed for --gpu={gpu}.

        This project pins mindspore==2.6.0 with uv. The Python package is
        reproducible, but GPU execution also needs the host system to provide a
        compatible NVIDIA driver, CUDA runtime and cuDNN before Python starts.
        For this repository, use a CUDA 11.x runtime with cuDNN 8; CUDA 11.6 is
        the recommended target for MindSpore 2.6.0.

        Clone-and-run checklist:
          1. nvidia-smi works on the host.
          2. CUDA 11.x and cuDNN 8 libraries are installed.
          3. CUDA_HOME points to that CUDA installation, or set
             FEDSEG_CUDA_HOME=/path/to/cuda-11.6 before running scripts.
          4. LD_LIBRARY_PATH contains the CUDA lib directory and the NVIDIA
             driver lib directory. The shell scripts in this directory try to
             detect common paths automatically.

        Verify the runtime with:
          uv sync
          bash check_mindspore_runtime.sh

        To intentionally run on CPU, pass --gpu "" or run:
          GPU_ID="" bash check_mindspore_runtime.sh

        Current CUDA_HOME={cuda_home}
        Current LD_LIBRARY_PATH={ld_library_path}
        Original error:
        {error}
        """
    ).strip().format(
        gpu=gpu_value,
        cuda_home=os.environ.get("CUDA_HOME", ""),
        ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
        error=exc,
    )


def setup_mindspore_device(
    gpu_value,
    mode=ms.PYNATIVE_MODE,
    deterministic=False,
    gpu_config=None,
    logger=None,
):
    if _is_cpu_request(gpu_value):
        ms.set_context(mode=mode, device_target="CPU")
        return "CPU"

    try:
        device_id = int(str(gpu_value).strip())
        ms.set_device(device_target="GPU", device_id=device_id)
        ms.set_context(mode=mode)
        if deterministic:
            ms.set_context(deterministic="ON")
        if gpu_config:
            ms.set_context(gpu_config=gpu_config)
        if logger is not None and (deterministic or gpu_config):
            logger.info(
                "MindSpore GPU context overrides: deterministic={}, gpu_config={}",
                bool(deterministic),
                gpu_config or {},
            )
        return "GPU:{}".format(device_id)
    except Exception as exc:
        if os.environ.get("FEDSEG_ALLOW_CPU_FALLBACK", "").lower() in {"1", "true", "yes"}:
            if logger is not None:
                logger.warning(
                    "MindSpore GPU initialization failed and FEDSEG_ALLOW_CPU_FALLBACK is set; falling back to CPU. Error: {}",
                    exc,
                )
            ms.set_context(mode=mode, device_target="CPU")
            return "CPU"
        raise RuntimeError(_gpu_error_message(gpu_value, exc)) from exc

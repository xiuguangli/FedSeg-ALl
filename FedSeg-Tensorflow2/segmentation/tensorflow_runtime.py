from __future__ import annotations

import os
import textwrap


def is_cpu_request(gpu_value: str | None) -> bool:
    if gpu_value is None:
        return True
    value = str(gpu_value).strip().lower()
    return value in {"", "cpu", "none", "-1"}


def _gpu_error_message(tf, requested_gpu: str | None) -> str:
    return textwrap.dedent(
        """
        TensorFlow GPU initialization failed for --gpu={gpu}.

        This project pins tensorflow[and-cuda]==2.20.0 with uv. The Python
        dependencies are reproducible, but GPU execution still needs a working
        NVIDIA driver on the host. The TensorFlow pip extra provides CUDA/cuDNN
        runtime libraries inside the virtual environment, and the shell scripts
        in this directory add those libraries to LD_LIBRARY_PATH before import.

        Clone-and-run checklist:
          1. nvidia-smi works on the host.
          2. Run uv sync after cloning.
          3. Run bash check_tensorflow_runtime.sh.
          4. Start scripts with bash eval_voc.sh or bash run_voc.sh so the
             TensorFlow runtime helper is loaded.

        To intentionally run on CPU:
          GPU_ID="" bash check_tensorflow_runtime.sh

        TensorFlow version: {tf_version}
        Built with CUDA: {built_cuda}
        CUDA_VISIBLE_DEVICES: {cuda_visible_devices}
        LD_LIBRARY_PATH: {ld_library_path}
        Visible GPUs: {visible_gpus}
        """
    ).strip().format(
        gpu=requested_gpu,
        tf_version=getattr(tf, "__version__", "unknown"),
        built_cuda=tf.test.is_built_with_cuda(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        ld_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
        visible_gpus=tf.config.list_physical_devices("GPU"),
    )


def require_tensorflow_device(tf, requested_gpu: str | None) -> str:
    if is_cpu_request(requested_gpu):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        return "CPU"

    gpus = tf.config.list_physical_devices("GPU")
    if not tf.test.is_built_with_cuda() or not gpus:
        raise RuntimeError(_gpu_error_message(tf, requested_gpu))

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    return "GPU:{}".format(requested_gpu)

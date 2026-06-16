from __future__ import annotations

import os
import textwrap

import torch


def is_cpu_request(gpu_value: str | None) -> bool:
    if gpu_value is None:
        return True
    value = str(gpu_value).strip().lower()
    return value in {"", "cpu", "none", "-1"}


def _gpu_error_message(requested_gpu: str | None) -> str:
    return textwrap.dedent(
        """
        PyTorch GPU initialization failed for --gpu={gpu}.

        This project pins torch==2.8.0 and torchvision==0.23.0. On Linux,
        pyproject.toml resolves them from the official PyTorch cu128 index, so
        clone-and-run GPU reproduction should install torch builds such as
        2.8.0+cu128.

        Clone-and-run checklist:
          1. nvidia-smi works on the host.
          2. Run uv sync after cloning.
          3. Run bash check_torch_runtime.sh.
          4. Start scripts with bash eval_voc.sh or bash run_voc.sh so the
             local uv environment is used.

        To intentionally run on CPU:
          GPU_ID="" bash check_torch_runtime.sh

        torch.__version__: {torch_version}
        torch.version.cuda: {torch_cuda}
        torch.cuda.is_available(): {cuda_available}
        torch.cuda.device_count(): {device_count}
        CUDA_VISIBLE_DEVICES: {cuda_visible_devices}
        """
    ).strip().format(
        gpu=requested_gpu,
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        device_count=torch.cuda.device_count(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )


def require_torch_device(requested_gpu: str | None) -> torch.device:
    if is_cpu_request(requested_gpu):
        return torch.device("cpu")

    if not torch.cuda.is_available():
        raise RuntimeError(_gpu_error_message(requested_gpu))

    device_id = int(str(requested_gpu).strip())
    if device_id < 0 or device_id >= torch.cuda.device_count():
        raise RuntimeError(_gpu_error_message(requested_gpu))

    torch.cuda.set_device(device_id)
    return torch.device("cuda")

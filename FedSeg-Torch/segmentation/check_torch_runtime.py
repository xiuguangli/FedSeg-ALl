from __future__ import annotations

import argparse

import torch

from torch_runtime import require_torch_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Check PyTorch runtime for FedSeg.")
    parser.add_argument("--gpu", type=str, default="0", help='GPU id, or "" for CPU')
    args = parser.parse_args()

    device = require_torch_device(args.gpu)
    value = torch.ones((2, 2), device=device).sum().item()
    print("PyTorch {} runtime OK on {}; cuda={}; test_sum={}".format(
        torch.__version__,
        device,
        torch.version.cuda,
        value,
    ))
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(device))
        print("Visible GPU count:", torch.cuda.device_count())


if __name__ == "__main__":
    main()

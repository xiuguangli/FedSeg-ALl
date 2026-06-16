from __future__ import annotations

import argparse
import json
from pathlib import Path

from runtime_utils import (
    configure_tensorflow_env,
    configure_tensorflow_runtime,
    install_tensorflow_stderr_filter,
)

_BOOTSTRAP_ARGS = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu", type=str, default="0")
    _BOOTSTRAP_ARGS, _ = parser.parse_known_args()
    configure_tensorflow_env(gpu=_BOOTSTRAP_ARGS.gpu)
    install_tensorflow_stderr_filter()

import tensorflow as tf

from logging_utils import logger, setup_logger
from tf2_tools import metadata_output_path, normalize_tf_backbone_path, save_torch_backbone_as_tf_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert BiSeNetV2 torch backbone weights to TensorFlow weights.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument(
        "--torch-backbone",
        type=str,
        default="segmentation/myseg/backbone_v2.pth",
        help="Path to the torch backbone_v2.pth file",
    )
    parser.add_argument(
        "--tf-backbone",
        type=str,
        default="segmentation/myseg/backbone_v2.weights.h5",
        help="Output TensorFlow .weights.h5 backbone path",
    )
    parser.add_argument("--root", type=str, default="./")
    return parser.parse_args()


def _resolve_path(root: Path, path: str) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    return (root / resolved).resolve()


def main() -> None:
    args = parse_args()
    setup_logger(verbose=False, logs_dir="logs/convert_backbone", log_name="convert")
    configure_tensorflow_runtime(tf)

    root = Path(args.root).resolve()
    torch_backbone = _resolve_path(root, args.torch_backbone)
    tf_backbone = normalize_tf_backbone_path(_resolve_path(root, args.tf_backbone))
    if not torch_backbone.exists():
        raise FileNotFoundError(f"torch backbone not found: {torch_backbone}")

    saved_path = save_torch_backbone_as_tf_weights(torch_backbone, tf_backbone)
    metadata = {
        "source_torch_backbone": str(torch_backbone),
        "type": "bisenetv2_backbone",
    }
    metadata_path = metadata_output_path(saved_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    logger.info("Converted torch backbone: {}", torch_backbone.name)
    logger.info("Saved tf backbone: {}", saved_path.name)
    logger.info("Saved metadata: {}", metadata_path.name)


if __name__ == "__main__":
    main()

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
import torch

from logging_utils import logger, setup_logger
from myseg.bisenet_utils import set_model_bisenetv2
from tf2_tools import metadata_output_path, normalize_tf_checkpoint_path, save_torch_checkpoint_as_tf_weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a FedSeg torch checkpoint to TensorFlow weights.")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--torch-checkpoint", type=str, required=True, help="Path to a torch .pth checkpoint")
    parser.add_argument("--tf-checkpoint", type=str, default="", help="Optional output .weights.h5 path")
    parser.add_argument("--num-classes", type=int, default=20)
    parser.add_argument("--proj-dim", type=int, default=256)
    parser.add_argument("--root", type=str, default="./")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    setup_logger(verbose=False, logs_dir="logs/convert_checkpoint", log_name="convert")
    configure_tensorflow_runtime(tf)

    torch_checkpoint = Path(args.torch_checkpoint)
    if not torch_checkpoint.is_absolute():
        torch_checkpoint = (Path(args.root) / torch_checkpoint).resolve()
    if not torch_checkpoint.exists():
        raise FileNotFoundError(f"torch checkpoint not found: {torch_checkpoint}")

    tf_checkpoint = Path(args.tf_checkpoint) if args.tf_checkpoint else normalize_tf_checkpoint_path(torch_checkpoint)
    if not tf_checkpoint.is_absolute():
        tf_checkpoint = (Path(args.root) / tf_checkpoint).resolve()

    model_args = argparse.Namespace(proj_dim=args.proj_dim)
    model = set_model_bisenetv2(model_args, num_classes=args.num_classes)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)

    saved_path = save_torch_checkpoint_as_tf_weights(model, torch_checkpoint, tf_checkpoint)

    checkpoint_payload = torch.load(str(torch_checkpoint), map_location="cpu")
    metadata = {
        "source_torch_checkpoint": str(torch_checkpoint),
        "epoch": int(checkpoint_payload.get("epoch", -1)),
        "exp_name": checkpoint_payload.get("exp_name"),
        "wandb_id": checkpoint_payload.get("wandb_id"),
    }
    metadata_path = metadata_output_path(saved_path)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")

    logger.info("Converted torch checkpoint: {}", torch_checkpoint.name)
    logger.info("Saved tf checkpoint: {}", saved_path.name)
    logger.info("Saved metadata: {}", metadata_path.name)


if __name__ == "__main__":
    main()

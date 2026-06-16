from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
SEG_DIR = ROOT / "segmentation"
TORCH_SEG_DIR = ROOT.parent / "FedSeg-torch" / "segmentation"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SEG_DIR) not in sys.path:
    sys.path.insert(0, str(SEG_DIR))

from segmentation.runtime_utils import (
    configure_tensorflow_env,
    configure_tensorflow_runtime,
    install_tensorflow_stderr_filter,
)

configure_tensorflow_env()
install_tensorflow_stderr_filter()

import tensorflow as tf

configure_tensorflow_runtime(tf)
tf.keras.backend.set_image_data_format("channels_first")
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


def pytest_configure():
    np.random.seed(0)

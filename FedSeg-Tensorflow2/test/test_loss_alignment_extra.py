from __future__ import annotations

import numpy as np
import tensorflow as tf
import torch
from pathlib import Path
import sys

from segmentation.myseg.bisenet_utils import DiceLoss as TFDiceLoss
from segmentation.myseg.bisenet_utils import FocalLoss as TFFocalLoss
from segmentation.myseg.bisenet_utils import LovaszLoss as TFLovaszLoss
from segmentation.myseg.bisenet_utils import SoftBCEWithLogitsLoss as TFSoftBCEWithLogitsLoss

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from reference_smp_losses import DiceLoss as TorchDiceLoss
from reference_smp_losses import FocalLoss as TorchFocalLoss
from reference_smp_losses import LovaszLoss as TorchLovaszLoss
from reference_smp_losses import SoftBCEWithLogitsLoss as TorchSoftBCEWithLogitsLoss


def test_soft_bce_loss_matches_smp_reference():
    np.random.seed(11)
    logits_np = np.random.randn(2, 4, 8, 8).astype(np.float32)
    labels_np = np.random.randint(0, 2, size=(2, 4, 8, 8)).astype(np.float32)
    labels_np[0, 0, 0, 0] = 255.0

    torch_loss_fn = TorchSoftBCEWithLogitsLoss(ignore_index=255)
    torch_loss = torch_loss_fn(torch.tensor(logits_np), torch.tensor(labels_np)).item()

    tf_loss_fn = TFSoftBCEWithLogitsLoss(ignore_index=255)
    tf_loss = tf_loss_fn(
        tf.convert_to_tensor(labels_np, dtype=tf.float32),
        tf.convert_to_tensor(logits_np, dtype=tf.float32),
    ).numpy()

    np.testing.assert_allclose(tf_loss, torch_loss, rtol=1e-5, atol=1e-5)


def test_focal_loss_matches_smp_reference():
    np.random.seed(12)
    logits_np = np.random.randn(2, 5, 8, 8).astype(np.float32)
    labels_np = np.random.randint(0, 5, size=(2, 8, 8), dtype=np.int64)
    labels_np[0, 0, 0] = 255

    torch_loss_fn = TorchFocalLoss("multiclass", alpha=0.25, ignore_index=255)
    torch_loss = torch_loss_fn(torch.tensor(logits_np), torch.tensor(labels_np)).item()

    tf_loss_fn = TFFocalLoss("multiclass", alpha=0.25, ignore_index=255)
    tf_loss = tf_loss_fn(
        tf.convert_to_tensor(labels_np, dtype=tf.int64),
        tf.convert_to_tensor(logits_np, dtype=tf.float32),
    ).numpy()

    np.testing.assert_allclose(tf_loss, torch_loss, rtol=1e-5, atol=1e-5)


def test_dice_loss_matches_smp_reference():
    np.random.seed(13)
    logits_np = np.random.randn(2, 4, 8, 8).astype(np.float32)
    labels_np = np.random.randint(0, 4, size=(2, 8, 8), dtype=np.int64)
    labels_np[0, 0, 0] = 255

    torch_loss_fn = TorchDiceLoss("multiclass", ignore_index=255)
    torch_loss = torch_loss_fn(torch.tensor(logits_np), torch.tensor(labels_np)).item()

    tf_loss_fn = TFDiceLoss("multiclass", ignore_index=255)
    tf_loss = tf_loss_fn(
        tf.convert_to_tensor(labels_np, dtype=tf.int64),
        tf.convert_to_tensor(logits_np, dtype=tf.float32),
    ).numpy()

    np.testing.assert_allclose(tf_loss, torch_loss, rtol=1e-5, atol=1e-5)


def test_lovasz_loss_matches_smp_reference():
    np.random.seed(14)
    logits_np = np.random.randn(2, 4, 8, 8).astype(np.float32)
    labels_np = np.random.randint(0, 4, size=(2, 8, 8), dtype=np.int64)
    labels_np[0, 0, 0] = 255

    torch_loss_fn = TorchLovaszLoss("multiclass", ignore_index=255)
    torch_loss = torch_loss_fn(torch.tensor(logits_np), torch.tensor(labels_np)).item()

    tf_loss_fn = TFLovaszLoss("multiclass", ignore_index=255)
    tf_loss = tf_loss_fn(
        tf.convert_to_tensor(labels_np, dtype=tf.int64),
        tf.convert_to_tensor(logits_np, dtype=tf.float32),
    ).numpy()

    np.testing.assert_allclose(tf_loss, torch_loss, rtol=1e-5, atol=1e-5)

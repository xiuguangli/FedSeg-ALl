from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import torch

from segmentation.myseg.bisenet_utils import BackCELoss as TFBackCELoss
from segmentation.myseg.bisenet_utils import OhemCELoss as TFOhemCELoss
from segmentation.myseg.bisenet_utils import SparseCEIgnore as TFSparseCEIgnore


def torch_backce_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_lb: int = 255) -> torch.Tensor:
    criteria = torch.nn.NLLLoss(ignore_index=ignore_lb, reduction="mean")
    total_labels = torch.unique(labels)
    new_labels = labels.clone()
    probs = torch.softmax(logits, dim=1)
    fore_ = []
    back_ = []
    for cls_idx in range(num_classes):
        if cls_idx in total_labels:
            fore_.append(probs[:, cls_idx : cls_idx + 1, :, :])
        else:
            back_.append(probs[:, cls_idx : cls_idx + 1, :, :])
    flag = False
    if len(fore_) != num_classes:
        fore_.append(sum(back_))
        flag = True
    for mapped_idx, label_value in enumerate(total_labels):
        if flag:
            new_labels[labels == label_value] = mapped_idx
        elif int(label_value) != ignore_lb:
            new_labels[labels == label_value] = mapped_idx
    probs = torch.cat(fore_, dim=1)
    log_probs = torch.log(probs + 1e-7)
    return criteria(log_probs, new_labels.long())


def test_backce_loss_matches_torch_reference():
    np.random.seed(42)
    logits_np = np.random.randn(2, 5, 8, 8).astype(np.float32)
    labels_np = np.random.randint(0, 5, size=(2, 8, 8), dtype=np.int64)
    labels_np[0, 0, 0] = 255

    torch_loss = torch_backce_loss(
        torch.tensor(logits_np),
        torch.tensor(labels_np),
        num_classes=5,
    ).item()

    tf_loss_fn = TFBackCELoss(SimpleNamespace(num_classes=5))
    tf_loss = tf_loss_fn(
        tf.convert_to_tensor(labels_np, dtype=tf.int64),
        tf.convert_to_tensor(logits_np, dtype=tf.float32),
    ).numpy()

    np.testing.assert_allclose(tf_loss, torch_loss, rtol=1e-5, atol=1e-5)


def torch_ohem_loss(logits: torch.Tensor, labels: torch.Tensor, thresh: float = 0.7, ignore_lb: int = 255) -> torch.Tensor:
    threshold = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float32))
    criteria = torch.nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")
    n_min = int(labels[labels != ignore_lb].numel() * 0.25)
    loss = criteria(logits, labels).view(-1)
    loss_hard = loss[loss > threshold]
    if loss_hard.numel() < n_min:
        loss_hard, _ = loss.topk(n_min)
    return torch.mean(loss_hard)


def test_training_ce_back_ohem_losses_match_torch_reference():
    np.random.seed(20260605)
    num_classes = 5
    logits_np = np.random.randn(2, num_classes, 4, 5).astype(np.float32) * 1.3
    valid_labels = np.random.randint(0, num_classes, size=(2, 4, 5), dtype=np.int64)
    labels_with_ignore = valid_labels.copy()
    labels_with_ignore[0, 0, 0] = 255
    labels_with_ignore[1, 1, 2] = 255
    labels_missing_classes = np.random.choice([0, 2, 255], size=(2, 4, 5), p=[0.45, 0.45, 0.10]).astype(np.int64)

    args = SimpleNamespace(num_classes=num_classes)
    tf_logits = tf.convert_to_tensor(logits_np, dtype=tf.float32)
    torch_logits = torch.tensor(logits_np, dtype=torch.float32)

    for labels_np in (valid_labels, labels_with_ignore, labels_missing_classes):
        tf_labels = tf.convert_to_tensor(labels_np, dtype=tf.int64)
        torch_labels = torch.tensor(labels_np, dtype=torch.long)

        torch_ce = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="mean")(torch_logits, torch_labels).item()
        tf_ce = TFSparseCEIgnore()(tf_labels, tf_logits).numpy()
        np.testing.assert_allclose(tf_ce, torch_ce, rtol=1e-5, atol=1e-5)

        torch_back = torch_backce_loss(torch_logits, torch_labels, num_classes=num_classes).item()
        tf_back = TFBackCELoss(args)(tf_labels, tf_logits).numpy()
        np.testing.assert_allclose(tf_back, torch_back, rtol=1e-5, atol=1e-5)

        torch_ohem = torch_ohem_loss(torch_logits, torch_labels).item()
        tf_ohem = TFOhemCELoss(0.7)(tf_labels, tf_logits).numpy()
        np.testing.assert_allclose(tf_ohem, torch_ohem, rtol=1e-5, atol=1e-5)

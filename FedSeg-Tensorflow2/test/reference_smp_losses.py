from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn


def _binary_focal_loss_with_logits(
    output: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[float] = 0.25,
    reduction: str = "mean",
    normalized: bool = False,
    reduced_threshold: Optional[float] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    target = target.type_as(output)
    logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
    pt = torch.exp(-logpt)

    if reduced_threshold is None:
        focal_term = (1.0 - pt).pow(gamma)
    else:
        focal_term = ((1.0 - pt) / reduced_threshold).pow(gamma)
        focal_term = torch.where(pt < reduced_threshold, torch.ones_like(focal_term), focal_term)

    loss = focal_term * logpt
    if alpha is not None:
        loss = loss * (alpha * target + (1.0 - alpha) * (1.0 - target))

    if normalized:
        loss = loss / focal_term.sum().clamp_min(eps)

    if reduction == "sum":
        return loss.sum()
    if reduction == "batchwise_mean":
        return loss.sum(0)
    if reduction == "none":
        return loss
    return loss.mean()


class SoftBCEWithLogitsLoss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = -100, reduction: str = "mean", smooth_factor: Optional[float] = None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.smooth_factor is not None:
            targets = (1.0 - y_true) * self.smooth_factor + y_true * (1.0 - self.smooth_factor)
        else:
            targets = y_true

        loss = F.binary_cross_entropy_with_logits(y_pred, targets, reduction="none")
        if self.ignore_index is not None:
            loss = loss * (y_true != self.ignore_index).type_as(loss)

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class FocalLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        ignore_index: Optional[int] = None,
        reduction: str = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        super().__init__()
        if mode != "multiclass":
            raise ValueError(f"unsupported focal mode: {mode}")
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        num_classes = y_pred.size(1)
        total = y_pred.new_tensor(0.0)
        not_ignored = None if self.ignore_index is None else (y_true != self.ignore_index)

        for cls_idx in range(num_classes):
            cls_true = (y_true == cls_idx).long()
            cls_pred = y_pred[:, cls_idx, ...]
            if not_ignored is not None:
                cls_true = cls_true[not_ignored]
                cls_pred = cls_pred[not_ignored]
            total = total + _binary_focal_loss_with_logits(
                cls_pred,
                cls_true,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=self.reduction,
                normalized=self.normalized,
                reduced_threshold=self.reduced_threshold,
            )
        return total


def _soft_tversky_score(
    output: torch.Tensor,
    target: torch.Tensor,
    alpha: float,
    beta: float,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    if dims is None:
        output_sum = output.sum()
        target_sum = target.sum()
        difference = torch.linalg.vector_norm(output - target, ord=1)
    else:
        output_sum = output.sum(dim=dims)
        target_sum = target.sum(dim=dims)
        difference = torch.linalg.vector_norm(output - target, ord=1, dim=dims)

    intersection = (output_sum + target_sum - difference) / 2.0
    fp = output_sum - intersection
    fn = target_sum - intersection
    return (intersection + smooth) / (intersection + alpha * fp + beta * fn + smooth).clamp_min(eps)


def _soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims: Optional[Iterable[int]] = None,
) -> torch.Tensor:
    return _soft_tversky_score(output, target, alpha=0.5, beta=0.5, smooth=smooth, eps=eps, dims=dims)


class DiceLoss(nn.Module):
    def __init__(
        self,
        mode: str,
        classes=None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        ignore_index: Optional[int] = None,
        eps: float = 1e-7,
    ):
        super().__init__()
        if mode != "multiclass":
            raise ValueError(f"unsupported dice mode: {mode}")
        self.mode = mode
        self.classes = classes
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=1).exp()

        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)
            safe_true = (y_true * mask).to(torch.long)
            y_true = F.one_hot(safe_true, num_classes).permute(0, 2, 1) * mask.unsqueeze(1)
        else:
            y_true = F.one_hot(y_true, num_classes).permute(0, 2, 1)

        y_true = y_true.type_as(y_pred)
        scores = _soft_dice_score(y_pred, y_true, smooth=self.smooth, eps=self.eps, dims=(0, 2))
        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        non_empty = y_true.sum((0, 2)) > 0
        loss = loss * non_empty.to(loss.dtype)

        if self.classes is not None:
            loss = loss[self.classes]
        return loss.mean()


def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
    gt_sorted = gt_sorted.float()
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1.0 - gt_sorted).cumsum(0)
    jaccard = 1.0 - intersection / union
    if len(gt_sorted) > 1:
        jaccard[1:] = jaccard[1:] - jaccard[:-1]
    return jaccard


def _flatten_probas(probas: torch.Tensor, labels: torch.Tensor, ignore: Optional[int] = None):
    if probas.dim() == 3:
        bsz, hgt, wid = probas.size()
        probas = probas.view(bsz, 1, hgt, wid)
    num_classes = probas.size(1)
    probas = torch.movedim(probas, 1, -1).contiguous().view(-1, num_classes)
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = labels != ignore
    return probas[valid], labels[valid]


def _lovasz_softmax_flat(probas: torch.Tensor, labels: torch.Tensor, classes="present") -> torch.Tensor:
    if probas.numel() == 0:
        return probas.sum() * 0.0

    num_classes = probas.size(1)
    class_ids = list(range(num_classes)) if classes in {"all", "present"} else classes
    losses = []
    for cls_idx in class_ids:
        fg = (labels == cls_idx).type_as(probas)
        if classes == "present" and fg.sum() == 0:
            continue
        errors = (fg - probas[:, cls_idx]).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return probas.sum() * 0.0
    return sum(losses) / len(losses)


class LovaszLoss(nn.Module):
    def __init__(self, mode: str, per_image: bool = False, ignore_index: Optional[int] = None, from_logits: bool = True):
        super().__init__()
        if mode != "multiclass":
            raise ValueError(f"unsupported lovasz mode: {mode}")
        self.mode = mode
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        probas = y_pred.softmax(dim=1) if self.from_logits else y_pred
        if self.per_image:
            losses = []
            for prob, label in zip(probas, y_true):
                prob_flat, label_flat = _flatten_probas(prob.unsqueeze(0), label.unsqueeze(0), self.ignore_index)
                losses.append(_lovasz_softmax_flat(prob_flat, label_flat, classes="present"))
            return sum(losses) / len(losses)
        prob_flat, label_flat = _flatten_probas(probas, y_true, self.ignore_index)
        return _lovasz_softmax_flat(prob_flat, label_flat, classes="present")

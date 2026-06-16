from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from segmentation.myseg.bisenet_utils import set_model_bisenetv2
from segmentation.update import LocalUpdate


class TinyDataset:
    def __init__(self, num_samples: int = 4, num_classes: int = 3):
        self.images = np.random.randn(num_samples, 3, 64, 64).astype(np.float32)
        self.labels = np.random.randint(0, num_classes, size=(num_samples, 64, 64), dtype=np.int64)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


def test_local_update_smoke_runs_one_step():
    np.random.seed(0)
    tf.random.set_seed(0)
    args = SimpleNamespace(
        local_bs=2,
        num_workers=0,
        losstype="ce",
        temp_dist=0.07,
        max_anchor=32,
        temperature=0.07,
        num_classes=3,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        model="bisenetv2",
        local_ep=1,
        is_proto=False,
        proto_start_epoch=1,
        kmean_num=0,
        pseudo_label=False,
        pseudo_label_start_epoch=1,
        con_lamb=0.1,
        con_lamb_local=0.1,
        fedprox_mu=0.0,
        distill=False,
        distill_lamb_pi=0.0,
        distill_lamb_pa=0.0,
        verbose=0,
        proj_dim=16,
    )
    dataset = TinyDataset(num_samples=4, num_classes=args.num_classes)
    updater = LocalUpdate(args, dataset, idxs=range(len(dataset)))
    model = set_model_bisenetv2(args, num_classes=args.num_classes)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)

    weights, loss = updater.update_weights(model=model, global_round=0)

    assert isinstance(weights, list)
    assert len(weights) > 0
    assert np.isfinite(loss)


def test_local_update_smoke_runs_one_step_with_bce_loss():
    np.random.seed(1)
    tf.random.set_seed(1)
    args = SimpleNamespace(
        local_bs=2,
        num_workers=0,
        losstype="bce",
        temp_dist=0.07,
        max_anchor=32,
        temperature=0.07,
        num_classes=3,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        model="bisenetv2",
        local_ep=1,
        is_proto=False,
        proto_start_epoch=1,
        kmean_num=0,
        pseudo_label=False,
        pseudo_label_start_epoch=1,
        con_lamb=0.1,
        con_lamb_local=0.1,
        fedprox_mu=0.0,
        distill=False,
        distill_lamb_pi=0.0,
        distill_lamb_pa=0.0,
        verbose=0,
        proj_dim=16,
    )
    dataset = TinyDataset(num_samples=4, num_classes=args.num_classes)
    updater = LocalUpdate(args, dataset, idxs=range(len(dataset)))
    model = set_model_bisenetv2(args, num_classes=args.num_classes)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)

    weights, loss = updater.update_weights(model=model, global_round=0)

    assert isinstance(weights, list)
    assert len(weights) > 0
    assert np.isfinite(loss)


def test_local_update_smoke_runs_one_step_with_focal_loss():
    np.random.seed(2)
    tf.random.set_seed(2)
    args = SimpleNamespace(
        local_bs=2,
        num_workers=0,
        losstype="focal",
        temp_dist=0.07,
        max_anchor=32,
        temperature=0.07,
        num_classes=3,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        model="bisenetv2",
        local_ep=1,
        is_proto=False,
        proto_start_epoch=1,
        kmean_num=0,
        pseudo_label=False,
        pseudo_label_start_epoch=1,
        con_lamb=0.1,
        con_lamb_local=0.1,
        fedprox_mu=0.0,
        distill=False,
        distill_lamb_pi=0.0,
        distill_lamb_pa=0.0,
        verbose=0,
        proj_dim=16,
    )
    dataset = TinyDataset(num_samples=4, num_classes=args.num_classes)
    updater = LocalUpdate(args, dataset, idxs=range(len(dataset)))
    model = set_model_bisenetv2(args, num_classes=args.num_classes)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)

    weights, loss = updater.update_weights(model=model, global_round=0)

    assert isinstance(weights, list)
    assert len(weights) > 0
    assert np.isfinite(loss)


def test_local_update_smoke_runs_one_step_with_dice_loss():
    np.random.seed(3)
    tf.random.set_seed(3)
    args = SimpleNamespace(
        local_bs=2,
        num_workers=0,
        losstype="dice",
        temp_dist=0.07,
        max_anchor=32,
        temperature=0.07,
        num_classes=3,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0,
        model="bisenetv2",
        local_ep=1,
        is_proto=False,
        proto_start_epoch=1,
        kmean_num=0,
        pseudo_label=False,
        pseudo_label_start_epoch=1,
        con_lamb=0.1,
        con_lamb_local=0.1,
        fedprox_mu=0.0,
        distill=False,
        distill_lamb_pi=0.0,
        distill_lamb_pa=0.0,
        verbose=0,
        proj_dim=16,
    )
    dataset = TinyDataset(num_samples=4, num_classes=args.num_classes)
    updater = LocalUpdate(args, dataset, idxs=range(len(dataset)))
    model = set_model_bisenetv2(args, num_classes=args.num_classes)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)

    weights, loss = updater.update_weights(model=model, global_round=0)

    assert isinstance(weights, list)
    assert len(weights) > 0
    assert np.isfinite(loss)

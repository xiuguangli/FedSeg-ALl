from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from segmentation.eval_voc import (
    build_voc_eval_tf_loader,
    evaluate_voc_dataset,
    run_eval_inference_fast,
)
from segmentation.eval_utils import build_tfdata_shape_batched_eval_loader, evaluate_fast_shape_bucket


class TinyVocDataset:
    def __init__(self):
        self.samples = [
            (
                np.zeros((3, 33, 35), dtype=np.float32),
                np.zeros((33, 35), dtype=np.int64),
            ),
            (
                np.ones((3, 47, 31), dtype=np.float32),
                np.ones((47, 31), dtype=np.int64),
            ),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, target = self.samples[idx]
        return tf.convert_to_tensor(image), tf.convert_to_tensor(target)


class TinyEvalModel(tf.keras.Model):
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.aux_mode = "train"

    def call(self, x, training=False):
        b = tf.shape(x)[0]
        h = tf.shape(x)[2]
        w = tf.shape(x)[3]
        zeros = tf.zeros([b, h, w], dtype=tf.float32)
        ones = tf.ones([b, h, w], dtype=tf.float32)
        logits = tf.stack([zeros, ones], axis=1)
        return (logits,)


class TinyPaddedEvalModel(TinyEvalModel):
    def __init__(self, num_classes: int):
        super().__init__(num_classes)
        self.assume_padded_input = False


def test_evaluate_voc_dataset_shape_bucket_smoke():
    dataset = TinyVocDataset()
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = evaluate_voc_dataset(
        model,
        dataset,
        num_classes=2,
        batch_size=2,
        dataset_size=len(dataset),
        eval_buckets=[(64, 64)],
    )
    assert model.aux_mode == "train"
    assert np.isfinite(acc_global)
    assert np.isfinite(iou_mean)
    assert "mean IoU" in confmat_text


def test_evaluate_voc_dataset_restores_assume_padded_input():
    dataset = TinyVocDataset()
    model = TinyPaddedEvalModel(num_classes=2)
    model.assume_padded_input = False

    evaluate_voc_dataset(
        model,
        dataset,
        num_classes=2,
        batch_size=2,
        dataset_size=len(dataset),
        eval_buckets=[(64, 64)],
    )

    assert model.aux_mode == "train"
    assert model.assume_padded_input is False


def test_build_voc_eval_tf_loader_compat_smoke():
    args = SimpleNamespace(num_classes=2, eval_bs=1)
    dataset = TinyVocDataset()
    loader = build_voc_eval_tf_loader(dataset, batch_size=1)
    assert loader is dataset
    assert args.eval_bs == 1


def test_run_eval_inference_fast_smoke():
    args = SimpleNamespace(num_classes=2, eval_bs=2, eval_buckets="64x64")
    dataset = TinyVocDataset()
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = run_eval_inference_fast(args, model, dataset, dataset)
    assert np.isfinite(acc_global)
    assert np.isfinite(iou_mean)
    assert "mean IoU" in confmat_text


def test_evaluate_voc_dataset_tail_pad_batch_smoke():
    dataset = TinyVocDataset()
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = evaluate_voc_dataset(
        model,
        dataset,
        num_classes=2,
        batch_size=4,
        dataset_size=len(dataset),
        eval_buckets=[(64, 64)],
        tail_pad_batch=True,
    )
    assert model.aux_mode == "train"
    assert np.isfinite(acc_global)
    assert np.isfinite(iou_mean)
    assert "mean IoU" in confmat_text


def test_evaluate_voc_dataset_tf_metric_smoke():
    dataset = TinyVocDataset()
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = evaluate_voc_dataset(
        model,
        dataset,
        num_classes=2,
        batch_size=2,
        dataset_size=len(dataset),
        eval_buckets=[(64, 64)],
        use_tf_metric=True,
    )
    assert model.aux_mode == "train"
    assert np.isfinite(acc_global)
    assert np.isfinite(iou_mean)
    assert "mean IoU" in confmat_text


def test_evaluate_voc_dataset_tfdata_batch_smoke():
    images = tf.zeros([2, 3, 64, 64], dtype=tf.float32)
    targets = tf.zeros([2, 64, 64], dtype=tf.int64)
    loader = tf.data.Dataset.from_tensors((images, targets))
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = evaluate_voc_dataset(
        model,
        loader,
        num_classes=2,
        batch_size=2,
        dataset_size=2,
        tfdata_batch=True,
    )
    assert model.aux_mode == "train"
    assert np.isfinite(acc_global)
    assert np.isfinite(iou_mean)
    assert "mean IoU" in confmat_text


def test_generic_fast_eval_handles_mixed_shapes():
    samples = [
        (np.zeros((3, 31, 33), dtype=np.float32), np.ones((31, 33), dtype=np.int64)),
        (np.zeros((3, 64, 32), dtype=np.float32), np.ones((64, 32), dtype=np.int64)),
    ]
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = evaluate_fast_shape_bucket(
        model,
        samples,
        num_classes=2,
        batch_size=2,
        dataset_size=len(samples),
        eval_buckets=[],
    )

    assert model.aux_mode == "train"
    assert np.isclose(acc_global, 100.0)
    assert np.isclose(iou_mean, 100.0)
    assert "mean IoU: 100.0" in confmat_text


def test_generic_tfdata_fast_eval_handles_buckets():
    samples = [
        (np.zeros((3, 31, 33), dtype=np.float32), np.ones((31, 33), dtype=np.int64)),
        (np.zeros((3, 64, 32), dtype=np.float32), np.ones((64, 32), dtype=np.int64)),
    ]
    loader = build_tfdata_shape_batched_eval_loader(
        samples,
        batch_size=2,
        eval_buckets=[(64, 64)],
        num_parallel_calls=1,
    )
    model = TinyEvalModel(num_classes=2)
    acc_global, iou_mean, confmat_text = evaluate_fast_shape_bucket(
        model,
        loader,
        num_classes=2,
        batch_size=2,
        dataset_size=len(samples),
        eval_buckets=[(64, 64)],
        tfdata_batch=True,
    )

    assert model.aux_mode == "train"
    assert np.isclose(acc_global, 100.0)
    assert np.isclose(iou_mean, 100.0)
    assert "mean IoU: 100.0" in confmat_text

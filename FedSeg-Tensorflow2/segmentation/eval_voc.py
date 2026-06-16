from __future__ import annotations

from collections import defaultdict
import copy
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from options import args_parser
from tqdm import tqdm
from runtime_utils import (
    configure_tensorflow_env,
    configure_tensorflow_runtime,
    install_tensorflow_stderr_filter,
    should_disable_tqdm,
    should_use_xla,
)

_BOOTSTRAP_ARGS = args_parser(allow_unknown=True) if __name__ == "__main__" else None

configure_tensorflow_env(gpu=_BOOTSTRAP_ARGS.gpu if _BOOTSTRAP_ARGS is not None else None)
os.environ.setdefault("FEDSEG_FORCE_NHWC_CONV", "1")
install_tensorflow_stderr_filter()

import tensorflow as tf

from logging_utils import logger, setup_logger
from myseg.bisenet_utils import set_model_bisenetv2
from tensorflow_runtime import require_tensorflow_device
from tf2_tools import build_fast_tf_bisenetv2_from_model

configure_tensorflow_runtime(tf)

warnings.filterwarnings("ignore")

TF_CHECKPOINT_SUFFIX = ".weights.h5"
SUPPORTED_CHECKPOINT_SUFFIXES = (TF_CHECKPOINT_SUFFIX,)

VOC_MEAN = np.asarray((0.3257, 0.3690, 0.3223), dtype=np.float32)[:, None, None]
VOC_STD = np.asarray((0.2112, 0.2148, 0.2115), dtype=np.float32)[:, None, None]
VOC_MEAN_TF = tf.constant((0.3257, 0.3690, 0.3223), dtype=tf.float32)
VOC_STD_TF = tf.constant((0.2112, 0.2148, 0.2115), dtype=tf.float32)


def _make_voc_colormap() -> np.ndarray:
    colormap = np.zeros((256, 3), dtype=np.uint8)
    for idx in range(256):
        r = g = b = 0
        value = idx
        for bit in range(8):
            r |= ((value >> 0) & 1) << (7 - bit)
            g |= ((value >> 1) & 1) << (7 - bit)
            b |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
        colormap[idx] = (r, g, b)
    return colormap


VOC_COLORMAP = _make_voc_colormap()
VOC_COLOR_CODES = (
    VOC_COLORMAP[:, 0].astype(np.int64) * 65536
    + VOC_COLORMAP[:, 1].astype(np.int64) * 256
    + VOC_COLORMAP[:, 2].astype(np.int64)
)
VOC_COLOR_TO_LABEL = {int(code): idx for idx, code in enumerate(VOC_COLOR_CODES.tolist())}
VOC_COLOR_KEYS_TF = tf.constant(VOC_COLOR_CODES, dtype=tf.int64)
VOC_COLOR_VALUES_TF = tf.constant(np.arange(256, dtype=np.int64), dtype=tf.int64)


def resolve_path(root_dir: str | Path, path_value: str | Path) -> Path:
    path_value = Path(path_value)
    if path_value.is_absolute():
        return path_value.resolve()
    return (Path(root_dir) / path_value).resolve()


def _read_images_dir(root_dir: Path, folder: str) -> list[Path]:
    image_dirs = []
    for city_name in sorted(os.listdir(root_dir / folder)):
        city_dir = root_dir / folder / city_name
        for image_name in os.listdir(city_dir):
            image_dirs.append(city_dir / image_name)
    return sorted(image_dirs)


def _normalize_voc_label_indices(label: np.ndarray) -> np.ndarray:
    label = label.copy()
    label[label == 255] = 0
    label = label.astype(np.uint8, copy=False) - 1
    return label.astype(np.int64, copy=False)


def _decode_voc_label_numpy(label_path: str | Path) -> np.ndarray:
    label_rgb = tf.io.decode_png(tf.io.read_file(str(label_path)), channels=3).numpy().astype(np.int64)
    color_codes = label_rgb[:, :, 0] * 65536 + label_rgb[:, :, 1] * 256 + label_rgb[:, :, 2]
    label = np.zeros(color_codes.shape, dtype=np.uint8)
    for color_code, label_id in VOC_COLOR_TO_LABEL.items():
        label[color_codes == color_code] = label_id
    return _normalize_voc_label_indices(label)


class VocEvalDataset:
    val_folder = "images/val"
    val_lb_folder = "labels/val"

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.image_dirs = _read_images_dir(self.root_dir, self.val_folder)
        self.label_dirs = _read_images_dir(self.root_dir, self.val_lb_folder)
        if len(self.image_dirs) != len(self.label_dirs):
            raise ValueError("VOC image and label counts do not match")
        logger.info("Found {} val examples", len(self.image_dirs))

    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        image = tf.io.decode_jpeg(
            tf.io.read_file(str(self.image_dirs[idx])),
            channels=3,
            dct_method="INTEGER_ACCURATE",
        ).numpy().astype(np.float32)
        label = _decode_voc_label_numpy(self.label_dirs[idx])

        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = ((image - VOC_MEAN) / VOC_STD).astype(np.float32, copy=False)
        return image, label


def build_voc_eval_dataset(eval_args, project_root: Path):
    if eval_args.dataset != "voc":
        raise ValueError("eval_voc.py only supports --dataset=voc")
    dataset_root = resolve_path(project_root, eval_args.root_dir)
    dataset = VocEvalDataset(dataset_root)
    logger.info("Prepared VOC val dataset with {} samples", len(dataset))
    return dataset


class SequentialEvalLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        for idx in range(len(self.dataset)):
            yield self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def build_voc_eval_loader(eval_args, dataset):
    num_parallel_calls = max(1, int(getattr(eval_args, "num_workers", 1)))
    label_lookup = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(VOC_COLOR_KEYS_TF, VOC_COLOR_VALUES_TF),
        default_value=tf.constant(0, dtype=tf.int64),
    )
    eval_buckets = _parse_eval_buckets(getattr(eval_args, "eval_buckets", ""))
    if bool(getattr(eval_args, "eval_tfdata_batch", False)):
        return build_voc_eval_batched_loader(
            dataset,
            batch_size=max(1, int(getattr(eval_args, "eval_bs", 1))),
            eval_buckets=eval_buckets,
            label_lookup=label_lookup,
            num_parallel_calls=num_parallel_calls,
        )

    loader = tf.data.Dataset.from_tensor_slices(
        (
            [str(path) for path in dataset.image_dirs],
            [str(path) for path in dataset.label_dirs],
        )
    )

    def load_sample(image_path, label_path):
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3, dct_method="INTEGER_ACCURATE")
        label_rgb = tf.cast(tf.io.decode_png(tf.io.read_file(label_path), channels=3), tf.int64)
        label_code = label_rgb[:, :, 0] * 65536 + label_rgb[:, :, 1] * 256 + label_rgb[:, :, 2]
        label = label_lookup.lookup(label_code)
        label = tf.where(tf.equal(label, 255), tf.zeros_like(label), label)
        label = tf.cast(tf.cast(label, tf.uint8) - tf.cast(1, tf.uint8), tf.int64)

        image = tf.cast(image, tf.float32) / 255.0
        image = (image - VOC_MEAN_TF) / VOC_STD_TF
        image = tf.transpose(image, [2, 0, 1])
        return image, tf.cast(label, tf.int64)

    loader = loader.map(load_sample, num_parallel_calls=num_parallel_calls)
    loader = loader.prefetch(tf.data.AUTOTUNE)
    logger.info(
        "Prepared VOC eval loader (tf.data single-sample stream, num_parallel_calls={})",
        num_parallel_calls,
    )
    return loader


def _reflect_indices(length: tf.Tensor, target_size: tf.Tensor | int) -> tf.Tensor:
    target_size = tf.cast(target_size, tf.int32)
    indices = tf.range(target_size, dtype=tf.int32)
    length = tf.cast(length, tf.int32)

    def one_pixel_axis():
        return tf.zeros([target_size], dtype=tf.int32)

    def reflected_axis():
        period = length * 2 - 2
        reflected = tf.math.floormod(indices, period)
        return tf.where(reflected < length, reflected, period - reflected)

    return tf.cond(length <= 1, one_pixel_axis, reflected_axis)


def _reflect_pad_hwc_to_shape(image: tf.Tensor, target_shape: tuple[int, int]) -> tf.Tensor:
    target_h, target_w = target_shape
    image = tf.gather(image, _reflect_indices(tf.shape(image)[0], target_h), axis=0)
    image = tf.gather(image, _reflect_indices(tf.shape(image)[1], target_w), axis=1)
    return image


def _reflect_pad_hwc_like_eval_path(image: tf.Tensor, target_shape: tuple[int, int]) -> tf.Tensor:
    image_shape = tf.shape(image)
    padded_h = ((image_shape[0] + 31) // 32) * 32
    padded_w = ((image_shape[1] + 31) // 32) * 32
    image = _reflect_pad_hwc_to_shape(image, (padded_h, padded_w))
    return _reflect_pad_hwc_to_shape(image, target_shape)


def _pad_label_to_shape(label: tf.Tensor, target_shape: tuple[int, int]) -> tf.Tensor:
    target_h, target_w = target_shape
    pad_h = target_h - tf.shape(label)[0]
    pad_w = target_w - tf.shape(label)[1]
    return tf.pad(label, [[0, pad_h], [0, pad_w]], mode="CONSTANT", constant_values=-1)


def _select_eval_bucket_tf(
    padded_h: tf.Tensor,
    padded_w: tf.Tensor,
    eval_buckets: list[tuple[int, int]],
) -> tuple[tf.Tensor, tf.Tensor]:
    if not eval_buckets:
        return padded_h, padded_w

    buckets = tf.constant(eval_buckets, dtype=tf.int32)
    bucket_h = buckets[:, 0]
    bucket_w = buckets[:, 1]
    fits = tf.logical_and(padded_h <= bucket_h, padded_w <= bucket_w)
    scores = bucket_h * bucket_w * 1_000_000 + bucket_h * 1_000 + bucket_w
    best_idx = tf.argmin(tf.where(fits, scores, tf.fill(tf.shape(scores), tf.reduce_max(scores) + 1)))

    def select_fitting_bucket():
        return bucket_h[best_idx], bucket_w[best_idx]

    def select_expanded_bucket():
        max_h = tf.maximum(padded_h, tf.reduce_max(bucket_h))
        max_w = tf.maximum(padded_w, tf.reduce_max(bucket_w))
        return ((max_h + 31) // 32) * 32, ((max_w + 31) // 32) * 32

    return tf.cond(tf.reduce_any(fits), select_fitting_bucket, select_expanded_bucket)


def build_voc_eval_batched_loader(
    dataset: VocEvalDataset,
    batch_size: int,
    eval_buckets: list[tuple[int, int]],
    label_lookup: tf.lookup.StaticHashTable,
    num_parallel_calls: int,
):
    loader = tf.data.Dataset.from_tensor_slices(
        (
            [str(path) for path in dataset.image_dirs],
            [str(path) for path in dataset.label_dirs],
        )
    )

    def load_pad_and_key_sample(image_path, label_path):
        image = tf.io.decode_jpeg(tf.io.read_file(image_path), channels=3, dct_method="INTEGER_ACCURATE")
        label_rgb = tf.cast(tf.io.decode_png(tf.io.read_file(label_path), channels=3), tf.int64)
        label_code = label_rgb[:, :, 0] * 65536 + label_rgb[:, :, 1] * 256 + label_rgb[:, :, 2]
        label = label_lookup.lookup(label_code)
        label = tf.where(tf.equal(label, 255), tf.zeros_like(label), label)
        label = tf.cast(tf.cast(label, tf.uint8) - tf.cast(1, tf.uint8), tf.int64)

        image = tf.cast(image, tf.float32) / 255.0
        image = (image - VOC_MEAN_TF) / VOC_STD_TF
        image_shape = tf.shape(image)
        padded_h = ((image_shape[0] + 31) // 32) * 32
        padded_w = ((image_shape[1] + 31) // 32) * 32
        target_h, target_w = _select_eval_bucket_tf(padded_h, padded_w, eval_buckets)
        target_shape = (target_h, target_w)
        image = _reflect_pad_hwc_like_eval_path(image, target_shape)
        image = tf.transpose(image, [2, 0, 1])
        label = _pad_label_to_shape(label, target_shape)
        key = tf.cast(target_h * 10000 + target_w, tf.int64)
        return key, image, label

    def key_func(key, image, label):
        return key

    def reduce_func(key, window):
        del key
        window = window.map(lambda sample_key, image, label: (image, label), num_parallel_calls=tf.data.AUTOTUNE)
        return window.padded_batch(
            batch_size,
            padded_shapes=([3, None, None], [None, None]),
            padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(-1, dtype=tf.int64)),
            drop_remainder=False,
        )

    loader = loader.map(load_pad_and_key_sample, num_parallel_calls=num_parallel_calls)
    loader = loader.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=batch_size,
    )
    loader = loader.prefetch(tf.data.AUTOTUNE)
    logger.info(
        "Prepared VOC eval loader (tf.data streaming shape-batched stream, eval_bs={} eval_buckets={} num_parallel_calls={})",
        batch_size,
        eval_buckets,
        num_parallel_calls,
    )
    return loader


def build_voc_eval_tf_loader(dataset, batch_size=1):
    return dataset


def _make_progress_bar(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", True)
    return tqdm(*args, **kwargs)


def _to_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.asarray(value)


def _env_flag(name: str) -> bool:
    value = os.environ.get(name, "")
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _should_profile_eval_batches(profile_runtime: bool) -> bool:
    return bool(profile_runtime) or _env_flag("FEDSEG_PROFILE_SHAPES")


def _append_eval_batch_profile(
    records: list[dict[str, object]],
    *,
    mode: str,
    padded_shape: tuple[int, int],
    batch_size: int,
    compute_batch_size: int | None = None,
    flush_kind: str,
    stack_elapsed: float,
    to_tensor_elapsed: float,
    graph_elapsed: float,
    pred_copy_elapsed: float,
    metric_elapsed: float,
) -> None:
    records.append(
        {
            "mode": mode,
            "shape": padded_shape,
            "batch_size": batch_size,
            "compute_batch_size": compute_batch_size if compute_batch_size is not None else batch_size,
            "flush_kind": flush_kind,
            "stack": stack_elapsed,
            "to_tensor": to_tensor_elapsed,
            "graph": graph_elapsed,
            "pred_copy": pred_copy_elapsed,
            "metric": metric_elapsed,
            "total": stack_elapsed + to_tensor_elapsed + graph_elapsed + pred_copy_elapsed + metric_elapsed,
        }
    )


def _format_eval_batch_group_summary(groups: dict[object, dict[str, float | int]], limit: int | None = None) -> str:
    items = sorted(groups.items())
    if limit is not None:
        items = items[:limit]
    return "; ".join(
        "{}:calls={} samples={} graph={:.3f}s graph+copy/img={:.4f}s total={:.3f}s".format(
            key,
            int(values["calls"]),
            int(values["samples"]),
            float(values["graph"]),
            (float(values["graph"]) + float(values["pred_copy"])) / max(1, int(values["samples"])),
            float(values["total"]),
        )
        for key, values in items
    )


def _log_eval_batch_profile(records: list[dict[str, object]], dataset_size: int | None) -> None:
    if not records:
        return

    by_batch_size: dict[int, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "samples": 0, "graph": 0.0, "pred_copy": 0.0, "total": 0.0}
    )
    by_shape: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"calls": 0, "samples": 0, "graph": 0.0, "pred_copy": 0.0, "total": 0.0}
    )
    flush_counts: dict[str, int] = defaultdict(int)
    flush_samples: dict[str, int] = defaultdict(int)

    for record in records:
        batch_size = int(record["batch_size"])
        shape_h, shape_w = record["shape"]
        shape_key = f"{shape_h}x{shape_w}"
        flush_kind = str(record["flush_kind"])
        compute_batch_size = int(record["compute_batch_size"])
        graph_elapsed = float(record["graph"])
        pred_copy_elapsed = float(record["pred_copy"])
        total_elapsed = float(record["total"])

        for groups, key in ((by_batch_size, compute_batch_size), (by_shape, shape_key)):
            groups[key]["calls"] += 1
            groups[key]["samples"] += batch_size
            groups[key]["graph"] += graph_elapsed
            groups[key]["pred_copy"] += pred_copy_elapsed
            groups[key]["total"] += total_elapsed

        flush_counts[flush_kind] += 1
        flush_samples[flush_kind] += batch_size

    total_calls = len(records)
    total_samples = sum(int(record["batch_size"]) for record in records)
    total_compute_samples = sum(int(record["compute_batch_size"]) for record in records)
    total_graph = sum(float(record["graph"]) for record in records)
    total_pred_copy = sum(float(record["pred_copy"]) for record in records)
    total_elapsed = sum(float(record["total"]) for record in records)
    logger.info(
        "VOC eval batch profile | samples={} profiled_samples={} graph_calls={} avg_bs={:.2f} "
        "avg_compute_bs={:.2f} graph={:.3f}s graph+copy/img={:.4f}s total_profiled={:.3f}s "
        "flush_calls={} flush_samples={}",
        dataset_size,
        total_samples,
        total_calls,
        total_samples / max(1, total_calls),
        total_compute_samples / max(1, total_calls),
        total_graph,
        (total_graph + total_pred_copy) / max(1, total_samples),
        total_elapsed,
        dict(sorted(flush_counts.items())),
        dict(sorted(flush_samples.items())),
    )
    logger.info("VOC eval batch profile by_bs | {}", _format_eval_batch_group_summary(by_batch_size))

    slow_shapes = dict(
        sorted(
            by_shape.items(),
            key=lambda item: (float(item[1]["graph"]) + float(item[1]["pred_copy"])),
            reverse=True,
        )[:8]
    )
    logger.info("VOC eval batch profile top_shapes_by_graph_copy | {}", _format_eval_batch_group_summary(slow_shapes))

    inefficient_shapes = dict(
        sorted(
            by_shape.items(),
            key=lambda item: (
                (float(item[1]["graph"]) + float(item[1]["pred_copy"])) / max(1, int(item[1]["samples"])),
                float(item[1]["graph"]) + float(item[1]["pred_copy"]),
            ),
            reverse=True,
        )[:8]
    )
    logger.info("VOC eval batch profile top_shapes_by_graph_copy_per_img | {}", _format_eval_batch_group_summary(inefficient_shapes))


def _make_eval_pred_fn(model):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 3, None, None], dtype=tf.float32),
        ],
        jit_compile=should_use_xla(),
        reduce_retracing=True,
    )
    def _eval_pred(images):
        logits = model(images, training=False)[0]
        return tf.argmax(logits, axis=1, output_type=tf.int64)

    return _eval_pred


def _make_eval_confmat_fn(model, num_classes: int):
    @tf.function(
        input_signature=[
            tf.TensorSpec(shape=[None, 3, None, None], dtype=tf.float32),
            tf.TensorSpec(shape=[None, None, None], dtype=tf.int64),
        ],
        jit_compile=should_use_xla(),
        reduce_retracing=True,
    )
    def _eval_confmat(images, targets):
        logits = model(images, training=False)[0]
        pred = tf.argmax(logits, axis=1, output_type=tf.int64)
        target_flat = tf.reshape(targets, [-1])
        pred_flat = tf.reshape(pred, [-1])
        valid_mask = tf.logical_and(target_flat >= 0, target_flat < num_classes)
        target_flat = tf.boolean_mask(target_flat, valid_mask)
        pred_flat = tf.boolean_mask(pred_flat, valid_mask)
        indices = tf.cast(num_classes, tf.int64) * target_flat + pred_flat
        batch_confmat = tf.math.bincount(
            indices,
            minlength=num_classes * num_classes,
            maxlength=num_classes * num_classes,
            dtype=tf.int64,
        )
        return tf.reshape(batch_confmat, [num_classes, num_classes])

    return _eval_confmat


def _pad_to_multiple_of_32(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = int(image.shape[1]), int(image.shape[2])
    padded_h = ((h + 31) // 32) * 32
    padded_w = ((w + 31) // 32) * 32
    if padded_h == h and padded_w == w:
        return image, (padded_h, padded_w)

    pad_bottom = padded_h - h
    pad_right = padded_w - w
    return np.pad(
        image,
        ((0, 0), (0, pad_bottom), (0, pad_right)),
        mode="reflect",
    ), (padded_h, padded_w)


def _parse_eval_buckets(value: str | None) -> list[tuple[int, int]]:
    if not value:
        return []
    buckets = []
    for raw_item in value.split(","):
        item = raw_item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"invalid eval bucket {raw_item!r}; expected HxW")
        h_text, w_text = item.split("x", 1)
        h, w = int(h_text), int(w_text)
        if h <= 0 or w <= 0 or h % 32 != 0 or w % 32 != 0:
            raise ValueError(f"invalid eval bucket {raw_item!r}; dimensions must be positive multiples of 32")
        buckets.append((h, w))
    return buckets


def _select_eval_bucket(shape: tuple[int, int], buckets: list[tuple[int, int]]) -> tuple[int, int]:
    if not buckets:
        return shape
    h, w = shape
    candidates = [(bh, bw) for bh, bw in buckets if h <= bh and w <= bw]
    if candidates:
        return min(candidates, key=lambda bucket: (bucket[0] * bucket[1], bucket[0], bucket[1]))
    max_h = max(h, max(bucket[0] for bucket in buckets))
    max_w = max(w, max(bucket[1] for bucket in buckets))
    return (((max_h + 31) // 32) * 32, ((max_w + 31) // 32) * 32)


def _pad_to_shape(image: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = int(image.shape[1]), int(image.shape[2])
    target_h, target_w = target_shape
    if target_h == h and target_w == w:
        return image
    if target_h < h or target_w < w:
        raise ValueError(f"target shape {target_shape} is smaller than image shape {(h, w)}")
    return np.pad(
        image,
        ((0, 0), (0, target_h - h), (0, target_w - w)),
        mode="reflect",
    )


def _update_confmat_numpy(confmat: np.ndarray, target: np.ndarray, pred: np.ndarray, num_classes: int) -> None:
    target_flat = target.reshape(-1)
    pred_flat = pred.reshape(-1)
    valid_mask = (target_flat >= 0) & (target_flat < num_classes)
    if not np.any(valid_mask):
        return

    indices = num_classes * target_flat[valid_mask].astype(np.int64) + pred_flat[valid_mask].astype(np.int64)
    confmat += np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _stack_targets_to_shape(
    targets: list[np.ndarray],
    padded_shape: tuple[int, int],
    batch_size: int | None = None,
) -> np.ndarray:
    target_h, target_w = padded_shape
    batch_size = len(targets) if batch_size is None else batch_size
    batch_targets = np.full((batch_size, target_h, target_w), -1, dtype=np.int64)
    for sample_idx, target in enumerate(targets):
        h, w = target.shape
        batch_targets[sample_idx, :h, :w] = target
    return batch_targets


def _iter_prebatched_eval_samples(
    data_loader,
    batch_size: int,
    eval_buckets: list[tuple[int, int]] | None,
):
    buckets = {}
    for image, target in data_loader:
        image = _to_numpy(image)
        target = _to_numpy(target)
        padded_image, padded_shape = _pad_to_multiple_of_32(image)
        bucket_shape = _select_eval_bucket(padded_shape, eval_buckets or [])
        if bucket_shape != padded_shape:
            padded_image = _pad_to_shape(padded_image, bucket_shape)
            padded_shape = bucket_shape
        bucket = buckets.setdefault(padded_shape, {"images": [], "targets": []})
        bucket["images"].append(padded_image)
        bucket["targets"].append(target.astype(np.int64, copy=False))

    for padded_shape in sorted(buckets):
        bucket = buckets[padded_shape]
        images = bucket["images"]
        targets = bucket["targets"]
        for start in range(0, len(images), batch_size):
            batch_images = np.stack(images[start : start + batch_size], axis=0).astype(np.float32, copy=False)
            batch_targets = targets[start : start + batch_size]
            yield padded_shape, batch_images, batch_targets


def _compute_confmat_metrics(confmat: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
    h = confmat.astype(np.float32, copy=False)
    total = float(h.sum())
    acc_global = float(np.diag(h).sum() / (total + 1e-10) * 100.0)
    acc = np.diag(h) / (h.sum(axis=1) + 1e-10)
    iu = np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h) + 1e-10)
    valid = np.isfinite(iu) & ((h.sum(axis=1) + h.sum(axis=0) - np.diag(h)) > 0)
    iou_mean = float(iu[valid].mean() * 100.0) if np.any(valid) else 0.0
    return acc_global, acc, iu, iou_mean


def _format_confmat_summary(acc_global: float, acc: np.ndarray, iu: np.ndarray, iou_mean: float) -> str:
    return (
        "global correct: {:.1f}\n"
        "average row correct: {}\n"
        "IoU: {}\n"
        "mean IoU: {:.1f}"
    ).format(
        acc_global,
        ["{:.1f}".format(i) for i in (acc * 100).tolist()],
        ["{:.1f}".format(i) for i in (iu * 100).tolist()],
        iou_mean,
    )


def evaluate_voc_dataset(
    model,
    data_loader,
    num_classes: int,
    batch_size: int,
    dataset_size: int | None = None,
    eval_buckets: list[tuple[int, int]] | None = None,
    prebatch: bool = False,
    profile_runtime: bool = False,
    tail_pad_batch: bool = False,
    use_tf_metric: bool = False,
    tfdata_batch: bool = False,
):
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.aux_mode = "eval"
    previous_assume_padded_input = getattr(model, "assume_padded_input", None)
    if previous_assume_padded_input is not None:
        model.assume_padded_input = True
    eval_pred = _make_eval_pred_fn(model)
    eval_confmat = _make_eval_confmat_fn(model, num_classes) if use_tf_metric else None
    timer = time.perf_counter
    stage = {
        "numpy": 0.0,
        "pad": 0.0,
        "stack": 0.0,
        "to_tensor": 0.0,
        "graph": 0.0,
        "graph_first": 0.0,
        "graph_repeat": 0.0,
        "pred_copy": 0.0,
        "metric": 0.0,
    }
    pending_batches = {}
    seen_shapes = set()
    graph_call_count = 0
    disable_tqdm = should_disable_tqdm()
    batch_profile_records: list[dict[str, object]] = []
    should_profile_batches = _should_profile_eval_batches(profile_runtime)

    if dataset_size is None:
        try:
            dataset_size = len(data_loader)
        except TypeError:
            dataset_size = None

    if tfdata_batch:
        infer_bar = _make_progress_bar(
            total=dataset_size,
            desc="Evaluating VOC",
            disable=disable_tqdm,
        )
        for batch_images, batch_targets in data_loader:
            actual_batch_size = int(batch_images.shape[0])
            padded_shape = (int(batch_images.shape[2]), int(batch_images.shape[3]))
            seen_shapes.add(padded_shape)

            t_graph = timer()
            if use_tf_metric:
                batch_confmat = eval_confmat(batch_images, batch_targets)
            else:
                batch_pred = eval_pred(batch_images)
            graph_elapsed = timer() - t_graph
            stage["graph"] += graph_elapsed
            if graph_call_count == 0:
                stage["graph_first"] += graph_elapsed
            else:
                stage["graph_repeat"] += graph_elapsed
            graph_call_count += 1

            t_pred_copy = timer()
            if use_tf_metric:
                batch_confmat = batch_confmat.numpy()
            else:
                batch_pred = batch_pred.numpy()
                batch_targets = batch_targets.numpy()
            pred_copy_elapsed = timer() - t_pred_copy
            stage["pred_copy"] += pred_copy_elapsed

            t_metric = timer()
            if use_tf_metric:
                confmat[...] += batch_confmat
            else:
                _update_confmat_numpy(confmat, batch_targets, batch_pred, num_classes)
            metric_elapsed = timer() - t_metric
            stage["metric"] += metric_elapsed

            if should_profile_batches:
                _append_eval_batch_profile(
                    batch_profile_records,
                    mode="tfdata_shape_batch",
                    padded_shape=padded_shape,
                    batch_size=actual_batch_size,
                    compute_batch_size=actual_batch_size,
                    flush_kind="full" if actual_batch_size >= batch_size else "tail",
                    stack_elapsed=0.0,
                    to_tensor_elapsed=0.0,
                    graph_elapsed=graph_elapsed,
                    pred_copy_elapsed=pred_copy_elapsed,
                    metric_elapsed=metric_elapsed,
                )
            infer_bar.update(actual_batch_size)

        infer_bar.close()
        model.aux_mode = "train"
        if previous_assume_padded_input is not None:
            model.assume_padded_input = previous_assume_padded_input
        acc_global, acc, iu, iou_mean = _compute_confmat_metrics(confmat)
        confmat_summary = _format_confmat_summary(acc_global, acc, iu, iou_mean)
        logger.info(
            "VOC eval runtime | mode=tfdata_shape_batch samples={} unique_padded_shapes={} eval_bs={} xla={} eval_buckets={} numpy={:.2f}s pad={:.2f}s stack={:.2f}s to_tensor={:.2f}s graph={:.2f}s first_graph={:.2f}s repeat_graph={:.2f}s pred_copy={:.2f}s metric={:.2f}s graph_calls={}",
            dataset_size,
            len(seen_shapes),
            batch_size,
            should_use_xla(),
            eval_buckets or [],
            stage["numpy"],
            stage["pad"],
            stage["stack"],
            stage["to_tensor"],
            stage["graph"],
            stage["graph_first"],
            stage["graph_repeat"],
            stage["pred_copy"],
            stage["metric"],
            graph_call_count,
        )
        _log_eval_batch_profile(batch_profile_records, dataset_size)
        return acc_global, iou_mean, confmat_summary

    if prebatch:
        infer_bar = _make_progress_bar(
            total=dataset_size,
            desc="Evaluating VOC",
            disable=disable_tqdm,
        )
        t_prep = timer()
        prebatched = list(_iter_prebatched_eval_samples(data_loader, batch_size, eval_buckets))
        stage["pad"] += timer() - t_prep
        for padded_shape, batch_images, batch_targets in prebatched:
            seen_shapes.add(padded_shape)
            stack_elapsed = 0.0
            stage["stack"] += stack_elapsed
            t_to_tensor = timer()
            batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
            to_tensor_elapsed = timer() - t_to_tensor
            stage["to_tensor"] += to_tensor_elapsed

            t_graph = timer()
            if use_tf_metric:
                batch_target_tensor = tf.convert_to_tensor(
                    _stack_targets_to_shape(batch_targets, padded_shape, batch_size=int(batch_images.shape[0])),
                    dtype=tf.int64,
                )
                batch_confmat = eval_confmat(batch_images, batch_target_tensor)
            else:
                batch_pred = eval_pred(batch_images)
            graph_elapsed = timer() - t_graph
            stage["graph"] += graph_elapsed
            if graph_call_count == 0:
                stage["graph_first"] += graph_elapsed
            else:
                stage["graph_repeat"] += graph_elapsed
            graph_call_count += 1

            t_pred_copy = timer()
            if use_tf_metric:
                batch_confmat = batch_confmat.numpy()
            else:
                batch_pred = batch_pred.numpy()
            pred_copy_elapsed = timer() - t_pred_copy
            stage["pred_copy"] += pred_copy_elapsed

            t_metric = timer()
            if use_tf_metric:
                confmat[...] += batch_confmat
            else:
                for sample_idx, target in enumerate(batch_targets):
                    h, w = target.shape
                    _update_confmat_numpy(confmat, target, batch_pred[sample_idx, :h, :w], num_classes)
            metric_elapsed = timer() - t_metric
            stage["metric"] += metric_elapsed
            if should_profile_batches:
                _append_eval_batch_profile(
                    batch_profile_records,
                    mode="prebatch_shape_bucket",
                    padded_shape=padded_shape,
                    batch_size=len(batch_targets),
                    compute_batch_size=len(batch_targets),
                    flush_kind="full" if len(batch_targets) >= batch_size else "tail",
                    stack_elapsed=stack_elapsed,
                    to_tensor_elapsed=to_tensor_elapsed,
                    graph_elapsed=graph_elapsed,
                    pred_copy_elapsed=pred_copy_elapsed,
                    metric_elapsed=metric_elapsed,
                )
            infer_bar.update(len(batch_targets))

        infer_bar.close()
        model.aux_mode = "train"
        if previous_assume_padded_input is not None:
            model.assume_padded_input = previous_assume_padded_input
        acc_global, acc, iu, iou_mean = _compute_confmat_metrics(confmat)
        confmat_summary = _format_confmat_summary(acc_global, acc, iu, iou_mean)
        logger.info(
            "VOC eval runtime | mode=prebatch_shape_bucket samples={} unique_padded_shapes={} eval_bs={} xla={} eval_buckets={} numpy={:.2f}s pad={:.2f}s stack={:.2f}s to_tensor={:.2f}s graph={:.2f}s first_graph={:.2f}s repeat_graph={:.2f}s pred_copy={:.2f}s metric={:.2f}s graph_calls={}",
            dataset_size,
            len(seen_shapes),
            batch_size,
            should_use_xla(),
            eval_buckets or [],
            stage["numpy"],
            stage["pad"],
            stage["stack"],
            stage["to_tensor"],
            stage["graph"],
            stage["graph_first"],
            stage["graph_repeat"],
            stage["pred_copy"],
            stage["metric"],
            graph_call_count,
        )
        _log_eval_batch_profile(batch_profile_records, dataset_size)
        return acc_global, iou_mean, confmat_summary

    def flush_shape_bucket(padded_shape: tuple[int, int], flush_kind: str = "full") -> int:
        nonlocal graph_call_count
        bucket = pending_batches.get(padded_shape)
        if bucket is None or not bucket["images"]:
            return 0

        t_stack = timer()
        batch_targets = bucket["targets"]
        processed = len(bucket["images"])
        images_for_inference = bucket["images"]
        if tail_pad_batch and flush_kind == "tail" and 0 < processed < batch_size:
            pad_count = batch_size - processed
            images_for_inference = bucket["images"] + [bucket["images"][-1]] * pad_count
            flush_kind = "tail_padded"
        batch_images = np.stack(images_for_inference, axis=0).astype(np.float32, copy=False)
        stack_elapsed = timer() - t_stack
        stage["stack"] += stack_elapsed

        t_to_tensor = timer()
        batch_images = tf.convert_to_tensor(batch_images, dtype=tf.float32)
        to_tensor_elapsed = timer() - t_to_tensor
        stage["to_tensor"] += to_tensor_elapsed

        t_graph = timer()
        if use_tf_metric:
            batch_target_tensor = tf.convert_to_tensor(
                _stack_targets_to_shape(batch_targets, padded_shape, batch_size=int(batch_images.shape[0])),
                dtype=tf.int64,
            )
            batch_confmat = eval_confmat(batch_images, batch_target_tensor)
        else:
            batch_pred = eval_pred(batch_images)
        graph_elapsed = timer() - t_graph
        stage["graph"] += graph_elapsed
        if graph_call_count == 0:
            stage["graph_first"] += graph_elapsed
        else:
            stage["graph_repeat"] += graph_elapsed
        graph_call_count += 1

        t_pred_copy = timer()
        if use_tf_metric:
            batch_confmat = batch_confmat.numpy()
        else:
            batch_pred = batch_pred.numpy()
        pred_copy_elapsed = timer() - t_pred_copy
        stage["pred_copy"] += pred_copy_elapsed

        t_metric = timer()
        if use_tf_metric:
            confmat[...] += batch_confmat
        else:
            for sample_idx, target in enumerate(batch_targets):
                h, w = target.shape
                _update_confmat_numpy(confmat, target, batch_pred[sample_idx, :h, :w], num_classes)
        metric_elapsed = timer() - t_metric
        stage["metric"] += metric_elapsed

        if should_profile_batches:
            _append_eval_batch_profile(
                batch_profile_records,
                mode="shape_bucket",
                padded_shape=padded_shape,
                batch_size=processed,
                compute_batch_size=int(batch_images.shape[0]),
                flush_kind=flush_kind,
                stack_elapsed=stack_elapsed,
                to_tensor_elapsed=to_tensor_elapsed,
                graph_elapsed=graph_elapsed,
                pred_copy_elapsed=pred_copy_elapsed,
                metric_elapsed=metric_elapsed,
            )
        bucket["images"].clear()
        bucket["targets"].clear()
        return processed

    infer_bar = _make_progress_bar(
        total=dataset_size,
        desc="Evaluating VOC",
        disable=disable_tqdm,
    )
    for image, target in data_loader:
        t_numpy = timer()
        image = _to_numpy(image)
        target = _to_numpy(target)
        stage["numpy"] += timer() - t_numpy

        t_pad = timer()
        padded_image, padded_shape = _pad_to_multiple_of_32(image)
        bucket_shape = _select_eval_bucket(padded_shape, eval_buckets or [])
        if bucket_shape != padded_shape:
            padded_image = _pad_to_shape(padded_image, bucket_shape)
            padded_shape = bucket_shape
        stage["pad"] += timer() - t_pad

        seen_shapes.add(padded_shape)
        bucket = pending_batches.setdefault(padded_shape, {"images": [], "targets": []})
        bucket["images"].append(padded_image)
        bucket["targets"].append(target.astype(np.int64, copy=False))
        if len(bucket["images"]) >= batch_size:
            infer_bar.update(flush_shape_bucket(padded_shape))

    for padded_shape in list(pending_batches):
        infer_bar.update(flush_shape_bucket(padded_shape, flush_kind="tail"))

    infer_bar.close()

    model.aux_mode = "train"
    if previous_assume_padded_input is not None:
        model.assume_padded_input = previous_assume_padded_input
    acc_global, acc, iu, iou_mean = _compute_confmat_metrics(confmat)
    confmat_summary = _format_confmat_summary(acc_global, acc, iu, iou_mean)
    logger.info(
        "VOC eval runtime | mode=shape_bucket samples={} unique_padded_shapes={} eval_bs={} xla={} eval_buckets={} numpy={:.2f}s pad={:.2f}s stack={:.2f}s to_tensor={:.2f}s graph={:.2f}s first_graph={:.2f}s repeat_graph={:.2f}s pred_copy={:.2f}s metric={:.2f}s graph_calls={}",
        dataset_size,
        len(seen_shapes),
        batch_size,
        should_use_xla(),
        eval_buckets or [],
        stage["numpy"],
        stage["pad"],
        stage["stack"],
        stage["to_tensor"],
        stage["graph"],
        stage["graph_first"],
        stage["graph_repeat"],
        stage["pred_copy"],
        stage["metric"],
        graph_call_count,
    )
    _log_eval_batch_profile(batch_profile_records, dataset_size)
    return acc_global, iou_mean, confmat_summary


def build_model(eval_args, sample_image):
    sample_image = _to_numpy(sample_image).astype(np.float32, copy=False)
    model_args = copy.copy(eval_args)
    model_args.rand_init = True
    model = set_model_bisenetv2(model_args, num_classes=eval_args.num_classes)
    _ = model(tf.convert_to_tensor(sample_image[None, ...], dtype=tf.float32), training=False)
    return model


def _append_unique_path(paths: list[Path], path: Path):
    normalized = Path(path)
    if normalized not in paths:
        paths.append(normalized)


def _checkpoint_stem(path: Path) -> str:
    name = path.name
    if name.endswith(TF_CHECKPOINT_SUFFIX):
        return name[: -len(TF_CHECKPOINT_SUFFIX)]
    return path.stem


def _candidate_checkpoint_paths(base_path: Path) -> list[Path]:
    base_path = Path(base_path)
    candidates: list[Path] = []
    _append_unique_path(candidates, base_path)

    base_text = str(base_path)
    if base_text.endswith(TF_CHECKPOINT_SUFFIX):
        pass
    elif base_path.suffix:
        raise ValueError(f"Only TensorFlow .weights.h5 checkpoints are supported by eval_voc.py: {base_path}")
    else:
        _append_unique_path(candidates, Path(base_text + TF_CHECKPOINT_SUFFIX))
    return candidates


def _case_insensitive_match(directory: Path, target_name: str) -> Path | None:
    if not directory.is_dir():
        return None
    target_name_lower = target_name.lower()
    matches = [entry for entry in directory.iterdir() if entry.name.lower() == target_name_lower]
    if len(matches) == 1:
        return matches[0].resolve()
    return None


def _discover_checkpoint_paths(checkpoint_dir: Path) -> list[Path]:
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    tf_checkpoints = []
    for entry in sorted(checkpoint_dir.iterdir()):
        if not entry.is_file():
            continue
        if entry.name.endswith(TF_CHECKPOINT_SUFFIX):
            tf_checkpoints.append(entry.resolve())

    if not tf_checkpoints:
        raise FileNotFoundError(
            f"No supported checkpoints ({', '.join(SUPPORTED_CHECKPOINT_SUFFIXES)}) found in {checkpoint_dir}"
        )
    return tf_checkpoints


def resolve_checkpoint_path(project_root: Path, checkpoint_dir: Path, checkpoint: str) -> Path:
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_absolute():
        base_path = checkpoint_path
    elif checkpoint_path.parent != Path("."):
        base_path = project_root / checkpoint_path
    else:
        base_path = checkpoint_dir / checkpoint_path

    candidate_paths = _candidate_checkpoint_paths(base_path)
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path.resolve()

    for candidate_path in candidate_paths:
        matched_path = _case_insensitive_match(candidate_path.parent, candidate_path.name)
        if matched_path is not None:
            return matched_path

    available = []
    if checkpoint_dir.is_dir():
        available = [path.name for path in _discover_checkpoint_paths(checkpoint_dir)]
    searched = ", ".join(str(path) for path in candidate_paths)
    message = f"Checkpoint not found for '{checkpoint}'. Searched: {searched}"
    if available:
        message += ". Available checkpoints: " + ", ".join(available)
    raise FileNotFoundError(message)


def resolve_checkpoint_paths(eval_args, project_root: Path) -> list[Path]:
    checkpoint_dir = resolve_path(project_root, eval_args.checkpoint_dir)
    requested_checkpoints = None
    if eval_args.checkpoints:
        requested_checkpoints = eval_args.checkpoints
    elif eval_args.checkpoint:
        requested_checkpoints = [eval_args.checkpoint]

    if requested_checkpoints:
        return [
            resolve_checkpoint_path(project_root, checkpoint_dir, checkpoint)
            for checkpoint in requested_checkpoints
        ]

    return _discover_checkpoint_paths(checkpoint_dir)


def load_checkpoint(model, checkpoint_path: Path):
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.name.endswith(TF_CHECKPOINT_SUFFIX):
        raise ValueError(f"Only TensorFlow weights are supported by eval_voc.py: {checkpoint_path}")
    model.load_weights(str(checkpoint_path))
    logger.info("Loaded tf checkpoint: {}", checkpoint_path.name)
    return checkpoint_path


def run_eval_inference_fast(args, model, test_dataset, test_loader):
    eval_buckets = _parse_eval_buckets(getattr(args, "eval_buckets", ""))
    prebatch = bool(getattr(args, "eval_prebatch", False))
    acc_global, iou_mean, confmat_summary = evaluate_voc_dataset(
        model,
        test_loader,
        num_classes=args.num_classes,
        batch_size=max(1, int(args.eval_bs)),
        dataset_size=len(test_dataset) if hasattr(test_dataset, "__len__") else None,
        eval_buckets=eval_buckets,
        prebatch=prebatch,
        profile_runtime=bool(getattr(args, "profile_runtime", False)),
        tail_pad_batch=bool(getattr(args, "eval_tail_pad_batch", False)),
        use_tf_metric=bool(getattr(args, "eval_tf_metric", False)),
        tfdata_batch=bool(getattr(args, "eval_tfdata_batch", False)),
    )
    return acc_global, iou_mean, confmat_summary


def evaluate_checkpoint(eval_args, checkpoint_path: Path, test_dataset, test_loader):
    logger.info("=" * 100)
    logger.info("Evaluating checkpoint: {}", checkpoint_path)

    global_model = build_model(eval_args, test_dataset[0][0])
    loaded_checkpoint_path = load_checkpoint(global_model, checkpoint_path)
    inference_model = global_model
    if getattr(eval_args, "fast_nhwc", False):
        inference_model = build_fast_tf_bisenetv2_from_model(global_model)
        logger.info("Using TensorFlow-native NHWC inference model")

    start_time = time.perf_counter()
    test_acc, test_iou, confmat = run_eval_inference_fast(
        eval_args,
        inference_model,
        test_dataset,
        test_loader,
    )
    eval_time = time.perf_counter() - start_time
    eval_mode_parts = [f"shape_bucket_eval_bs={max(1, int(eval_args.eval_bs))}"]
    if getattr(eval_args, "eval_tfdata_batch", False):
        eval_mode_parts.append("tfdata_batch")
    if getattr(eval_args, "eval_tf_metric", False):
        eval_mode_parts.append("tf_metric")
    if getattr(eval_args, "eval_tail_pad_batch", False):
        eval_mode_parts.append("tail_pad_batch")
    eval_buckets = getattr(eval_args, "eval_buckets", "")
    if eval_buckets:
        eval_mode_parts.append(f"buckets={eval_buckets}")
    eval_mode = "+".join(eval_mode_parts)

    logger.debug("Confusion matrix:\n{}", confmat)
    logger.info("Evaluated checkpoint: {}", loaded_checkpoint_path.name)
    logger.info("Eval Mode: {}", eval_mode)
    logger.info("Global Test Accuracy: {:.2f}%", test_acc)
    logger.info("Global Test IoU: {:.2f}%", test_iou)
    logger.info("Eval Time: {:.2f}s", eval_time)

    return {
        "checkpoint": loaded_checkpoint_path,
        "acc": test_acc,
        "iou": test_iou,
        "eval_time": eval_time,
        "eval_mode": eval_mode,
    }


def main():
    eval_args = args_parser()
    setup_logger(verbose=False, logs_dir="logs/eval_voc", log_name="eval")

    project_root = resolve_path(Path.cwd(), eval_args.root)
    start_time = time.time()
    device = require_tensorflow_device(tf, eval_args.gpu)
    logger.info(
        "VOC eval config: device={} gpu={} root={} root_dir={} eval_bs={}",
        device,
        eval_args.gpu,
        project_root,
        eval_args.root_dir,
        eval_args.eval_bs,
    )

    checkpoint_paths = resolve_checkpoint_paths(eval_args, project_root)
    logger.info("checkpoints to evaluate:")
    for checkpoint_path in checkpoint_paths:
        logger.info("  - {}", checkpoint_path)

    test_dataset = build_voc_eval_dataset(eval_args, project_root)
    test_loader = build_voc_eval_loader(eval_args, test_dataset)

    results = []
    for checkpoint_path in checkpoint_paths:
        results.append(
            evaluate_checkpoint(
                eval_args,
                checkpoint_path,
                test_dataset,
                test_loader,
            )
        )

    logger.info("=" * 100)
    logger.info("Summary")
    for result in results:
        logger.info(
            "{} | mode={} | Acc={:.2f}% | mIoU={:.2f}% | Eval Time={:.2f}s",
            result["checkpoint"].name,
            result["eval_mode"],
            result["acc"],
            result["iou"],
            result["eval_time"],
        )
    logger.info("Total Run Time: {:.2f}s", time.time() - start_time)


if __name__ == "__main__":
    main()

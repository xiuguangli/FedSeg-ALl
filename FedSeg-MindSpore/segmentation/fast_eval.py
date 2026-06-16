import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import mindspore as ms
import numpy as np
from tqdm import tqdm

from eval_utils import ConfusionMatrix
from myseg.datasplit import build_dataset_for_split


_EVAL_GROUP_CACHE = {}


def _cache_key(args, split, bucket_align, num_workers):
    return (
        str(getattr(args, "dataset", "")),
        os.path.abspath(str(getattr(args, "root_dir", ""))),
        str(split or getattr(args, "data", "")),
        bool(getattr(args, "USE_ERASE_DATA", False)),
        int(bucket_align or 0),
        int(getattr(args, "num_classes", 0)),
        int(getattr(args, "num_workers", 1) if num_workers is None else num_workers),
    )


def build_eval_groups(args, split=None, bucket_align=32, num_workers=None):
    split_name = split or args.data
    cache_key = _cache_key(args, split_name, bucket_align, num_workers)
    cached = _EVAL_GROUP_CACHE.get(cache_key)
    if cached is not None:
        return cached

    dataset = build_dataset_for_split(args, split_name)
    grouped_samples = defaultdict(list)
    indices = list(range(len(dataset)))
    worker_count = max(1, int(getattr(args, "num_workers", 1) if num_workers is None else num_workers))

    if worker_count == 1:
        samples = (dataset[idx] for idx in indices)
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            samples = executor.map(dataset.__getitem__, indices)

    for image_np, label_np in samples:
        image_np = image_np.astype(np.float32, copy=False)
        label_np = label_np.astype(np.int64, copy=False)
        image_h, image_w = image_np.shape[1], image_np.shape[2]
        if bucket_align and bucket_align > 1:
            bucket_shape = (
                (image_h + bucket_align - 1) // bucket_align * bucket_align,
                (image_w + bucket_align - 1) // bucket_align * bucket_align,
            )
        else:
            bucket_shape = (image_h, image_w)
        grouped_samples[bucket_shape].append((image_np, label_np))

    cached_value = (dataset, dict(grouped_samples))
    _EVAL_GROUP_CACHE[cache_key] = cached_value
    return cached_value


def iter_eval_batches(grouped_samples, batch_size, pad_mode="reflect"):
    del pad_mode
    for (bucket_h, bucket_w), bucket_samples in grouped_samples.items():
        for start in range(0, len(bucket_samples), batch_size):
            batch_samples = bucket_samples[start:start + batch_size]
            batch_len = len(batch_samples)
            image_batch = np.zeros((batch_len, 3, bucket_h, bucket_w), dtype=np.float32)
            label_batch = np.full((batch_len, bucket_h, bucket_w), 255, dtype=np.int32)
            for batch_index, (image_np, label_np) in enumerate(batch_samples):
                image_h, image_w = image_np.shape[1], image_np.shape[2]
                image_batch[batch_index, :, :image_h, :image_w] = image_np
                label_batch[batch_index, :image_h, :image_w] = label_np
            yield image_batch, label_batch, 0.0


def evaluate_grouped_dataset(
    args,
    model,
    split=None,
    batch_size=None,
    num_workers=None,
    bucket_align=32,
    profile_runtime=False,
    progress_desc=None,
):
    overall_start = time.perf_counter()
    eval_batch_size = max(1, int(batch_size if batch_size is not None else getattr(args, "eval_batch_size", 1)))

    loader_start = time.perf_counter()
    dataset, grouped_samples = build_eval_groups(
        args,
        split=split,
        bucket_align=bucket_align,
        num_workers=num_workers,
    )
    loader_build_time = time.perf_counter() - loader_start

    confmat = ConfusionMatrix(args.num_classes)
    num_batches = sum(
        (len(samples) + eval_batch_size - 1) // eval_batch_size
        for samples in grouped_samples.values()
    )
    progress = tqdm(
        iter_eval_batches(grouped_samples, eval_batch_size),
        total=num_batches,
        desc=progress_desc or "Test",
        leave=False,
        dynamic_ncols=True,
    )

    forward_time = 0.0
    metric_time = 0.0
    batch_prepare_time = 0.0
    tensor_create_time = 0.0
    model_forward_time = 0.0
    argmax_time = 0.0
    pred_to_numpy_time = 0.0

    original_aux_mode = getattr(model, "aux_mode", None)
    model.set_train(False)
    for image_batch, labels_batch, prepare_time in progress:
        batch_prepare_time += prepare_time
        tensor_start = time.perf_counter()
        images = ms.Tensor.from_numpy(image_batch)
        tensor_create_time += time.perf_counter() - tensor_start

        if original_aux_mode is not None:
            model.aux_mode = "eval"
        forward_start = time.perf_counter()
        outputs = model(images)[0]
        model_forward_time += time.perf_counter() - forward_start
        if original_aux_mode is not None:
            model.aux_mode = original_aux_mode

        argmax_start = time.perf_counter()
        preds = outputs.argmax(axis=1)
        argmax_time += time.perf_counter() - argmax_start

        pred_cast_start = time.perf_counter()
        preds = preds.astype(ms.int32)
        argmax_time += time.perf_counter() - pred_cast_start

        pred_to_numpy_start = time.perf_counter()
        preds_np = preds.asnumpy()
        pred_to_numpy_time += time.perf_counter() - pred_to_numpy_start
        forward_time = model_forward_time + argmax_time + pred_to_numpy_time

        metric_start = time.perf_counter()
        confmat.update_numpy(
            labels_batch.reshape(-1),
            preds_np.reshape(-1),
        )
        metric_time += time.perf_counter() - metric_start

    confmat.compute()
    total_time = time.perf_counter() - overall_start
    runtime = {
        "build_eval_groups": loader_build_time,
        "forward": forward_time,
        "metrics": metric_time,
        "total": total_time,
        "batch_prepare": batch_prepare_time,
        "tensor_create": tensor_create_time,
        "model_forward": model_forward_time,
        "argmax": argmax_time,
        "pred_to_numpy": pred_to_numpy_time,
        "num_groups": len(grouped_samples),
        "num_batches": num_batches,
        "batch_size": eval_batch_size,
        "bucket_align": bucket_align,
        "num_samples": len(dataset),
    }
    return {
        "num_samples": len(dataset),
        "acc": confmat.acc_global,
        "iou": confmat.iou_mean,
        "confmat": str(confmat),
        "runtime": runtime if profile_runtime else runtime,
    }


def format_runtime_profile(runtime):
    return (
        "Runtime profile: build_eval_groups={:.2f}s forward={:.2f}s metrics={:.2f}s total={:.2f}s "
        "groups={} batches={} eval_batch_size={} bucket_align={}".format(
            runtime["build_eval_groups"],
            runtime["forward"],
            runtime["metrics"],
            runtime["total"],
            runtime["num_groups"],
            runtime["num_batches"],
            runtime["batch_size"],
            runtime["bucket_align"],
        )
    )


def format_runtime_detail(runtime):
    return (
        "Runtime detail: batch_prepare={:.2f}s tensor_create={:.2f}s model_forward={:.2f}s "
        "argmax={:.2f}s pred_to_numpy={:.2f}s".format(
            runtime["batch_prepare"],
            runtime["tensor_create"],
            runtime["model_forward"],
            runtime["argmax"],
            runtime["pred_to_numpy"],
        )
    )


def checkpoint_label(checkpoint_path):
    return os.path.basename(checkpoint_path)

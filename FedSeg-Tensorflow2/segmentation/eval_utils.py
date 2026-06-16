import numpy as np
# import torch
# import torch.utils.data
import sys
import time
import tensorflow as tf
from tqdm import tqdm
from runtime_utils import should_disable_tqdm
from logging_utils import logger


def _make_progress_bar(*args, **kwargs):
    kwargs.setdefault("file", sys.stdout)
    kwargs.setdefault("dynamic_ncols", True)
    kwargs.setdefault("leave", True)
    return tqdm(*args, **kwargs)


def evaluate(model, data_loader, num_classes):
    confmat = ConfusionMatrix(num_classes)
    model.aux_mode = 'eval'
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 3, None, None], dtype=tf.float32), 
        tf.TensorSpec(shape=[None, None, None], dtype=tf.int64) 
    ])
    def model_inference(inputs,target):
        output = model(inputs, training=False)[0]
        confmat.update(tf.reshape(target, [-1]), tf.reshape(tf.argmax(output, axis=1), [-1]))
        # return confmat

    for image, target in _make_progress_bar(
        data_loader,
        desc="Evaluating VOC",
        disable=should_disable_tqdm(),
    ):
    # for image, target in data_loader:
        # output = model(image, training=False)
        model_inference(image, target)
        # break
    
    model.aux_mode = 'train'
    confmat.compute()

    return confmat


def _to_numpy(value):
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def parse_eval_buckets(value):
    if not value:
        return []
    buckets = []
    for raw_item in str(value).split(","):
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


def _iter_eval_samples(data_loader):
    for image, target in data_loader:
        image = _to_numpy(image)
        target = _to_numpy(target)
        if image.ndim == 4:
            for sample_image, sample_target in zip(image, target):
                yield sample_image, sample_target
        elif image.ndim == 3:
            yield image, target
        else:
            raise ValueError(f"expected image shape [C,H,W] or [B,C,H,W], got {image.shape}")


def _load_pad_key_eval_sample(dataset, index, eval_buckets):
    image, target = dataset[int(np.asarray(index).item())]
    image = _to_numpy(image).astype(np.float32, copy=False)
    target = _to_numpy(target).astype(np.int64, copy=False)
    if image.ndim != 3:
        raise ValueError(f"expected dataset image shape [C,H,W], got {image.shape}")
    if image.shape[0] != 3:
        raise ValueError(f"expected NCHW image with 3 channels, got {image.shape}")

    padded_image, padded_shape = _pad_to_multiple_of_32(image)
    bucket_shape = _select_eval_bucket(padded_shape, eval_buckets)
    if bucket_shape != padded_shape:
        padded_image = _pad_to_shape(padded_image, bucket_shape)
        padded_shape = bucket_shape

    padded_target = _stack_targets_to_shape([target], padded_shape, batch_size=1)[0]
    key = np.int64(padded_shape[0] * 10000 + padded_shape[1])
    return key, padded_image, padded_target


def build_tfdata_shape_batched_eval_loader(
    dataset,
    batch_size=8,
    eval_buckets=None,
    num_parallel_calls=1,
):
    batch_size = max(1, int(batch_size))
    eval_buckets = eval_buckets or []
    dataset_size = len(dataset)
    num_parallel_calls = max(1, int(num_parallel_calls))
    loader = tf.data.Dataset.range(dataset_size)

    def load_pad_and_key_sample(index):
        key, image, target = tf.numpy_function(
            lambda sample_index: _load_pad_key_eval_sample(dataset, sample_index, eval_buckets),
            [index],
            [tf.int64, tf.float32, tf.int64],
        )
        key.set_shape(())
        image.set_shape([3, None, None])
        target.set_shape([None, None])
        return key, image, target

    def key_func(key, image, target):
        del image, target
        return key

    def reduce_func(key, window):
        del key
        window = window.map(lambda sample_key, image, target: (image, target), num_parallel_calls=tf.data.AUTOTUNE)
        return window.padded_batch(
            batch_size,
            padded_shapes=([3, None, None], [None, None]),
            padding_values=(tf.constant(0.0, dtype=tf.float32), tf.constant(-1, dtype=tf.int64)),
            drop_remainder=False,
        )

    loader = loader.map(load_pad_and_key_sample, num_parallel_calls=num_parallel_calls, deterministic=True)
    loader = loader.group_by_window(
        key_func=key_func,
        reduce_func=reduce_func,
        window_size=batch_size,
    )
    loader = loader.prefetch(tf.data.AUTOTUNE)
    logger.info(
        "Prepared generic eval loader (tf.data shape-batched stream, samples={} eval_bs={} eval_buckets={} num_parallel_calls={})",
        dataset_size,
        batch_size,
        eval_buckets,
        num_parallel_calls,
    )
    return loader


def _pad_to_multiple_of_32(image):
    h, w = int(image.shape[1]), int(image.shape[2])
    padded_h = ((h + 31) // 32) * 32
    padded_w = ((w + 31) // 32) * 32
    if padded_h == h and padded_w == w:
        return image, (padded_h, padded_w)
    return np.pad(
        image,
        ((0, 0), (0, padded_h - h), (0, padded_w - w)),
        mode="reflect",
    ), (padded_h, padded_w)


def _select_eval_bucket(shape, buckets):
    if not buckets:
        return shape
    h, w = shape
    candidates = [(bh, bw) for bh, bw in buckets if h <= bh and w <= bw]
    if candidates:
        return min(candidates, key=lambda bucket: (bucket[0] * bucket[1], bucket[0], bucket[1]))
    max_h = max(h, max(bucket[0] for bucket in buckets))
    max_w = max(w, max(bucket[1] for bucket in buckets))
    return (((max_h + 31) // 32) * 32, ((max_w + 31) // 32) * 32)


def _pad_to_shape(image, target_shape):
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


def _stack_targets_to_shape(targets, padded_shape, batch_size=None):
    target_h, target_w = padded_shape
    batch_size = len(targets) if batch_size is None else batch_size
    batch_targets = np.full((batch_size, target_h, target_w), -1, dtype=np.int64)
    for sample_idx, target in enumerate(targets):
        h, w = target.shape
        batch_targets[sample_idx, :h, :w] = target
    return batch_targets


def _update_confmat_numpy(confmat, target, pred, num_classes):
    target_flat = target.reshape(-1)
    pred_flat = pred.reshape(-1)
    valid_mask = (target_flat >= 0) & (target_flat < num_classes)
    if not np.any(valid_mask):
        return
    indices = num_classes * target_flat[valid_mask].astype(np.int64) + pred_flat[valid_mask].astype(np.int64)
    confmat += np.bincount(indices, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def _compute_confmat_metrics(confmat):
    h = confmat.astype(np.float32, copy=False)
    total = float(h.sum())
    acc_global = float(np.diag(h).sum() / (total + 1e-10) * 100.0)
    acc = np.diag(h) / (h.sum(axis=1) + 1e-10)
    iu = np.diag(h) / (h.sum(axis=1) + h.sum(axis=0) - np.diag(h) + 1e-10)
    valid = np.isfinite(iu) & ((h.sum(axis=1) + h.sum(axis=0) - np.diag(h)) > 0)
    iou_mean = float(iu[valid].mean() * 100.0) if np.any(valid) else 0.0
    return acc_global, acc, iu, iou_mean


def _format_confmat_summary(acc_global, acc, iu, iou_mean):
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


def _make_fast_eval_pred_fn(model):
    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None, 3, None, None], dtype=tf.float32)],
        reduce_retracing=True,
    )
    def eval_pred(images):
        logits = model(images, training=False)[0]
        return tf.argmax(logits, axis=1, output_type=tf.int64)

    return eval_pred


def evaluate_fast_shape_bucket(
    model,
    data_loader,
    num_classes,
    batch_size=8,
    dataset_size=None,
    eval_buckets=None,
    profile_runtime=False,
    tfdata_batch=False,
    desc="Evaluating fast",
):
    """Fast generic segmentation eval by batching samples with the same padded shape."""
    batch_size = max(1, int(batch_size))
    eval_buckets = eval_buckets or []
    confmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    previous_aux_mode = getattr(model, "aux_mode", None)
    previous_assume_padded_input = getattr(model, "assume_padded_input", None)
    if previous_aux_mode is not None:
        model.aux_mode = "eval"
    if previous_assume_padded_input is not None:
        model.assume_padded_input = True

    eval_pred = _make_fast_eval_pred_fn(model)
    pending_batches = {}
    seen_shapes = set()
    graph_calls = 0
    stage = {"numpy_pad": 0.0, "stack": 0.0, "graph": 0.0, "pred_copy": 0.0, "metric": 0.0}
    timer = time.perf_counter

    if tfdata_batch:
        try:
            infer_bar = _make_progress_bar(
                total=dataset_size,
                desc=desc,
                disable=should_disable_tqdm(),
            )
            for batch_images, batch_targets in data_loader:
                actual_batch_size = int(batch_images.shape[0])
                padded_shape = (int(batch_images.shape[2]), int(batch_images.shape[3]))
                seen_shapes.add(padded_shape)

                t_graph = timer()
                batch_pred = eval_pred(batch_images)
                stage["graph"] += timer() - t_graph
                graph_calls += 1

                t_copy = timer()
                batch_pred = batch_pred.numpy()
                batch_targets = batch_targets.numpy()
                stage["pred_copy"] += timer() - t_copy

                t_metric = timer()
                _update_confmat_numpy(confmat, batch_targets, batch_pred, num_classes)
                stage["metric"] += timer() - t_metric
                infer_bar.update(actual_batch_size)
            infer_bar.close()
        finally:
            if previous_aux_mode is not None:
                model.aux_mode = previous_aux_mode
            if previous_assume_padded_input is not None:
                model.assume_padded_input = previous_assume_padded_input

        acc_global, acc, iu, iou_mean = _compute_confmat_metrics(confmat)
        confmat_summary = _format_confmat_summary(acc_global, acc, iu, iou_mean)
        logger.info(
            "Fast eval runtime | mode=tfdata_shape_batch samples={} unique_padded_shapes={} eval_bs={} eval_buckets={} "
            "numpy_pad={:.2f}s stack={:.2f}s graph={:.2f}s pred_copy={:.2f}s metric={:.2f}s graph_calls={}",
            dataset_size,
            len(seen_shapes),
            batch_size,
            eval_buckets,
            stage["numpy_pad"],
            stage["stack"],
            stage["graph"],
            stage["pred_copy"],
            stage["metric"],
            graph_calls,
        )
        return acc_global, iou_mean, confmat_summary

    def flush_shape_bucket(padded_shape, flush_kind="full"):
        nonlocal graph_calls
        bucket = pending_batches.get(padded_shape)
        if bucket is None or not bucket["images"]:
            return 0

        t_stack = timer()
        batch_images = np.stack(bucket["images"], axis=0).astype(np.float32, copy=False)
        batch_targets = bucket["targets"]
        processed = len(batch_targets)
        stage["stack"] += timer() - t_stack

        t_graph = timer()
        batch_pred = eval_pred(tf.convert_to_tensor(batch_images, dtype=tf.float32))
        stage["graph"] += timer() - t_graph
        graph_calls += 1

        t_copy = timer()
        batch_pred = batch_pred.numpy()
        stage["pred_copy"] += timer() - t_copy

        t_metric = timer()
        for sample_idx, target in enumerate(batch_targets):
            h, w = target.shape
            _update_confmat_numpy(confmat, target, batch_pred[sample_idx, :h, :w], num_classes)
        stage["metric"] += timer() - t_metric

        if profile_runtime:
            logger.info(
                "Fast eval batch | shape={} samples={} flush={} graph={:.3f}s",
                padded_shape,
                processed,
                flush_kind,
                stage["graph"],
            )

        bucket["images"].clear()
        bucket["targets"].clear()
        return processed

    try:
        infer_bar = _make_progress_bar(
            total=dataset_size,
            desc=desc,
            disable=should_disable_tqdm(),
        )
        for image, target in _iter_eval_samples(data_loader):
            t_prepare = timer()
            image = image.astype(np.float32, copy=False)
            target = target.astype(np.int64, copy=False)
            padded_image, padded_shape = _pad_to_multiple_of_32(image)
            bucket_shape = _select_eval_bucket(padded_shape, eval_buckets)
            if bucket_shape != padded_shape:
                padded_image = _pad_to_shape(padded_image, bucket_shape)
                padded_shape = bucket_shape
            stage["numpy_pad"] += timer() - t_prepare

            seen_shapes.add(padded_shape)
            bucket = pending_batches.setdefault(padded_shape, {"images": [], "targets": []})
            bucket["images"].append(padded_image)
            bucket["targets"].append(target)
            if len(bucket["images"]) >= batch_size:
                infer_bar.update(flush_shape_bucket(padded_shape))

        for padded_shape in list(pending_batches):
            infer_bar.update(flush_shape_bucket(padded_shape, flush_kind="tail"))
        infer_bar.close()
    finally:
        if previous_aux_mode is not None:
            model.aux_mode = previous_aux_mode
        if previous_assume_padded_input is not None:
            model.assume_padded_input = previous_assume_padded_input

    acc_global, acc, iu, iou_mean = _compute_confmat_metrics(confmat)
    confmat_summary = _format_confmat_summary(acc_global, acc, iu, iou_mean)
    logger.info(
        "Fast eval runtime | samples={} unique_padded_shapes={} eval_bs={} eval_buckets={} "
        "numpy_pad={:.2f}s stack={:.2f}s graph={:.2f}s pred_copy={:.2f}s metric={:.2f}s graph_calls={}",
        dataset_size,
        len(seen_shapes),
        batch_size,
        eval_buckets,
        stage["numpy_pad"],
        stage["stack"],
        stage["graph"],
        stage["pred_copy"],
        stage["metric"],
        graph_calls,
    )
    return acc_global, iou_mean, confmat_summary


class ConfusionMatrix1(object):
    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0.0
        self.iou_mean = 0.0
        self.acc = np.array(0)
        self.iu = np.array(0)

    def update(self, a, b):
        # Tensor转换为Numpy array
        a_np = a
        b_np = b
        
        n = self.num_classes
        if self.mat is None:
            self.mat = np.zeros((n, n), dtype=np.int64)
        
        k = (a_np >= 0) & (a_np < n)
        inds = n * a_np[k].astype(np.int64) + b_np[k]
        update_matrix = np.bincount(inds, minlength=n * n).reshape(n, n)
        self.mat += update_matrix

    def compute(self):
        """ 根据混淆矩阵计算并更新度量指标 (Numpy实现) """
        if self.mat is None:
            logger.warning("Confusion matrix is not updated. Call update() first.")
            return

        h = self.mat.astype(np.float32)
        
        # 全局准确率
        self.acc_global = np.diag(h).sum() / h.sum() * 100

        # 各类别准确率
        self.acc = np.diag(h) / (h.sum(axis=1) + 1e-10)

        # 各类别交并比 (IoU)
        denominator = h.sum(axis=1) + h.sum(axis=0) - np.diag(h)
        self.iu = np.diag(h) / (denominator + 1e-10)

        # 平均交并比 (mIoU)
        iu_not_nan = self.iu[~np.isnan(self.iu)]
        if iu_not_nan.size == 0:
            self.iou_mean = 0.0
        else:
            self.iou_mean = iu_not_nan.mean() * 100

    def __str__(self):
        self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            self.acc_global,
            ['{:.1f}'.format(i) for i in (self.acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.iu * 100).tolist()],
            self.iou_mean)


class ConfusionMatrix(object):
    def __init__(self, num_classes: int):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        # 使用 tf.Variable 存储混淆矩阵，以便在 tf.function 中更新
        self.mat = tf.Variable(
            initial_value=tf.zeros((num_classes, num_classes), dtype=tf.int64),
            trainable=False,
            dtype=tf.int64
        )
        self.acc_global = tf.Variable(0.0, trainable=False)
        self.iou_mean = tf.Variable(0.0, trainable=False)
        self.acc = tf.Variable(tf.zeros(num_classes, dtype=tf.float32), trainable=False)
        self.iu = tf.Variable(tf.zeros(num_classes, dtype=tf.float32), trainable=False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.int64), # 真实标签 a
        tf.TensorSpec(shape=[None], dtype=tf.int64)  # 预测标签 b
    ])
    def update(self, a_flat: tf.Tensor, b_flat: tf.Tensor):
        """ 使用 tf.math.confusion_matrix 累加到 Variable 中 """
        n = self.num_classes
        # 1. 过滤有效范围内的标签
        valid_mask = (a_flat >= 0) & (a_flat < n)
        a_valid = tf.boolean_mask(a_flat, valid_mask)
        b_valid = tf.boolean_mask(b_flat, valid_mask)
        
        # 2. 直接计算当前批次的混淆矩阵，并累加到 Variable 中 (assign_add)
        current_confmat = tf.math.confusion_matrix(
            a_valid, 
            b_valid, 
            num_classes=n, 
            dtype=tf.int64
        )
        self.mat.assign_add(current_confmat)

    @tf.function
    def compute(self):
        """ 根据混淆矩阵计算并更新度量指标 (TensorFlow实现) """
        h = tf.cast(self.mat.read_value(), tf.float32) # 从 Variable 读取值

        # 全局准确率
        global_correct = tf.reduce_sum(tf.linalg.diag_part(h))
        total_sum = tf.reduce_sum(h)
        self.acc_global.assign(global_correct / (total_sum + 1e-10) * 100.0)

        # 各类别准确率
        row_sum = tf.reduce_sum(h, axis=1)
        self.acc.assign(tf.linalg.diag_part(h) / (row_sum + 1e-10))

        # 各类别交并比 (IoU)
        col_sum = tf.reduce_sum(h, axis=0)
        intersection = tf.linalg.diag_part(h)
        union = row_sum + col_sum - intersection
        iu_tensor = intersection / (union + 1e-10)
        self.iu.assign(iu_tensor)

        # 平均交并比 (mIoU)
        # 过滤 NaN 值，只计算有效类别的均值
        is_valid = tf.math.is_finite(iu_tensor) & (union > 0)
        iu_valid = tf.boolean_mask(iu_tensor, is_valid)
        
        # 检查是否所有 IoU 都为 0 或 NaN
        if tf.reduce_sum(tf.cast(is_valid, tf.int32)) == 0:
            self.iou_mean.assign(0.0)
        else:
            self.iou_mean.assign(tf.reduce_mean(iu_valid) * 100.0)

    def reset(self):
        self.mat.assign(tf.zeros_like(self.mat))
        self.acc_global.assign(0.0)
        self.iou_mean.assign(0.0)
        self.acc.assign(tf.zeros_like(self.acc))
        self.iu.assign(tf.zeros_like(self.iu))
    
    def _get_metric_values(self):
        # 显式调用 tf.function 来更新指标
        self.compute() 
        
        # 使用 tf.identity 确保读取的是最新的值，并用 .numpy() 转换为 Python/NumPy
        acc_global = self.acc_global.read_value().numpy()
        acc = self.acc.read_value().numpy()
        iu = self.iu.read_value().numpy()
        iou_mean = self.iou_mean.read_value().numpy()
        
        return acc_global, acc, iu, iou_mean
        
    def __str__(self):        
        acc_global, acc, iu, iou_mean = self._get_metric_values()
        
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iou_mean)

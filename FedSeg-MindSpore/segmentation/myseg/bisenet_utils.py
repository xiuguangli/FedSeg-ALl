import numpy as np

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

from logging_utils import logger
from myseg.bisenetv2 import BiSeNetV2

L2_NORM_EPS = 1e-12


def fedseg_l2_normalize(axis):
    return ops.L2Normalize(axis=axis, epsilon=L2_NORM_EPS)


def set_model_bisenetv2(args, num_classes):
    return BiSeNetV2(args, num_classes)


def set_optimizer(model, args, learning_rate=None):
    learning_rate = args.lr if learning_rate is None else learning_rate
    if hasattr(model, "get_params"):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        params_list = [
            {"params": wd_params},
            {"params": nowd_params, "weight_decay": 0.0},
            {"params": lr_mul_wd_params, "lr": learning_rate * 10},
            {"params": lr_mul_nowd_params, "lr": learning_rate * 10, "weight_decay": 0.0},
        ]
    else:
        wd_params, non_wd_params = [], []
        for _, param in model.parameters_and_names():
            if param.ndim == 1:
                non_wd_params.append(param)
            elif param.ndim in {2, 4}:
                wd_params.append(param)
        params_list = [
            {"params": wd_params},
            {"params": non_wd_params, "weight_decay": 0.0},
        ]
    return nn.SGD(
        params_list,
        learning_rate=learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )


class BackCELoss(nn.Cell):
    def __init__(self, args, ignore_lb=255):
        super().__init__()
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes
        self.eps = ms.Tensor(1e-7, ms.float32)
        self.clip_min = ms.Tensor(0, ms.int32)
        self.clip_max = ms.Tensor(self.class_num - 1, ms.int32)
        self.denom_floor = ms.Tensor(1.0, ms.float32)

    def construct(self, logits, labels):
        probs = ops.softmax(logits, axis=1)
        labels = labels.astype(ms.int32)
        valid_mask = labels != self.ignore_lb
        ignore_mask = labels == self.ignore_lb

        present_flags = []
        for cls_idx in range(self.class_num):
            present_flags.append(ops.any(ops.logical_and(valid_mask, labels == cls_idx)))
        present_mask = ops.stack(present_flags)
        absent_mask = ops.logical_not(present_mask)
        bg_exists = ops.any(absent_mask)

        absent_weights = absent_mask.astype(probs.dtype).reshape(1, self.class_num, 1, 1)
        bg_prob = (probs * absent_weights).sum(axis=1)

        labels_clamped = ops.clip_by_value(labels, self.clip_min, self.clip_max)
        true_prob = ops.gather_elements(probs, 1, labels_clamped.unsqueeze(1)).squeeze(1)
        true_loss = -ops.log(true_prob + self.eps)
        bg_loss = -ops.log(bg_prob + self.eps)

        include_ignore = ops.logical_and(ignore_mask, bg_exists)
        loss_map = ops.where(valid_mask, true_loss, ops.zeros_like(true_loss))
        loss_map = ops.where(include_ignore, bg_loss, loss_map)
        include_mask = ops.logical_or(valid_mask, include_ignore)

        denom = include_mask.astype(probs.dtype).sum()
        denom = ops.maximum(denom, self.denom_floor)
        return loss_map.sum() / denom


class OhemCELoss(nn.Cell):
    def __init__(self, thresh, ignore_lb=255):
        super().__init__()
        self.thresh = -ops.log(ms.Tensor(thresh, dtype=ms.float32))
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction="none")

    def construct(self, logits, labels):
        valid_labels = ops.masked_select(labels, labels != self.ignore_lb)
        n_min = int(valid_labels.size * 0.25)
        loss = self.criteria(logits, labels).reshape(-1)
        loss_hard = ops.masked_select(loss, loss > self.thresh)
        if loss_hard.size < n_min:
            loss_hard, _ = ops.TopK(sorted=True)(loss, n_min)
        return loss_hard.mean()


class CriterionPixelPair(nn.Cell):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        self.normalize = fedseg_l2_normalize(axis=1)

    def pair_wise_sim_map(self, fea_0, fea_1):
        channels, _, _ = fea_0.shape
        fea_0 = fea_0.reshape(channels, -1).transpose(1, 0)
        fea_1 = fea_1.reshape(channels, -1).transpose(1, 0)
        return ops.matmul(fea_0, fea_1.transpose(1, 0))

    def construct(self, feat_s, feat_t):
        batch, _, _, _ = feat_s.shape
        feat_s = self.normalize(self.avg_pool(feat_s))
        feat_t = self.normalize(self.avg_pool(feat_t))

        sim_dis = ms.Tensor(0.0, ms.float32)
        for idx in range(batch):
            s_sim_map = self.pair_wise_sim_map(feat_s[idx], feat_s[idx])
            t_sim_map = self.pair_wise_sim_map(feat_t[idx], feat_t[idx])
            p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
            p_t = ops.softmax(t_sim_map / self.temperature, axis=1)
            sim_dis = sim_dis + ops.kl_div(p_s, p_t, reduction="batchmean")
        return sim_dis / batch


class CriterionPixelPairSeq(nn.Cell):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args
        self.avg_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0, ceil_mode=True)
        self.normalize = fedseg_l2_normalize(axis=1)

    def construct(self, feat_s, feat_t, pixel_seq):
        _, channels, _, _ = feat_s.shape
        feat_s = self.normalize(self.avg_pool(feat_s))
        feat_t = self.normalize(self.avg_pool(feat_t))

        feat_s = feat_s.permute(0, 2, 3, 1).reshape(-1, channels)
        feat_t = feat_t.permute(0, 2, 3, 1).reshape(-1, channels)

        num_pixels = feat_t.shape[0]
        num_select = min(4000, num_pixels)
        sampled_idx = np.random.choice(num_pixels, num_select, replace=False)
        sampled_idx = ms.Tensor(sampled_idx.astype(np.int32), ms.int32)
        feat_t = ops.gather(feat_t, sampled_idx, 0)
        pixel_seq.extend([item for item in ops.split(feat_t, 1, axis=0)])
        if len(pixel_seq) > 20000:
            del pixel_seq[: len(pixel_seq) - 20000]

        proto_mem = ops.cat(pixel_seq, axis=0)
        s_sim_map = ops.matmul(feat_s, proto_mem.transpose(1, 0))
        t_sim_map = ops.matmul(feat_t, proto_mem.transpose(1, 0))
        p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
        p_t = ops.softmax(t_sim_map / self.temperature, axis=1)
        return ops.kl_div(p_s, p_t, reduction="batchmean"), pixel_seq


class CriterionPixelRegionPair(nn.Cell):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        channels, _, _ = fea_0.shape
        fea_0 = fea_0.reshape(channels, -1).transpose(1, 0)
        fea_1 = fea_1.transpose(1, 0)
        return ops.matmul(fea_0, fea_1)

    def construct(self, feat_s, feat_t, proto_mem, proto_mask):
        batch, _, _, _ = feat_s.shape
        if self.args.kmean_num > 0:
            _, users, km_num, channels = proto_mem.shape
            proto_mem_flat = proto_mem.reshape(-1, channels)
            proto_mask_flat = ops.tile(proto_mask.unsqueeze(-1), (1, 1, km_num)).reshape(-1)
        else:
            _, users, channels = proto_mem.shape
            proto_mem_flat = proto_mem.reshape(-1, channels)
            proto_mask_flat = proto_mask.reshape(-1)

        selected = ops.masked_select(ops.arange(proto_mask_flat.shape[0]), proto_mask_flat.astype(ms.bool_))
        proto_mem_flat = proto_mem_flat[selected]

        sim_dis = ms.Tensor(0.0, ms.float32)
        for idx in range(batch):
            s_sim_map = self.pair_wise_sim_map(feat_s[idx], proto_mem_flat)
            t_sim_map = self.pair_wise_sim_map(feat_t[idx], proto_mem_flat)
            p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
            p_t = ops.softmax(t_sim_map / self.temperature, axis=1)
            sim_dis = sim_dis + ops.kl_div(p_s, p_t, reduction="batchmean")
        return sim_dis / batch


class ContrastLoss(nn.Cell):
    def __init__(self, args, ignore_lb=255):
        super().__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature
        self.num_classes = int(args.num_classes)
        self.num_per_class = max(1, int(args.max_anchor) // max(1, self.num_classes))
        self.topk = ops.TopK(sorted=True)
        self.one = ms.Tensor(1.0, ms.float32)
        self.zero = ms.Tensor(0.0, ms.float32)
        self.eps = ms.Tensor(1e-6, ms.float32)
        self.very_neg = ms.Tensor(-1e4, ms.float32)

    def _gather_rows(self, tensor, indices):
        return ops.gather_nd(tensor, indices.reshape(-1, 1))

    def _anchor_sampling(self, embeddings, labels):
        _, channels, _, _ = embeddings.shape
        embeddings = embeddings.permute(0, 2, 3, 1).reshape(-1, channels)
        labels_np = labels.reshape(-1).asnumpy().astype(np.int32, copy=False)

        class_indices = []
        for class_id in range(self.num_classes):
            selected_index_np = np.flatnonzero(labels_np == class_id).astype(np.int32, copy=False)
            if selected_index_np.size > 0:
                class_indices.append((class_id, selected_index_np))

        if not class_indices:
            return None, None

        num_per_class = max(1, int(self.max_anchor) // len(class_indices))
        sampled_embeddings = []
        sampled_labels = []
        for class_id, selected_index_np in class_indices:
            if selected_index_np.size > num_per_class:
                if getattr(self.args, "debug_deterministic_contrast", False):
                    selected_index_np = selected_index_np[:num_per_class]
                else:
                    selected_index_np = np.random.permutation(selected_index_np)[:num_per_class]
            if selected_index_np.size > 0:
                max_index = int(selected_index_np.max())
                if max_index >= int(embeddings.shape[0]):
                    raise ValueError(
                        "ContrastLoss anchor index out of range: class_id={}, max_index={}, embedding_rows={}".format(
                            class_id,
                            max_index,
                            int(embeddings.shape[0]),
                        )
                    )

            selected_index = ms.Tensor(selected_index_np.astype(np.int32, copy=False), ms.int32)
            sampled_embeddings.append(self._gather_rows(embeddings, selected_index))
            sampled_labels.append(ms.Tensor(np.full(selected_index_np.shape[0], float(class_id), dtype=np.float32)))

        return ops.concat(sampled_embeddings, axis=0), ops.concat(sampled_labels, axis=0)

    def _flatten_proto_memory(self, proto_mem, proto_mask):
        if self.args.kmean_num > 0:
            classes, proto_slots, channels = proto_mem.shape
            proto_mem_flat = proto_mem.reshape(-1, channels)
            proto_labels_np = np.repeat(np.arange(classes, dtype=np.float32), proto_slots)
            selected_np = np.flatnonzero(proto_mask.reshape(-1).asnumpy().astype(bool, copy=False)).astype(np.int32, copy=False)
            if selected_np.size == 0:
                return None, None
            selected = ms.Tensor(selected_np, ms.int32)
            return (
                self._gather_rows(proto_mem_flat, selected),
                ms.Tensor(proto_labels_np[selected_np], ms.float32),
            )

        classes, channels = proto_mem.shape
        proto_mem_flat = proto_mem
        selected_np = np.flatnonzero(proto_mask.reshape(-1).asnumpy().astype(bool, copy=False)).astype(np.int32, copy=False)
        if selected_np.size == 0:
            return None, None
        selected = ms.Tensor(selected_np, ms.int32)
        return (
            self._gather_rows(proto_mem_flat, selected),
            ms.Tensor(np.arange(classes, dtype=np.float32)[selected_np], ms.float32),
        )

    def construct(self, embeddings, labels, proto_mem, proto_mask):
        anchors, anchor_labels = self._anchor_sampling(embeddings, labels)
        if anchors is None:
            return self.zero

        proto_mem_flat, proto_labels = self._flatten_proto_memory(proto_mem, proto_mask)
        if proto_mem_flat is None:
            return self.zero

        anchor_dot = ops.matmul(anchors, proto_mem_flat.transpose(1, 0)) / self.temperature
        mask = (anchor_labels.unsqueeze(1) == proto_labels.unsqueeze(0)).astype(ms.float32)
        neg_mask = 1.0 - mask

        logits_max = anchor_dot.max(axis=1, keepdims=True)
        logits = anchor_dot - ops.stop_gradient(logits_max)

        exp_logits = ops.exp(logits)
        neg_logits = (exp_logits * neg_mask).sum(axis=1, keepdims=True)
        pos_exp_logits = exp_logits * mask
        log_prob = logits - ops.log(pos_exp_logits + neg_logits + self.eps)

        mask_sum = mask.sum(axis=1)
        valid_anchor = (mask_sum > 0).astype(ms.float32)
        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / ops.maximum(mask_sum, self.one)
        loss = -(mean_log_prob_pos * valid_anchor).sum() / ops.maximum(valid_anchor.sum(), self.one)

        return loss

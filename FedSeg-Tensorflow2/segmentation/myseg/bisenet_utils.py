from pathlib import Path
# import torch
# from torch import nn
import numpy as np
# import torch.nn.functional as F

from myseg.bisenetv2 import BiSeNetV2
import tensorflow as tf
from logging_utils import logger
from tf2_tools import load_tf_backbone_into_tf


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _default_tf_backbone_path() -> Path:
    return Path(__file__).resolve().with_name("backbone_v2.weights.h5")


def _resolve_tf_backbone_path(args) -> Path:
    configured = getattr(args, "backbone_checkpoint", "") or ""
    if configured:
        checkpoint = Path(configured)
        if checkpoint.is_absolute():
            return checkpoint
        root = Path(getattr(args, "root", "") or ".")
        rooted = root / checkpoint
        if rooted.exists():
            return rooted.resolve()
        return checkpoint.resolve()
    return _default_tf_backbone_path()


def set_model_bisenetv2(args, num_classes):
    net = BiSeNetV2(proj_dim=args.proj_dim, n_classes=num_classes) # num_classes = 19
    if not _as_bool(getattr(args, "rand_init", True)):
        backbone_checkpoint = _resolve_tf_backbone_path(args)
        if not backbone_checkpoint.exists():
            raise FileNotFoundError(
                f"TensorFlow backbone checkpoint not found: {backbone_checkpoint}. "
                "Run segmentation/convert_torch_backbone_to_tf.py first."
            )
        load_tf_backbone_into_tf(net, backbone_checkpoint)
        logger.info("loaded TensorFlow BiSeNetV2 backbone: {}", backbone_checkpoint)

    # if not args.finetune_from is None:
    #     logger.info(f'load pretrained weights from {args.finetune_from}')
    #     net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))
    # if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # net.cuda()
    # net.train()
    # criteria_pre = OhemCELoss(0.7)
    # criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4
    # return net, criteria_pre, criteria_aux

    # TensorFlow2 版本直接返回模型实例
    return net


def set_optimizer(model, args):
    if model is not None and hasattr(model, "get_params"):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        var_lr_multipliers = {}
        var_weight_decays = {}

        for var in wd_params:
            var_lr_multipliers[id(var)] = 1.0
            var_weight_decays[id(var)] = args.weight_decay
        for var in nowd_params:
            var_lr_multipliers[id(var)] = 1.0
            var_weight_decays[id(var)] = 0.0
        for var in lr_mul_wd_params:
            var_lr_multipliers[id(var)] = 10.0
            var_weight_decays[id(var)] = args.weight_decay
        for var in lr_mul_nowd_params:
            var_lr_multipliers[id(var)] = 10.0
            var_weight_decays[id(var)] = 0.0
    else:
        var_lr_multipliers = {}
        var_weight_decays = {}
        if model is not None:
            for var in model.trainable_variables:
                var_lr_multipliers[id(var)] = 1.0
                var_weight_decays[id(var)] = args.weight_decay if len(var.shape) in {2, 4} else 0.0

    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=0.0,
    )
    optimizer._fedseg_var_lr_multipliers = var_lr_multipliers
    optimizer._fedseg_var_weight_decays = var_weight_decays
    return optimizer

# class BackCELoss(nn.Module):
#     def __init__(self, args, ignore_lb=255):
#         super(BackCELoss, self).__init__()
#         self.ignore_lb = ignore_lb
#         self.class_num = args.num_classes
#         self.criteria = nn.NLLLoss(ignore_index=ignore_lb, reduction='mean')
#     def forward(self, logits, labels):
#         total_labels = torch.unique(labels)
#         new_labels = labels.clone()
#         probs = torch.softmax(logits,1)
#         fore_ = []
#         back_ = []
#         for l in range(self.class_num):
#             if l in total_labels:
#                 fore_.append(probs[:,l,:,:].unsqueeze(1))
#             else:
#                 back_.append(probs[:,l,:,:].unsqueeze(1))
#         Flag=False
#         if not  len(fore_)==self.class_num:
#             fore_.append(sum(back_))
#             Flag=True
#         for i,l in enumerate(total_labels):
#             if Flag :
#                 new_labels[labels==l]=i
#             else:
#                 if l!=255:
#                     new_labels[labels==l]=i
#         probs  =torch.cat(fore_,1)
#         logprobs = torch.log(probs+1e-7)
#         return self.criteria(logprobs,new_labels.long())

class BackCELoss(tf.keras.layers.Layer):
    def __init__(self, args, ignore_lb=255, name='BackCELoss'):
        super().__init__(name=name)
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes

    def call(self,labels,logits):
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])  # NCHW to NHWC
        # 转换类型
        labels = tf.cast(labels, tf.int32)
        logits = tf.cast(logits, tf.float32)
        
        # 1. Softmax 概率
        probs = tf.nn.softmax(logits, axis=-1)
        
        # 2. 获取并排序 Unique Labels
        # 先展平以便 unique 和后续处理
        flat_labels_raw = tf.reshape(labels, [-1])
        unique_labels, _ = tf.unique(flat_labels_raw)
        unique_labels = tf.sort(unique_labels) # [N_unique]
        
        # 3. 区分 Foreground 和 Background 通道
        all_classes = tf.range(self.class_num, dtype=tf.int32)
        
        # 判断每个类别是否存在: [C, 1] == [1, N_unique] -> [C, N_unique] -> reduce_any -> [C]
        is_present = tf.reduce_any(tf.equal(tf.expand_dims(all_classes, 1), tf.expand_dims(unique_labels, 0)), axis=1)
        
        # 获取索引并提取通道 (使用 tf.gather 避免 Shape Inference 问题)
        fore_indices = tf.reshape(tf.where(is_present), [-1])
        back_indices = tf.reshape(tf.where(tf.logical_not(is_present)), [-1])
        
        fore_probs = tf.gather(probs, fore_indices, axis=-1)
        back_probs = tf.gather(probs, back_indices, axis=-1)
        
        # 4. 合并逻辑
        has_back = tf.size(back_indices) > 0
        
        def merge_back():
            back_sum = tf.reduce_sum(back_probs, axis=-1, keepdims=True)
            return tf.concat([fore_probs, back_sum], axis=-1)
            
        def keep_fore():
            return fore_probs
            
        probs_cat = tf.cond(has_back, merge_back, keep_fore)
        
        # 5. 重新映射标签 (修复 searchsorted 报错的关键)
        # 将 labels 展平为 1D 进行 searchsorted，避免 Batch 维度匹配错误
        flat_labels = tf.reshape(labels, [-1])
        
        # searchsorted: 查找 flat_labels 中每个元素在 unique_labels 中的索引
        flat_new_labels = tf.searchsorted(unique_labels, flat_labels)
        
        # 恢复原始形状 [B, H, W]
        new_labels = tf.reshape(flat_new_labels, tf.shape(labels))
        
        # 6. 计算 Loss
        logprobs = tf.math.log(probs_cat + 1e-7)
        depth = tf.shape(probs_cat)[-1]
        
        # 生成 One-Hot
        # 如果 new_labels 的值 >= depth (例如 Flag=False 时的 ignore_lb)，one_hot 全为 0
        one_hot = tf.one_hot(new_labels, depth)
        
        per_pixel_loss = -tf.reduce_sum(one_hot * logprobs, axis=-1)
        
        # 7. 归一化 (valid_mask 排除 ignore 区域)
        valid_mask = tf.reduce_sum(one_hot, axis=-1) # [B, H, W]
        num_valid = tf.reduce_sum(valid_mask)
        
        loss = tf.reduce_sum(per_pixel_loss)
        
        return loss / (num_valid + 1e-7)




# class OhemCELoss(nn.Module):
#     '''
#     Feddrive: We apply OHEM (Online Hard-Negative Mining) [56], selecting 25%
#     of the pixels having the highest loss for the optimization.
#     '''
#     def __init__(self, thresh, ignore_lb=255):
#         super(OhemCELoss, self).__init__()
#         self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
#         self.ignore_lb = ignore_lb
#         self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')
#     def forward(self, logits, labels):
#         # n_min = labels[labels != self.ignore_lb].numel() // 16
#         n_min = int(labels[labels != self.ignore_lb].numel() * 0.25)
#         loss = self.criteria(logits, labels).view(-1)
#         loss_hard = loss[loss > self.thresh]
#         if loss_hard.numel() < n_min:
#             loss_hard, _ = loss.topk(n_min)
#         return torch.mean(loss_hard)

class OhemCELoss(tf.keras.layers.Layer):
    '''
    Feddrive: We apply OHEM (Online Hard-Negative Mining) [56], selecting 25%
    of the pixels having the highest loss for the optimization.
    '''
    def __init__(self, thresh, ignore_lb=255):
        super().__init__()
        self.thresh = -tf.math.log(tf.constant(thresh, dtype=tf.float32))
        self.ignore_lb = ignore_lb

    def call(self, labels, logits):
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])
        mask = tf.not_equal(labels, self.ignore_lb)
        new_labels = tf.where(mask, labels, tf.zeros_like(labels))
        loss = tf.keras.losses.sparse_categorical_crossentropy(new_labels, logits, from_logits=True)
        loss = tf.where(mask, loss, tf.zeros_like(loss))
        loss_flat = tf.reshape(loss, [-1])
        hard_loss = tf.boolean_mask(loss_flat, loss_flat > self.thresh)
        valid_count = tf.reduce_sum(tf.cast(mask, tf.float32))
        n_min = tf.cast(valid_count * 0.25, tf.int32)
        if tf.size(hard_loss) < n_min:
            hard_loss = tf.sort(loss_flat, direction='DESCENDING')[:n_min]
        return tf.reduce_mean(hard_loss)


class SparseCEIgnore(tf.keras.layers.Layer):
    def __init__(self, ignore_lb=255):
        super().__init__()
        self.ignore_lb = ignore_lb

    def call(self, labels, logits):
        logits = tf.transpose(logits, perm=[0, 2, 3, 1])
        labels = tf.cast(labels, tf.int32)
        valid_mask = tf.not_equal(labels, self.ignore_lb)
        safe_labels = tf.where(valid_mask, labels, tf.zeros_like(labels))
        loss = tf.keras.losses.sparse_categorical_crossentropy(safe_labels, logits, from_logits=True)
        loss = tf.where(valid_mask, loss, tf.zeros_like(loss))
        denom = tf.reduce_sum(tf.cast(valid_mask, tf.float32))
        return tf.reduce_sum(loss) / (denom + 1e-7)


def _soft_tversky_score(output, target, alpha, beta, smooth=0.0, eps=1e-7, axes=None):
    if axes is None:
        output_sum = tf.reduce_sum(output)
        target_sum = tf.reduce_sum(target)
        difference = tf.reduce_sum(tf.abs(output - target))
    else:
        output_sum = tf.reduce_sum(output, axis=axes)
        target_sum = tf.reduce_sum(target, axis=axes)
        difference = tf.reduce_sum(tf.abs(output - target), axis=axes)

    intersection = (output_sum + target_sum - difference) / 2.0
    fp = output_sum - intersection
    fn = target_sum - intersection
    return (intersection + smooth) / tf.maximum(intersection + alpha * fp + beta * fn + smooth, eps)


def _soft_dice_score(output, target, smooth=0.0, eps=1e-7, axes=None):
    return _soft_tversky_score(output, target, 0.5, 0.5, smooth=smooth, eps=eps, axes=axes)


def _focal_loss_with_logits(
    output,
    target,
    gamma=2.0,
    alpha=0.25,
    reduction="mean",
    normalized=False,
    reduced_threshold=None,
    eps=1e-6,
):
    target = tf.cast(target, output.dtype)
    logpt = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)
    pt = tf.exp(-logpt)

    if reduced_threshold is None:
        focal_term = tf.pow(1.0 - pt, gamma)
    else:
        focal_term = tf.pow((1.0 - pt) / reduced_threshold, gamma)
        focal_term = tf.where(pt < reduced_threshold, tf.ones_like(focal_term), focal_term)

    loss = focal_term * logpt
    if alpha is not None:
        loss *= alpha * target + (1.0 - alpha) * (1.0 - target)

    if normalized:
        loss = loss / tf.maximum(tf.reduce_sum(focal_term), eps)

    if reduction == "sum":
        return tf.reduce_sum(loss)
    if reduction == "batchwise_mean":
        return tf.reduce_sum(loss, axis=0)
    if reduction == "none":
        return loss
    return tf.reduce_mean(loss)


class SoftBCEWithLogitsLoss(tf.keras.layers.Layer):
    def __init__(self, ignore_index=-100, reduction="mean", smooth_factor=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor

    def call(self, labels, logits):
        labels = tf.cast(labels, tf.float32)
        logits = tf.cast(logits, tf.float32)
        if len(labels.shape) == 4 and labels.shape[-1] is not None and labels.shape[1] != logits.shape[1]:
            labels = tf.transpose(labels, perm=[0, 3, 1, 2])

        if self.smooth_factor is not None:
            soft_targets = (1.0 - labels) * self.smooth_factor + labels * (1.0 - self.smooth_factor)
        else:
            soft_targets = labels

        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=soft_targets, logits=logits)
        if self.ignore_index is not None:
            not_ignored_mask = tf.cast(tf.not_equal(labels, tf.cast(self.ignore_index, labels.dtype)), loss.dtype)
            loss = loss * not_ignored_mask

        if self.reduction == "sum":
            return tf.reduce_sum(loss)
        if self.reduction == "none":
            return loss
        return tf.reduce_mean(loss)


class FocalLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        mode,
        alpha=None,
        gamma=2.0,
        ignore_index=None,
        reduction="mean",
        normalized=False,
        reduced_threshold=None,
    ):
        super().__init__()
        self.mode = mode
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.normalized = normalized
        self.reduced_threshold = reduced_threshold

    def call(self, labels, logits):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)

        if self.mode != "multiclass":
            raise ValueError(f"unsupported focal mode: {self.mode}")

        num_classes = logits.shape[1]
        total_loss = tf.constant(0.0, dtype=logits.dtype)
        not_ignored = None
        if self.ignore_index is not None:
            not_ignored = tf.not_equal(labels, self.ignore_index)

        for cls_idx in range(num_classes):
            cls_y_true = tf.cast(tf.equal(labels, cls_idx), logits.dtype)
            cls_y_pred = logits[:, cls_idx, ...]

            if not_ignored is not None:
                cls_y_true = tf.boolean_mask(cls_y_true, not_ignored)
                cls_y_pred = tf.boolean_mask(cls_y_pred, not_ignored)

            total_loss += _focal_loss_with_logits(
                cls_y_pred,
                cls_y_true,
                gamma=self.gamma,
                alpha=self.alpha,
                reduction=self.reduction,
                normalized=self.normalized,
                reduced_threshold=self.reduced_threshold,
            )
        return total_loss


class DiceLoss(tf.keras.layers.Layer):
    def __init__(
        self,
        mode,
        classes=None,
        log_loss=False,
        from_logits=True,
        smooth=0.0,
        ignore_index=None,
        eps=1e-7,
    ):
        super().__init__()
        self.mode = mode
        self.classes = classes
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def call(self, labels, logits):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)

        if self.mode != "multiclass":
            raise ValueError(f"unsupported dice mode: {self.mode}")

        if self.from_logits:
            probs = tf.exp(tf.nn.log_softmax(logits, axis=1))
        else:
            probs = logits

        batch_size = tf.shape(labels)[0]
        num_classes = tf.shape(probs)[1]
        labels = tf.reshape(labels, [batch_size, -1])
        probs = tf.reshape(probs, [batch_size, num_classes, -1])

        if self.ignore_index is not None:
            mask = tf.not_equal(labels, self.ignore_index)
            probs = probs * tf.cast(tf.expand_dims(mask, axis=1), probs.dtype)
            labels_safe = tf.where(mask, labels, tf.zeros_like(labels))
            labels_one_hot = tf.one_hot(labels_safe, depth=num_classes, dtype=probs.dtype)
            labels_one_hot = tf.transpose(labels_one_hot, [0, 2, 1]) * tf.cast(tf.expand_dims(mask, axis=1), probs.dtype)
        else:
            labels_one_hot = tf.one_hot(labels, depth=num_classes, dtype=probs.dtype)
            labels_one_hot = tf.transpose(labels_one_hot, [0, 2, 1])

        scores = _soft_dice_score(probs, labels_one_hot, smooth=self.smooth, eps=self.eps, axes=(0, 2))
        if self.log_loss:
            loss = -tf.math.log(tf.maximum(scores, self.eps))
        else:
            loss = 1.0 - scores

        non_empty = tf.reduce_sum(labels_one_hot, axis=(0, 2)) > 0
        loss = loss * tf.cast(non_empty, loss.dtype)

        if self.classes is not None:
            loss = tf.gather(loss, self.classes)
        return tf.reduce_mean(loss)


def _lovasz_grad(gt_sorted):
    gt_sorted = tf.cast(gt_sorted, tf.float32)
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1.0 - gt_sorted)
    jaccard = 1.0 - intersection / union
    return tf.cond(
        tf.shape(jaccard)[0] > 1,
        lambda: tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0),
        lambda: jaccard,
    )


def _flatten_probas(probas, labels, ignore=None):
    if len(probas.shape) == 3:
        probas = tf.expand_dims(probas, axis=1)

    num_classes = tf.shape(probas)[1]
    probas = tf.transpose(probas, [0, 2, 3, 1])
    probas = tf.reshape(probas, [-1, num_classes])
    labels = tf.reshape(labels, [-1])

    if ignore is None:
        return probas, labels

    valid = tf.not_equal(labels, ignore)
    return tf.boolean_mask(probas, valid), tf.boolean_mask(labels, valid)


def _lovasz_softmax_flat(probas, labels, class_num, classes="present"):
    if tf.size(probas) == 0:
        return tf.reduce_sum(probas) * 0.0

    losses = []
    class_to_sum = range(class_num) if classes in {"all", "present"} else classes
    for cls_idx in class_to_sum:
        fg = tf.cast(tf.equal(labels, cls_idx), probas.dtype)
        if classes == "present" and float(tf.reduce_sum(fg).numpy()) == 0.0:
            continue

        class_pred = probas[:, cls_idx]
        errors = tf.abs(fg - class_pred)
        perm = tf.argsort(errors, direction="DESCENDING")
        errors_sorted = tf.gather(errors, perm)
        fg_sorted = tf.gather(fg, perm)
        losses.append(tf.tensordot(errors_sorted, _lovasz_grad(fg_sorted), axes=1))

    if not losses:
        return tf.reduce_sum(probas) * 0.0
    return tf.add_n(losses) / len(losses)


class LovaszLoss(tf.keras.layers.Layer):
    def __init__(self, mode, per_image=False, ignore_index=None, from_logits=True):
        super().__init__()
        self.mode = mode
        self.per_image = per_image
        self.ignore_index = ignore_index
        self.from_logits = from_logits

    def call(self, labels, logits):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.int32)

        if self.mode != "multiclass":
            raise ValueError(f"unsupported lovasz mode: {self.mode}")

        probas = tf.nn.softmax(logits, axis=1) if self.from_logits else logits
        if self.per_image:
            losses = []
            for prob, label in zip(tf.unstack(probas, axis=0), tf.unstack(labels, axis=0)):
                prob_flat, label_flat = _flatten_probas(tf.expand_dims(prob, axis=0), tf.expand_dims(label, axis=0), self.ignore_index)
                losses.append(_lovasz_softmax_flat(prob_flat, label_flat, class_num=logits.shape[1], classes="present"))
            return tf.add_n(losses) / len(losses)

        prob_flat, label_flat = _flatten_probas(probas, labels, self.ignore_index)
        return _lovasz_softmax_flat(prob_flat, label_flat, class_num=logits.shape[1], classes="present")


#################################################

# class CriterionPixelPair(nn.Module):
#     def __init__(self, args,temperature=0.1,ignore_index=255, ):
#         super(CriterionPixelPair, self).__init__()
#         self.ignore_index = ignore_index
#         self.temperature = temperature
#         self.args= args
#     def pair_wise_sim_map(self, fea_0, fea_1):
#         C, H, W = fea_0.size()
#         fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
#         fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
#         sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
#         return sim_map_0_1
#     def forward(self, feat_S, feat_T):
#         #feat_T = self.concat_all_gather(feat_T)
#         #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
#         B, C, H, W = feat_S.size()
#         device = feat_S.device
#         patch_w = 2
#         patch_h = 2
#         #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
#         maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
#         feat_S = maxpool(feat_S)
#         feat_T= maxpool(feat_T)
#         feat_S = F.normalize(feat_S, p=2, dim=1)
#         feat_T = F.normalize(feat_T, p=2, dim=1)
#         sim_dis = torch.tensor(0.).to(device)
#         for i in range(B):
#             s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[i])
#             t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[i])
#             p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
#             p_t = F.softmax(t_sim_map / self.temperature, dim=1)
#             sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
#             sim_dis += sim_dis_
#         sim_dis = sim_dis / B
#         return sim_dis

class CriterionPixelPair(tf.keras.layers.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # fea_0, fea_1: [C, H, W]
        C = tf.shape(fea_0)[0]
        H = tf.shape(fea_0)[1]
        W = tf.shape(fea_0)[2]
        fea_0 = tf.reshape(fea_0, [C, -1])
        fea_1 = tf.reshape(fea_1, [C, -1])
        sim_map_0_1 = tf.matmul(fea_0, fea_1, transpose_b=True)
        return sim_map_0_1

    def call(self, feat_S, feat_T):
        # feat_S, feat_T: [B, C, H, W]
        patch_w = 2
        patch_h = 2
        feat_S = tf.nn.avg_pool(feat_S, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_T = tf.nn.avg_pool(feat_T, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_S = tf.nn.l2_normalize(feat_S, axis=1)
        feat_T = tf.nn.l2_normalize(feat_T, axis=1)
        sim_dis = 0.0
        B = tf.shape(feat_S)[0]
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[i])
            t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[i])
            p_s = tf.nn.log_softmax(s_sim_map / self.temperature, axis=1)
            p_t = tf.nn.softmax(t_sim_map / self.temperature, axis=1)
            sim_dis += tf.reduce_mean(tf.keras.losses.KLD(p_t, p_s))
        sim_dis = sim_dis / tf.cast(B, tf.float32)
        return sim_dis
######################################################

# class CriterionPixelPairSeq(nn.Module):
#     def __init__(self, args,temperature=0.1,ignore_index=255, ):
#         super(CriterionPixelPairSeq, self).__init__()
#         self.ignore_index = ignore_index
#         self.temperature = temperature
#         self.args= args
#     def pair_wise_sim_map(self, fea_0, fea_1):
#         C, H, W = fea_0.size()
#         fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
#         fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
#         sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
#         return sim_map_0_1
#     def forward(self, feat_S, feat_T, pixel_seq):
#         #feat_T = self.concat_all_gather(feat_T)
#         #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
#         B, C, H, W = feat_S.size()
#         device = feat_S.device
#         patch_w = 2
#         patch_h = 2
#         #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
#         maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
#         feat_S = maxpool(feat_S)
#         feat_T= maxpool(feat_T)
#         feat_S = F.normalize(feat_S, p=2, dim=1)
#         feat_T = F.normalize(feat_T, p=2, dim=1)
#         feat_S = feat_S.permute(0,2,3,1).reshape(-1,C)
#         feat_T = feat_T.permute(0,2,3,1).reshape(-1,C)
#         split_T = feat_T
#         idx = np.random.choice(len(split_T),4000,replace=False)
#         split_T = split_T[idx]
#         split_T = torch.split(split_T,1,dim=0)
#         pixel_seq.extend(split_T)
#         if len(pixel_seq)>20000:
#             del pixel_seq[:len(pixel_seq)-20000]
#         proto_mem_ = torch.cat(pixel_seq,0)
#         s_sim_map = torch.matmul(feat_S,proto_mem_.T)
#         t_sim_map = torch.matmul(feat_T,proto_mem_.T)
#         p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
#         p_t = F.softmax(t_sim_map / self.temperature, dim=1)
#         sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
#         return sim_dis,pixel_seq

class CriterionPixelPairSeq(tf.keras.layers.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # fea_0, fea_1: [C, H, W]
        C = tf.shape(fea_0)[0]
        H = tf.shape(fea_0)[1]
        W = tf.shape(fea_0)[2]
        fea_0 = tf.reshape(fea_0, [C, -1])
        fea_1 = tf.reshape(fea_1, [C, -1])
        sim_map_0_1 = tf.matmul(fea_0, fea_1, transpose_b=True)
        return sim_map_0_1

    def call(self, feat_S, feat_T, pixel_seq):
        # feat_S, feat_T: [B, C, H, W]
        patch_w = 2
        patch_h = 2
        feat_S = tf.nn.avg_pool(feat_S, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_T = tf.nn.avg_pool(feat_T, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_S = tf.nn.l2_normalize(feat_S, axis=1)
        feat_T = tf.nn.l2_normalize(feat_T, axis=1)
        B = tf.shape(feat_S)[0]
        C = tf.shape(feat_S)[1]
        feat_S_flat = tf.reshape(tf.transpose(feat_S, [0, 2, 3, 1]), [-1, C])
        feat_T_flat = tf.reshape(tf.transpose(feat_T, [0, 2, 3, 1]), [-1, C])
        split_T = feat_T_flat
        idx = np.random.choice(split_T.shape[0], 4000, replace=False)
        split_T = tf.gather(split_T, idx)
        split_T = tf.split(split_T, num_or_size_splits=split_T.shape[0], axis=0)
        pixel_seq.extend(split_T)
        if len(pixel_seq) > 20000:
            del pixel_seq[:len(pixel_seq) - 20000]
        proto_mem_ = tf.concat(pixel_seq, axis=0)
        s_sim_map = tf.matmul(feat_S_flat, proto_mem_, transpose_b=True)
        t_sim_map = tf.matmul(feat_T_flat, proto_mem_, transpose_b=True)
        p_s = tf.nn.log_softmax(s_sim_map / self.temperature, axis=1)
        p_t = tf.nn.softmax(t_sim_map / self.temperature, axis=1)
        sim_dis = tf.reduce_mean(tf.keras.losses.KLD(p_t, p_s))
        return sim_dis, pixel_seq
######################################################

# class CriterionPixelPairG(nn.Module):
#     def __init__(self, args,temperature=0.1,ignore_index=255, ):
#         super(CriterionPixelPairG, self).__init__()
#         self.ignore_index = ignore_index
#         self.temperature = temperature
#         self.args= args
#     def pair_wise_sim_map(self, fea_0, fea_1):
#         C, H, W = fea_0.size()
#         fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
#         fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
#         sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
#         return sim_map_0_1
#     def forward(self, feat_S, feat_T,proto_mem,proto_mask):
#         #feat_T = self.concat_all_gather(feat_T)
#         #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
#         B, C, H, W = feat_S.size()
#         device = feat_S.device
#         patch_w = 2
#         patch_h = 2
#         #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
#         maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
#         feat_S = maxpool(feat_S)
#         feat_T= maxpool(feat_T)
#         feat_S = F.normalize(feat_S, p=2, dim=1)
#         feat_T = F.normalize(feat_T, p=2, dim=1)
#         feat_S = feat_S.permute(0,2,3,1).reshape(-1,C)
#         feat_T = feat_T.permute(0,2,3,1).reshape(-1,C)
#         if self.args.kmean_num>0:
#             C_,km_,c_ = proto_mem.size()
#             proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,km_)
#             proto_mem_ = proto_mem.reshape(-1,c_)
#             proto_mask = proto_mask.view(-1)
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#         else:
#             C_,c_ = proto_mem.size()
#             proto_labels = torch.arange(C_)
#             proto_mem_ = proto_mem
#             proto_mask = proto_mask
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#         s_sim_map = torch.matmul(feat_S,proto_mem_.T)
#         t_sim_map = torch.matmul(feat_T,proto_mem_.T)
#         p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
#         p_t = F.softmax(t_sim_map / self.temperature, dim=1)
#         sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
#         return sim_dis

class CriterionPixelPairG(tf.keras.layers.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # fea_0, fea_1: [C, H, W]
        C = tf.shape(fea_0)[0]
        H = tf.shape(fea_0)[1]
        W = tf.shape(fea_0)[2]
        fea_0 = tf.reshape(fea_0, [C, -1])
        fea_1 = tf.reshape(fea_1, [C, -1])
        sim_map_0_1 = tf.matmul(fea_0, fea_1, transpose_b=True)
        return sim_map_0_1

    def call(self, feat_S, feat_T, proto_mem, proto_mask):
        # feat_S, feat_T: [B, C, H, W]
        patch_w = 2
        patch_h = 2
        feat_S = tf.nn.avg_pool(feat_S, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_T = tf.nn.avg_pool(feat_T, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_S = tf.nn.l2_normalize(feat_S, axis=1)
        feat_T = tf.nn.l2_normalize(feat_T, axis=1)
        B = tf.shape(feat_S)[0]
        C = tf.shape(feat_S)[1]
        feat_S_flat = tf.reshape(tf.transpose(feat_S, [0, 2, 3, 1]), [-1, C])
        feat_T_flat = tf.reshape(tf.transpose(feat_T, [0, 2, 3, 1]), [-1, C])
        if self.args.kmean_num > 0:
            C_ = tf.shape(proto_mem)[0]
            km_ = tf.shape(proto_mem)[1]
            c_ = tf.shape(proto_mem)[2]
            proto_mem_ = tf.reshape(proto_mem, [-1, c_])
            proto_mask_flat = tf.reshape(proto_mask, [-1])
            proto_idx = tf.range(tf.shape(proto_mask_flat)[0])
            sel_idx = tf.boolean_mask(proto_idx, tf.cast(proto_mask_flat, tf.bool))
            proto_mem_ = tf.gather(proto_mem_, sel_idx)
        else:
            C_ = tf.shape(proto_mem)[0]
            c_ = tf.shape(proto_mem)[1]
            proto_mem_ = proto_mem
            proto_mask_flat = tf.reshape(proto_mask, [-1])
            proto_idx = tf.range(tf.shape(proto_mask_flat)[0])
            sel_idx = tf.boolean_mask(proto_idx, tf.cast(proto_mask_flat, tf.bool))
            proto_mem_ = tf.gather(proto_mem_, sel_idx)
        s_sim_map = tf.matmul(feat_S_flat, proto_mem_, transpose_b=True)
        t_sim_map = tf.matmul(feat_T_flat, proto_mem_, transpose_b=True)
        p_s = tf.nn.log_softmax(s_sim_map / self.temperature, axis=1)
        p_t = tf.nn.softmax(t_sim_map / self.temperature, axis=1)
        sim_dis = tf.reduce_mean(tf.keras.losses.KLD(p_t, p_s))
        return sim_dis
######################################################

# class CriterionPixelRegionPair(nn.Module):
#     def __init__(self,args, temperature=0.1,ignore_index=255, ):
#         super(CriterionPixelRegionPair, self).__init__()
#         self.ignore_index = ignore_index
#         self.temperature = temperature
#         self.args = args
#     def pair_wise_sim_map(self, fea_0, fea_1):
#         C, H, W = fea_0.size()
#         fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
#         fea_1 = fea_1.transpose(0, 1)
#         sim_map_0_1 = torch.mm(fea_0, fea_1)
#         return sim_map_0_1
#     def forward(self, feat_S, feat_T,proto_mem,proto_mask):
#         #feat_T = self.concat_all_gather(feat_T)
#         #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
#         B, C, H, W = feat_S.size()
#         device = feat_S.device
#         if self.args.kmean_num>0:
#             C_,U_,km_,c_ = proto_mem.size()
#             proto_mem_ = proto_mem.reshape(-1,c_)
#             proto_mask = proto_mask.unsqueeze(-1).repeat(1,1,km_).view(-1)
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#         else:
#             C_,U_,c_ = proto_mem.size()
#             proto_mem_ = proto_mem.reshape(-1,c_)
#             proto_mask = proto_mask.view(-1)
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#         sim_dis = torch.tensor(0.).to(device)
#         for i in range(B):
#             s_sim_map = self.pair_wise_sim_map(feat_S[i], proto_mem_)
#             t_sim_map = self.pair_wise_sim_map(feat_T[i], proto_mem_)
#             p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
#             p_t = F.softmax(t_sim_map / self.temperature, dim=1)
#             sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
#             sim_dis += sim_dis_
#         sim_dis = sim_dis / B
#         return sim_dis

class CriterionPixelRegionPair(tf.keras.layers.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # fea_0, fea_1: [C, H, W]
        C = tf.shape(fea_0)[0]
        H = tf.shape(fea_0)[1]
        W = tf.shape(fea_0)[2]
        fea_0 = tf.reshape(fea_0, [C, -1])
        fea_1 = tf.transpose(fea_1, [1, 0])
        sim_map_0_1 = tf.matmul(fea_0, fea_1)
        return sim_map_0_1

    def call(self, feat_S, feat_T, proto_mem, proto_mask):
        # feat_S, feat_T: [B, C, H, W]
        B = tf.shape(feat_S)[0]
        sim_dis = 0.0
        if self.args.kmean_num > 0:
            C_ = tf.shape(proto_mem)[0]
            U_ = tf.shape(proto_mem)[1]
            km_ = tf.shape(proto_mem)[2]
            c_ = tf.shape(proto_mem)[3]
            proto_mem_ = tf.reshape(proto_mem, [-1, c_])
            proto_mask_flat = tf.reshape(proto_mask, [-1])
            proto_mask_flat = tf.tile(proto_mask_flat, [km_])
            proto_idx = tf.range(tf.shape(proto_mask_flat)[0])
            sel_idx = tf.boolean_mask(proto_idx, tf.cast(proto_mask_flat, tf.bool))
            proto_mem_ = tf.gather(proto_mem_, sel_idx)
        else:
            C_ = tf.shape(proto_mem)[0]
            U_ = tf.shape(proto_mem)[1]
            c_ = tf.shape(proto_mem)[2]
            proto_mem_ = tf.reshape(proto_mem, [-1, c_])
            proto_mask_flat = tf.reshape(proto_mask, [-1])
            proto_idx = tf.range(tf.shape(proto_mask_flat)[0])
            sel_idx = tf.boolean_mask(proto_idx, tf.cast(proto_mask_flat, tf.bool))
            proto_mem_ = tf.gather(proto_mem_, sel_idx)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], proto_mem_)
            t_sim_map = self.pair_wise_sim_map(feat_T[i], proto_mem_)
            p_s = tf.nn.log_softmax(s_sim_map / self.temperature, axis=1)
            p_t = tf.nn.softmax(t_sim_map / self.temperature, axis=1)
            sim_dis += tf.reduce_mean(tf.keras.losses.KLD(p_t, p_s))
        sim_dis = sim_dis / tf.cast(B, tf.float32)
        return sim_dis

######################################################


# def L2(f_):
#     return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

# def similarity(feat):
#     feat = feat.float()
#     tmp = L2(feat).detach()
#     feat = feat/tmp
#     feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
#     return torch.einsum('icm,icn->imn', [feat, feat])

# def sim_dis_compute(f_S, f_T):
#     sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
#     sim_dis = sim_err.sum()
#     return sim_dis

def L2(f_):
    return tf.sqrt(tf.reduce_sum(tf.square(f_), axis=1, keepdims=True)) + 1e-8

def similarity(feat):
    feat = tf.cast(feat, tf.float32)
    tmp = L2(feat)
    feat = feat / tmp
    feat = tf.reshape(feat, [tf.shape(feat)[0], tf.shape(feat)[1], -1])
    sim = tf.einsum('icm,icn->imn', feat, feat)
    return sim

def sim_dis_compute(f_S, f_T):
    sim_err = tf.square(similarity(f_T) - similarity(f_S)) / (tf.cast(tf.shape(f_T)[-1] * tf.shape(f_T)[-2], tf.float32) ** 2) / tf.cast(tf.shape(f_T)[0], tf.float32)
    sim_dis = tf.reduce_sum(sim_err)
    return sim_dis

# class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
#     def __init__(self, scale):
#         '''inter pair-wise loss from inter feature maps'''
#         super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
#         self.criterion = sim_dis_compute
#         self.scale = scale
#     def forward(self, preds_S, preds_T):
#         feat_S = preds_S
#         feat_T = preds_T
#         feat_T.detach()
#         total_w, total_h = feat_T.shape[2], feat_T.shape[3]
#         patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
#         maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
#         loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
#         return loss

class CriterionPairWiseforWholeFeatAfterPool(tf.keras.layers.Layer):
    def __init__(self, scale):
        super().__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def call(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        total_w = tf.shape(feat_T)[2]
        total_h = tf.shape(feat_T)[1]
        patch_w = tf.cast(total_w * self.scale, tf.int32)
        patch_h = tf.cast(total_h * self.scale, tf.int32)
        feat_S_pool = tf.nn.max_pool(feat_S, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        feat_T_pool = tf.nn.max_pool(feat_T, ksize=[1, patch_h, patch_w, 1], strides=[1, patch_h, patch_w, 1], padding='VALID')
        loss = self.criterion(feat_S_pool, feat_T_pool)
        return loss

# class ContrastLoss(nn.Module):
#     def __init__(self, args, ignore_lb=255):
#         super(ContrastLoss, self).__init__()
#         self.ignore_lb = ignore_lb
#         self.args = args
#         self.max_anchor = args.max_anchor
#         self.temperature = args.temperature
#     def _anchor_sampling(self,embs,labels):
#         device = embs.device
#         b_,c_,h_,w_ = embs.size()
#         class_u = torch.unique(labels)
#         class_u_num = len(class_u)
#         if 255 in class_u:
#             class_u_num =class_u_num - 1
#         if class_u_num==0:
#             return None,None
#         num_p_c = self.max_anchor//class_u_num
#         embs = embs.permute(0,2,3,1).reshape(-1,c_)
#         labels = labels.view(-1)
#         index_ = torch.arange(len(labels))
#         index_ = index_.to(device)
#         sampled_list = []
#         sampled_label_list = []
#         for cls_ in class_u:
#             if cls_ != 255:
#                 mask_ = labels==cls_
#                 selected_index_ = torch.masked_select(index_,mask_)
#                 if len(selected_index_)>num_p_c:
#                     sel_i_i = torch.arange(len(selected_index_))
#                     sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
#                     sel_i = sel_i_i[sel_i_i_i]
#                     selected_index_ = selected_index_[sel_i]
#                 embs_tmp = embs[selected_index_]
#                 sampled_list.append(embs_tmp)
#                 sampled_label_list.append(torch.ones(len(selected_index_)).to(device)*cls_)
#         sampled_list = torch.cat(sampled_list,0)
#         sampled_label_list = torch.cat(sampled_label_list,0)
#         return sampled_list,sampled_label_list
#     def forward(self,embs,labels,proto_mem,proto_mask):
#         device = proto_mem.device
#         anchors,anchor_labels = self._anchor_sampling(embs,labels)
#         if anchors is None:
#             loss =torch.tensor(0).to(device)
#             return loss
#         if self.args.kmean_num>0:
#             C_,km_,c_ = proto_mem.size()
#             proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,km_)
#             proto_mem_ = proto_mem.reshape(-1,c_)
#             proto_labels = proto_labels.view(-1)
#             proto_mask = proto_mask.view(-1)
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_labels =proto_labels.to(device)
#             proto_mem_ = proto_mem_[sel_idx]
#             proto_labels = proto_labels[sel_idx]
#             proto_labels =proto_labels.to(device)
#         else:
#             C_,c_ = proto_mem.size()
#             proto_labels = torch.arange(C_)
#             proto_mem_ = proto_mem
#             proto_labels = proto_labels
#             proto_labels = proto_labels[sel_idx]
#             proto_labels =proto_labels.to(device)
#             proto_mask = proto_mask
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#             proto_labels = proto_labels[sel_idx]
#             proto_labels =proto_labels.to(device)
#         anchor_dot_contrast = torch.div(torch.matmul(anchors,proto_mem_.T),self.temperature)
#         mask = anchor_labels.unsqueeze(1)==proto_labels.unsqueeze(0)
#         mask = mask.float()
#         mask = mask.to(device)
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#         neg_mask = 1 - mask
#         neg_logits = torch.exp(logits) * neg_mask
#         neg_logits = neg_logits.sum(1, keepdim=True)
#         exp_logits = torch.exp(logits) * mask
#         log_prob = logits - torch.log(exp_logits + neg_logits)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#         loss = - mean_log_prob_pos
#         loss = loss.mean()
#         if torch.isnan(loss):
#             print('!'*10)
#             print(torch.unique(logits))
#             print(torch.unique(exp_logits))
#             print(torch.unique(neg_logits))
#             print(torch.unique(log_prob))
#             print(torch.unique(mask.sum(1)))
#             print(mask)
#             print(torch.unique(anchor_labels))
#             print(proto_labels)
#             print(torch.unique(proto_labels))
#             exit()
#         return loss



class ContrastLoss(tf.keras.layers.Layer):
    def __init__(self, args, ignore_lb=255):
        super().__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self, embs, labels):
        embs = tf.convert_to_tensor(embs, dtype=tf.float32)
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        c_ = tf.shape(embs)[1]
        embs_ = tf.reshape(tf.transpose(embs, [0, 2, 3, 1]), [-1, c_])
        labels_ = tf.reshape(labels, [-1])
        class_u, _ = tf.unique(labels_)
        class_u = tf.boolean_mask(class_u, tf.not_equal(class_u, 255))
        class_u_num = tf.shape(class_u)[0]
        num_p_c = tf.maximum(1, self.max_anchor // tf.maximum(1, class_u_num))
        max_classes = self.args.num_classes
        sampled_embs = tf.TensorArray(
            tf.float32,
            size=max_classes,
            dynamic_size=False,
            clear_after_read=False,
            element_shape=tf.TensorShape([None, None]),
        )
        sampled_lbls = tf.TensorArray(
            tf.float32,
            size=max_classes,
            dynamic_size=False,
            clear_after_read=False,
            element_shape=tf.TensorShape([None]),
        )
        sampled_counts = tf.TensorArray(
            tf.int32,
            size=max_classes,
            dynamic_size=False,
            clear_after_read=False,
            element_shape=tf.TensorShape([]),
        )

        def cond(i, *_args):
            return i < class_u_num

        def body(i, ta_embs, ta_lbls, ta_counts):
            cls_ = class_u[i]
            selected_index_ = tf.reshape(tf.where(tf.equal(labels_, cls_)), [-1])
            selected_count = tf.shape(selected_index_)[0]
            selected_index_ = tf.cond(
                selected_count > num_p_c,
                lambda: tf.random.shuffle(selected_index_)[:num_p_c],
                lambda: selected_index_,
            )
            current_count = tf.shape(selected_index_)[0]
            padded_index = tf.pad(selected_index_, [[0, num_p_c - current_count]])
            sampled_emb = tf.gather(embs_, padded_index)
            sampled_lbl = tf.fill([num_p_c], tf.cast(cls_, tf.float32))
            return (
                i + 1,
                ta_embs.write(i, sampled_emb),
                ta_lbls.write(i, sampled_lbl),
                ta_counts.write(i, current_count),
            )

        def zero_return():
            return tf.zeros([0, c_], dtype=tf.float32), tf.zeros([0], dtype=tf.float32), class_u_num

        def sampled_return():
            _, ta_embs, ta_lbls, ta_counts = tf.while_loop(
                cond,
                body,
                loop_vars=(0, sampled_embs, sampled_lbls, sampled_counts),
                parallel_iterations=1,
            )
            embs_stack = ta_embs.gather(tf.range(class_u_num))
            lbls_stack = ta_lbls.gather(tf.range(class_u_num))
            counts_stack = ta_counts.gather(tf.range(class_u_num))
            valid_mask = tf.sequence_mask(counts_stack, maxlen=num_p_c)
            gathered_embs = tf.boolean_mask(embs_stack, valid_mask)
            gathered_lbls = tf.boolean_mask(lbls_stack, valid_mask)
            return gathered_embs, gathered_lbls, class_u_num

        return tf.cond(class_u_num > 0, sampled_return, zero_return)

    def preprocess_prototypes(self, proto_mem, proto_mask):
        proto_mem = tf.cast(tf.convert_to_tensor(proto_mem), tf.float32)
        proto_mask = tf.convert_to_tensor(proto_mask)

        if self.args.kmean_num > 0:
            c_ = tf.shape(proto_mem)[0]
            km_ = tf.shape(proto_mem)[1]
            feat_dim = tf.shape(proto_mem)[2]
            proto_labels = tf.reshape(tf.repeat(tf.range(c_), repeats=km_), [-1])
            proto_mem_flat = tf.reshape(proto_mem, [-1, feat_dim])
            proto_mask_flat = tf.reshape(tf.cast(proto_mask, tf.bool), [-1])
            proto_mem_flat = tf.boolean_mask(proto_mem_flat, proto_mask_flat)
            proto_labels = tf.boolean_mask(proto_labels, proto_mask_flat)
        else:
            c_ = tf.shape(proto_mem)[0]
            proto_labels = tf.range(c_)
            proto_mask_flat = tf.reshape(tf.cast(proto_mask, tf.bool), [-1])
            proto_mem_flat = tf.boolean_mask(proto_mem, proto_mask_flat)
            proto_labels = tf.boolean_mask(proto_labels, proto_mask_flat)

        return proto_mem_flat, tf.cast(proto_labels, tf.int32)

    def call(self, embs, labels, proto_mem, proto_mask, preprocessed_proto=None):
        embs = tf.cast(tf.convert_to_tensor(embs), tf.float32)
        labels = tf.cast(tf.convert_to_tensor(labels), tf.int32)

        anchors, anchor_labels, class_u_num = self._anchor_sampling(embs, labels)

        def zero_loss():
            return tf.constant(0.0, dtype=tf.float32)

        def compute_loss():
            local_anchor_labels = tf.cast(anchor_labels, tf.int32)
            if preprocessed_proto is not None:
                proto_mem_, proto_labels = preprocessed_proto
            else:
                proto_mem_, proto_labels = self.preprocess_prototypes(proto_mem, proto_mask)

            anchor_dot_contrast = tf.matmul(anchors, proto_mem_, transpose_b=True) / self.temperature
            mask = tf.cast(tf.equal(tf.expand_dims(local_anchor_labels, 1), tf.expand_dims(proto_labels, 0)), tf.float32)

            logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
            logits = anchor_dot_contrast - tf.stop_gradient(logits_max)
            neg_mask = 1.0 - mask

            exp_logits = tf.exp(logits) * mask
            neg_logits = tf.exp(logits) * neg_mask
            neg_logits = tf.reduce_sum(neg_logits, axis=1, keepdims=True)
            log_prob = logits - tf.math.log(exp_logits + neg_logits + 1e-12)
            mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-12)
            return tf.reduce_mean(-mean_log_prob_pos)

        return tf.cond(class_u_num > 0, compute_loss, zero_loss)



# class ContrastLossLocal(nn.Module):
#     def __init__(self, args, ignore_lb=255):
#         super(ContrastLossLocal, self).__init__()
#         self.ignore_lb = ignore_lb
#         self.args = args
#         self.max_anchor = args.max_anchor
#         self.temperature = args.temperature
#     def _anchor_sampling(self,embs,labels):
#         device = embs.device
#         b_,c_,h_,w_ = embs.size()
#         class_u = torch.unique(labels)
#         class_u_num = len(class_u)
#         if 255 in class_u:
#             class_u_num =class_u_num - 1
#         if class_u_num==0:
#             return None,None
#         num_p_c = self.max_anchor//class_u_num
#         embs = embs.permute(0,2,3,1).reshape(-1,c_)
#         labels = labels.view(-1)
#         index_ = torch.arange(len(labels))
#         index_ = index_.to(device)
#         sampled_list = []
#         sampled_label_list = []
#         for cls_ in class_u:
#             if cls_ != 255:
#                 mask_ = labels==cls_
#                 selected_index_ = torch.masked_select(index_,mask_)
#                 if len(selected_index_)>num_p_c:
#                     sel_i_i = torch.arange(len(selected_index_))
#                     sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
#                     sel_i = sel_i_i[sel_i_i_i]
#                     selected_index_ = selected_index_[sel_i]
#                 embs_tmp = embs[selected_index_]
#                 sampled_list.append(embs_tmp)
#                 sampled_label_list.append(torch.ones(len(selected_index_)).to(device)*cls_)
#         sampled_list = torch.cat(sampled_list,0)
#         sampled_label_list = torch.cat(sampled_label_list,0)
#         return sampled_list,sampled_label_list
#     def forward(self,embs,labels,proto_mem,proto_mask,local_mem):
#         device = proto_mem.device
#         anchors,anchor_labels = self._anchor_sampling(embs,labels)
#         if anchors is None:
#             loss =torch.tensor(0).to(device)
#             return loss
#         if self.args.kmean_num>0:
#             C_,U_,km_,c_ = proto_mem.size()
#             proto_labels = torch.arange(C_).unsqueeze(1).unsqueeze(1).repeat(1,U_,km_)
#             proto_mem_ = proto_mem.reshape(-1,c_)
#             proto_labels = proto_labels.view(-1)
#             proto_mask = proto_mask.unsqueeze(-1).repeat(1,1,km_).view(-1)
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#             proto_labels = proto_labels[sel_idx]
#             proto_labels =proto_labels.to(device)
#         else:
#             C_,U_,c_ = proto_mem.size()
#             proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,U_)
#             proto_mem_ = proto_mem.reshape(-1,c_)
#             proto_labels = proto_labels.view(-1)
#             proto_mask = proto_mask.view(-1)
#             proto_idx = torch.arange(len(proto_mask))
#             proto_idx = proto_idx.to(device)
#             sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
#             proto_mem_ = proto_mem_[sel_idx]
#             proto_labels = proto_labels[sel_idx]
#             proto_labels =proto_labels.to(device)
#         anchor_dot_contrast = torch.div(torch.matmul(anchors,proto_mem_.T),self.temperature)
#         mask = anchor_labels.unsqueeze(1)==proto_labels.unsqueeze(0)
#         mask = mask.float()
#         mask = mask.to(device)
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()
#         exp_logits = torch.exp(logits) * mask
#         C_,N_,c_= local_mem.size()
#         local_labels = torch.arange(C_).unsqueeze(1).repeat(1,N_)
#         local_mem = local_mem.reshape(-1,c_)
#         local_labels = local_labels.view(-1)
#         local_labels = local_labels.to(device)
#         anchor_dot_contrast_l = torch.div(torch.matmul(anchors,local_mem.T),self.temperature)
#         mask_l = anchor_labels.unsqueeze(1)==local_labels.unsqueeze(0)
#         mask_l = mask_l.float().to(device)
#         logits_l = anchor_dot_contrast_l - logits_max.detach()
#         neg_logits = torch.exp(logits_l) * mask_l
#         neg_logits = neg_logits.sum(1, keepdim=True)
#         log_prob = logits - torch.log(exp_logits + neg_logits)
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
#         loss = - mean_log_prob_pos
#         loss = loss.mean()
#         if torch.isnan(loss):
#             print('!'*10)
#             print(torch.unique(logits))
#             print(torch.unique(exp_logits))
#             print(torch.unique(neg_logits))
#             print(torch.unique(log_prob))
#             exit()
#         return loss

class ContrastLossLocal(tf.keras.layers.Layer):
    def __init__(self, args, ignore_lb=255):
        super().__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self, embs, labels):
        b_ = tf.shape(embs)[0]
        h_ = tf.shape(embs)[1]
        w_ = tf.shape(embs)[2]
        c_ = tf.shape(embs)[3]
        embs_flat = tf.reshape(embs, [-1, c_])
        labels_flat = tf.reshape(labels, [-1])
        class_u, _ = tf.unique(labels_flat)
        class_u_num = tf.size(class_u)
        if tf.reduce_any(tf.equal(class_u, 255)):
            class_u_num = class_u_num - 1
        if class_u_num == 0:
            return None, None
        num_p_c = self.max_anchor // class_u_num
        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u.numpy():
            if cls_ != 255:
                mask_ = tf.equal(labels_flat, cls_)
                selected_index_ = tf.where(mask_)
                selected_index_ = tf.reshape(selected_index_, [-1])
                if tf.size(selected_index_) > num_p_c:
                    sel_i_i = tf.range(tf.size(selected_index_))
                    sel_i_i_i = tf.random.shuffle(sel_i_i)[:num_p_c]
                    sel_i = tf.gather(selected_index_, sel_i_i_i)
                    selected_index_ = sel_i
                embs_tmp = tf.gather(embs_flat, selected_index_)
                sampled_list.append(embs_tmp)
                sampled_label_list.append(tf.ones(tf.size(selected_index_), dtype=tf.float32) * cls_)
        sampled_list = tf.concat(sampled_list, axis=0)
        sampled_label_list = tf.concat(sampled_label_list, axis=0)
        return sampled_list, sampled_label_list

    def call(self, embs, labels, proto_mem, proto_mask, local_mem):
        anchors, anchor_labels = self._anchor_sampling(embs, labels)
        if anchors is None:
            return tf.constant(0.0)
        if self.args.kmean_num > 0:
            C_ = tf.shape(proto_mem)[0]
            U_ = tf.shape(proto_mem)[1]
            km_ = tf.shape(proto_mem)[2]
            c_ = tf.shape(proto_mem)[3]
            proto_labels = tf.reshape(tf.repeat(tf.range(C_), U_ * km_), [-1])
            proto_mem_ = tf.reshape(proto_mem, [-1, c_])
            proto_mask_flat = tf.reshape(proto_mask, [-1])
            proto_idx = tf.range(tf.shape(proto_mask_flat)[0])
            sel_idx = tf.boolean_mask(proto_idx, tf.cast(proto_mask_flat, tf.bool))
            proto_mem_ = tf.gather(proto_mem_, sel_idx)
            proto_labels = tf.gather(proto_labels, sel_idx)
        else:
            C_ = tf.shape(proto_mem)[0]
            U_ = tf.shape(proto_mem)[1]
            c_ = tf.shape(proto_mem)[2]
            proto_labels = tf.reshape(tf.repeat(tf.range(C_), U_), [-1])
            proto_mem_ = tf.reshape(proto_mem, [-1, c_])
            proto_mask_flat = tf.reshape(proto_mask, [-1])
            proto_idx = tf.range(tf.shape(proto_mask_flat)[0])
            sel_idx = tf.boolean_mask(proto_idx, tf.cast(proto_mask_flat, tf.bool))
            proto_mem_ = tf.gather(proto_mem_, sel_idx)
            proto_labels = tf.gather(proto_labels, sel_idx)
        anchor_dot_contrast = tf.matmul(anchors, proto_mem_, transpose_b=True) / self.temperature
        anchor_labels = tf.expand_dims(anchor_labels, 1)
        proto_labels = tf.expand_dims(proto_labels, 0)
        mask = tf.equal(anchor_labels, proto_labels)
        mask = tf.cast(mask, tf.float32)
        logits_max = tf.reduce_max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max
        exp_logits = tf.exp(logits) * mask
        C_ = tf.shape(local_mem)[0]
        N_ = tf.shape(local_mem)[1]
        c_ = tf.shape(local_mem)[2]
        local_labels = tf.reshape(tf.repeat(tf.range(C_), N_), [-1])
        local_mem_flat = tf.reshape(local_mem, [-1, c_])
        anchor_dot_contrast_l = tf.matmul(anchors, local_mem_flat, transpose_b=True) / self.temperature
        mask_l = tf.equal(anchor_labels, tf.expand_dims(local_labels, 0))
        mask_l = tf.cast(mask_l, tf.float32)
        logits_l = anchor_dot_contrast_l - logits_max
        neg_logits = tf.exp(logits_l) * mask_l
        neg_logits = tf.reduce_sum(neg_logits, axis=1, keepdims=True)
        log_prob = logits - tf.math.log(exp_logits + neg_logits + 1e-8)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-8)
        loss = -mean_log_prob_pos
        loss = tf.reduce_mean(loss)
        return loss

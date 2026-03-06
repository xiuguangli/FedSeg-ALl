import sys

sys.path.append("/home/pjl/project/FedSeg/paddle_project")
import numpy as np
import paddle
from myseg.bisenetv2 import BiSeNetV2
from paddle_utils import *


def set_model_bisenetv2(args, num_classes):
    net = BiSeNetV2(args, num_classes)
    return net


def set_optimizer(model, args):
    if hasattr(model, "get_params"):
        (
            wd_params,
            nowd_params,
            lr_mul_wd_params,
            lr_mul_nowd_params,
        ) = model.get_params()
        wd_val = 0
        params_list = [
            {"params": wd_params},
            {"params": nowd_params, "weight_decay": wd_val},
            {"params": lr_mul_wd_params, "lr": args.lr * 10},
            {"params": lr_mul_nowd_params, "weight_decay": wd_val, "lr": args.lr * 10},
        ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {"params": wd_params},
            {"params": non_wd_params, "weight_decay": 0},
        ]
    optim = paddle.optimizer.SGD(
        parameters=params_list,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    return optim


class BackCELoss(paddle.nn.Layer):
    def __init__(self, args, ignore_lb=255):
        super(BackCELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes
        self.criteria = paddle.nn.NLLLoss(ignore_index=ignore_lb, reduction="mean")

    def forward(self, logits, labels):
        total_labels = paddle.unique(x=labels)
        new_labels = labels.clone()
        probs = paddle.nn.functional.softmax(x=logits, axis=1)
        fore_ = []
        back_ = []
        for l in range(self.class_num):
            if l in total_labels:
                fore_.append(probs[:, l, :, :].unsqueeze(axis=1))
            else:
                back_.append(probs[:, l, :, :].unsqueeze(axis=1))
        Flag = False
        if not len(fore_) == self.class_num:
            fore_.append(sum(back_))
            Flag = True
        for i, l in enumerate(total_labels):
            if Flag:
                new_labels[labels == l] = i
            elif l != 255:
                new_labels[labels == l] = i
        probs = paddle.concat(x=fore_, axis=1)
        logprobs = paddle.log(x=probs + 1e-07)
        return self.criteria(logprobs, new_labels.astype(dtype="int64"))


class OhemCELoss(paddle.nn.Layer):
    """
    Feddrive: We apply OHEM (Online Hard-Negative Mining) [56], selecting 25%
    of the pixels having the highest loss for the optimization.
    """

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        self.thresh = -paddle.log(
            x=paddle.to_tensor(data=thresh, dtype="float32", stop_gradient=not False)
        ).cuda()
        self.ignore_lb = ignore_lb
        self.criteria = paddle.nn.CrossEntropyLoss(
            ignore_index=ignore_lb, reduction="none"
        )

    def forward(self, logits, labels):
        n_min = int(labels[labels != self.ignore_lb].size * 0.25)
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.size < n_min:
            loss_hard, _ = loss.topk(k=n_min)
        return paddle.mean(x=loss_hard)


class CriterionPixelPair(paddle.nn.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super(CriterionPixelPair, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = tuple(fea_0.shape)
        fea_0 = fea_0.reshape(C, -1).transpose(
            perm=dim2perm(fea_0.reshape(C, -1).ndim, 0, 1)
        )
        fea_1 = fea_1.reshape(C, -1).transpose(
            perm=dim2perm(fea_1.reshape(C, -1).ndim, 0, 1)
        )
        sim_map_0_1 = paddle.mm(
            input=fea_0, mat2=fea_1.transpose(perm=dim2perm(fea_1.ndim, 0, 1))
        )
        return sim_map_0_1

    def forward(self, feat_S, feat_T):
        B, C, H, W = tuple(feat_S.shape)
        device = feat_S.place
        patch_w = 2
        patch_h = 2
        maxpool = paddle.nn.AvgPool2D(
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
            padding=0,
            ceil_mode=True,
            exclusive=False,
        )
        feat_S = maxpool(feat_S)
        feat_T = maxpool(feat_T)
        feat_S = paddle.nn.functional.normalize(x=feat_S, p=2, axis=1)
        feat_T = paddle.nn.functional.normalize(x=feat_T, p=2, axis=1)
        sim_dis = paddle.to_tensor(data=0.0).to(device)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[i])
            t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[i])
            p_s = paddle.nn.functional.log_softmax(
                x=s_sim_map / self.temperature, axis=1
            )
            p_t = paddle.nn.functional.softmax(x=t_sim_map / self.temperature, axis=1)
            sim_dis_ = paddle.nn.functional.kl_div(
                input=p_s, label=p_t, reduction="batchmean"
            )
            sim_dis += sim_dis_
        sim_dis = sim_dis / B
        return sim_dis


class CriterionPixelPairSeq(paddle.nn.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super(CriterionPixelPairSeq, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = tuple(fea_0.shape)
        fea_0 = fea_0.reshape(C, -1).transpose(
            perm=dim2perm(fea_0.reshape(C, -1).ndim, 0, 1)
        )
        fea_1 = fea_1.reshape(C, -1).transpose(
            perm=dim2perm(fea_1.reshape(C, -1).ndim, 0, 1)
        )
        sim_map_0_1 = paddle.mm(
            input=fea_0, mat2=fea_1.transpose(perm=dim2perm(fea_1.ndim, 0, 1))
        )
        return sim_map_0_1

    def forward(self, feat_S, feat_T, pixel_seq):
        B, C, H, W = tuple(feat_S.shape)
        device = feat_S.place
        patch_w = 2
        patch_h = 2
        maxpool = paddle.nn.AvgPool2D(
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
            padding=0,
            ceil_mode=True,
            exclusive=False,
        )
        feat_S = maxpool(feat_S)
        feat_T = maxpool(feat_T)
        feat_S = paddle.nn.functional.normalize(x=feat_S, p=2, axis=1)
        feat_T = paddle.nn.functional.normalize(x=feat_T, p=2, axis=1)
        feat_S = feat_S.transpose(perm=[0, 2, 3, 1]).reshape(-1, C)
        feat_T = feat_T.transpose(perm=[0, 2, 3, 1]).reshape(-1, C)
        split_T = feat_T
        idx = np.random.choice(len(split_T), 4000, replace=False)
        split_T = split_T[idx]
        split_T = paddle_split(x=split_T, num_or_sections=1, axis=0)
        pixel_seq.extend(split_T)
        if len(pixel_seq) > 20000:
            del pixel_seq[: len(pixel_seq) - 20000]
        proto_mem_ = paddle.concat(x=pixel_seq, axis=0)
        s_sim_map = paddle.matmul(x=feat_S, y=proto_mem_.T)
        t_sim_map = paddle.matmul(x=feat_T, y=proto_mem_.T)
        p_s = paddle.nn.functional.log_softmax(x=s_sim_map / self.temperature, axis=1)
        p_t = paddle.nn.functional.softmax(x=t_sim_map / self.temperature, axis=1)
        sim_dis = paddle.nn.functional.kl_div(
            input=p_s, label=p_t, reduction="batchmean"
        )
        return sim_dis, pixel_seq


class CriterionPixelPairG(paddle.nn.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super(CriterionPixelPairG, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = tuple(fea_0.shape)
        fea_0 = fea_0.reshape(C, -1).transpose(
            perm=dim2perm(fea_0.reshape(C, -1).ndim, 0, 1)
        )
        fea_1 = fea_1.reshape(C, -1).transpose(
            perm=dim2perm(fea_1.reshape(C, -1).ndim, 0, 1)
        )
        sim_map_0_1 = paddle.mm(
            input=fea_0, mat2=fea_1.transpose(perm=dim2perm(fea_1.ndim, 0, 1))
        )
        return sim_map_0_1

    def forward(self, feat_S, feat_T, proto_mem, proto_mask):
        B, C, H, W = tuple(feat_S.shape)
        device = feat_S.place
        patch_w = 2
        patch_h = 2
        maxpool = paddle.nn.AvgPool2D(
            kernel_size=(patch_h, patch_w),
            stride=(patch_h, patch_w),
            padding=0,
            ceil_mode=True,
            exclusive=False,
        )
        feat_S = maxpool(feat_S)
        feat_T = maxpool(feat_T)
        feat_S = paddle.nn.functional.normalize(x=feat_S, p=2, axis=1)
        feat_T = paddle.nn.functional.normalize(x=feat_T, p=2, axis=1)
        feat_S = feat_S.transpose(perm=[0, 2, 3, 1]).reshape(-1, C)
        feat_T = feat_T.transpose(perm=[0, 2, 3, 1]).reshape(-1, C)
        if self.args.kmean_num > 0:
            C_, km_, c_ = tuple(proto_mem.shape)
            proto_labels = (
                paddle.arange(end=C_).unsqueeze(axis=1).tile(repeat_times=[1, km_])
            )
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_mask = proto_mask.view(-1)
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
        else:
            C_, c_ = tuple(proto_mem.shape)
            proto_labels = paddle.arange(end=C_)
            proto_mem_ = proto_mem
            proto_mask = proto_mask
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
        s_sim_map = paddle.matmul(x=feat_S, y=proto_mem_.T)
        t_sim_map = paddle.matmul(x=feat_T, y=proto_mem_.T)
        p_s = paddle.nn.functional.log_softmax(x=s_sim_map / self.temperature, axis=1)
        p_t = paddle.nn.functional.softmax(x=t_sim_map / self.temperature, axis=1)
        sim_dis = paddle.nn.functional.kl_div(
            input=p_s, label=p_t, reduction="batchmean"
        )
        return sim_dis


class CriterionPixelRegionPair(paddle.nn.Layer):
    def __init__(self, args, temperature=0.1, ignore_index=255):
        super(CriterionPixelRegionPair, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        C, H, W = tuple(fea_0.shape)
        fea_0 = fea_0.reshape(C, -1).transpose(
            perm=dim2perm(fea_0.reshape(C, -1).ndim, 0, 1)
        )
        fea_1 = fea_1.transpose(perm=dim2perm(fea_1.ndim, 0, 1))
        sim_map_0_1 = paddle.mm(input=fea_0, mat2=fea_1)
        return sim_map_0_1

    def forward(self, feat_S, feat_T, proto_mem, proto_mask):
        B, C, H, W = tuple(feat_S.shape)
        device = feat_S.place
        if self.args.kmean_num > 0:
            C_, U_, km_, c_ = tuple(proto_mem.shape)
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_mask = (
                proto_mask.unsqueeze(axis=-1).tile(repeat_times=[1, 1, km_]).view(-1)
            )
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
        else:
            C_, U_, c_ = tuple(proto_mem.shape)
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_mask = proto_mask.view(-1)
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
        sim_dis = paddle.to_tensor(data=0.0).to(device)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], proto_mem_)
            t_sim_map = self.pair_wise_sim_map(feat_T[i], proto_mem_)
            p_s = paddle.nn.functional.log_softmax(
                x=s_sim_map / self.temperature, axis=1
            )
            p_t = paddle.nn.functional.softmax(x=t_sim_map / self.temperature, axis=1)
            sim_dis_ = paddle.nn.functional.kl_div(
                input=p_s, label=p_t, reduction="batchmean"
            )
            sim_dis += sim_dis_
        sim_dis = sim_dis / B
        return sim_dis


def L2(f_):
    return ((f_**2).sum(axis=1) ** 0.5).reshape(
        tuple(f_.shape)[0], 1, tuple(f_.shape)[2], tuple(f_.shape)[3]
    ) + 1e-08


def similarity(feat):
    feat = feat.astype(dtype="float32")
    tmp = L2(feat).detach()
    feat = feat / tmp
    feat = feat.reshape(tuple(feat.shape)[0], tuple(feat.shape)[1], -1)
    return paddle.einsum("icm,icn->imn", [feat, feat])


def sim_dis_compute(f_S, f_T):
    sim_err = (
        (similarity(f_T) - similarity(f_S)) ** 2
        / (tuple(f_T.shape)[-1] * tuple(f_T.shape)[-2]) ** 2
        / tuple(f_T.shape)[0]
    )
    sim_dis = sim_err.sum()
    return sim_dis


class CriterionPairWiseforWholeFeatAfterPool(paddle.nn.Layer):
    def __init__(self, scale):
        """inter pair-wise loss from inter feature maps"""
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    def forward(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        feat_T.detach()
        total_w, total_h = tuple(feat_T.shape)[2], tuple(feat_T.shape)[3]
        patch_w, patch_h = int(total_w * self.scale), int(total_h * self.scale)
        maxpool = paddle.nn.MaxPool2D(
            kernel_size=(patch_w, patch_h),
            stride=(patch_w, patch_h),
            padding=0,
            ceil_mode=True,
        )
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss


class ContrastLoss(paddle.nn.Layer):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self, embs, labels):
        device = embs.place
        b_, c_, h_, w_ = tuple(embs.shape)
        class_u = paddle.unique(x=labels)
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num = class_u_num - 1
        if class_u_num == 0:
            return None, None
        num_p_c = self.max_anchor // class_u_num
        embs = embs.transpose(perm=[0, 2, 3, 1]).reshape(-1, c_)
        labels = labels.view(-1)
        index_ = paddle.arange(end=len(labels))
        index_ = index_.to(device)
        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
            if cls_ != 255:
                mask_ = labels == cls_
                selected_index_ = paddle.masked_select(x=index_, mask=mask_)
                if len(selected_index_) > num_p_c:
                    sel_i_i = paddle.arange(end=len(selected_index_))
                    sel_i_i_i = paddle.randperm(n=len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]
                    selected_index_ = selected_index_[sel_i]
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                sampled_label_list.append(
                    paddle.ones(shape=len(selected_index_)).to(device) * cls_
                )
        sampled_list = paddle.concat(x=sampled_list, axis=0)
        sampled_label_list = paddle.concat(x=sampled_label_list, axis=0)
        return sampled_list, sampled_label_list

    def forward(self, embs, labels, proto_mem, proto_mask):
        device = proto_mem.place
        anchors, anchor_labels = self._anchor_sampling(embs, labels)
        if anchors is None:
            loss = paddle.to_tensor(data=0).to(device)
            return loss
        if self.args.kmean_num > 0:
            C_, km_, c_ = tuple(proto_mem.shape)
            proto_labels = (
                paddle.arange(end=C_).unsqueeze(axis=1).tile(repeat_times=[1, km_])
            )
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_labels = proto_labels.to(device)
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
        else:
            C_, c_ = tuple(proto_mem.shape)
            proto_labels = paddle.arange(end=C_)
            proto_mem_ = proto_mem
            proto_labels = proto_labels
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
            proto_mask = proto_mask
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
        anchor_dot_contrast = paddle.divide(
            x=paddle.matmul(x=anchors, y=proto_mem_.T),
            y=paddle.to_tensor(self.temperature),
        )
        mask = anchor_labels.unsqueeze(axis=1) == proto_labels.unsqueeze(axis=0)
        mask = mask.astype(dtype="float32")
        mask = mask.to(device)
        logits_max, _ = paddle.max(
            keepdim=True, x=anchor_dot_contrast, axis=1
        ), paddle.argmax(keepdim=True, x=anchor_dot_contrast, axis=1)
        logits = anchor_dot_contrast - logits_max.detach()
        neg_mask = 1 - mask
        neg_logits = paddle.exp(x=logits) * neg_mask
        neg_logits = neg_logits.sum(axis=1, keepdim=True)
        exp_logits = paddle.exp(x=logits) * mask
        log_prob = logits - paddle.log(x=exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / mask.sum(axis=1)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        if paddle.isnan(x=loss):
            print("!" * 10)
            print(paddle.unique(x=logits))
            print(paddle.unique(x=exp_logits))
            print(paddle.unique(x=neg_logits))
            print(paddle.unique(x=log_prob))
            print(paddle.unique(x=mask.sum(axis=1)))
            print(mask)
            print(paddle.unique(x=anchor_labels))
            print(proto_labels)
            print(paddle.unique(x=proto_labels))
            exit()
        return loss


class ContrastLossLocal(paddle.nn.Layer):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLossLocal, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self, embs, labels):
        device = embs.place
        b_, c_, h_, w_ = tuple(embs.shape)
        class_u = paddle.unique(x=labels)
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num = class_u_num - 1
        if class_u_num == 0:
            return None, None
        num_p_c = self.max_anchor // class_u_num
        embs = embs.transpose(perm=[0, 2, 3, 1]).reshape(-1, c_)
        labels = labels.view(-1)
        index_ = paddle.arange(end=len(labels))
        index_ = index_.to(device)
        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
            if cls_ != 255:
                mask_ = labels == cls_
                selected_index_ = paddle.masked_select(x=index_, mask=mask_)
                if len(selected_index_) > num_p_c:
                    sel_i_i = paddle.arange(end=len(selected_index_))
                    sel_i_i_i = paddle.randperm(n=len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]
                    selected_index_ = selected_index_[sel_i]
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                sampled_label_list.append(
                    paddle.ones(shape=len(selected_index_)).to(device) * cls_
                )
        sampled_list = paddle.concat(x=sampled_list, axis=0)
        sampled_label_list = paddle.concat(x=sampled_label_list, axis=0)
        return sampled_list, sampled_label_list

    def forward(self, embs, labels, proto_mem, proto_mask, local_mem):
        device = proto_mem.place
        anchors, anchor_labels = self._anchor_sampling(embs, labels)
        if anchors is None:
            loss = paddle.to_tensor(data=0).to(device)
            return loss
        if self.args.kmean_num > 0:
            C_, U_, km_, c_ = tuple(proto_mem.shape)
            proto_labels = (
                paddle.arange(end=C_)
                .unsqueeze(axis=1)
                .unsqueeze(axis=1)
                .tile(repeat_times=[1, U_, km_])
            )
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = (
                proto_mask.unsqueeze(axis=-1).tile(repeat_times=[1, 1, km_]).view(-1)
            )
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
        else:
            C_, U_, c_ = tuple(proto_mem.shape)
            proto_labels = (
                paddle.arange(end=C_).unsqueeze(axis=1).tile(repeat_times=[1, U_])
            )
            proto_mem_ = proto_mem.reshape(-1, c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            proto_idx = paddle.arange(end=len(proto_mask))
            proto_idx = proto_idx.to(device)
            sel_idx = paddle.masked_select(
                x=proto_idx, mask=proto_mask.astype(dtype="bool")
            )
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            proto_labels = proto_labels.to(device)
        anchor_dot_contrast = paddle.divide(
            x=paddle.matmul(x=anchors, y=proto_mem_.T),
            y=paddle.to_tensor(self.temperature),
        )
        mask = anchor_labels.unsqueeze(axis=1) == proto_labels.unsqueeze(axis=0)
        mask = mask.astype(dtype="float32")
        mask = mask.to(device)
        logits_max, _ = paddle.max(
            keepdim=True, x=anchor_dot_contrast, axis=1
        ), paddle.argmax(keepdim=True, x=anchor_dot_contrast, axis=1)
        logits = anchor_dot_contrast - logits_max.detach()
        exp_logits = paddle.exp(x=logits) * mask
        C_, N_, c_ = tuple(local_mem.shape)
        local_labels = (
            paddle.arange(end=C_).unsqueeze(axis=1).tile(repeat_times=[1, N_])
        )
        local_mem = local_mem.reshape(-1, c_)
        local_labels = local_labels.view(-1)
        local_labels = local_labels.to(device)
        anchor_dot_contrast_l = paddle.divide(
            x=paddle.matmul(x=anchors, y=local_mem.T),
            y=paddle.to_tensor(self.temperature),
        )
        mask_l = anchor_labels.unsqueeze(axis=1) == local_labels.unsqueeze(axis=0)
        mask_l = mask_l.astype(dtype="float32").to(device)
        logits_l = anchor_dot_contrast_l - logits_max.detach()
        neg_logits = paddle.exp(x=logits_l) * mask_l
        neg_logits = neg_logits.sum(axis=1, keepdim=True)
        log_prob = logits - paddle.log(x=exp_logits + neg_logits)
        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / mask.sum(axis=1)
        loss = -mean_log_prob_pos
        loss = loss.mean()
        if paddle.isnan(x=loss):
            print("!" * 10)
            print(paddle.unique(x=logits))
            print(paddle.unique(x=exp_logits))
            print(paddle.unique(x=neg_logits))
            print(paddle.unique(x=log_prob))
            exit()
        return loss

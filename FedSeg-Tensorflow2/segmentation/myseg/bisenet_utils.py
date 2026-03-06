# import torch
# from torch import nn
import numpy as np
# import torch.nn.functional as F

from myseg.bisenetv2 import BiSeNetV2
import tensorflow as tf


def set_model_bisenetv2(args, num_classes):
    net = BiSeNetV2(proj_dim=args.proj_dim, n_classes=num_classes) # num_classes = 19

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
    # if hasattr(model, 'get_params'):
    #     wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
    #     wd_val = 0
    #     params_list = [
    #         {'params': wd_params, },
    #         {'params': nowd_params, 'weight_decay': wd_val},
    #         {'params': lr_mul_wd_params, 'lr': args.lr * 10},
    #         {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': args.lr * 10},
    #     ]
    # else:
    #     wd_params, non_wd_params = [], []
    #     for name, param in model.named_parameters():
    #         if param.dim() == 1:
    #             non_wd_params.append(param)
    #         elif param.dim() == 2 or param.dim() == 4:
    #             wd_params.append(param)
    #     params_list = [
    #         {'params': wd_params, },
    #         {'params': non_wd_params, 'weight_decay': 0},
    #     ]
    # optim = torch.optim.SGD(
    #     params_list,
    #     lr=args.lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )
    # return optim

    # TensorFlow2 优化器迁移
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=args.lr,
        momentum=args.momentum,
        decay=args.weight_decay
    )
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

    def call(self, logits, labels):
        # logits: [batch, h, w, num_classes], labels: [batch, h, w]
        mask = tf.not_equal(labels, self.ignore_lb)
        new_labels = tf.where(mask, labels, tf.zeros_like(labels))
        loss = tf.keras.losses.sparse_categorical_crossentropy(new_labels, logits, from_logits=True)
        loss = tf.where(mask, loss, tf.zeros_like(loss))
        loss_flat = tf.reshape(loss, [-1])
        hard_loss = tf.boolean_mask(loss_flat, loss_flat > self.thresh)
        n_min = tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)) * 0.25, tf.int32)
        if tf.size(hard_loss) < n_min:
            hard_loss = tf.sort(loss_flat, direction='DESCENDING')[:n_min]
        return tf.reduce_mean(hard_loss)


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
        # 将 Tensor 转为 Numpy 进行采样操作
        embs_np = embs.numpy() if hasattr(embs, "numpy") else embs
        labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
        
        b_, c_, h_, w_ = embs_np.shape
        class_u = np.unique(labels_np)
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num -= 1
        if class_u_num == 0:
            return None, None
            
        num_p_c = self.max_anchor // class_u_num
        embs_ = np.transpose(embs_np, (0,2,3,1)).reshape(-1, c_)
        labels_ = labels_np.reshape(-1)
        index_ = np.arange(len(labels_))
        
        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
            if cls_ != 255:
                mask_ = labels_ == cls_
                selected_index_ = index_[mask_]
                if len(selected_index_) > num_p_c:
                    # 注意：np.random.permutation 与 torch.randperm 结果不同
                    sel_i_i = np.arange(len(selected_index_))
                    sel_i_i_i = np.random.permutation(len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]
                    selected_index_ = selected_index_[sel_i]
                embs_tmp = embs_[selected_index_]
                sampled_list.append(embs_tmp)
                sampled_label_list.append(np.ones(len(selected_index_)) * cls_)
        
        if len(sampled_list) == 0:
            return None, None

        sampled_list = np.concatenate(sampled_list, axis=0)
        sampled_label_list = np.concatenate(sampled_label_list, axis=0)
        return sampled_list, sampled_label_list

    def call(self, embs, labels, proto_mem, proto_mask):
        # 转 Numpy
        embs_np = embs.numpy() if hasattr(embs, "numpy") else embs
        labels_np = labels.numpy() if hasattr(labels, "numpy") else labels
        proto_mem_np = proto_mem.numpy() if hasattr(proto_mem, "numpy") else proto_mem
        proto_mask_np = proto_mask.numpy() if hasattr(proto_mask, "numpy") else proto_mask

        anchors, anchor_labels = self._anchor_sampling(embs_np, labels_np)
        if anchors is None:
            return tf.constant(0.0, dtype=tf.float32)

        # 原型筛选逻辑
        if self.args.kmean_num > 0:
            C_, km_, c_ = proto_mem_np.shape
            proto_labels = np.arange(C_).reshape(-1,1).repeat(km_, axis=1)
            proto_mem_ = proto_mem_np.reshape(-1, c_)
            proto_labels = proto_labels.reshape(-1)
            proto_mask_ = proto_mask_np.reshape(-1)
            proto_idx = np.arange(len(proto_mask_))
            sel_idx = proto_idx[proto_mask_.astype(bool)]
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
        else:
            C_, c_ = proto_mem_np.shape
            proto_labels = np.arange(C_)
            proto_mem_ = proto_mem_np
            # 保持逻辑一致
            proto_idx = np.arange(len(proto_mask_np))
            sel_idx = proto_idx[proto_mask_np.astype(bool)]
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]

        # 核心计算
        anchor_dot_contrast = np.matmul(anchors, proto_mem_.T) / self.temperature
        mask = (anchor_labels[:, None] == proto_labels[None, :]).astype(np.float32)
        
        logits_max = np.max(anchor_dot_contrast, axis=1, keepdims=True)
        logits = anchor_dot_contrast - logits_max

        # [修复点 1]：exp_logits 保持矩阵形状，不求和
        exp_logits = np.exp(logits) * mask
        
        neg_mask = 1 - mask
        neg_logits = np.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(axis=1, keepdims=True)

        # [修复点 2]：广播计算 Log Prob
        # exp_logits (N, M) + neg_logits (N, 1) -> (N, M)
        # 添加 1e-12 防止 log(0)
        log_prob = logits - np.log(exp_logits + neg_logits + 1e-12)

        # 计算均值
        mean_log_prob_pos = (mask * log_prob).sum(axis=1) / (mask.sum(axis=1) + 1e-12)

        loss = -mean_log_prob_pos
        loss = loss.mean()
        
        return tf.convert_to_tensor(loss, dtype=tf.float32)



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






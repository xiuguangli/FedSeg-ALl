# import torch
# from torch import nn
# import torch.nn.functional as F
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import numpy as np

from myseg.bisenetv2 import BiSeNetV2


def set_model_bisenetv2(args,num_classes) -> nn.Cell:
    net:nn.Cell = BiSeNetV2(args,num_classes) # num_classes = 19
    # print(net)
    # exit()

    # if not args.finetune_from is None:
    #     logger.info(f'load pretrained weights from {args.finetune_from}')
    #     net.load_state_dict(torch.load(args.finetune_from, map_location='cpu'))

    # if cfg.use_sync_bn: net = nn.SyncBatchNorm.convert_sync_batchnorm(net)

    # net.cuda()
    # net.train()

    # criteria_pre = OhemCELoss(0.7)
    # criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4
    # return net, criteria_pre, criteria_aux

    return net


def set_optimizer0(model, args):
    if hasattr(model, 'get_params'):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        #  wd_val = cfg.weight_decay
        wd_val = 0
        params_list = [
            {'params': wd_params, },
            {'params': nowd_params, 'weight_decay': wd_val},
            {'params': lr_mul_wd_params, 'lr': args.lr * 10},
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': args.lr * 10},
        ]
        # params_list = [
        #     {'params': wd_params, },
        #     {'params': nowd_params, 'weight_decay': wd_val},
        #     {'params': lr_mul_wd_params, 'lr': current_lr * 10},
        #     {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': current_lr * 10},
        # ]
    else:
        wd_params, non_wd_params = [], []
        for name, param in model.named_parameters():
            if param.dim() == 1:
                non_wd_params.append(param)
            elif param.dim() == 2 or param.dim() == 4:
                wd_params.append(param)
        params_list = [
            {'params': wd_params, },
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    optim = torch.optim.SGD(
        params_list,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    # optim = torch.optim.SGD(
    #     params_list,
    #     lr=current_lr,
    #     momentum=args.momentum,
    #     weight_decay=args.weight_decay,
    # )
    return optim

import mindspore.nn as nn
def set_optimizer(model:nn.Cell, args):
    # 分支一：模型有自定义的 get_params 方法

    if hasattr(model, 'get_params'):
        # 假设 model.get_params() 已经被转换为 MindSpore 版本
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
        
        wd_val = 0  # 为不应用 weight decay 的参数组设置的值
        
        # 参数分组的字典结构与 PyTorch 兼容
        params_list = [
            # 该组使用优化器全局的 learning_rate 和 weight_decay
            {'params': wd_params}, 
            # 该组覆盖 weight_decay
            {'params': nowd_params, 'weight_decay': wd_val},
            # 该组覆盖 learning_rate
            {'params': lr_mul_wd_params, 'lr': args.lr * 10},
            # 该组同时覆盖 learning_rate 和 weight_decay
            {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': args.lr * 10},
        ]
        
    # 分支二：通用参数分组逻辑
    else:
        wd_params, non_wd_params = [], []
        # 使用 parameters_and_names() 遍历所有可训练参数
        for name, param in model.parameters_and_names():
            # 使用 param.ndim 获取维度
            if param.ndim == 1:
                non_wd_params.append(param)
            elif param.ndim == 2 or param.ndim == 4:
                wd_params.append(param)
        
        params_list = [
            {'params': wd_params},
            {'params': non_wd_params, 'weight_decay': 0},
        ]
    # 使用 mindspore.nn.SGD
    def print_params(params_list):
        for i, param_group in enumerate(params_list):
            print(f"Parameter group {i}:")
            for key, value in param_group.items():
                if key == 'params':
                    # print(f"  {key}: {len([p.shape for p in value])}")
                    for idx, p in enumerate(value):
                        print(f"     {idx} : {p.shape}")
                else:
                    print(f"  {key}: {value}")

    optim = nn.SGD(
        # params_list,
        model.trainable_params(),  # 如果参数分组有问题，可以直接传入所有可训练参数 
        # learning_rate=args.lr_scheduler_,  # 参数名从 lr 变为 learning_rate
        # learning_rate=args.lr,  # 参数名从 lr 变为 learning_rate 2ka
        learning_rate=0.0002,  # 参数名从 lr 变为 learning_rate 2ka
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    return optim

# class BackCELoss(nn.Module):
class BackCELoss(nn.Cell):
    def __init__(self, args, ignore_lb=255):
        super(BackCELoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.class_num = args.num_classes
        self.criteria = nn.NLLLoss(ignore_index=ignore_lb, reduction='mean')
    # def forward(self, logits, labels):
    def construct(self, logits, labels):
        # total_labels = torch.unique(labels)
        total_labels, _ = ops.unique(labels)
        # new_labels = labels.clone()
        new_labels = ms.Tensor(labels.asnumpy()) 
        # probs = torch.softmax(logits,1)
        probs = ops.softmax(logits,1)
        fore_ = []
        back_ = []
        
        for l in range(self.class_num):
            if l in total_labels:
                fore_.append(probs[:,l,:,:].unsqueeze(1))
            else: 
                back_.append(probs[:,l,:,:].unsqueeze(1))
        Flag=False
        if not  len(fore_)==self.class_num:
            fore_.append(sum(back_))
            Flag=True
        
        for i,l in enumerate(total_labels):
            if Flag :
                new_labels[labels==l]=i
            else: 
                if l!=255:
                    new_labels[labels==l]=i
            
        # probs  =torch.cat(fore_,1)
        probs  =ops.cat(fore_,1)
        # logprobs = torch.log(probs+1e-7)
        logprobs = ops.log(probs+1e-7)
        # return self.criteria(logprobs,new_labels.long())
        # print(logprobs.shape)
        # print(ops.cast(new_labels, ms.int64).shape)
        # exit()
        return self.criteria(logprobs,ops.cast(new_labels, ms.int64))




# class OhemCELoss(nn.Module):
class OhemCELoss(nn.Cell):
    '''
    Feddrive: We apply OHEM (Online Hard-Negative Mining) [56], selecting 25%
    of the pixels having the highest loss for the optimization.
    '''

    def __init__(self, thresh, ignore_lb=255):
        super(OhemCELoss, self).__init__()
        # self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.thresh = -ops.log(ms.Tensor(thresh, dtype=ms.float32))
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward0(self, logits, labels):
        # n_min = labels[labels != self.ignore_lb].numel() // 16
        n_min = int(labels[labels != self.ignore_lb].numel() * 0.25)
        loss = self.criteria(logits, labels).view(-1)
        loss_hard = loss[loss > self.thresh]
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)
    
    def construct(self, logits, labels):
        # 步骤 1: 筛选有效标签，然后计数，最后计算 n_min (与 PyTorch 逻辑完全一致)
        valid_labels = ops.MaskedSelect()(labels, labels != self.ignore_lb)
        num_valid = valid_labels.size  # .numel() -> .size
        n_min = int(num_valid * 0.25)

        # 步骤 2: 计算原始损失并展平
        loss = self.criteria(logits, labels)
        loss = ops.Flatten()(loss)  # .view(-1) -> .flatten()

        # 步骤 3: 筛选出损失值大于阈值的 "硬" 样本
        loss_hard = ops.MaskedSelect()(loss, loss > self.thresh)  # loss[...] -> ops.masked_select(loss, ...)

        # 步骤 4: 如果硬样本数量不足，则用 topk 补充
        if loss_hard.size < n_min:  # .numel() -> .size
            loss_hard, _ = ops.TopK(sorted=True)(loss, n_min)

        # 步骤 5: 返回最终样本的平均损失
        return ops.ReduceMean()(loss_hard)  # torch.mean -> ops.reduce_mean


#################################################

# class CriterionPixelPair(nn.Module):
class CriterionPixelPair(nn.Cell):
    def __init__(self, args,temperature=0.1,ignore_index=255, ):
        super(CriterionPixelPair, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args= args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # C, H, W = fea_0.size()
        C, H, W = fea_0.shape

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        # sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        sim_map_0_1 = ops.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    # def forward(self, feat_S, feat_T):
    def construct(self, feat_S, feat_T):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        # B, C, H, W = feat_S.size()
        B, C, H, W = feat_S.shape

        # device = feat_S.device
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        # feat_S = F.normalize(feat_S, p=2, dim=1)
        feat_S = ops.normalize(feat_S, p=2, dim=1)
        # feat_T = F.normalize(feat_T, p=2, dim=1)
        feat_T = ops.normalize(feat_T, p=2, dim=1)
        
        # sim_dis = torch.tensor(0.).to(device)
        sim_dis = ms.tensor(0.)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], feat_S[i])
            t_sim_map = self.pair_wise_sim_map(feat_T[i], feat_T[i])

            # p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
            p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
            # p_t = F.softmax(t_sim_map / self.temperature, dim=1)
            p_t = ops.softmax(t_sim_map / self.temperature, axis=1)

            # sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis_ = ops.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis += sim_dis_
        sim_dis = sim_dis / B 
        return sim_dis
######################################################

# class CriterionPixelPairSeq(nn.Module):
class CriterionPixelPairSeq(nn.Cell):
    def __init__(self, args,temperature=0.1,ignore_index=255, ):
        super(CriterionPixelPairSeq, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args= args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # C, H, W = fea_0.size()
        C, H, W = fea_0.shape
        
        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        # sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        sim_map_0_1 = ops.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    def forward(self, feat_S, feat_T, pixel_seq):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        # B, C, H, W = feat_S.size()
        B, C, H, W = feat_S.shape

        # device = feat_S.device
        patch_w = 2
        patch_h = 2
        maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        # feat_S = F.normalize(feat_S, p=2, dim=1)
        normalize_op = ops.L2Normalize(axis=1)
        feat_S = normalize_op(feat_S)
        # feat_T = F.normalize(feat_T, p=2, dim=1)
        feat_T = normalize_op(feat_T)

        feat_S = feat_S.permute(0,2,3,1).reshape(-1,C)
        feat_T = feat_T.permute(0,2,3,1).reshape(-1,C)

#        split_T = torch.split(feat_T,1,dim=0)
        split_T = feat_T
        idx = np.random.choice(len(split_T),4000,replace=False)
        
        split_T = split_T[idx]
        # split_T = torch.split(split_T,1,dim=0)
        split_T = ops.split(split_T,1,axis=0)
        pixel_seq.extend(split_T)
        if len(pixel_seq)>20000:
            del pixel_seq[:len(pixel_seq)-20000]
        

        # proto_mem_ = torch.cat(pixel_seq,0)
        proto_mem_ = ops.cat(pixel_seq,0)
        # s_sim_map = torch.matmul(feat_S,proto_mem_.T)
        s_sim_map = ops.matmul(feat_S,proto_mem_.T)
        # t_sim_map = torch.matmul(feat_T,proto_mem_.T)
        t_sim_map = ops.matmul(feat_T,proto_mem_.T)


        # p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
        # p_t = F.softmax(t_sim_map / self.temperature, dim=1)
        p_t = ops.softmax(t_sim_map / self.temperature, axis=1)

        # sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
        sim_dis = ops.kl_div(p_s, p_t, reduction='batchmean')
        return sim_dis,pixel_seq
######################################################

# class CriterionPixelPairG(nn.Module):
class CriterionPixelPairG(nn.Cell):
    def __init__(self, args,temperature=0.1,ignore_index=255, ):
        super(CriterionPixelPairG, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args= args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # C, H, W = fea_0.size()
        C, H, W = fea_0.shape

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.reshape(C, -1).transpose(0, 1)
        
        # sim_map_0_1 = torch.mm(fea_0, fea_1.transpose(0, 1))
        sim_map_0_1 = ops.mm(fea_0, fea_1.transpose(0, 1))
        return sim_map_0_1

    # def forward(self, feat_S, feat_T,proto_mem,proto_mask):
    def construct(self, feat_S, feat_T,proto_mem,proto_mask):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        # B, C, H, W = feat_S.size()
        B, C, H, W = feat_S.shape
        
        # device = feat_S.device
        patch_w = 2
        patch_h = 2
        #maxpool = nn.MaxPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        maxpool = nn.AvgPool2d(kernel_size=(patch_h, patch_w), stride=(patch_h, patch_w), padding=0, ceil_mode=True)
        feat_S = maxpool(feat_S)
        feat_T= maxpool(feat_T)
        # feat_S = F.normalize(feat_S, p=2, dim=1)
        # feat_T = F.normalize(feat_T, p=2, dim=1)
        normalize_op = ops.L2Normalize(axis=1)
        feat_S = normalize_op(feat_S)
        feat_T = normalize_op(feat_T)

        feat_S = feat_S.permute(0,2,3,1).reshape(-1,C)
        feat_T = feat_T.permute(0,2,3,1).reshape(-1,C)

        if self.args.kmean_num>0:
            C_,km_,c_ = proto_mem.size()
            # proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,km_)
            proto_labels = ops.arange(C_).unsqueeze(1).repeat(1,km_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_mask = proto_mask.view(-1)
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]


        else:
            # C_,c_ = proto_mem.size()
            C_,c_ = proto_mem.shape
            # proto_labels = torch.arange(C_)
            proto_labels = ops.arange(C_)
            proto_mem_ = proto_mem
            proto_mask = proto_mask
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]

        # s_sim_map = torch.matmul(feat_S,proto_mem_.T)
        # t_sim_map = torch.matmul(feat_T,proto_mem_.T)
        s_sim_map = ops.matmul(feat_S,proto_mem_.T)
        t_sim_map = ops.matmul(feat_T,proto_mem_.T)


        # p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
        p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
        # p_t = F.softmax(t_sim_map / self.temperature, dim=1)
        p_t = ops.softmax(t_sim_map / self.temperature, axis=1)

        # sim_dis = F.kl_div(p_s, p_t, reduction='batchmean')
        sim_dis = ops.kl_div(p_s, p_t, reduction='batchmean')
        return sim_dis
######################################################

# class CriterionPixelRegionPair(nn.Module):
class CriterionPixelRegionPair(nn.Cell):
    def __init__(self,args, temperature=0.1,ignore_index=255, ):
        super(CriterionPixelRegionPair, self).__init__()
        self.ignore_index = ignore_index
        self.temperature = temperature
        self.args = args

    def pair_wise_sim_map(self, fea_0, fea_1):
        # C, H, W = fea_0.size()
        C, H, W = fea_0.shape

        fea_0 = fea_0.reshape(C, -1).transpose(0, 1)
        fea_1 = fea_1.transpose(0, 1)
        
        # sim_map_0_1 = torch.mm(fea_0, fea_1)
        sim_map_0_1 = ops.mm(fea_0, fea_1)
        return sim_map_0_1

    # def forward(self, feat_S, feat_T,proto_mem,proto_mask):
    def construct(self, feat_S, feat_T,proto_mem,proto_mask):
        #feat_T = self.concat_all_gather(feat_T)
        #feat_S = torch.cat(GatherLayer.apply(feat_S), dim=0)
        # B, C, H, W = feat_S.size()
        B, C, H, W = feat_S.shape

        # device = feat_S.device
        
        if self.args.kmean_num>0:
            # C_,U_,km_,c_ = proto_mem.size()
            C_,U_,km_,c_ = proto_mem.shape
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_mask = proto_mask.unsqueeze(-1).repeat(1,1,km_).view(-1)
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]

        else:
            # C_,U_,c_ = proto_mem.size()
            C_,U_,c_ = proto_mem.shape
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_mask = proto_mask.view(-1)
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]


        # sim_dis = torch.tensor(0.).to(device)
        sim_dis = ms.tensor(0.)
        for i in range(B):
            s_sim_map = self.pair_wise_sim_map(feat_S[i], proto_mem_)
            t_sim_map = self.pair_wise_sim_map(feat_T[i], proto_mem_)

            # p_s = F.log_softmax(s_sim_map / self.temperature, dim=1)
            p_s = ops.log_softmax(s_sim_map / self.temperature, axis=1)
            # p_t = F.softmax(t_sim_map / self.temperature, dim=1)
            p_t = ops.softmax(t_sim_map / self.temperature, axis=1)

            # sim_dis_ = F.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis_ = ops.kl_div(p_s, p_t, reduction='batchmean')
            sim_dis += sim_dis_
        sim_dis = sim_dis / B 
        return sim_dis

######################################################


def L2(f_):
    return (((f_**2).sum(dim=1))**0.5).reshape(f_.shape[0],1,f_.shape[2],f_.shape[3]) + 1e-8

def similarity(feat):
    feat = feat.float()
    # tmp = L2(feat).detach()
    tmp = ops.stop_gradient(L2(feat))
    feat = feat/tmp
    feat = feat.reshape(feat.shape[0],feat.shape[1],-1)
    # return torch.einsum('icm,icn->imn', [feat, feat])
    return ops.einsum('icm,icn->imn', [feat, feat])

def sim_dis_compute(f_S, f_T):
    sim_err = ((similarity(f_T) - similarity(f_S))**2)/((f_T.shape[-1]*f_T.shape[-2])**2)/f_T.shape[0]
    sim_dis = sim_err.sum()
    return sim_dis

# class CriterionPairWiseforWholeFeatAfterPool(nn.Module):
class CriterionPairWiseforWholeFeatAfterPool(nn.Cell):
    def __init__(self, scale):
        '''inter pair-wise loss from inter feature maps'''
        super(CriterionPairWiseforWholeFeatAfterPool, self).__init__()
        self.criterion = sim_dis_compute
        self.scale = scale

    # def forward(self, preds_S, preds_T):
    def construct(self, preds_S, preds_T):
        feat_S = preds_S
        feat_T = preds_T
        # feat_T.detach()

        total_w, total_h = feat_T.shape[2], feat_T.shape[3]
        patch_w, patch_h = int(total_w*self.scale), int(total_h*self.scale)
        maxpool = nn.MaxPool2d(kernel_size=(patch_w, patch_h), stride=(patch_w, patch_h), padding=0, ceil_mode=True) # change
        loss = self.criterion(maxpool(feat_S), maxpool(feat_T))
        return loss


# class ContrastLoss(nn.Module):
class ContrastLoss(nn.Cell):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLoss, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self,embs,labels):
        # device = embs.device
        # b_,c_,h_,w_ = embs.size()
        b_,c_,h_,w_ = embs.shape
        # class_u = torch.unique(labels)
        class_u = ops.unique(labels)[0]
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num =class_u_num - 1

        if class_u_num==0:
            return None,None

        num_p_c = self.max_anchor//class_u_num


        embs = embs.permute(0,2,3,1).reshape(-1,c_)

        labels = labels.view(-1)
        # index_ = torch.arange(len(labels))
        index_ = ops.arange(len(labels))
        # index_ = index_.to(device)

        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
       #     print(cls_)
            if cls_ != 255:
                mask_ = labels==cls_
                # selected_index_ = torch.masked_select(index_,mask_)
                selected_index_ = ops.masked_select(index_,mask_)
                if len(selected_index_)>num_p_c:
                    # sel_i_i = torch.arange(len(selected_index_))
                    sel_i_i = ops.arange(len(selected_index_))
                    # sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
                    sel_i_i_i = ms.Tensor(np.random.permutation(len(sel_i_i))[:num_p_c])
                    sel_i = sel_i_i[sel_i_i_i]     
                    selected_index_ = selected_index_[sel_i]
       #             print(selected_index_.size())
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                # sampled_label_list.append(torch.ones(len(selected_index_)).to(device)*cls_)
                sampled_label_list.append(ops.ones(len(selected_index_))*cls_)
       # print('&'*10)
        # sampled_list = torch.cat(sampled_list,0)
        sampled_list = ops.cat(sampled_list,0)
        # sampled_label_list = torch.cat(sampled_label_list,0)
        sampled_label_list = ops.cat(sampled_label_list,0)

        return sampled_list,sampled_label_list


    # def forward(self,embs,labels,proto_mem,proto_mask):
    def construct(self,embs,labels,proto_mem,proto_mask):
        # device = proto_mem.device
        
        anchors,anchor_labels = self._anchor_sampling(embs,labels)
        
        
        if anchors is None:
            # loss =torch.tensor(0).to(device)
            loss =ms.tensor(0)
            return loss 

        #print(anchors.size())
        #print(anchor_labels.size())
        #exit()

        if self.args.kmean_num>0:
            # C_,km_,c_ = proto_mem.size()
            C_,km_,c_ = proto_mem.shape
            # proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,km_)
            # proto_labels = ops.arange(C_).unsqueeze(1).repeat(1,km_)
            proto_labels = ops.Tile()(ops.expand_dims(ops.arange(C_), 1), (1, km_))
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())

            
            # proto_labels =proto_labels.to(device)
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            # proto_labels =proto_labels.to(device)

            
        else:
            C_,c_ = proto_mem.size()
            # proto_labels = torch.arange(C_)
            proto_labels = ops.arange(C_)
            proto_mem_ = proto_mem
            proto_labels = proto_labels
            proto_labels = proto_labels[sel_idx]
            # proto_labels =proto_labels.to(device)
            proto_mask = proto_mask
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            # proto_labels =proto_labels.to(device)


#        print(proto_mem_.size())
#        print(proto_labels.size())
#        exit()
        # anchor_dot_contrast = torch.div(torch.matmul(anchors,proto_mem_.T),self.temperature)
        
        anchor_dot_contrast = ops.div(ops.matmul(anchors,proto_mem_.T),self.temperature)
        mask = anchor_labels.unsqueeze(1)==proto_labels.unsqueeze(0)
        mask = mask.float()
        # mask = mask.to(device)
        
        
        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_max, _ = ops.max(anchor_dot_contrast, axis=1, keepdims=True)

        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast - ops.stop_gradient(logits_max)

        # mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        # logits_mask = torch.ones_like(mask).scatter_(1,
        #                                              torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(),
        #                                              0)

        # mask = mask * logits_mask

        # neg_logits = torch.exp(logits) * neg_mask
        neg_logits = ops.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        # exp_logits = torch.exp(logits) * mask
        exp_logits = ops.exp(logits) * mask
#        print(exp_logits.size())
#        print(logits.size())
#        print(neg_logits.size())
#        exit()
        # log_prob = logits - torch.log(exp_logits + neg_logits)
        log_prob = logits - ops.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        loss = - mean_log_prob_pos
        loss = loss.mean()
        # if torch.isnan(loss):
        if ops.isnan(loss):
            print('!'*10)
            # print(torch.unique(logits))
            # print(torch.unique(exp_logits))
            # print(torch.unique(neg_logits))
            # print(torch.unique(log_prob))
            # print(torch.unique(mask.sum(1)))
            # print(mask)
            # print(torch.unique(anchor_labels))
            # print(proto_labels)
            # print(torch.unique(proto_labels))
            print(ops.unique(logits))
            print(ops.unique(exp_logits))
            print(ops.unique(neg_logits))
            print(ops.unique(log_prob))
            print(ops.unique(mask.sum(1)))
            print(mask)
            print(ops.unique(anchor_labels))
            print(proto_labels)
            print(ops.unique(proto_labels))
              
            exit()
#        print(loss)
#        print('*'*10)
        
        return loss





# class ContrastLossLocal(nn.Module):
class ContrastLossLocal(nn.Cell):
    def __init__(self, args, ignore_lb=255):
        super(ContrastLossLocal, self).__init__()
        self.ignore_lb = ignore_lb
        self.args = args
        self.max_anchor = args.max_anchor
        self.temperature = args.temperature

    def _anchor_sampling(self,embs,labels):
        # device = embs.device
        # b_,c_,h_,w_ = embs.size()
        b_,c_,h_,w_ = embs.shape
        # class_u = torch.unique(labels)
        class_u = ops.unique(labels)
        class_u_num = len(class_u)
        if 255 in class_u:
            class_u_num =class_u_num - 1

        if class_u_num==0:
            return None,None

        num_p_c = self.max_anchor//class_u_num


        embs = embs.permute(0,2,3,1).reshape(-1,c_)

        labels = labels.view(-1)
        # index_ = torch.arange(len(labels))
        index_ = ops.arange(len(labels))
        # index_ = index_.to(device)

        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
       #     print(cls_)
            if cls_ != 255:
                mask_ = labels==cls_
                # selected_index_ = torch.masked_select(index_,mask_)
                selected_index_ = ops.masked_select(index_,mask_)
                if len(selected_index_)>num_p_c:
                    # sel_i_i = torch.arange(len(selected_index_))
                    sel_i_i = ops.arange(len(selected_index_))
                    # sel_i_i_i = torch.randperm(len(sel_i_i))[:num_p_c]
                    sel_i_i_i = ops.randperm(len(sel_i_i))[:num_p_c]
                    sel_i = sel_i_i[sel_i_i_i]     
                    selected_index_ = selected_index_[sel_i]
       #             print(selected_index_.size())
                embs_tmp = embs[selected_index_]
                sampled_list.append(embs_tmp)
                # sampled_label_list.append(torch.ones(len(selected_index_)).to(device)*cls_)
                sampled_label_list.append(ops.ones(len(selected_index_))*cls_)
       # print('&'*10)
        # sampled_list = torch.cat(sampled_list,0)
        sampled_list = ops.cat(sampled_list,0)
        # sampled_label_list = torch.cat(sampled_label_list,0)
        sampled_label_list = ops.cat(sampled_label_list,0)

        return sampled_list,sampled_label_list


    def forward(self,embs,labels,proto_mem,proto_mask,local_mem):
        # device = proto_mem.device
        anchors,anchor_labels = self._anchor_sampling(embs,labels)
        if anchors is None:
            # loss =torch.tensor(0).to(device)
            loss = ms.tensor(0)
            return loss 

        #print(anchors.size())
        #print(anchor_labels.size())
        #exit()

        if self.args.kmean_num>0:
            # C_,U_,km_,c_ = proto_mem.size()
            C_,U_,km_,c_ = proto_mem.shape
            # proto_labels = torch.arange(C_).unsqueeze(1).unsqueeze(1).repeat(1,U_,km_)
            proto_labels = ops.arange(C_).unsqueeze(1).unsqueeze(1).repeat(1,U_,km_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.unsqueeze(-1).repeat(1,1,km_).view(-1)
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            # proto_labels =proto_labels.to(device)

        else:
            # C_,U_,c_ = proto_mem.size()
            C_,U_,c_ = proto_mem.shape
            # proto_labels = torch.arange(C_).unsqueeze(1).repeat(1,U_)
            proto_labels = ops.arange(C_).unsqueeze(1).repeat(1,U_)
            proto_mem_ = proto_mem.reshape(-1,c_)
            proto_labels = proto_labels.view(-1)
            proto_mask = proto_mask.view(-1)
            # proto_idx = torch.arange(len(proto_mask))
            proto_idx = ops.arange(len(proto_mask))
            # proto_idx = proto_idx.to(device)
            # sel_idx = torch.masked_select(proto_idx,proto_mask.bool())
            sel_idx = ops.masked_select(proto_idx,proto_mask.bool())
            proto_mem_ = proto_mem_[sel_idx]
            proto_labels = proto_labels[sel_idx]
            # proto_labels =proto_labels.to(device)


        # anchor_dot_contrast = torch.div(torch.matmul(anchors,proto_mem_.T),self.temperature)
        anchor_dot_contrast = ops.div(ops.matmul(anchors,proto_mem_.T),self.temperature)
        mask = anchor_labels.unsqueeze(1)==proto_labels.unsqueeze(0)
        mask = mask.float()
        # mask = mask.to(device)

        # logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits_max, _ = ops.max(anchor_dot_contrast, dim=1, keepdims=True)

        # logits = anchor_dot_contrast - logits_max.detach()
        logits = anchor_dot_contrast - ops.stop_gradient(logits_max)

        # exp_logits = torch.exp(logits) * mask
        exp_logits = ops.exp(logits) * mask

       ################## 
        # C_,N_,c_= local_mem.size()
        C_,N_,c_= local_mem.shape
        # local_labels = torch.arange(C_).unsqueeze(1).repeat(1,N_)
        local_labels = ops.arange(C_).unsqueeze(1).repeat(1,N_)
        local_mem = local_mem.reshape(-1,c_)
        local_labels = local_labels.view(-1)
        # local_labels = local_labels.to(device)

        # anchor_dot_contrast_l = torch.div(torch.matmul(anchors,local_mem.T),self.temperature)
        anchor_dot_contrast_l = ops.div(ops.matmul(anchors,local_mem.T),self.temperature)
        mask_l = anchor_labels.unsqueeze(1)==local_labels.unsqueeze(0)
        # mask_l = mask_l.float().to(device)
        # logits_l = anchor_dot_contrast_l - logits_max.detach()
        logits_l = anchor_dot_contrast_l - ops.stop_gradient(logits_max)

        # neg_logits = torch.exp(logits_l) * mask_l
        neg_logits = ops.exp(logits_l) * mask_l
        neg_logits = neg_logits.sum(1, keepdim=True)

######################################
        # log_prob = logits - torch.log(exp_logits + neg_logits)
        log_prob = logits - ops.log(exp_logits + neg_logits)

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - mean_log_prob_pos
        loss = loss.mean()
        # if torch.isnan(loss):
        if ops.isnan(loss):
            print('!'*10)
            # print(torch.unique(logits))
            # print(torch.unique(exp_logits))
            # print(torch.unique(neg_logits))
            # print(torch.unique(log_prob))
            print(ops.unique(logits))
            print(ops.unique(exp_logits))
            print(ops.unique(neg_logits))
            print(ops.unique(log_prob))

            exit()
#        print(loss)
#        print('*'*10)
        return loss






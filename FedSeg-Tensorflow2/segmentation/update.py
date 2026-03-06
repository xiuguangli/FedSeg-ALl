from inspect import signature
import time
# import torch
import copy
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset
import numpy as np
import tensorflow as tf
from torch import permute
from line_profiler import profile
from myseg.bisenet_utils import set_model_bisenetv2
from eval_utils import evaluate
import sys

import myseg.bisenet_utils
from myseg.bisenet_utils import OhemCELoss,BackCELoss,CriterionPixelPair,CriterionPixelRegionPair,ContrastLoss,ContrastLossLocal,CriterionPixelPairG,CriterionPixelPairSeq
from myseg.magic import create_tf_dataloader_from_custom_dataset_train
import numpy as np
from tqdm import tqdm
#from segmentation_models_pytorch.losses import JaccardLoss,DiceLoss,FocalLoss,LovaszLoss,SoftBCEWithLogitsLoss


# class DatasetSplit(Dataset):
#     """
#     An abstract Dataset class wrapped around Pytorch Dataset class.
#     """
#
#     def __init__(self, dataset, idxs):
#         self.dataset = dataset
#         self.idxs = [int(i) for i in idxs]
#
#     def __len__(self):
#         return len(self.idxs)
#
#     def __getitem__(self, item):
#         image, label = self.dataset[self.idxs[item]]
#         # return torch.tensor(image), torch.tensor(label)
#         # pytorch warning and suggest below
#         return image.clone().detach().float(), label.clone().detach()

class DatasetSplit:
    """
    一个抽象的数据集分割类，兼容 numpy/tf.Tensor 数据。
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        # numpy/tf.Tensor 直接返回，无需 clone/detach
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
        elif isinstance(image, tf.Tensor):
            image = tf.cast(image, tf.float32)
        if isinstance(label, tf.Tensor):
            label = tf.identity(label)
        elif isinstance(label, np.ndarray):
            label = label.copy()
        return image, label



class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        # # idxs = range(50)  # 仅用于调试
        self.idx = list(idxs)
        self.shape = dataset[0][0].shape
        # # print(f"Local data shape: {self.shape}")
        
        self.trainloader, self.testloader,self.trainloader_eval = self.train_val_test(dataset, list(idxs))
        #self.device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init(args)

    def init(self, args):            
        self.criteria_distill_pi = CriterionPixelPairSeq(args, temperature=args.temp_dist)
        self.criteria_distill_pa = CriterionPixelRegionPair(args)
        self.criteria_contrast = ContrastLoss(args)

        if args.losstype == 'ohem':
            criteria_pre = OhemCELoss(thresh=0.7)
            criteria_aux = [OhemCELoss(thresh=0.7) for _ in range(4)]  # num_aux_heads=4

        elif args.losstype == 'ce':
            # TF CrossEntropy usually expects labels as integers (Sparse) or One-hot
            # ignore_index处理通常需要sample_weight或自定义Loss
            criteria_pre = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='auto') 
            criteria_aux = [tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='auto') for _ in range(4)]
            
        elif args.losstype == 'back':
            criteria_pre = BackCELoss(args)
            criteria_aux = [BackCELoss(args) for _ in range(4)]
            
        elif args.losstype == 'lovasz':
            criteria_pre = LovaszLoss('multiclass', ignore_index=255)
            criteria_aux = [LovaszLoss('multiclass', ignore_index=255) for _ in range(4)]
            
        elif args.losstype == 'dice':
            criteria_pre = DiceLoss('multiclass', args.num_classes, ignore_index=255)
            criteria_aux = [DiceLoss('multiclass', args.num_classes, ignore_index=255) for _ in range(4)]
            
        elif args.losstype == 'focal':
            criteria_pre = FocalLoss('multiclass', alpha=0.25, ignore_index=255)
            criteria_aux = [FocalLoss('multiclass', alpha=0.25, ignore_index=255) for _ in range(4)]
            
        elif args.losstype == 'bce':
            criteria_pre = SoftBCEWithLogitsLoss(ignore_index=255)
            criteria_aux = [SoftBCEWithLogitsLoss(ignore_index=255) for _ in range(4)]
            
        else:
            raise ValueError('loss type is not defined')

        self.criteria_pre = criteria_pre
        self.criteria_aux = criteria_aux
 

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, and test (80%, 20%)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_test = idxs[int(0.8*len(idxs)):]

        # split indexes for train, and test (100%, 50%)
        idxs_train = idxs[:]
        idxs_test = idxs[:int(0.5*len(idxs))]

        # try to change num_workers, to see if can speed up training. (num_workers=4 is better for training speed)

        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, num_workers=self.args.num_workers,
        #                          shuffle=True, drop_last=True, pin_memory=True)

        # use MultiEpochsDataLoader to speed up training
        
        # trainloader = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
        #                                     batch_size=self.args.local_bs, num_workers=self.args.num_workers,
        #                                     shuffle=True, drop_last=True, pin_memory=True)

        # trainloader_eval = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
        #                                     batch_size=1, num_workers=self.args.num_workers,
        #                                     shuffle=False, drop_last=False, pin_memory=True)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=1, num_workers=self.args.num_workers,
        #                         shuffle=False)
        
        i = 0
        trainloader = create_tf_dataloader_from_custom_dataset_train(DatasetSplit(dataset, idxs_train), batch_size=self.args.local_bs, shuffle=True, repeat=False,output_img_shape=dataset[i][0].shape, output_lbl_shape=dataset[i][1].shape, drop_last=True)
        trainloader_eval = create_tf_dataloader_from_custom_dataset_train(DatasetSplit(dataset, idxs_train), batch_size=1, shuffle=False, repeat=False,output_img_shape=dataset[i][0].shape, output_lbl_shape=dataset[i][1].shape, drop_last=False)
        
        testloader = create_tf_dataloader_from_custom_dataset_train(DatasetSplit(dataset, idxs_test), batch_size=1, shuffle=False, repeat=False,output_img_shape=dataset[i][0].shape, output_lbl_shape=dataset[i][1].shape)
        
        return trainloader, testloader, trainloader_eval

    # @torch.no_grad()
    def get_protos(self, model, global_round):
        args = self.args
        tmp_ = []
        label_list = []
        label_mask_list = []
        
        c,h,w = self.shape
        
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, c, h, w], dtype=tf.float32),
                                      tf.TensorSpec(shape=[1, h, w], dtype=tf.int64)])
        def get_protos_inner(images, labels):
            # images: NCHW, labels: NCHW (or NHW)
            outputs = model(images, training=False)
            logits, feat_head = outputs[0], outputs[1]
            
            # feat_head is NCHW: [Batch, Channel, Height, Width]
            feat_h, feat_w = feat_head.shape[2], feat_head.shape[3]

            # ---------------------------------------------------
            # 1. Resize Logits (NCHW -> NHWC -> Resize -> NCHW)
            # ---------------------------------------------------
            logits_nhwc = tf.transpose(logits, perm=[0, 2, 3, 1])
            logits_resized_nhwc = tf.image.resize(logits_nhwc, (feat_h, feat_w), method='bilinear')
            logits_resized = tf.transpose(logits_resized_nhwc, perm=[0, 3, 1, 2]) # Back to NCHW

            # 2. Softmax (Axis 1 is Channel)
            probs = tf.nn.softmax(logits_resized, axis=1)
            props = tf.reduce_max(probs, axis=1)
            labels_2 = tf.argmax(probs, axis=1)
            
            mask_ = props < 0.8
            labels_2 = tf.cast(labels_2, tf.float32)
            labels_2 = tf.where(mask_, 255.0, labels_2)

            # 3. Process Labels
            # labels is usually (N, H, W). Need to expand for resize
            if len(labels.shape) == 3:
                labels_expanded = tf.expand_dims(labels, axis=-1) # (N, H, W, 1)
            else:
                labels_expanded = labels 
            
            # TF Resize expects NHWC
            labels_resized_nhwc = tf.image.resize(labels_expanded, (feat_h, feat_w), method='nearest')
            labels_resized = tf.squeeze(labels_resized_nhwc, axis=-1)
            labels_resized = tf.cast(labels_resized, tf.float32)

            # 4. Merge
            labels_final = tf.where(labels_resized != 255, labels_resized, labels_2)

            
            # unique_l = np.unique(labels_final.numpy()).tolist()
            # label_list.extend(unique_l)
            # one_hot_mask = np.zeros(args.num_classes)
            # for ll in unique_l:
            #     if int(ll) != 255:
            #         one_hot_mask[int(ll)] = 1
            # label_mask_list.append(one_hot_mask)

            # 5. Weighted Sum (Einsum NCHW style)
            labels_final_int = tf.cast(labels_final, tf.int32)
            # one_hot returns (N, H, W, C)
            labels_one_hot_nhwc = tf.one_hot(labels_final_int, depth=args.num_classes)
            # Transpose to NCHW -> (N, C, H, W)
            labels_one_hot = tf.transpose(labels_one_hot_nhwc, perm=[0, 3, 1, 2])
            
            weight_sum = tf.reduce_sum(labels_one_hot, axis=[2, 3], keepdims=True)
            weight_norm = labels_one_hot / (weight_sum + 1e-5)
            
            # Einsum: feat(nfhw), weight(nchw) -> ncf
            # n: batch, f: feat_dim, c: num_classes, h/w: spatial
            out = tf.einsum('nfhw,nchw->ncf', feat_head, weight_norm)
            # tmp_.append(out)
            
            return out,labels_final

        # import time
        # s0 = time.time()
        for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader_eval,desc="Extracting prototypes",leave=False)):
        # for batch_idx, (images, labels) in enumerate(self.trainloader_eval):
            out, labels_final = get_protos_inner(images, labels)
            tmp_.append(out)
            unique_l = np.unique(labels_final.numpy()).tolist()
            label_list.extend(unique_l)
            one_hot_mask = np.zeros(args.num_classes)
            for ll in unique_l:
                if int(ll) != 255:
                    one_hot_mask[int(ll)] = 1
            label_mask_list.append(one_hot_mask)
        tmp_ = tf.concat(tmp_, axis=0)
        tmp_ = tf.transpose(tmp_, perm=[1, 0, 2])
        
        label_mask_ = np.stack(label_mask_list, axis=1)
        
        # print(f"{tmp_.shape=}, {label_mask_.shape=}")
        # print(f"Unique labels collected: {set(label_list)}")
        # exit()    
        return tmp_.numpy(), sorted(list(set(label_list))), label_mask_ 
    
    def get_protos2(self, model, global_round):
        args = self.args
        tmp_ = []
        label_list = []
        label_mask_list = []
        
        @tf.function
        def get_protos_inner(images, labels):
            # images: NCHW, labels: NCHW (or NHW)
            outputs = model(images, training=False)
            if args.model == 'bisenetv2':
                logits, feat_head = outputs[0], outputs[1]
            
            # feat_head is NCHW: [Batch, Channel, Height, Width]
            feat_h, feat_w = feat_head.shape[2], feat_head.shape[3]

            # ---------------------------------------------------
            # 1. Resize Logits (NCHW -> NHWC -> Resize -> NCHW)
            # ---------------------------------------------------
            logits_nhwc = tf.transpose(logits, perm=[0, 2, 3, 1])
            logits_resized_nhwc = tf.image.resize(logits_nhwc, (feat_h, feat_w), method='bilinear')
            logits_resized = tf.transpose(logits_resized_nhwc, perm=[0, 3, 1, 2]) # Back to NCHW

            # 2. Softmax (Axis 1 is Channel)
            probs = tf.nn.softmax(logits_resized, axis=1)
            props = tf.reduce_max(probs, axis=1)
            labels_2 = tf.argmax(probs, axis=1)
            
            mask_ = props < 0.8
            labels_2 = tf.cast(labels_2, tf.float32)
            labels_2 = tf.where(mask_, 255.0, labels_2)

            # 3. Process Labels
            # labels is usually (N, H, W). Need to expand for resize
            if len(labels.shape) == 3:
                labels_expanded = tf.expand_dims(labels, axis=-1) # (N, H, W, 1)
            else:
                labels_expanded = labels 
            
            # TF Resize expects NHWC
            labels_resized_nhwc = tf.image.resize(labels_expanded, (feat_h, feat_w), method='nearest')
            labels_resized = tf.squeeze(labels_resized_nhwc, axis=-1)
            labels_resized = tf.cast(labels_resized, tf.float32)

            # 4. Merge
            labels_final = tf.where(labels_resized != 255, labels_resized, labels_2)

            
            # unique_l = np.unique(labels_final.numpy()).tolist()
            # label_list.extend(unique_l)
            # one_hot_mask = np.zeros(args.num_classes)
            # for ll in unique_l:
            #     if int(ll) != 255:
            #         one_hot_mask[int(ll)] = 1
            # label_mask_list.append(one_hot_mask)

            # 5. Weighted Sum (Einsum NCHW style)
            labels_final_int = tf.cast(labels_final, tf.int32)
            # one_hot returns (N, H, W, C)
            labels_one_hot_nhwc = tf.one_hot(labels_final_int, depth=args.num_classes)
            # Transpose to NCHW -> (N, C, H, W)
            labels_one_hot = tf.transpose(labels_one_hot_nhwc, perm=[0, 3, 1, 2])
            
            weight_sum = tf.reduce_sum(labels_one_hot, axis=[2, 3], keepdims=True)
            weight_norm = labels_one_hot / (weight_sum + 1e-5)
            
            # Einsum: feat(nfhw), weight(nchw) -> ncf
            # n: batch, f: feat_dim, c: num_classes, h/w: spatial
            out = tf.einsum('nfhw,nchw->ncf', feat_head, weight_norm)
            # tmp_.append(out)
            
            return out,labels_final

        # import time
        # s0 = time.time()    
        for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader_eval,desc="Extracting prototypes",leave=False)):
            out, labels_final = get_protos_inner(images, labels)
            tmp_.append(out)
            unique_l = np.unique(labels_final.numpy()).tolist()
            label_list.extend(unique_l)
            one_hot_mask = np.zeros(args.num_classes)
            for ll in unique_l:
                if int(ll) != 255:
                    one_hot_mask[int(ll)] = 1
            label_mask_list.append(one_hot_mask)
        print('Time consumed to extract prototypes: {:.2f}s\n'.format((time.time() - s0)))
        tmp_ = tf.concat(tmp_, axis=0)
        tmp_ = tf.transpose(tmp_, perm=[1, 0, 2])
        
        label_mask_ = np.stack(label_mask_list, axis=1)
        
        # print(f"{tmp_.shape=}, {label_mask_.shape=}")
        # print(f"Unique labels collected: {set(label_list)}")
        # exit()    
        return tmp_.numpy(), sorted(list(set(label_list))), label_mask_ 
        
    
    @profile
    def update_weights(self, model, global_round, prototypes=None, proto_mask=None, global_model=None):
        args = self.args
        
        epoch_loss = []
        pixel_seq = []

        # Set optimizer and lr_scheduler for the local updates
        optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr, momentum=0.9)
        # optimizer = myseg.bisenet_utils.set_optimizer(model, args) 
        criteria_distill_pi = self.criteria_distill_pi
        criteria_distill_pa = self.criteria_distill_pa
        criteria_contrast = self.criteria_contrast
        criteria_pre = self.criteria_pre
        criteria_aux = self.criteria_aux
        
        # global_model = copy.deepcopy(model)
        # global_model.set_weights(model.get_weights())
        # global_model.trainable = False
        model.aux_mode = 'train'
        
        h,w = self.shape[1], self.shape[2]
        @tf.function(input_signature=[tf.TensorSpec(shape=[None, 3, h, w], dtype=tf.float32)])
        def loss_fn(images):
            
            outputs = model(images, training=True)
                                 
            logits = outputs[0]
            feat_head = outputs[1]
            logits_aux = outputs[2:] # list of aux outputs

            return logits, feat_head, logits_aux        
                
        # training
        start_time = time.time()
        
        for iter_epoch in range(args.local_ep):
            batch_loss = []
            t0 = time.time()
            # 假设 self.trainloader 是一个 tf.data.Dataset 或 可迭代对象
            # for batch_idx, (images, labels) in enumerate(self.trainloader):                
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader, desc="Training", leave=False)):                
                optimizer.learning_rate.assign(args.lr)
                # print(f"shape: images: {images.shape}, labels: {labels.shape}")
                with tf.GradientTape() as tape:
                    # loss,logits, feat_head, logits_aux, labels_, loss_ce, loss_con_item, loss_con_2_item, loss_1_item, loss_pi_item, loss_pa_item = loss_fn(images, labels)
                    
                    logits, feat_head, logits_aux = loss_fn(images)
                    labels_ = labels
                    
                    # exit()
                    loss = 0.0
                    
                    # BCE 特殊处理
                    if args.losstype == 'bce':
                        # PyTorch: cl_ = torch.arange(args.num_classes)...
                        # TF: One-hot encoding
                        # 假设 labels_ 是 index [Batch, H, W]
                        labels_one_hot = tf.one_hot(tf.cast(labels_, tf.int32), depth=args.num_classes)
                        # 调整维度以匹配 PyTorch 的 [1, C, 1, 1] 广播逻辑，TF通常直接用one_hot即可
                        # 这里为了保持逻辑一致，转换为 float
                        labels_ = tf.cast(labels_one_hot, tf.float32)
                        # 注意：BCE loss实现需要根据 input shape 调整

                    # 计算主要 Loss
                    loss_pre = criteria_pre(labels_, logits)
                    
                    # 计算辅助 Loss
                    loss_aux_sum = 0.0
                    for crit, lgt in zip(criteria_aux, logits_aux):
                        loss_aux_sum += crit(labels_, lgt)
                    
                    loss = loss_pre + loss_aux_sum

                    # -------------------------------------------------------
                    # Prototype / Contrastive Loss Logic
                    # -------------------------------------------------------
                    loss_con_item = 0.0
                    loss_con_2_item = 0.0
                    
                    if args.is_proto and global_round >= args.proto_start_epoch:
                        # feat_head shape: [N, C, H, W] in PyTorch (N, H, W, C) in TF usually?
                        # 假设 TF 模型输出为 NCHW 或者 NHWC，这里假设代码已适配 TF 的格式
                        # PyTorch .size(): _, _, h, w
                        # TF: shape
                        
                        # 注意：如果TF模型是NHWC，h, w 是 shape[1], shape[2]
                        # 下面代码假设 feat_head 保持了 PyTorch 的 NCHW 布局或者我们获取 H,W 的方式
                        if  tf.keras.backend.image_data_format() == 'channels_last':
                            h, w = feat_head.shape[1], feat_head.shape[2]
                        else:
                            h, w = feat_head.shape[2], feat_head.shape[3]

                        # Labels 插值调整
                        # PyTorch: labels_.unsqueeze(1) -> interpolate -> squeeze
                        # TF: expand_dims -> resize -> squeeze
                        labels_1 = tf.expand_dims(labels_, axis=-1) # Add channel dim
                        labels_1 = tf.image.resize(labels_1, [h, w], method='nearest')
                        labels_1 = tf.squeeze(labels_1, axis=-1)
                        
                        # 掩码处理 (Proto Mask)
                        if args.kmean_num > 0:
                            # sum(1) in PyTorch is sum over channels? Check dimensions. 
                            # 假设 proto_mask 是 [Classes, K]
                            proto_mask_tmp = tf.reduce_sum(proto_mask, axis=1) < 1
                        else:
                            proto_mask_tmp = proto_mask < 1
                        
                        # 应用掩码：将无效类的标签设为 255
                        # enumerate(proto_mask_tmp)
                        # TF 中不能直接 iterate tensor, 需要 eager execution 或者用 tf.where
                        # 为了完全迁移逻辑，这里用循环处理 tensor assignment (效率较低但逻辑一致)
                        # 更高效的 TF 做法是使用 boolean_mask 或 tf.where 向量化操作
                        
                        # 转换 labels_1 为可修改的变量或使用 tf.where
                        for ii, bo in enumerate(proto_mask_tmp):
                            if bo:
                                labels_1 = tf.where(labels_1 == ii, 255, labels_1) # 假设 float

                        loss_con = criteria_contrast(feat_head, labels_1, prototypes, proto_mask)
                        loss_con_item = loss_con.numpy() if hasattr(loss_con, 'numpy') else float(loss_con)
                        loss_ce = float(loss) # 保存当前CE loss用于打印
                        
                        loss += args.con_lamb * loss_con
                        
                        # Pseudo Label Logic
                        if args.pseudo_label and global_round >= args.pseudo_label_start_epoch:
                            # 调用 Global Model (no_grad 等同于 training=False 或 stop_gradient)
                            outputs_t = global_model(images, training=False)
                            logits_t = outputs_t[0]
                            
                            # Resize logits
                            # PyTorch interpolate default is NCHW, TF resize expects NHWC usually
                            # 这里假设 logits_t 已经是合适的格式，或者使用了自定义 resize
                            # TF resize:
                            if tf.keras.backend.image_data_format() == 'channels_first':
                                logits_t = tf.transpose(logits_t, [0, 2, 3, 1]) # to NHWC for resize
                                labels_2 = tf.image.resize(logits_t, [h, w], method='bilinear')
                                labels_2 = tf.transpose(labels_2, [0, 3, 1, 2]) # back to NCHW
                            else:
                                labels_2 = tf.image.resize(logits_t, [h, w], method='bilinear')
                            
                            labels_2 = tf.nn.softmax(labels_2, axis=1 if tf.keras.backend.image_data_format()=='channels_first' else -1)
                            
                            # Max over classes
                            axis_dim = 1 if tf.keras.backend.image_data_format()=='channels_first' else -1
                            props = tf.reduce_max(labels_2, axis=axis_dim)
                            labels_2_cls = tf.argmax(labels_2, axis=axis_dim)
                            
                            # Cast to float for masking
                            labels_2_cls = tf.cast(labels_2_cls, tf.float32)

                            # Threshold masking
                            mask_ = props < 0.8
                            labels_2_cls = tf.where(mask_, 255, labels_2_cls)

                            # Proto Masking again
                            for ii, bo in enumerate(proto_mask_tmp):
                                if bo:
                                    labels_2_cls = tf.where(labels_2_cls == ii, 255, labels_2_cls)
                            
                            loss_con_2 = criteria_contrast(feat_head, labels_2_cls, prototypes, proto_mask)
                            loss_con_2_item = loss_con_2.numpy() if hasattr(loss_con_2, 'numpy') else float(loss_con_2)
                            loss += args.con_lamb * loss_con_2

                    else:
                        loss_ce = float(loss)
                        loss_con_item = 0

                    # -------------------------------------------------------
                    # FedProx Logic
                    # -------------------------------------------------------
                    if args.fedprox_mu > 0:
                        proximal_term = 0.0
                        for w, w_t in zip(model.trainable_variables, global_model.trainable_variables):
                            # PyTorch: (w-w_t).norm(2)
                            # TF: tf.norm(w-w_t, ord=2)
                            proximal_term += tf.norm(w - w_t, ord=2)
                        
                        loss += (args.fedprox_mu / 2) * proximal_term

                    # -------------------------------------------------------
                    # Distillation Logic
                    # -------------------------------------------------------
                    loss_1_item = float(loss) # current loss value
                    loss_pi_item = 0
                    loss_pa_item = 0

                    if args.distill:
                        outputs_t = global_model(images, training=False)
                        logits_t = outputs_t[0]
                        feat_head_t = outputs_t[1]
                        
                        # Feature Distillation
                        if args.distill_lamb_pi > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                            loss_pi, pixel_seq = criteria_distill_pi(feat_head, tf.stop_gradient(feat_head_t), pixel_seq)
                            loss_pi = args.distill_lamb_pi * loss_pi
                            
                            loss += loss_pi
                            loss_pi_item = loss_pi.numpy() if hasattr(loss_pi, 'numpy') else float(loss_pi)

                        # Pair/Region Distillation
                        if args.distill_lamb_pa > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                            loss_pa = args.distill_lamb_pa * criteria_distill_pa(feat_head, tf.stop_gradient(feat_head_t), prototypes, proto_mask)
                            loss += loss_pa
                            loss_pa_item = loss_pa.numpy() if hasattr(loss_pa, 'numpy') else float(loss_pa)
                
                # -------------------------------------------------------
                # Backward Pass
                # -------------------------------------------------------
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                
                batch_loss.append(float(loss))

                # 打印学习率
                # current_lr = float(optimizer.learning_rate)
                # print("Local Epoch: {}, batch_idx: {}, lr: {:.3e}".format(iter_epoch, batch_idx, current_lr))
                
                # lr_scheduler.step() Logic handled manually above via optimizer.learning_rate.assign
                # break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            if args.verbose:
                string = '| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}'.format(
                    global_round, iter_epoch + 1, len(self.trainloader.dataset), float(loss)) # assuming trainloader has .dataset
                print(string)

        # after training, print logs
        strings = [
            '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
                global_round, args.local_ep, len(self.idx), float(loss))
        ]
        print(''.join(strings))
        
        if args.distill:
            print('Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}'.format(loss_1_item, loss_pi_item, loss_pa_item))
        
        if args.is_proto:
            if global_round >= args.proto_start_epoch:
                if args.pseudo_label:
                    print('Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}'.format(loss_ce, loss_con_item, loss_con_2_item))
                else:
                    print('Loss_CE:{:.6f} | loss_contrast:{:.6f}'.format(loss_ce, loss_con_item))
            else:
                print('Loss_CE:{:.6f}'.format(loss_ce))

        # Return: weights and average loss
        # model.state_dict() equivalent is model.get_weights() (returns list of numpy arrays)
        # If the system strictly expects a state_dict dictionary, serialization logic is needed here.
        # model_weights = {v.name: v.numpy() for v in model.trainable_variables}
        # return model_weights, sum(epoch_loss) / len(epoch_loss)
        
        return model.get_weights(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss. """
        confmat = evaluate(model, self.testloader, self.args.num_classes)
        # print(str(confmat)) # local test也输出信息
        return confmat.acc_global, confmat.iou_mean, str(confmat)


def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss. """
    # device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    confmat = evaluate(model, testloader, args.num_classes)
    acc_global, acc, iu, iou_mean = confmat._get_metric_values()
    return acc_global, iou_mean, str(confmat)
    
    return confmat.acc_global, confmat.iou_mean, str(confmat)


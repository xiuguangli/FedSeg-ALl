import copy
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eval_utils import evaluate
from myseg.bisenet_utils import (
    BackCELoss,
    ContrastLoss,
    CriterionPixelRegionPair,
    CriterionPixelPairSeq,
    OhemCELoss,
    set_optimizer,
)
from logging_utils import logger
from myseg.magic import MultiEpochsDataLoader
from seed_utils import make_torch_generator, make_worker_init_fn


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        # 固定排序后的索引切片能让日志预览、评估抽样和复现实验保持一致。
        self.idxs = [int(i) for i in sorted(idxs)]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.clone().detach().float(), label.clone().detach()


class Client:
    def __init__(self, args, dataset, idxs, client_id=None):
        self.args = args
        self.dataset = dataset
        self.idxs = [int(i) for i in sorted(idxs)]
        self.client_id = int(client_id if client_id is not None else 0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._profile_runtime = bool(getattr(args, "profile_runtime", False))
        self._debug_train_order_spec = None
        self._debug_train_order_logged = False

    @property
    def num_samples(self):
        return len(self.idxs)

    @property
    def num_eval_samples(self):
        return int(0.5 * len(self.idxs))

    def preview_indices(self, limit=8):
        return self.idxs[:limit]

    def _loader_seed(self, seed_offset=0):
        return self.args.seed + self.client_id * 1000 + seed_offset

    def _num_train_batches(self):
        return len(self.idxs) // int(self.args.local_bs)

    def _load_debug_train_order_spec(self):
        path = str(getattr(self.args, "debug_train_order_file", "") or "").strip()
        if not path:
            return None
        if self._debug_train_order_spec is None:
            with open(path, "r", encoding="utf-8") as fin:
                self._debug_train_order_spec = json.load(fin)
        return self._debug_train_order_spec

    def _resolve_debug_train_order_entry(self, spec, global_round):
        if spec is None:
            return None
        if "round_client_orders" in spec:
            round_map = spec["round_client_orders"].get(str(int(global_round)))
            if round_map is None:
                raise ValueError("debug_train_order_file is missing round {}".format(global_round))
            client_entry = round_map.get(str(self.client_id))
            if client_entry is None:
                raise ValueError(
                    "debug_train_order_file is missing client {} for round {}".format(
                        self.client_id, global_round
                    )
                )
            return client_entry
        return spec.get("metadata", spec)

    def debug_train_epoch_relative_indices(self, global_round, local_epoch=0):
        spec = self._load_debug_train_order_spec()
        if spec is not None:
            spec = self._resolve_debug_train_order_entry(spec, global_round)
            expected_client_id = spec.get("client_id")
            if expected_client_id is not None and int(expected_client_id) != self.client_id:
                raise ValueError(
                    "debug_train_order_file client_id mismatch: expected {}, got {}".format(
                        expected_client_id, self.client_id
                    )
                )
            expected_round = spec.get("global_round")
            if expected_round is not None and int(expected_round) != int(global_round):
                raise ValueError(
                    "debug_train_order_file global_round mismatch: expected {}, got {}".format(
                        expected_round, global_round
                    )
                )
            epoch_orders = spec.get("relative_epoch_orders") or spec.get("epoch_orders")
            if not epoch_orders:
                raise ValueError("debug_train_order_file is missing relative_epoch_orders")
            if local_epoch < 0 or local_epoch >= len(epoch_orders):
                raise ValueError(
                    "debug_train_order_file local_epoch {} out of range [0, {})".format(
                        local_epoch, len(epoch_orders)
                    )
                )
            relative_indices = [int(idx) for idx in epoch_orders[int(local_epoch)]]
            if len(relative_indices) != len(self.idxs):
                raise ValueError(
                    "debug_train_order_file epoch {} length mismatch: expected {}, got {}".format(
                        local_epoch, len(self.idxs), len(relative_indices)
                    )
                )
            if sorted(relative_indices) != list(range(len(self.idxs))):
                raise ValueError(
                    "debug_train_order_file epoch {} must be a permutation of [0, {})".format(
                        local_epoch, len(self.idxs)
                    )
                )
            if not self._debug_train_order_logged:
                logger.info(
                    "Client {} using explicit train order from {} for round {}",
                    self.client_id,
                    self.args.debug_train_order_file,
                    global_round,
                )
                self._debug_train_order_logged = True
            return relative_indices

        generator = make_torch_generator(self._loader_seed(global_round))
        # Match DataLoader iterator creation: it draws one base_seed from the same
        # generator before RandomSampler starts consuming permutations.
        torch.empty((), dtype=torch.int64).random_(generator=generator)
        for _ in range(int(local_epoch)):
            torch.randperm(len(self.idxs), generator=generator)
            # RandomSampler(replacement=False) calls an additional empty-slice
            # randperm when num_samples % n == 0, so the generator advances one
            # more full permutation per sampler pass.
            torch.randperm(len(self.idxs), generator=generator)
        permutation = torch.randperm(len(self.idxs), generator=generator)
        return permutation.detach().cpu().tolist()

    def debug_train_epoch_absolute_indices(self, global_round, local_epoch=0):
        return [self.idxs[idx] for idx in self.debug_train_epoch_relative_indices(global_round, local_epoch)]

    def _build_loader(self, idxs, batch_size, shuffle, drop_last, multi_epoch=False, seed_offset=0):
        # 每个客户端都从独立种子派生 DataLoader 随机性，预创建全部客户端也不会互相干扰。
        seed = self._loader_seed(seed_offset)
        generator = make_torch_generator(seed)
        worker_init_fn = make_worker_init_fn(seed)
        pin_memory = torch.cuda.is_available()

        loader_cls = MultiEpochsDataLoader if multi_epoch else DataLoader
        loader_kwargs = {
            "batch_size": batch_size,
            "num_workers": self.args.num_workers,
            "shuffle": shuffle,
            "drop_last": drop_last,
            "pin_memory": pin_memory,
            "worker_init_fn": worker_init_fn,
        }
        if shuffle:
            loader_kwargs["generator"] = generator

        return loader_cls(DatasetSplit(self.dataset, idxs), **loader_kwargs)

    def _build_trainloader(self, global_round):
        return self._build_loader(
            self.idxs,
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True,
            multi_epoch=True,
            seed_offset=global_round,
        )

    def _build_trainloader_eval(self, global_round):
        return self._build_loader(
            self.idxs,
            batch_size=1,
            shuffle=False,
            drop_last=False,
            multi_epoch=True,
            seed_offset=global_round,
        )

    def _build_testloader(self):
        # 本地评估沿用原实现，只取客户端前半部分样本做快速验证。
        return self._build_loader(
            self.idxs[: self.num_eval_samples],
            batch_size=1,
            shuffle=False,
            drop_last=False,
            multi_epoch=False,
        )

    def _build_losses(self):
        args = self.args
        if args.model != "bisenetv2":
            raise ValueError("unrecognized model")

        if args.losstype == "ohem":
            criteria_pre = OhemCELoss(0.7)
            criteria_aux = [OhemCELoss(0.7) for _ in range(4)]
        elif args.losstype == "ce":
            criteria_pre = nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
            criteria_aux = [nn.CrossEntropyLoss(ignore_index=255, reduction="mean") for _ in range(4)]
        elif args.losstype == "back":
            criteria_pre = BackCELoss(args)
            criteria_aux = [BackCELoss(args) for _ in range(4)]
        elif args.losstype in {"lovasz", "dice", "focal", "bce"}:
            from segmentation_models_pytorch.losses import (
                DiceLoss,
                FocalLoss,
                LovaszLoss,
                SoftBCEWithLogitsLoss,
            )

            if args.losstype == "lovasz":
                criteria_pre = LovaszLoss("multiclass", ignore_index=255)
                criteria_aux = [LovaszLoss("multiclass", ignore_index=255) for _ in range(4)]
            elif args.losstype == "dice":
                criteria_pre = DiceLoss("multiclass", args.num_classes, ignore_index=255)
                criteria_aux = [DiceLoss("multiclass", args.num_classes, ignore_index=255) for _ in range(4)]
            elif args.losstype == "focal":
                criteria_pre = FocalLoss("multiclass", alpha=0.25, ignore_index=255)
                criteria_aux = [FocalLoss("multiclass", alpha=0.25, ignore_index=255) for _ in range(4)]
            else:
                criteria_pre = SoftBCEWithLogitsLoss(ignore_index=255)
                criteria_aux = [SoftBCEWithLogitsLoss(ignore_index=255) for _ in range(4)]
        else:
            raise ValueError("loss type is not defined")

        return criteria_pre, criteria_aux

    def _build_scheduler(self, optimizer, trainloader, global_round):
        if self.args.lr_scheduler == "step":
            return torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda x: 1 if global_round < 1000 else 0.1,
            )
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda x: (1 - x / (len(trainloader) * max(1, self.args.local_ep))) ** 0.9,
        )

    def _forward(self, model, images):
        if self.args.model != "bisenetv2":
            raise ValueError("unrecognized model")
        logits, feat_head, *logits_aux = model(images)
        return logits, feat_head, logits_aux

    def _apply_debug_train_overrides(self, model):
        if not getattr(self.args, "debug_freeze_bn_stats", False):
            return
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def _sync_device(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def extract_prototypes(self, model, global_round):
        args = self.args
        model.eval()
        tmp_ = []
        label_list = []
        label_mask_list = []
        # 原型提取不打乱样本顺序，方便在相同 seed 下得到稳定的类别特征统计。
        trainloader_eval = self._build_trainloader_eval(global_round)
        proto_loader = tqdm(
            trainloader_eval,
            desc=f"Client {self.client_id} prototypes",
            leave=False,
            dynamic_ncols=True,
        )
        for images, labels in proto_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits, feat_head, _ = self._forward(model, images)

            _, _, h, w = feat_head.size()
            labels_2 = F.interpolate(logits.float(), size=(h, w), mode="bilinear")
            labels_2 = torch.softmax(labels_2, dim=1)
            props, labels_2 = torch.max(labels_2, dim=1)
            labels_2[props < 0.8] = 255

            # 先用高置信预测补齐缺失标签，再把像素特征按类别压成 prototype。
            feat_head = feat_head.unsqueeze(1)
            labels = labels.unsqueeze(1)
            labels = F.interpolate(labels.float(), size=(h, w), mode="nearest")
            labels = labels.unsqueeze(1)
            labels_2 = labels_2.unsqueeze(1).unsqueeze(1)

            labels = torch.where(labels.float() != 255, labels.float(), labels_2.float())
            unique_l = torch.unique(labels.cpu()).numpy().tolist()
            label_list.extend(unique_l)

            one_hot_ = torch.zeros(args.num_classes, device=self.device)
            for ll in unique_l:
                if ll != 255:
                    one_hot_[int(ll)] = 1
            label_mask_list.append(one_hot_)

            class_ = torch.arange(args.num_classes, device=self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_ = class_ == labels
            weight_ = weight_ / (weight_.sum(3, keepdim=True).sum(4, keepdim=True) + 1e-5)
            out = (weight_ * feat_head).sum(-1).sum(-1)
            tmp_.append(out)

        tmp_ = torch.cat(tmp_, 0).permute(1, 0, 2)
        label_mask_ = torch.stack(label_mask_list, 1)
        return tmp_, label_list, label_mask_

    def train(self, model, global_round, prototypes=None, proto_mask=None):
        model.train()
        self._apply_debug_train_overrides(model)
        args = self.args
        epoch_loss = []
        stage_times = {}
        timer = time.perf_counter

        start = timer()
        trainloader = self._build_trainloader(global_round)
        stage_times["build_trainloader"] = timer() - start
        start = timer()
        criteria_pre, criteria_aux = self._build_losses()
        stage_times["build_losses"] = timer() - start
        start = timer()
        optimizer = set_optimizer(model, args)
        stage_times["build_optimizer"] = timer() - start
        lr_scheduler = self._build_scheduler(optimizer, trainloader, global_round)

        reference_model = None
        if args.distill or args.fedprox_mu > 0 or (args.is_proto and args.pseudo_label):
            # 蒸馏、FedProx 和伪标签都需要一份冻结的“全局快照”作为教师或约束目标。
            start = timer()
            reference_model = copy.deepcopy(model)
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False
            stage_times["clone_reference_model"] = timer() - start
        else:
            stage_times["clone_reference_model"] = 0.0

        if args.is_proto:
            criteria_contrast = ContrastLoss(args)

        if args.distill:
            criteria_distill_pi = CriterionPixelPairSeq(args, temperature=args.temp_dist)
            criteria_distill_pa = CriterionPixelRegionPair(args)
            pixel_seq = []

        loss_ce = 0
        loss_con_item = 0
        loss_con_2_item = 0
        loss_1_item = 0
        loss_pi_item = 0
        loss_pa_item = 0

        epoch_bar = tqdm(
            range(args.local_ep),
            desc=f"Client {self.client_id} local epochs",
            leave=False,
            dynamic_ncols=True,
        )
        for local_epoch in epoch_bar:
            batch_loss = []
            batch_bar = tqdm(
                trainloader,
                desc=f"Client {self.client_id} epoch {local_epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )
            iter_end = timer()
            data_wait_time = 0.0
            compute_time = 0.0
            for batch_idx, (images, labels) in enumerate(batch_bar):
                batch_start = timer()
                data_wait_time += batch_start - iter_end
                images, labels = images.to(self.device), labels.to(self.device)
                logits, feat_head, logits_aux = self._forward(model, images)

                labels_ = labels
                if args.losstype == "bce":
                    cl_ = torch.arange(args.num_classes, device=labels_.device).view(1, -1, 1, 1)
                    labels_ = (labels_.unsqueeze(1) == cl_).float()

                loss_pre = criteria_pre(logits, labels_)
                loss_aux = [crit(lgt, labels_) for crit, lgt in zip(criteria_aux, logits_aux)]
                loss = loss_pre + sum(loss_aux)
                loss_ce = loss.item()

                if args.is_proto and global_round >= args.proto_start_epoch:
                    _, _, h, w = feat_head.size()
                    labels_1 = F.interpolate(labels_.unsqueeze(1).float(), size=(h, w), mode="nearest").squeeze(1)
                    if args.kmean_num > 0:
                        proto_mask_tmp = proto_mask.sum(1) < 1
                    else:
                        proto_mask_tmp = proto_mask < 1
                    # 客户端本地不存在的类别直接屏蔽掉，避免拿空 prototype 做对比学习。
                    for class_idx, missing in enumerate(proto_mask_tmp):
                        if missing:
                            labels_1[labels_1 == class_idx] = 255

                    loss_con = criteria_contrast(feat_head, labels_1, prototypes, proto_mask)
                    loss_con_item = loss_con.item()
                    loss += args.con_lamb * loss_con

                    if args.pseudo_label and global_round >= args.pseudo_label_start_epoch:
                        with torch.no_grad():
                            logits_t, feat_head_t, _ = self._forward(reference_model, images)
                        labels_2 = F.interpolate(logits_t.float(), size=(h, w), mode="bilinear")
                        labels_2 = torch.softmax(labels_2, dim=1)
                        props, labels_2 = torch.max(labels_2, dim=1)
                        labels_2[props < 0.8] = 255
                        for class_idx, missing in enumerate(proto_mask_tmp):
                            if missing:
                                labels_2[labels_2 == class_idx] = 255
                        loss_con_2 = criteria_contrast(feat_head, labels_2, prototypes, proto_mask)
                        loss_con_2_item = loss_con_2.item()
                        loss += args.con_lamb * loss_con_2
                else:
                    loss_con_item = 0

                if args.fedprox_mu > 0:
                    proximal_term = 0.0
                    for w, w_t in zip(model.parameters(), reference_model.parameters()):
                        proximal_term += (w - w_t).norm(2)
                    loss += (args.fedprox_mu / 2) * proximal_term

                if args.distill:
                    loss_1_item = loss.item()
                    with torch.no_grad():
                        logits_t, feat_head_t, _ = self._forward(reference_model, images)
                    # 像素级和区域级蒸馏都复用冻结教师特征，约束本地更新不要偏离全局初始化过快。
                    if args.distill_lamb_pi > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                        loss_pi, pixel_seq = criteria_distill_pi(feat_head, feat_head_t.detach(), pixel_seq)
                        loss_pi = args.distill_lamb_pi * loss_pi
                        loss += loss_pi
                        loss_pi_item = loss_pi.item()
                    if args.distill_lamb_pa > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                        loss_pa = args.distill_lamb_pa * criteria_distill_pa(feat_head, feat_head_t.detach(), prototypes, proto_mask)
                        loss += loss_pa
                        loss_pa_item = loss_pa.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

                batch_bar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                )
                lr_scheduler.step()
                self._sync_device()
                iter_end = timer()
                compute_time += iter_end - batch_start

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_bar.set_postfix(loss=f"{epoch_loss[-1]:.4f}")
            if args.verbose:
                logger.debug(
                    "| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}",
                    global_round,
                    local_epoch + 1,
                    len(trainloader.dataset),
                    loss.item(),
                )
            if self._profile_runtime:
                logger.info(
                    "Runtime profile | client={} round={} epoch={} data_wait={:.3f}s compute={:.3f}s batches={}",
                    self.client_id,
                    global_round,
                    local_epoch + 1,
                    data_wait_time,
                    compute_time,
                    len(batch_loss),
                )

        logger.info(
            "| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}",
            global_round,
            args.local_ep,
            len(trainloader.dataset),
            loss.item(),
        )
        if args.distill:
            logger.info("Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}", loss_1_item, loss_pi_item, loss_pa_item)
        if args.is_proto:
            if global_round >= args.proto_start_epoch:
                if args.pseudo_label:
                    logger.info(
                        "Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}",
                        loss_ce,
                        loss_con_item,
                        loss_con_2_item,
                    )
                else:
                    logger.info("Loss_CE:{:.6f} | loss_contrast:{:.6f}", loss_ce, loss_con_item)
            else:
                logger.info("Loss_CE:{:.6f}", loss_ce)

        if self._profile_runtime:
            logger.info(
                "Runtime profile | client={} round={} loader={:.3f}s losses={:.3f}s optimizer={:.3f}s reference={:.3f}s",
                self.client_id,
                global_round,
                stage_times.get("build_trainloader", 0.0),
                stage_times.get("build_losses", 0.0),
                stage_times.get("build_optimizer", 0.0),
                stage_times.get("clone_reference_model", 0.0),
            )

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        confmat = evaluate(model, self._build_testloader(), self.device, self.args.num_classes)
        return confmat.acc_global, confmat.iou_mean, str(confmat)

    evaluate = inference
    update_weights = train


def test_inference(args, model, testloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    confmat = evaluate(model, testloader, device, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)

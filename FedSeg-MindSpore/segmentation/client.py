import copy
import json
import time

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import amp
import numpy as np
from tqdm import tqdm

from batch_utils import build_batches, num_batches
from checkpoint_utils import clone_state_dict, load_state_into_net
from eval_utils import evaluate
from logging_utils import logger
from myseg.bisenet_utils import (
    BackCELoss,
    ContrastLoss,
    CriterionPixelPairSeq,
    CriterionPixelRegionPair,
    OhemCELoss,
    set_optimizer,
)
from seed_utils import torch_multi_epoch_loader_indices


def _apply_available_class_mask(labels, available_mask, ignore_value=255):
    labels_int = labels.astype(ms.int32)
    available_mask_np = np.asarray(available_mask.asnumpy(), dtype=bool).reshape(-1)
    if available_mask_np.size == 0:
        return ops.ones_like(labels_int) * int(ignore_value)

    masked_labels = labels_int
    ignore_fill = ops.ones_like(labels_int) * int(ignore_value)
    for class_idx, is_available in enumerate(available_mask_np.tolist()):
        if not is_available:
            masked_labels = ops.where(masked_labels == int(class_idx), ignore_fill, masked_labels)
    return masked_labels


class DatasetSplit:
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(idx) for idx in sorted(idxs)]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image.astype("float32"), label.astype("int32")


class Client:
    def __init__(self, args, dataset, idxs, client_id=None):
        # Keep clients lightweight: they only own dataset/index metadata.
        # The model is always passed into train/evaluate methods on demand.
        self.args = args
        self.dataset = dataset
        self.idxs = [int(idx) for idx in sorted(idxs)]
        self.client_id = int(client_id if client_id is not None else 0)
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

    def _train_epoch_seed_offset(self, global_round, local_epoch=0):
        return global_round * max(1, int(self.args.local_ep)) + int(local_epoch)

    def _num_train_batches(self):
        return num_batches(len(self.idxs), self.args.local_bs, drop_last=True)

    def _load_debug_train_order_spec(self):
        path = str(getattr(self.args, "debug_train_order_file", "") or "").strip()
        if not path:
            return None
        if self._debug_train_order_spec is None:
            with open(path, "r", encoding="utf-8") as fin:
                loaded = json.load(fin)
            self._debug_train_order_spec = loaded
        return self._debug_train_order_spec

    def _resolve_debug_train_order_entry(self, spec, global_round):
        if spec is None:
            return None
        if "round_client_orders" in spec:
            round_map = spec["round_client_orders"].get(str(int(global_round)))
            if round_map is None:
                raise ValueError(
                    "debug_train_order_file is missing round {}".format(global_round)
                )
            client_entry = round_map.get(str(self.client_id))
            if client_entry is None:
                raise ValueError(
                    "debug_train_order_file is missing client {} for round {}".format(
                        self.client_id, global_round
                    )
                )
            return client_entry
        return spec.get("metadata", spec)

    def _default_train_epoch_relative_indices(self, global_round, local_epoch=0):
        return torch_multi_epoch_loader_indices(
            seed=self._loader_seed(global_round),
            size=len(self.idxs),
            local_epoch=local_epoch,
        )

    def debug_train_epoch_relative_indices(self, global_round, local_epoch=0):
        spec = self._load_debug_train_order_spec()
        if spec is None:
            return self._default_train_epoch_relative_indices(global_round, local_epoch)
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

    def debug_train_epoch_absolute_indices(self, global_round, local_epoch=0):
        return [self.idxs[idx] for idx in self.debug_train_epoch_relative_indices(global_round, local_epoch)]

    def _build_loader(
        self,
        idxs,
        batch_size,
        shuffle,
        drop_last,
        seed_offset=0,
        pad_to_max_shape=False,
        ordered_indices=None,
        synthetic_num_workers=0,
        batch_worker_offset=0,
        worker_seed_base=None,
        worker_states=None,
    ):
        split = DatasetSplit(self.dataset, idxs)
        return build_batches(
            split,
            idxs=ordered_indices,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            seed=self._loader_seed(seed_offset),
            pad_to_max_shape=pad_to_max_shape,
            synthetic_num_workers=synthetic_num_workers,
            batch_worker_offset=batch_worker_offset,
            worker_seed_base=worker_seed_base,
            worker_states=worker_states,
        )

    def _build_trainloader(self, global_round, local_epoch=0, worker_states=None):
        ordered_indices = self.debug_train_epoch_relative_indices(global_round, local_epoch)
        synthetic_num_workers = 0
        batch_worker_offset = 0
        worker_seed_base = None
        if getattr(self.args, "debug_emulate_torch_worker_rng", False) and int(getattr(self.args, "num_workers", 0)) > 0:
            synthetic_num_workers = int(self.args.num_workers)
            batch_worker_offset = int(local_epoch) * self._num_train_batches()
            worker_seed_base = self._loader_seed(global_round)
        return self._build_loader(
            self.idxs,
            batch_size=self.args.local_bs,
            shuffle=False,
            drop_last=True,
            seed_offset=self._train_epoch_seed_offset(global_round, local_epoch),
            ordered_indices=ordered_indices,
            synthetic_num_workers=synthetic_num_workers,
            batch_worker_offset=batch_worker_offset,
            worker_seed_base=worker_seed_base,
            worker_states=worker_states,
        )

    def _build_trainloader_eval(self, global_round):
        synthetic_num_workers = 0
        batch_worker_offset = 0
        worker_seed_base = None
        if getattr(self.args, "debug_emulate_torch_worker_rng", False) and int(getattr(self.args, "num_workers", 0)) > 0:
            synthetic_num_workers = int(self.args.num_workers)
            worker_seed_base = self._loader_seed(global_round)
        return self._build_loader(
            self.idxs,
            # Match the torch implementation exactly: prototype extraction
            # keeps one sample per step so the prototype memory bank stores
            # per-image slots rather than per-batch aggregates.
            batch_size=1,
            shuffle=False,
            drop_last=False,
            seed_offset=global_round,
            synthetic_num_workers=synthetic_num_workers,
            batch_worker_offset=batch_worker_offset,
            worker_seed_base=worker_seed_base,
        )

    def _build_testloader(self):
        return self._build_loader(
            self.idxs[: self.num_eval_samples],
            batch_size=1,
            shuffle=False,
            drop_last=False,
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
        else:
            raise ValueError("loss type is not defined for the MindSpore refactor")
        return criteria_pre, criteria_aux

    def _make_learning_rate(self, global_round, num_train_batches):
        if self.args.lr_scheduler == "step":
            lr_value = self.args.lr if global_round < 1000 else self.args.lr * 0.1
            return ms.Tensor(lr_value, ms.float32)
        total_step = max(1, num_train_batches * max(1, self.args.local_ep))
        lr_values = [
            self.args.lr * ((1 - step / total_step) ** 0.9)
            for step in range(total_step)
        ]
        return ms.Tensor(lr_values, ms.float32)

    def _forward(self, model, images):
        if self.args.model != "bisenetv2":
            raise ValueError("unrecognized model")
        logits, feat_head, *logits_aux = model(images)
        return logits, feat_head, logits_aux

    def _apply_debug_train_overrides(self, model):
        if not getattr(self.args, "debug_freeze_bn_stats", False):
            return
        for _, cell in model.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.use_batch_statistics = False

    def _lr_snapshot(self, learning_rate, lr_step):
        if hasattr(learning_rate, "asnumpy"):
            lr_np = learning_rate.asnumpy()
            if getattr(lr_np, "size", 0) == 0:
                return 0.0
            lr_values = np.asarray(lr_np).reshape(-1)
            return float(lr_values[min(int(lr_step), len(lr_values) - 1)])
        try:
            return float(learning_rate)
        except Exception:
            return 0.0

    def _should_update_progress(self, batch_idx, total_batches):
        if total_batches <= 0:
            return False
        if self._profile_runtime or self.args.verbose:
            return True
        if self._is_graph_mode():
            return batch_idx + 1 == total_batches
        stride = max(1, total_batches // 4)
        return ((batch_idx + 1) % stride == 0) or (batch_idx + 1 == total_batches)

    def _is_graph_mode(self):
        try:
            return ms.get_context("mode") == ms.GRAPH_MODE
        except Exception:
            return False

    def _anchor_mask(self, labels, proto_mask):
        if self.args.kmean_num > 0:
            available_mask = proto_mask.sum(axis=1) > 0
        else:
            available_mask = proto_mask > 0
        return _apply_available_class_mask(labels, available_mask, ignore_value=255)

    def extract_prototypes(self, model, global_round):
        args = self.args
        model.set_train(False)
        tmp_list = []
        label_list = []
        label_mask_list = []
        class_idx = ops.arange(args.num_classes, dtype=ms.int32).reshape(1, args.num_classes, 1, 1, 1)
        trainloader_eval = self._build_trainloader_eval(global_round)
        forward_batch_size = max(1, int(getattr(args, "prototype_forward_batch_size", 1)))
        proto_loader = tqdm(
            trainloader_eval,
            desc=f"Client {self.client_id} prototypes",
            leave=False,
            dynamic_ncols=True,
        )
        image_buffer = []
        label_buffer = []

        def flush_buffer():
            if not image_buffer:
                return
            images = ops.concat(tuple(image_buffer), axis=0)
            labels = ops.concat(tuple(label_buffer), axis=0)
            logits, feat_head, _ = self._forward(model, images)
            _, _, height, width = feat_head.shape
            labels_2 = ops.interpolate(logits.astype(ms.float32), size=(height, width), mode="bilinear", align_corners=False)
            labels_2 = ops.softmax(labels_2, axis=1)
            props, labels_2 = ops.max(labels_2, axis=1)
            labels_2 = ops.where(props < 0.8, ops.ones_like(labels_2) * 255, labels_2)

            feat_head = feat_head.unsqueeze(1)
            labels_main = ops.interpolate(labels.unsqueeze(1).astype(ms.float32), size=(height, width), mode="nearest")
            labels_main = labels_main.unsqueeze(1)
            labels_2 = labels_2.unsqueeze(1).unsqueeze(1).astype(ms.float32)

            labels_main = ops.where(labels_main != 255, labels_main, labels_2)
            weight = class_idx == labels_main.astype(ms.int32)
            label_mask_list.append(
                ops.stop_gradient((weight.sum(axis=(3, 4)) > 0).astype(ms.float32).squeeze(-1))
            )
            weight = weight.astype(ms.float32) / (weight.sum(axis=3, keepdims=True).sum(axis=4, keepdims=True) + 1e-5)
            out = (weight * feat_head).sum(axis=-1).sum(axis=-1)
            tmp_list.append(ops.stop_gradient(out))
            image_buffer.clear()
            label_buffer.clear()

        for images, labels in proto_loader:
            image_buffer.append(images)
            label_buffer.append(labels)
            if len(image_buffer) >= forward_batch_size:
                flush_buffer()

        flush_buffer()
        tmp_ = ops.stop_gradient(ops.cat(tmp_list, axis=0).permute(1, 0, 2))
        label_mask_ = ops.stop_gradient(ops.cat(label_mask_list, axis=0).transpose(1, 0))
        return tmp_, label_list, label_mask_

    def train(self, model, global_round, prototypes=None, proto_mask=None):
        model = copy.deepcopy(model)
        model.set_train(True)
        self._apply_debug_train_overrides(model)
        args = self.args
        epoch_loss = []
        stage_times = {}
        timer = time.perf_counter

        stage_times["build_trainloader"] = 0.0
        start = timer()
        criteria_pre, criteria_aux = self._build_losses()
        stage_times["build_losses"] = timer() - start
        start = timer()
        train_batches = num_batches(len(self.idxs), self.args.local_bs, drop_last=True)
        learning_rate = self._make_learning_rate(global_round, train_batches)
        optimizer = set_optimizer(model, args, learning_rate=learning_rate)
        stage_times["build_optimizer"] = timer() - start

        reference_model = None
        if args.distill or args.fedprox_mu > 0 or (args.is_proto and args.pseudo_label):
            start = timer()
            reference_model = copy.deepcopy(model)
            reference_model.set_train(False)
            stage_times["clone_reference_model"] = timer() - start
        else:
            stage_times["clone_reference_model"] = 0.0

        criteria_contrast = ContrastLoss(args) if args.is_proto else None
        criteria_distill_pi = CriterionPixelPairSeq(args, temperature=args.temp_dist) if args.distill else None
        criteria_distill_pa = CriterionPixelRegionPair(args) if args.distill else None
        pixel_seq = []

        params = optimizer.parameters

        def forward_fn(images, labels):
            logits, feat_head, logits_aux = self._forward(model, images)
            labels_work = labels
            if args.losstype == "bce":
                raise ValueError("bce is not supported in the MindSpore refactor yet")

            loss_pre = criteria_pre(logits, labels_work)
            loss_aux = [crit(lgt, labels_work) for crit, lgt in zip(criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
            metrics = {
                "loss_ce": loss,
                "loss_con": ms.Tensor(0.0, ms.float32),
                "loss_con_2": ms.Tensor(0.0, ms.float32),
                "loss_pi": ms.Tensor(0.0, ms.float32),
                "loss_pa": ms.Tensor(0.0, ms.float32),
            }

            if args.is_proto and global_round >= args.proto_start_epoch and prototypes is not None and proto_mask is not None:
                _, _, height, width = feat_head.shape
                labels_1 = ops.interpolate(labels_work.unsqueeze(1).astype(ms.float32), size=(height, width), mode="nearest").squeeze(1)
                labels_1 = self._anchor_mask(labels_1.astype(ms.int32), proto_mask)
                loss_con = criteria_contrast(feat_head, labels_1, prototypes, proto_mask)
                metrics["loss_con"] = loss_con
                loss = loss + args.con_lamb * loss_con

                if args.pseudo_label and global_round >= args.pseudo_label_start_epoch and reference_model is not None:
                    logits_t, _, _ = self._forward(reference_model, images)
                    labels_2 = ops.interpolate(logits_t.astype(ms.float32), size=(height, width), mode="bilinear", align_corners=False)
                    labels_2 = ops.softmax(labels_2, axis=1)
                    props, labels_2 = ops.max(labels_2, axis=1)
                    labels_2 = ops.where(props < 0.8, ops.ones_like(labels_2) * 255, labels_2)
                    labels_2 = self._anchor_mask(labels_2.astype(ms.int32), proto_mask)
                    loss_con_2 = criteria_contrast(feat_head, labels_2, prototypes, proto_mask)
                    metrics["loss_con_2"] = loss_con_2
                    loss = loss + args.con_lamb * loss_con_2

            if args.fedprox_mu > 0 and reference_model is not None:
                proximal_term = ms.Tensor(0.0, ms.float32)
                for param, ref_param in zip(model.get_parameters(), reference_model.get_parameters()):
                    proximal_term = proximal_term + ops.norm(param - ref_param, ord=2)
                loss = loss + (args.fedprox_mu / 2.0) * proximal_term

            if args.distill and reference_model is not None:
                logits_t, feat_head_t, _ = self._forward(reference_model, images)
                if args.distill_lamb_pi > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                    loss_pi, pixel_seq_out = criteria_distill_pi(feat_head, ops.stop_gradient(feat_head_t), pixel_seq)
                    pixel_seq.clear()
                    pixel_seq.extend(pixel_seq_out)
                    loss_pi = args.distill_lamb_pi * loss_pi
                    metrics["loss_pi"] = loss_pi
                    loss = loss + loss_pi
                if args.distill_lamb_pa > 0 and args.is_proto and global_round >= args.proto_start_epoch and prototypes is not None and proto_mask is not None:
                    loss_pa = args.distill_lamb_pa * criteria_distill_pa(feat_head, ops.stop_gradient(feat_head_t), prototypes, proto_mask)
                    metrics["loss_pa"] = loss_pa
                    loss = loss + loss_pa

            return loss, metrics

        grad_fn = ms.value_and_grad(forward_fn, None, params, has_aux=True)
        last_metrics = None
        epoch_bar = tqdm(
            range(args.local_ep),
            desc=f"Client {self.client_id} local epochs",
            leave=False,
            dynamic_ncols=True,
        )
        lr_step = 0
        worker_states = None
        for local_epoch in epoch_bar:
            loader_start = timer()
            trainloader = self._build_trainloader(global_round, local_epoch=local_epoch, worker_states=worker_states)
            if worker_states is None and getattr(trainloader, "worker_states", None) is not None:
                worker_states = trainloader.worker_states
            stage_times["build_trainloader"] += timer() - loader_start
            loss_sum = ms.Tensor(0.0, ms.float32)
            batch_count = 0
            batch_bar = tqdm(
                trainloader,
                desc=f"Client {self.client_id} epoch {local_epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )
            data_wait_time = 0.0
            compute_time = 0.0
            iter_end = timer()
            total_batches = len(trainloader)
            for batch_idx, (images, labels) in enumerate(batch_bar):
                batch_start = timer()
                data_wait_time += batch_start - iter_end
                (loss, metrics), grads = grad_fn(images, labels.astype(ms.int32))
                optimizer(grads)
                loss_sum = loss_sum + loss
                batch_count += 1
                last_metrics = metrics

                if self._should_update_progress(batch_idx, total_batches):
                    loss_value = float(loss.asnumpy())
                    batch_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        lr=f"{self._lr_snapshot(learning_rate, lr_step):.2e}",
                    )
                lr_step += 1
                iter_end = timer()
                compute_time += iter_end - batch_start

            epoch_loss_value = float((loss_sum / max(1, batch_count)).asnumpy()) if batch_count > 0 else 0.0
            epoch_loss.append(epoch_loss_value)
            epoch_bar.set_postfix(loss=f"{epoch_loss[-1]:.4f}")
            if args.verbose and batch_count > 0:
                logger.debug(
                    "| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}",
                    global_round,
                    local_epoch + 1,
                    len(self.idxs),
                    epoch_loss_value,
                )
            if self._profile_runtime:
                logger.info(
                    "Runtime profile | client={} round={} epoch={} data_wait={:.3f}s compute={:.3f}s batches={}",
                    self.client_id,
                    global_round,
                    local_epoch + 1,
                    data_wait_time,
                    compute_time,
                    batch_count,
                )

        last_loss = epoch_loss[-1] if epoch_loss else 0.0
        logger.info(
            "| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}",
            global_round,
            args.local_ep,
            len(self.idxs),
            last_loss,
        )
        if last_metrics is not None and args.distill:
            logger.info(
                "Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}",
                float(last_metrics["loss_ce"].asnumpy()),
                float(last_metrics["loss_pi"].asnumpy()),
                float(last_metrics["loss_pa"].asnumpy()),
            )
        if last_metrics is not None and args.is_proto:
            if global_round >= args.proto_start_epoch:
                if args.pseudo_label:
                    logger.info(
                        "Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}",
                        float(last_metrics["loss_ce"].asnumpy()),
                        float(last_metrics["loss_con"].asnumpy()),
                        float(last_metrics["loss_con_2"].asnumpy()),
                    )
                else:
                    logger.info(
                        "Loss_CE:{:.6f} | loss_contrast:{:.6f}",
                        float(last_metrics["loss_ce"].asnumpy()),
                        float(last_metrics["loss_con"].asnumpy()),
                    )
            else:
                logger.info("Loss_CE:{:.6f}", float(last_metrics["loss_ce"].asnumpy()))

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

        state_dict = {name: param.data.clone() for name, param in model.parameters_and_names()}
        mean_epoch_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        return state_dict, mean_epoch_loss

    def inference(self, model):
        confmat = evaluate(model, self._build_testloader(), self.args.num_classes)
        return confmat.acc_global, confmat.iou_mean, str(confmat)

    evaluate = inference
    update_weights = train


class ReusableLocalTrainer:
    def __init__(self, args, train_model, reference_model):
        self.args = args
        self.train_model = train_model
        self.reference_model = reference_model
        self._profile_runtime = bool(getattr(args, "profile_runtime", False))
        self._profile_runtime_detail = bool(getattr(args, "profile_runtime_detail", False))
        self._profile_runtime_sync = bool(getattr(args, "profile_runtime_sync", False))
        self._debug_log_memory = bool(getattr(args, "debug_log_memory", False))
        self._debug_empty_cache_per_client = bool(getattr(args, "debug_empty_cache_per_client", False))

        self.criteria_pre, self.criteria_aux = self._build_losses()
        self.criteria_contrast = ContrastLoss(args) if args.is_proto else None
        self.criteria_distill_pi = CriterionPixelPairSeq(args, temperature=args.temp_dist) if args.distill else None
        self.criteria_distill_pa = CriterionPixelRegionPair(args) if args.distill else None
        self.graph_plain_loss_net = None
        self.graph_plain_train_step = None
        self.graph_proto_loss_nets = {}
        self.graph_proto_train_steps = {}
        if self._supports_graph_plain_training():
            self.graph_plain_loss_net = _PlainSegLossCell(self.train_model, self.criteria_pre, self.criteria_aux)

        self.current_global_round = 0
        self.current_prototypes = None
        self.current_proto_mask = None
        self.pixel_seq = []
        self.learning_rate = None
        self.optimizer = None
        self._reference_state_token = None
        self._optimizer_mode = None
        self._optimizer_reuse_logged = False
        self._invalid_step_count = 0
        self._low_conf = ms.Tensor(0.8, ms.float32)
        self._active_forward_timing = None
        self.last_runtime_detail = {}

        self.grad_params = ()
        self.grad_fn = None

    def _new_runtime_detail(self):
        return {
            "forward_total_s": 0.0,
            "forward_model_s": 0.0,
            "forward_seg_loss_s": 0.0,
            "forward_proto_target_s": 0.0,
            "forward_contrast_s": 0.0,
            "forward_teacher_forward_s": 0.0,
            "forward_pseudo_target_s": 0.0,
            "forward_pseudo_contrast_s": 0.0,
            "grad_fn_s": 0.0,
            "finite_check_s": 0.0,
            "optimizer_step_s": 0.0,
        }

    def _accumulate_runtime_detail(self, total, batch_detail):
        for key, value in batch_detail.items():
            total[key] = total.get(key, 0.0) + float(value)

    def _finalize_runtime_detail(self, total, batch_count):
        if batch_count <= 0:
            self.last_runtime_detail = {}
            return
        averaged = {key: float(value) / float(batch_count) for key, value in total.items()}
        averaged["profiled_batches"] = int(batch_count)
        averaged["backward_residual_s"] = max(0.0, averaged["grad_fn_s"] - averaged["forward_total_s"])
        self.last_runtime_detail = averaged

    def _format_runtime_detail(self):
        if not self.last_runtime_detail:
            return ""
        detail = self.last_runtime_detail
        return (
            "model_fwd={:.3f}s seg_loss={:.3f}s proto_target={:.3f}s contrast={:.3f}s "
            "teacher_fwd={:.3f}s pseudo_target={:.3f}s pseudo_contrast={:.3f}s "
            "forward_total={:.3f}s grad_fn={:.3f}s finite_check={:.3f}s backward_residual={:.3f}s opt_step={:.3f}s batches={}"
        ).format(
            detail.get("forward_model_s", 0.0),
            detail.get("forward_seg_loss_s", 0.0),
            detail.get("forward_proto_target_s", 0.0),
            detail.get("forward_contrast_s", 0.0),
            detail.get("forward_teacher_forward_s", 0.0),
            detail.get("forward_pseudo_target_s", 0.0),
            detail.get("forward_pseudo_contrast_s", 0.0),
            detail.get("forward_total_s", 0.0),
            detail.get("grad_fn_s", 0.0),
            detail.get("finite_check_s", 0.0),
            detail.get("backward_residual_s", 0.0),
            detail.get("optimizer_step_s", 0.0),
            detail.get("profiled_batches", 0),
        )

    def _sync_device(self):
        if not self._profile_runtime_sync:
            return
        runtime_api = getattr(ms, "runtime", None)
        hal_api = getattr(ms, "hal", None)
        sync_fn = getattr(runtime_api, "synchronize", None)
        if sync_fn is None:
            sync_fn = getattr(hal_api, "synchronize", None)
        if sync_fn is None:
            return
        try:
            sync_fn()
        except Exception:
            return

    def _ordered_train_params(self):
        if hasattr(self.train_model, "get_params"):
            wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = self.train_model.get_params()
            return tuple(list(wd_params) + list(nowd_params) + list(lr_mul_wd_params) + list(lr_mul_nowd_params))

        wd_params, non_wd_params = [], []
        for _, param in self.train_model.parameters_and_names():
            if param.ndim == 1:
                non_wd_params.append(param)
            elif param.ndim in {2, 4}:
                wd_params.append(param)
        return tuple(wd_params + non_wd_params)

    def _active_grad_params(self):
        if self.optimizer is not None and hasattr(self.optimizer, "parameters"):
            try:
                return tuple(self.optimizer.parameters)
            except TypeError:
                return tuple(list(self.optimizer.parameters))
        return self._ordered_train_params()

    def _rebuild_grad_fn(self):
        self.grad_params = self._active_grad_params()
        self.grad_fn = ms.value_and_grad(self.forward_fn, None, self.grad_params, has_aux=True)

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
        else:
            raise ValueError("loss type is not defined for the MindSpore refactor")
        return criteria_pre, criteria_aux

    def _make_learning_rate(self, global_round, num_train_batches):
        if self.args.lr_scheduler == "step":
            lr_value = self.args.lr if global_round < 1000 else self.args.lr * 0.1
            return ms.Tensor(lr_value, ms.float32)
        total_step = max(1, num_train_batches * max(1, self.args.local_ep))
        lr_values = [
            self.args.lr * ((1 - step / total_step) ** 0.9)
            for step in range(total_step)
        ]
        return ms.Tensor(lr_values, ms.float32)

    def _forward(self, model, images):
        if self.args.model != "bisenetv2":
            raise ValueError("unrecognized model")
        logits, feat_head, *logits_aux = model(images)
        return logits, feat_head, logits_aux

    def _apply_debug_train_overrides(self, model):
        if not getattr(self.args, "debug_freeze_bn_stats", False):
            return
        for _, cell in model.cells_and_names():
            if isinstance(cell, nn.BatchNorm2d):
                cell.use_batch_statistics = False

    def _lr_snapshot(self, learning_rate, lr_step):
        if hasattr(learning_rate, "asnumpy"):
            lr_np = learning_rate.asnumpy()
            if getattr(lr_np, "size", 0) == 0:
                return 0.0
            lr_values = np.asarray(lr_np).reshape(-1)
            return float(lr_values[min(int(lr_step), len(lr_values) - 1)])
        try:
            return float(learning_rate)
        except Exception:
            return 0.0

    def _anchor_mask(self, labels, proto_mask):
        if self.args.kmean_num > 0:
            available_mask = proto_mask.sum(axis=1) > 0
        else:
            available_mask = proto_mask > 0
        return _apply_available_class_mask(labels, available_mask, ignore_value=255)

    def _needs_reference_model(self):
        args = self.args
        return args.distill or args.fedprox_mu > 0 or (args.is_proto and args.pseudo_label)

    def _is_graph_mode(self):
        try:
            return ms.get_context("mode") == ms.GRAPH_MODE
        except Exception:
            return False

    def _supports_graph_plain_training(self):
        return self._is_graph_mode() and (not self.args.is_proto) and (not self.args.distill) and self.args.fedprox_mu <= 0

    def _supports_graph_proto_training(self, global_round, prototypes, proto_mask):
        return (
            self._is_graph_mode()
            and self.args.is_proto
            and (not self.args.distill)
            and self.args.fedprox_mu <= 0
            and global_round >= self.args.proto_start_epoch
            and prototypes is not None
            and proto_mask is not None
        )

    def _optimizer_param_dict(self):
        if self.optimizer is None:
            return {}
        try:
            return self.optimizer.parameters_dict()
        except Exception:
            return {}

    def _reset_optimizer_state(self):
        if self.optimizer is None:
            return
        param_dict = self._optimizer_param_dict()
        seen = set()

        def _set_slot(param, value):
            if param is None or not hasattr(param, "set_data"):
                return
            param_id = id(param)
            if param_id in seen:
                return
            seen.add(param_id)
            param.set_data(value)

        global_step = param_dict.get("global_step")
        _set_slot(global_step, ops.zeros_like(global_step))

        accum_state = getattr(self.optimizer, "accum", None)
        if accum_state is not None:
            for param in accum_state:
                _set_slot(param, ops.zeros_like(param))

        for name, param in param_dict.items():
            if name.startswith("accum."):
                _set_slot(param, ops.zeros_like(param))
            elif name.startswith("stat."):
                # A freshly built MindSpore SGD initializes these slots to 1.
                # When we reuse one optimizer across clients, restoring that
                # exact initial value preserves the same update rule as creating
                # a brand-new optimizer for each local client.
                _set_slot(param, ops.ones_like(param))

    def _set_optimizer_learning_rate(self, learning_rate):
        if self.optimizer is None:
            return
        base_lr = learning_rate if isinstance(learning_rate, ms.Tensor) else ms.Tensor(learning_rate, ms.float32)
        high_lr = base_lr * 10

        base_lr_np = None
        try:
            base_lr_np = np.asarray(base_lr.asnumpy())
        except Exception:
            base_lr_np = None

        if base_lr_np is not None and base_lr_np.ndim == 0:
            updated = False
            seen = set()
            for lr_param in getattr(self.optimizer, "learning_rate", ()):
                if id(lr_param) in seen or not hasattr(lr_param, "set_data"):
                    continue
                seen.add(id(lr_param))
                updated = True
                name = getattr(lr_param, "name", "")
                if name in {"learning_rate_group_2", "learning_rate_group_3"}:
                    lr_param.set_data(high_lr)
                else:
                    lr_param.set_data(base_lr)
            if updated:
                return

        param_dict = self._optimizer_param_dict()
        lr_param = param_dict.get("learning_rate")
        if lr_param is not None and hasattr(lr_param, "set_data"):
            lr_param.set_data(base_lr)
        if "learning_rate_group_2" in param_dict:
            param_dict["learning_rate_group_2"].set_data(high_lr)
        if "learning_rate_group_3" in param_dict:
            param_dict["learning_rate_group_3"].set_data(high_lr)

    def _should_update_progress(self, batch_idx, total_batches):
        if total_batches <= 0:
            return False
        if self._profile_runtime or self.args.verbose:
            return True
        if self._is_graph_mode():
            return batch_idx + 1 == total_batches
        stride = max(1, total_batches // 4)
        return ((batch_idx + 1) % stride == 0) or (batch_idx + 1 == total_batches)

    def _tensor_is_finite(self, tensor):
        return self._tensors_are_finite((tensor,))

    def _tensors_are_finite(self, tensors):
        if not tensors:
            return True
        try:
            return bool(amp.all_finite(tuple(tensors)).asnumpy())
        except Exception:
            finite_flags = [ops.all(ops.isfinite(tensor)) for tensor in tensors]
            return bool(ops.all(ops.stack(finite_flags)).asnumpy())

    def _sanitize_tensor(self, tensor):
        finite_mask = ops.isfinite(tensor)
        return ops.where(finite_mask, tensor, ops.zeros_like(tensor))

    def _sanitize_prototypes(self, prototypes, proto_mask):
        if prototypes is None or proto_mask is None:
            return prototypes, proto_mask
        prototypes = ops.stop_gradient(self._sanitize_tensor(prototypes))
        proto_mask = ops.stop_gradient(proto_mask.astype(ms.float32))
        if self.args.kmean_num > 0:
            prototypes = prototypes * proto_mask.unsqueeze(-1).astype(prototypes.dtype)
        else:
            prototypes = prototypes * proto_mask.reshape(proto_mask.shape[0], 1).astype(prototypes.dtype)
        return prototypes, proto_mask

    def _memory_snapshot(self):
        runtime_api = getattr(ms, "runtime", None)
        hal_api = getattr(ms, "hal", None)
        memory_api = runtime_api if runtime_api is not None else hal_api
        if not self._debug_log_memory or memory_api is None:
            return None
        try:
            return {
                "allocated_mb": float(memory_api.memory_allocated() / (1024 ** 2)),
                "reserved_mb": float(memory_api.memory_reserved() / (1024 ** 2)),
                "peak_allocated_mb": float(memory_api.max_memory_allocated() / (1024 ** 2)),
                "peak_reserved_mb": float(memory_api.max_memory_reserved() / (1024 ** 2)),
            }
        except Exception:
            return None

    def _reset_peak_memory_stats(self):
        runtime_api = getattr(ms, "runtime", None)
        hal_api = getattr(ms, "hal", None)
        memory_api = runtime_api if runtime_api is not None else hal_api
        if not self._debug_log_memory or memory_api is None:
            return
        try:
            memory_api.reset_peak_memory_stats()
        except Exception:
            try:
                if hasattr(memory_api, "reset_max_memory_allocated"):
                    memory_api.reset_max_memory_allocated()
                if hasattr(memory_api, "reset_max_memory_reserved"):
                    memory_api.reset_max_memory_reserved()
            except Exception:
                return

    def _empty_device_cache(self):
        hal_api = getattr(ms, "hal", None)
        if not self._debug_empty_cache_per_client or hal_api is None:
            return
        try:
            hal_api.empty_cache()
        except Exception:
            return

    def _log_invalid_step(self, client_id, global_round, batch_idx, reason, metrics=None):
        self._invalid_step_count += 1
        if metrics is None:
            metrics = {}
        metric_pairs = []
        for key, value in metrics.items():
            try:
                metric_pairs.append("{}={:.6f}".format(key, float(value.asnumpy())))
            except Exception:
                continue
        logger.warning(
            "Skip invalid local step #{}, client={}, round={}, batch={}, reason={}{}",
            self._invalid_step_count,
            client_id,
            global_round,
            batch_idx,
            reason,
            ", metrics=" + ", ".join(metric_pairs) if metric_pairs else "",
        )

    def _ensure_optimizer(self, global_round, num_train_batches):
        learning_rate = self._make_learning_rate(global_round, num_train_batches)
        if not self._is_graph_mode():
            if self.optimizer is None or self._optimizer_mode != "pynative_reuse":
                self.learning_rate = learning_rate
                self.optimizer = set_optimizer(self.train_model, self.args, learning_rate=self.learning_rate)
                self._optimizer_mode = "pynative_reuse"
            else:
                self.learning_rate = learning_rate
                self._set_optimizer_learning_rate(self.learning_rate)
                self._reset_optimizer_state()
            if not self._optimizer_reuse_logged:
                logger.info("ReusableLocalTrainer: reusing a single MindSpore optimizer in pynative mode to avoid per-client optimizer allocation growth")
                self._optimizer_reuse_logged = True
            return

        if self._is_graph_mode() and self.args.lr_scheduler == "step":
            if self.optimizer is None or self._optimizer_mode != "step_graph_reuse":
                self.learning_rate = learning_rate
                self.optimizer = set_optimizer(self.train_model, self.args, learning_rate=self.learning_rate)
                self._optimizer_mode = "step_graph_reuse"
                if self._supports_graph_plain_training():
                    self.graph_plain_train_step = nn.TrainOneStepCell(self.graph_plain_loss_net, self.optimizer)
                    self.graph_plain_train_step.set_train(True)
                self.graph_proto_train_steps = {}
            else:
                self.learning_rate = learning_rate
                if hasattr(self.optimizer, "learning_rate"):
                    for lr_param in self.optimizer.learning_rate:
                        lr_param.set_data(learning_rate)
                self._reset_optimizer_state()
            return

        self.learning_rate = learning_rate
        self.optimizer = set_optimizer(self.train_model, self.args, learning_rate=self.learning_rate)
        self._optimizer_mode = "fresh"

    def _ensure_graph_proto_train_step(self, global_round):
        use_pseudo_label = bool(self.args.pseudo_label and global_round >= self.args.pseudo_label_start_epoch)
        if use_pseudo_label not in self.graph_proto_loss_nets:
            loss_net = _ProtoSegLossCell(
                self.args,
                self.train_model,
                self.reference_model,
                self.criteria_pre,
                self.criteria_aux,
                self.criteria_contrast,
                use_pseudo_label=use_pseudo_label,
            )
            self.graph_proto_loss_nets[use_pseudo_label] = loss_net
            train_step = nn.TrainOneStepCell(loss_net, self.optimizer)
            train_step.set_train(True)
            self.graph_proto_train_steps[use_pseudo_label] = train_step
        return self.graph_proto_train_steps[use_pseudo_label]

    def prepare_reference_model(self, state_dict):
        load_state_into_net(self.reference_model, state_dict, strict=False)
        self.reference_model.set_train(False)
        self._reference_state_token = id(state_dict)

    def _ensure_reference_model(self, state_dict):
        if not self._needs_reference_model():
            return 0.0
        if self._reference_state_token == id(state_dict):
            return 0.0
        start = time.perf_counter()
        self.prepare_reference_model(state_dict)
        return time.perf_counter() - start

    def forward_fn(self, images, labels):
        args = self.args
        detail = self._active_forward_timing if self._profile_runtime_detail else None
        forward_total_start = time.perf_counter() if detail is not None else None

        stage_start = time.perf_counter() if detail is not None else None
        logits, feat_head, logits_aux = self._forward(self.train_model, images)
        if detail is not None:
            self._sync_device()
            detail["forward_model_s"] += time.perf_counter() - stage_start
        labels_work = labels
        if args.losstype == "bce":
            raise ValueError("bce is not supported in the MindSpore refactor yet")

        stage_start = time.perf_counter() if detail is not None else None
        loss_pre = self.criteria_pre(logits, labels_work)
        loss_aux = [crit(lgt, labels_work) for crit, lgt in zip(self.criteria_aux, logits_aux)]
        if detail is not None:
            self._sync_device()
            detail["forward_seg_loss_s"] += time.perf_counter() - stage_start
        loss = loss_pre + sum(loss_aux)
        metrics = {
            "loss_ce": loss,
            "loss_con": ms.Tensor(0.0, ms.float32),
            "loss_con_2": ms.Tensor(0.0, ms.float32),
            "loss_pi": ms.Tensor(0.0, ms.float32),
            "loss_pa": ms.Tensor(0.0, ms.float32),
        }

        if (
            args.is_proto
            and self.current_global_round >= args.proto_start_epoch
            and self.current_prototypes is not None
            and self.current_proto_mask is not None
        ):
            _, _, height, width = feat_head.shape
            stage_start = time.perf_counter() if detail is not None else None
            labels_1 = ops.interpolate(
                labels_work.unsqueeze(1).astype(ms.float32),
                size=(height, width),
                mode="nearest",
            ).squeeze(1)
            labels_1 = self._anchor_mask(labels_1.astype(ms.int32), self.current_proto_mask)
            if detail is not None:
                self._sync_device()
                detail["forward_proto_target_s"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter() if detail is not None else None
            loss_con = self.criteria_contrast(feat_head, labels_1, self.current_prototypes, self.current_proto_mask)
            if detail is not None:
                self._sync_device()
                detail["forward_contrast_s"] += time.perf_counter() - stage_start
            metrics["loss_con"] = loss_con
            loss = loss + args.con_lamb * loss_con

            if args.pseudo_label and self._needs_reference_model():
                stage_start = time.perf_counter() if detail is not None else None
                logits_t, _, _ = self._forward(self.reference_model, images)
                logits_t = ops.stop_gradient(self._sanitize_tensor(logits_t))
                if detail is not None:
                    self._sync_device()
                    detail["forward_teacher_forward_s"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter() if detail is not None else None
                labels_2 = ops.interpolate(
                    logits_t.astype(ms.float32),
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
                labels_2 = self._sanitize_tensor(labels_2)
                labels_2 = ops.softmax(labels_2, axis=1)
                props, labels_2 = ops.max(labels_2, axis=1)
                labels_2 = ops.where(props < self._low_conf, ops.ones_like(labels_2) * 255, labels_2)
                labels_2 = self._anchor_mask(labels_2.astype(ms.int32), self.current_proto_mask)
                if detail is not None:
                    self._sync_device()
                    detail["forward_pseudo_target_s"] += time.perf_counter() - stage_start

                stage_start = time.perf_counter() if detail is not None else None
                loss_con_2 = self.criteria_contrast(feat_head, labels_2, self.current_prototypes, self.current_proto_mask)
                if detail is not None:
                    self._sync_device()
                    detail["forward_pseudo_contrast_s"] += time.perf_counter() - stage_start
                metrics["loss_con_2"] = loss_con_2
                if self._tensor_is_finite(loss_con_2):
                    loss = loss + args.con_lamb * loss_con_2
                else:
                    metrics["loss_con_2"] = ms.Tensor(0.0, ms.float32)

        if args.fedprox_mu > 0 and self._needs_reference_model():
            proximal_term = ms.Tensor(0.0, ms.float32)
            for param, ref_param in zip(self.train_model.get_parameters(), self.reference_model.get_parameters()):
                proximal_term = proximal_term + ops.norm(param - ref_param, ord=2)
            loss = loss + (args.fedprox_mu / 2.0) * proximal_term

        if args.distill and self._needs_reference_model():
            logits_t, feat_head_t, _ = self._forward(self.reference_model, images)
            logits_t = ops.stop_gradient(self._sanitize_tensor(logits_t))
            feat_head_t = ops.stop_gradient(self._sanitize_tensor(feat_head_t))
            if args.distill_lamb_pi > 0 and args.is_proto and self.current_global_round >= args.proto_start_epoch:
                loss_pi, pixel_seq_out = self.criteria_distill_pi(feat_head, ops.stop_gradient(feat_head_t), self.pixel_seq)
                self.pixel_seq.clear()
                self.pixel_seq.extend(pixel_seq_out)
                loss_pi = args.distill_lamb_pi * loss_pi
                metrics["loss_pi"] = loss_pi
                loss = loss + loss_pi
            if (
                args.distill_lamb_pa > 0
                and args.is_proto
                and self.current_global_round >= args.proto_start_epoch
                and self.current_prototypes is not None
                and self.current_proto_mask is not None
            ):
                loss_pa = args.distill_lamb_pa * self.criteria_distill_pa(
                    feat_head,
                    ops.stop_gradient(feat_head_t),
                    self.current_prototypes,
                    self.current_proto_mask,
                )
                metrics["loss_pa"] = loss_pa
                loss = loss + loss_pa

        if detail is not None:
            self._sync_device()
            detail["forward_total_s"] += time.perf_counter() - forward_total_start

        return loss, metrics

    def train(self, client, model_state, global_round, prototypes=None, proto_mask=None):
        args = self.args
        epoch_loss = []
        stage_times = {}
        timer = time.perf_counter

        start = timer()
        load_state_into_net(self.train_model, model_state, strict=False)
        self._sync_device()
        self.train_model.set_train(True)
        self._apply_debug_train_overrides(self.train_model)
        stage_times["load_local_model"] = timer() - start

        stage_times["build_trainloader"] = 0.0

        start = timer()
        train_batches = num_batches(client.num_samples, args.local_bs, drop_last=True)
        self._ensure_optimizer(global_round, train_batches)
        self._rebuild_grad_fn()
        self._sync_device()
        stage_times["build_optimizer"] = timer() - start

        stage_times["load_reference_model"] = self._ensure_reference_model(model_state)
        self._sync_device()

        self.current_global_round = global_round
        prototypes, proto_mask = self._sanitize_prototypes(prototypes, proto_mask)
        self.current_prototypes = prototypes
        self.current_proto_mask = proto_mask
        self.pixel_seq.clear()
        self._reset_peak_memory_stats()
        self.last_runtime_detail = {}
        graph_proto_train_step = None
        if self._supports_graph_proto_training(global_round, prototypes, proto_mask):
            graph_proto_train_step = self._ensure_graph_proto_train_step(global_round)

        last_metrics = None
        metrics_available = False
        epoch_bar = tqdm(
            range(args.local_ep),
            desc=f"Client {client.client_id} local epochs",
            leave=False,
            dynamic_ncols=True,
        )
        lr_step = 0
        total_valid_batches = 0
        runtime_detail_totals = self._new_runtime_detail() if self._profile_runtime_detail else None
        runtime_detail_batches = 0
        worker_states = None
        for local_epoch in epoch_bar:
            loader_start = timer()
            trainloader = client._build_trainloader(
                global_round,
                local_epoch=local_epoch,
                worker_states=worker_states,
            )
            if worker_states is None and getattr(trainloader, "worker_states", None) is not None:
                worker_states = trainloader.worker_states
            stage_times["build_trainloader"] += timer() - loader_start
            loss_sum = ms.Tensor(0.0, ms.float32)
            batch_count = 0
            skipped_batches = 0
            batch_bar = tqdm(
                trainloader,
                desc=f"Client {client.client_id} epoch {local_epoch + 1}",
                leave=False,
                dynamic_ncols=True,
            )
            data_wait_time = 0.0
            compute_time = 0.0
            iter_end = timer()
            total_batches = len(trainloader)
            for batch_idx, (images, labels) in enumerate(batch_bar):
                self._sync_device()
                batch_start = timer()
                data_wait_time += batch_start - iter_end
                if self._supports_graph_plain_training():
                    loss = self.graph_plain_train_step(images, labels.astype(ms.int32))
                    loss_sum = loss_sum + loss
                    batch_count += 1
                    total_valid_batches += 1
                    last_metrics = {
                        "loss_ce": loss,
                        "loss_con": ms.Tensor(0.0, ms.float32),
                        "loss_con_2": ms.Tensor(0.0, ms.float32),
                        "loss_pi": ms.Tensor(0.0, ms.float32),
                        "loss_pa": ms.Tensor(0.0, ms.float32),
                    }
                    metrics_available = False
                elif graph_proto_train_step is not None:
                    loss = graph_proto_train_step(
                        images,
                        labels.astype(ms.int32),
                        prototypes,
                        proto_mask,
                    )
                    loss_sum = loss_sum + loss
                    batch_count += 1
                    total_valid_batches += 1
                    last_metrics = {
                        "loss_ce": loss,
                        "loss_con": ms.Tensor(0.0, ms.float32),
                        "loss_con_2": ms.Tensor(0.0, ms.float32),
                        "loss_pi": ms.Tensor(0.0, ms.float32),
                        "loss_pa": ms.Tensor(0.0, ms.float32),
                    }
                    metrics_available = False
                else:
                    batch_detail = self._new_runtime_detail() if self._profile_runtime_detail else None
                    if batch_detail is not None:
                        self._active_forward_timing = batch_detail
                    grad_start = timer()
                    (loss, metrics), grads = self.grad_fn(images, labels.astype(ms.int32))
                    self._sync_device()
                    grad_time = timer() - grad_start
                    if batch_detail is not None:
                        batch_detail["grad_fn_s"] += grad_time
                        self._active_forward_timing = None
                    if getattr(self.args, "check_finite_per_batch", True):
                        finite_start = timer() if batch_detail is not None else None
                        grad_is_finite = self._tensors_are_finite(grads)
                        loss_is_finite = self._tensor_is_finite(loss)
                        metric_is_finite = self._tensors_are_finite(list(metrics.values()))
                        if batch_detail is not None:
                            self._sync_device()
                            batch_detail["finite_check_s"] += timer() - finite_start
                        if not loss_is_finite or not grad_is_finite or not metric_is_finite:
                            skipped_batches += 1
                            self._log_invalid_step(
                                client.client_id,
                                global_round,
                                batch_idx,
                                reason="non-finite loss/grad/metric",
                                metrics=metrics,
                            )
                            last_metrics = metrics
                            metrics_available = True
                            self._sync_device()
                            iter_end = timer()
                            compute_time += iter_end - batch_start
                            continue
                    opt_start = timer()
                    self.optimizer(grads)
                    self._sync_device()
                    if batch_detail is not None:
                        batch_detail["optimizer_step_s"] += timer() - opt_start
                        self._accumulate_runtime_detail(runtime_detail_totals, batch_detail)
                        runtime_detail_batches += 1
                    loss_sum = loss_sum + loss
                    batch_count += 1
                    total_valid_batches += 1
                    last_metrics = metrics
                    metrics_available = True

                if self._should_update_progress(batch_idx, total_batches):
                    loss_value = float(loss.asnumpy())
                    batch_bar.set_postfix(
                        loss=f"{loss_value:.4f}",
                        lr=f"{self._lr_snapshot(self.learning_rate, lr_step):.2e}",
                    )
                lr_step += 1
                self._sync_device()
                iter_end = timer()
                compute_time += iter_end - batch_start

            epoch_loss_value = float((loss_sum / max(1, batch_count)).asnumpy()) if batch_count > 0 else 0.0
            epoch_loss.append(epoch_loss_value)
            epoch_bar.set_postfix(loss=f"{epoch_loss[-1]:.4f}")
            if args.verbose and batch_count > 0:
                logger.debug(
                    "| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}",
                    global_round,
                    local_epoch + 1,
                    client.num_samples,
                    epoch_loss_value,
                )
            if self._profile_runtime:
                logger.info(
                    "Runtime profile | client={} round={} epoch={} data_wait={:.3f}s compute={:.3f}s batches={}",
                    client.client_id,
                    global_round,
                    local_epoch + 1,
                    data_wait_time,
                    compute_time,
                    batch_count,
                )
            if skipped_batches > 0:
                logger.warning(
                    "Client {} round {} epoch {} skipped {} invalid batches",
                    client.client_id,
                    global_round,
                    local_epoch + 1,
                    skipped_batches,
                )

        last_loss = epoch_loss[-1] if epoch_loss else 0.0
        logger.info(
            "| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}",
            global_round,
            args.local_ep,
            client.num_samples,
            last_loss,
        )
        if last_metrics is not None and metrics_available and args.distill:
            logger.info(
                "Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}",
                float(last_metrics["loss_ce"].asnumpy()),
                float(last_metrics["loss_pi"].asnumpy()),
                float(last_metrics["loss_pa"].asnumpy()),
            )
        if last_metrics is not None and metrics_available and args.is_proto:
            if global_round >= args.proto_start_epoch:
                if args.pseudo_label:
                    logger.info(
                        "Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}",
                        float(last_metrics["loss_ce"].asnumpy()),
                        float(last_metrics["loss_con"].asnumpy()),
                        float(last_metrics["loss_con_2"].asnumpy()),
                    )
                else:
                    logger.info(
                        "Loss_CE:{:.6f} | loss_contrast:{:.6f}",
                        float(last_metrics["loss_ce"].asnumpy()),
                        float(last_metrics["loss_con"].asnumpy()),
                    )
            else:
                logger.info("Loss_CE:{:.6f}", float(last_metrics["loss_ce"].asnumpy()))

        if self._profile_runtime:
            logger.info(
                "Runtime profile | client={} round={} state_load={:.3f}s loader={:.3f}s optimizer={:.3f}s reference={:.3f}s",
                client.client_id,
                global_round,
                stage_times.get("load_local_model", 0.0),
                stage_times.get("build_trainloader", 0.0),
                stage_times.get("build_optimizer", 0.0),
                stage_times.get("load_reference_model", 0.0),
            )
        if self._profile_runtime_detail:
            self._finalize_runtime_detail(runtime_detail_totals, runtime_detail_batches)
            if self.last_runtime_detail:
                logger.info(
                    "Runtime detail | client={} round={} {}",
                    client.client_id,
                    global_round,
                    self._format_runtime_detail(),
                )

        mem_stats = self._memory_snapshot()
        if mem_stats is not None:
            logger.info(
                "Memory profile | client={} round={} alloc={:.1f}MB reserved={:.1f}MB peak_alloc={:.1f}MB peak_reserved={:.1f}MB",
                client.client_id,
                global_round,
                mem_stats["allocated_mb"],
                mem_stats["reserved_mb"],
                mem_stats["peak_allocated_mb"],
                mem_stats["peak_reserved_mb"],
            )

        self.current_prototypes = None
        self.current_proto_mask = None
        self.pixel_seq.clear()
        self._empty_device_cache()

        mean_epoch_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else 0.0
        if total_valid_batches == 0:
            mean_epoch_loss = float("nan")
        return clone_state_dict(self.train_model.parameters_dict(), host=True), mean_epoch_loss


class _PlainSegLossCell(nn.Cell):
    def __init__(self, model, criteria_pre, criteria_aux):
        super().__init__()
        self.model = model
        self.criteria_pre = criteria_pre
        self.criteria_aux = nn.CellList(criteria_aux)

    def construct(self, images, labels):
        logits, _, *logits_aux = self.model(images)
        loss_pre = self.criteria_pre(logits, labels)
        loss_aux = ms.Tensor(0.0, ms.float32)
        for crit, lgt in zip(self.criteria_aux, logits_aux):
            loss_aux = loss_aux + crit(lgt, labels)
        return loss_pre + loss_aux


class _ProtoSegLossCell(nn.Cell):
    def __init__(self, args, model, reference_model, criteria_pre, criteria_aux, criteria_contrast, use_pseudo_label):
        super().__init__()
        self.args = args
        self.model = model
        self.reference_model = reference_model
        self.criteria_pre = criteria_pre
        self.criteria_aux = nn.CellList(criteria_aux)
        self.criteria_contrast = criteria_contrast
        self.use_pseudo_label = bool(use_pseudo_label)
        self.num_classes = int(args.num_classes)
        self.zero_i32 = ms.Tensor(0, ms.int32)
        self.max_class_i32 = ms.Tensor(self.num_classes - 1, ms.int32)
        self.ignore_fill = ms.Tensor(255, ms.int32)
        self.low_conf = ms.Tensor(0.8, ms.float32)
        self.zero_f32 = ms.Tensor(0.0, ms.float32)

    def _finite_or_zero(self, tensor):
        return ops.where(ops.isfinite(tensor), tensor, ops.zeros_like(tensor))

    def _anchor_mask(self, labels, proto_mask):
        if self.args.kmean_num > 0:
            available_mask = proto_mask.sum(axis=1) > 0
        else:
            available_mask = proto_mask > 0
        return _apply_available_class_mask(labels, available_mask, ignore_value=255)

    def construct(self, images, labels, prototypes, proto_mask):
        logits, feat_head, *logits_aux = self.model(images)
        loss_pre = self.criteria_pre(logits, labels)
        loss_aux = self.zero_f32
        for crit, lgt in zip(self.criteria_aux, logits_aux):
            loss_aux = loss_aux + crit(lgt, labels)
        loss = loss_pre + loss_aux

        _, _, height, width = feat_head.shape
        labels_1 = ops.interpolate(
            labels.unsqueeze(1).astype(ms.float32),
            size=(height, width),
            mode="nearest",
        ).squeeze(1)
        labels_1 = self._anchor_mask(labels_1.astype(ms.int32), proto_mask)
        loss_con = self._finite_or_zero(self.criteria_contrast(feat_head, labels_1, prototypes, proto_mask))
        loss = loss + self.args.con_lamb * loss_con

        if self.use_pseudo_label:
            logits_t, _, _ = self.reference_model(images)
            logits_t = self._finite_or_zero(logits_t.astype(ms.float32))
            labels_2 = ops.interpolate(
                logits_t,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            labels_2 = ops.softmax(labels_2, axis=1)
            props, labels_2 = ops.max(labels_2, axis=1)
            labels_2 = ops.where(props < self.low_conf, ops.ones_like(labels_2) * 255, labels_2)
            labels_2 = self._anchor_mask(labels_2.astype(ms.int32), proto_mask)
            loss_con_2 = self._finite_or_zero(self.criteria_contrast(feat_head, labels_2, prototypes, proto_mask))
            loss = loss + self.args.con_lamb * loss_con_2

        return loss


def test_inference(args, model, testloader):
    confmat = evaluate(model, testloader, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)

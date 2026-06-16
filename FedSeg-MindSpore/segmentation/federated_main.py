import os
import time
import warnings
import json

import mindspore as ms
import mindspore.ops as ops
import numpy as np
from tqdm import tqdm

from checkpoint_utils import (
    clone_state_dict,
    copy_training_checkpoint,
    load_state_into_net,
    load_training_checkpoint,
    save_training_checkpoint,
)
from client import Client, ReusableLocalTrainer
from fast_eval import evaluate_grouped_dataset, format_runtime_detail, format_runtime_profile
from logging_utils import logger, setup_logger
from mindspore_runtime import setup_mindspore_device
from myseg.bisenet_utils import set_model_bisenetv2
from myseg.datasplit import get_dataset_ade20k, get_dataset_camvid, get_dataset_cityscapes, get_test_dataset
from options import args_parser
from seed_utils import seed_everything
from utils import EMA, average_weights, exp_details, weighted_average_weights

warnings.filterwarnings("ignore")

L2_NORM_EPS = 1e-12


def fedseg_l2_normalize(axis):
    return ops.L2Normalize(axis=axis, epsilon=L2_NORM_EPS)


def _to_ms_tensor(value):
    if isinstance(value, ms.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return ms.Tensor(value)
    return ms.Tensor(np.array(value))


def set_seed(seed):
    seed_everything(seed)


def make_model(args):
    if args.model == "bisenetv2":
        return set_model_bisenetv2(args=args, num_classes=args.num_classes)
    raise ValueError("unrecognized model")


def get_exp_name(args):
    return "fed_{}_{}_{}_c{}_e{}_frac[{}]_iid[{}]_E[{}]_B[{}]_lr[{}]_users[{}]_opti[{}]_sche[{}]".format(
        args.date_now,
        args.data,
        args.model,
        args.num_classes,
        args.epochs,
        args.frac_num,
        args.iid,
        args.local_ep,
        args.local_bs,
        args.lr,
        args.num_users,
        args.optimizer,
        args.lr_scheduler,
    )


def init_wandb(args, wandb_id, project_name="myseg"):
    import wandb

    if wandb_id is None:
        wandb.init(project=project_name, name=args.date_now)
    else:
        wandb.init(project=project_name, resume="must", id=wandb_id)
    return wandb


def load_datasets(args):
    if args.dataset == "cityscapes":
        return get_dataset_cityscapes(args)
    if args.dataset == "camvid":
        return get_dataset_camvid(args)
    if args.dataset in {"ade20k", "voc"}:
        return get_dataset_ade20k(args)
    raise ValueError("unrecognized dataset")


def _build_ms_gpu_config(args):
    config = {}
    if getattr(args, "ms_disable_tf32", False):
        config["conv_allow_tf32"] = False
        config["matmul_allow_tf32"] = False

    for attr_name, config_key in (
        ("ms_conv_fprop_algo", "conv_fprop_algo"),
        ("ms_conv_dgrad_algo", "conv_dgrad_algo"),
        ("ms_conv_wgrad_algo", "conv_wgrad_algo"),
    ):
        value = str(getattr(args, attr_name, "") or "").strip()
        if value:
            config[config_key] = value
    return config


class FederatedTrainer:
    _RESUME_SENSITIVE_CONFIG_FIELDS = (
        "globalema",
        "aggregate_bn_stats",
        "reuse_local_trainer",
        "debug_emulate_torch_worker_rng",
        "debug_train_order_file",
        "ms_deterministic",
        "ms_disable_tf32",
        "ms_conv_fprop_algo",
        "ms_conv_dgrad_algo",
        "ms_conv_wgrad_algo",
    )

    def __init__(self, args):
        self.args = args
        self.start_time = time.time()
        self._profile_runtime = bool(getattr(args, "profile_runtime", False))
        self._eval_only = bool(getattr(args, "eval_only", False))
        self.ms_mode_name = "pynative"
        self.device = self._setup_device()

        if self._eval_only:
            self.train_dataset = None
            self.user_groups = {}
            self.client_records = {}
            self.test_dataset = get_test_dataset(args)
        else:
            self.train_dataset, self.test_dataset, self.user_groups = load_datasets(args)
            self.client_records = self._build_client_records()
        self.test_loader = None

        self.global_model = make_model(args)
        self.global_model.set_train(not self._eval_only)
        self.local_train_model = None
        self.prototype_model = None
        self.reference_model = None
        self.local_trainer = None
        if not self._eval_only:
            self.local_train_model = make_model(args)
            self.prototype_model = make_model(args)
            self.reference_model = make_model(args)
            self.local_trainer = ReusableLocalTrainer(args, self.local_train_model, self.reference_model)
        self._prototype_state_token = None
        self.start_epoch = 0
        self.wandb_id = None
        self.exp_name = get_exp_name(args)
        self.global_weights = clone_state_dict(self.global_model.parameters_dict(), host=True)
        self._debug_train_order_spec = None
        self._loaded_ema_shadow = None
        self._load_checkpoint()

        self.wandb = init_wandb(args, self.wandb_id, project_name="Fedavg_seg") if args.USE_WANDB and not self._eval_only else None
        if self.wandb is not None:
            try:
                self.wandb_id = self.wandb.run.id
            except Exception:
                self.wandb_id = None

        self.ema = None
        if args.globalema and not self._eval_only:
            self.ema = EMA(self.global_model, args.momentum)
            self.ema.register()
            if self._loaded_ema_shadow is not None:
                self.ema.load_shadow(self._loaded_ema_shadow)
                logger.info(
                    "Restored EMA shadow weights from checkpoint for resume continuity: {} tensors",
                    len(self.ema.shadow),
                )
            elif self.args.checkpoint != "":
                logger.warning(
                    "Checkpoint {} does not contain EMA shadow weights; resumed globalema training may diverge from uninterrupted training",
                    self.args.checkpoint,
                )

        self.train_loss = []
        self.local_test_accuracy = []
        self.local_test_iou = []
        self.global_test_acc = []
        self.global_test_iou = []
        if self.client_records:
            self._log_clients()

    def _resolve_ms_mode(self):
        requested_mode = str(getattr(self.args, "ms_mode", "pynative")).lower()
        self.args.ms_mode_requested = requested_mode
        if requested_mode != "graph":
            self.ms_mode_name = "pynative"
            self.args.ms_mode = self.ms_mode_name
            return ms.PYNATIVE_MODE

        unsupported_features = []
        if getattr(self.args, "is_proto", False):
            unsupported_features.append("is_proto")
        if getattr(self.args, "distill", False):
            unsupported_features.append("distill")
        if float(getattr(self.args, "fedprox_mu", 0.0)) > 0:
            unsupported_features.append("fedprox_mu>0")

        if unsupported_features:
            self.ms_mode_name = "pynative"
            self.args.ms_mode = self.ms_mode_name
            logger.warning(
                "Requested ms_mode=graph, but the current MindSpore refactor does not support graph training with {}. Falling back to ms_mode=pynative.",
                ", ".join(unsupported_features),
            )
            return ms.PYNATIVE_MODE

        self.ms_mode_name = "graph"
        self.args.ms_mode = self.ms_mode_name
        return ms.GRAPH_MODE

    def _setup_device(self):
        gpu_value = getattr(self.args, "gpu", "")
        ms_mode = self._resolve_ms_mode()
        return setup_mindspore_device(
            gpu_value,
            mode=ms_mode,
            deterministic=getattr(self.args, "ms_deterministic", False),
            gpu_config=_build_ms_gpu_config(self.args),
            logger=logger,
        )

    def _load_checkpoint(self):
        auto_init_checkpoint = self._maybe_auto_align_torch_init()
        if auto_init_checkpoint is not None:
            self.args.init_checkpoint = auto_init_checkpoint

        if self.args.checkpoint == "" and str(getattr(self.args, "init_checkpoint", "") or "").strip() == "":
            return

        if str(getattr(self.args, "init_checkpoint", "") or "").strip():
            checkpoint_path = os.path.join(self.args.root, "save/checkpoints", self.args.init_checkpoint)
            checkpoint_info = load_training_checkpoint(self.global_model, checkpoint_path, strict=False)
            self.global_weights = clone_state_dict(self.global_model.parameters_dict(), host=True)
            logger.info(
                "Loaded initialization checkpoint without resume metadata: {} (missing={}, unexpected={}, ignored={})",
                checkpoint_path,
                len(checkpoint_info["missing"]),
                len(checkpoint_info["unexpected"]),
                len(checkpoint_info["ignored"]),
            )
            return

        if self.args.checkpoint == "":
            return
        checkpoint_path = os.path.join(self.args.root, "save/checkpoints", self.args.checkpoint)
        checkpoint_info = load_training_checkpoint(self.global_model, checkpoint_path, strict=False)
        meta = checkpoint_info["meta"]
        if "epoch" in meta:
            self.start_epoch = int(meta["epoch"]) + 1
        self.wandb_id = meta.get("wandb_id")
        self.global_weights = clone_state_dict(self.global_model.parameters_dict(), host=True)
        self._loaded_ema_shadow = checkpoint_info.get("ema_shadow")
        self._log_resume_config_mismatch(meta)
        logger.info("Resume from checkpoint: {}", checkpoint_path)

    def _maybe_auto_align_torch_init(self):
        if getattr(self.args, "checkpoint", ""):
            return None
        if str(getattr(self.args, "init_checkpoint", "") or "").strip():
            return None
        if not bool(getattr(self.args, "auto_align_torch_init", True)):
            return None
        if getattr(self.args, "dataset", "") != "voc":
            return None
        if getattr(self.args, "model", "") != "bisenetv2":
            return None
        candidate = os.path.join(self.args.root, "save/checkpoints", "fedseg_torch_init_full.ckpt")
        if not os.path.exists(candidate):
            return None
        logger.info(
            "Auto-align torch init: using {} as round-0 initialization to match the torch baseline start state",
            candidate,
        )
        return os.path.basename(candidate)

    def _resume_config_snapshot(self):
        snapshot = {}
        for field in self._RESUME_SENSITIVE_CONFIG_FIELDS:
            snapshot[field] = getattr(self.args, field, None)
        return snapshot

    def _log_resume_config_mismatch(self, meta):
        saved_snapshot = meta.get("config_snapshot")
        if not isinstance(saved_snapshot, dict):
            logger.warning(
                "Checkpoint {} has no config_snapshot metadata; resume-time parity checks for BN/EMA/runtime flags are unavailable.",
                self.args.checkpoint,
            )
            return

        current_snapshot = self._resume_config_snapshot()
        mismatches = []
        for field, current_value in current_snapshot.items():
            saved_value = saved_snapshot.get(field)
            if saved_value != current_value:
                mismatches.append((field, saved_value, current_value))

        if mismatches:
            mismatch_text = ", ".join(
                "{}: saved={} current={}".format(field, repr(saved_value), repr(current_value))
                for field, saved_value, current_value in mismatches
            )
            logger.warning(
                "Resume config differs from checkpoint {}: {}",
                self.args.checkpoint,
                mismatch_text,
            )
        else:
            logger.info(
                "Resume config matches checkpoint metadata for {} sensitive fields",
                len(current_snapshot),
            )

    def _build_client_records(self):
        return {
            int(idx): {
                "idxs": tuple(int(sample_idx) for sample_idx in sorted(self.user_groups[idx])),
                "num_samples": len(self.user_groups[idx]),
                "num_eval_samples": int(0.5 * len(self.user_groups[idx])),
            }
            for idx in range(self.args.num_users)
        }

    def _log_clients(self):
        if not self.client_records:
            return
        sample_counts = [record["num_samples"] for record in self.client_records.values()]
        preview_client_ids = sorted(self.client_records)[:2]
        preview_count = 8
        logger.info(
            "Prepared {} clients: total_samples={}, min={}, max={}, mean={:.2f}, selected_per_round={}",
            len(self.client_records),
            sum(sample_counts),
            min(sample_counts),
            max(sample_counts),
            float(np.mean(sample_counts)),
            self.args.frac_num,
        )
        logger.info(
            "Client split preview for reproducibility: seed={}, clients={}, first_{}_indices",
            self.args.seed,
            preview_client_ids,
            preview_count,
        )
        for client_id in preview_client_ids:
            client = self.client_records[client_id]
            logger.info(
                "Client {}: train_samples={}, first_{}_indices={}",
                client_id,
                client["num_samples"],
                preview_count,
                list(client["idxs"][:preview_count]),
            )
        for client_id, client in self.client_records.items():
            logger.debug(
                "Client {}: train_samples={}, local_eval_samples={}",
                client_id,
                client["num_samples"],
                client["num_eval_samples"],
            )

    def _select_clients(self, epoch):
        debug_spec = self._load_debug_train_order_spec()
        if debug_spec is not None:
            selected_clients = debug_spec.get("selected_clients", {}).get(str(int(epoch)))
            if selected_clients is not None:
                return np.asarray([int(idx) for idx in selected_clients], dtype=np.int32)
        rng = np.random.default_rng(self.args.seed + epoch)
        return rng.choice(range(self.args.num_users), int(self.args.frac_num), replace=False)

    def _load_debug_train_order_spec(self):
        path = str(getattr(self.args, "debug_train_order_file", "") or "").strip()
        if not path:
            return None
        if self._debug_train_order_spec is None:
            with open(path, "r", encoding="utf-8") as fin:
                self._debug_train_order_spec = json.load(fin)
        return self._debug_train_order_spec

    def _make_client(self, idx):
        record = self.client_records[int(idx)]
        return Client(
            self.args,
            self.train_dataset,
            record["idxs"],
            client_id=int(idx),
        )

    def _prepare_prototypes(self, client, epoch, model_state):
        if not self.args.is_proto or not self.args.localmem or epoch < self.args.proto_start_epoch:
            return None, None
        logger.info("Extracting prototypes for client {}", client.client_id)
        if self._prototype_state_token != id(model_state):
            load_state_into_net(self.prototype_model, model_state, strict=False)
            self._prototype_state_token = id(model_state)
        proto_tmp, _, label_mask_ = client.extract_prototypes(self.prototype_model, epoch)
        normalize = fedseg_l2_normalize(axis=2 if self.args.kmean_num > 0 else 1)
        if self.args.kmean_num > 0:
            proto_tmp = normalize(proto_tmp)
        else:
            proto_tmp = normalize(proto_tmp.mean(axis=0))
            label_mask_ = label_mask_.sum(axis=0) > 0
        return ops.stop_gradient(proto_tmp), ops.stop_gradient(label_mask_.astype(ms.float32))

    def _aggregate(self, local_weights, client_dataset_len, reference_state=None):
        if self.args.iid:
            logger.info("Aggregating with average_weights")
            averaged = average_weights(local_weights)
        else:
            logger.info("Aggregating with weighted_average_weights")
            averaged = weighted_average_weights(local_weights, client_dataset_len)

        if not getattr(self.args, "aggregate_bn_stats", True):
            if getattr(self.args, "checkpoint", ""):
                logger.warning(
                    "aggregate_bn_stats=False while resuming from checkpoint {}. This setting is not numerically comparable to the torch baseline that aggregates BatchNorm moving stats.",
                    self.args.checkpoint,
                )
            preserved_source = reference_state if reference_state is not None else self.global_weights
            preserved = 0
            for name in list(averaged.keys()):
                if name.endswith(".moving_mean") or name.endswith(".moving_variance"):
                    if name in preserved_source:
                        averaged[name] = np.array(preserved_source[name], copy=True)
                        preserved += 1
            logger.info(
                "Preserved {} BatchNorm moving-stat tensors from {} during aggregation",
                preserved,
                "the client-start round state" if reference_state is not None else "self.global_weights",
            )
        return averaged

    def _state_dict_is_finite(self, state_dict):
        for tensor in state_dict.values():
            if hasattr(tensor, "asnumpy"):
                if not bool(ops.all(ops.isfinite(tensor)).asnumpy()):
                    return False
                continue
            if not np.all(np.isfinite(tensor)):
                return False
        return True

    def _load_state_dict(self, state_dict):
        load_state_into_net(self.global_model, state_dict, strict=False)

    def _save_checkpoint(self, epoch):
        checkpoint_path = os.path.join(self.args.root, "save/checkpoints", self.exp_name + ".ckpt")
        ema_shadow = self.ema.shadow if self.ema is not None else None
        save_training_checkpoint(
            self.global_model,
            checkpoint_path=checkpoint_path,
            epoch=epoch,
            exp_name=self.exp_name,
            wandb_id=self.wandb_id,
            ema_shadow=ema_shadow,
            config_snapshot=self._resume_config_snapshot(),
        )
        logger.info("Global model weights saved to checkpoint: {}", os.path.basename(checkpoint_path))
        if getattr(self.args, "keep_round_checkpoints", False):
            round_tagged_path = os.path.join(
                self.args.root,
                "save/checkpoints",
                "{}__round_{:04d}.ckpt".format(self.exp_name, int(epoch) + 1),
            )
            copy_training_checkpoint(checkpoint_path, round_tagged_path)
            logger.info(
                "Saved immutable round checkpoint snapshot: {}",
                os.path.basename(round_tagged_path),
            )

    def _log(self, payload, step, commit=False):
        if self.wandb is not None:
            self.wandb.log(payload, commit=commit, step=step)

    def _global_eval(self, batch_size, bucket_align=32, log_prefix="Global test", log_runtime=False):
        eval_result = evaluate_grouped_dataset(
            self.args,
            self.global_model,
            split="val",
            batch_size=batch_size,
            num_workers=self.args.num_workers,
            bucket_align=bucket_align,
            profile_runtime=log_runtime,
            progress_desc=log_prefix,
        )
        if log_runtime:
            logger.info("{} | {}", log_prefix, format_runtime_profile(eval_result["runtime"]))
            logger.info("{} | {}", log_prefix, format_runtime_detail(eval_result["runtime"]))
        return eval_result

    def run_eval_only(self):
        if self.args.checkpoint == "":
            raise ValueError("eval_only requires --checkpoint to be set")

        self.global_model.set_train(False)
        logger.info("Eval-only mode: skip federated training and evaluate the loaded checkpoint")
        logger.info("Checkpoint: {}", self.args.checkpoint)
        logger.info("Evaluate global model on global test dataset")
        eval_start_time = time.time()
        eval_result = self._global_eval(
            batch_size=self.args.final_eval_batch_size,
            bucket_align=1,
            log_prefix="Eval-only precise global test",
            log_runtime=self._profile_runtime,
        )
        eval_time = time.time() - eval_start_time
        logger.info("Results after {} global rounds of training", self.start_epoch)
        logger.info("Global Test Accuracy: {:.2f}%", eval_result["acc"])
        logger.info("Global Test IoU: {:.2f}%", eval_result["iou"])
        logger.info("Eval Time: {:.2f}s", eval_time)
        logger.info("Total Run Time: {:.2f}s", time.time() - self.start_time)
        logger.debug("Global test confusion matrix:\n{}", eval_result["confmat"])

    def run(self):
        if self._eval_only:
            self.run_eval_only()
            return

        logger.info(
            "Training global model on {} of {} users locally for {} epochs",
            self.args.frac_num,
            self.args.num_users,
            self.args.epochs,
        )

        ema = self.ema if self.ema is not None else None
        global_bar = tqdm(
            range(self.start_epoch, self.args.epochs),
            desc="Global rounds",
            leave=False,
            dynamic_ncols=True,
        )
        for epoch in global_bar:
            round_start_time = time.time()
            local_weights, local_losses = [], []
            client_dataset_len = []
            logger.info("Global training round {}", epoch)

            if ema is not None:
                ema.apply_shadow()
            self.global_model.set_train(True)
            # Torch uses the EMA shadow weights as the client-side starting
            # point for both prototype extraction and local training whenever
            # globalema is enabled. Capture that exact round state once here
            # so every selected client in this round sees the same weights.
            round_model_state = clone_state_dict(self.global_model.parameters_dict(), host=True)

            idxs_users = self._select_clients(epoch)
            logger.info("Selected clients: {}", [int(idx) for idx in idxs_users])
            client_bar = tqdm(
                idxs_users,
                desc=f"Round {epoch} clients",
                leave=False,
                dynamic_ncols=True,
            )
            for idx in client_bar:
                client_bar.set_postfix(client=int(idx))
                client = self._make_client(idx)
                logger.debug("Training client {} with {} samples", client.client_id, client.num_samples)
                stage_start = time.perf_counter()
                local_mem, local_mask = self._prepare_prototypes(client, epoch, round_model_state)
                prototype_time = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                if self.args.reuse_local_trainer:
                    weights, loss = self.local_trainer.train(
                        client=client,
                        model_state=round_model_state,
                        global_round=epoch,
                        prototypes=local_mem,
                        proto_mask=local_mask,
                    )
                else:
                    weights, loss = client.train(
                        model=self.global_model,
                        global_round=epoch,
                        prototypes=local_mem,
                        proto_mask=local_mask,
                    )
                train_time = time.perf_counter() - stage_start
                if not np.isfinite(loss):
                    logger.warning(
                        "Skip client {} in round {} because local loss is non-finite: {}",
                        client.client_id,
                        epoch,
                        loss,
                    )
                    continue
                if not self._state_dict_is_finite(weights):
                    logger.warning(
                        "Skip client {} in round {} because local weights contain non-finite values",
                        client.client_id,
                        epoch,
                    )
                    continue
                local_weights.append(clone_state_dict(weights, host=True))
                local_losses.append(loss)
                client_dataset_len.append(client.num_samples)
                if self._profile_runtime:
                    logger.info(
                        "Runtime profile | round={} client={} prototypes={:.3f}s train={:.3f}s samples={}",
                        epoch,
                        client.client_id,
                        prototype_time,
                        train_time,
                        client.num_samples,
                    )

            if not local_weights:
                raise RuntimeError("All local client updates were invalid in round {}".format(epoch))

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)
            global_bar.set_postfix(loss=f"{loss_avg:.4f}")
            logger.info("Global round {} summary: local_train_loss_avg={:.6f}", epoch, loss_avg)
            self._log({"train_loss": loss_avg, "epoch_time (s)": time.time() - round_start_time}, epoch + 1)

            logger.info("Weight averaging")
            self.global_weights = self._aggregate(
                local_weights,
                client_dataset_len,
                reference_state=round_model_state,
            )
            if ema is not None:
                for name, tensor in self.global_weights.items():
                    ema.model.parameters_dict()[name].set_data(_to_ms_tensor(tensor))
                ema.update()
                self._load_state_dict(self.global_weights)
            else:
                self._load_state_dict(self.global_weights)

            if (epoch + 1) % self.args.save_frequency == 0 or epoch == self.args.epochs - 1:
                self._save_checkpoint(epoch)

            self.global_model.set_train(False)
            if (epoch + 1) % self.args.local_test_frequency == 0:
                local_test_start_time = time.time()
                logger.info(
                    "Testing global model on 50% train data from {} selected clients after {} epochs",
                    len(idxs_users),
                    epoch + 1,
                )
                list_acc, list_iou = [], []
                eval_bar = tqdm(
                    idxs_users,
                    desc=f"Round {epoch} local eval",
                    leave=False,
                    dynamic_ncols=True,
                )
                for idx in eval_bar:
                    eval_bar.set_postfix(client=int(idx))
                    client = self._make_client(idx)
                    logger.debug("Local test client {} indices: {}", int(idx), self.user_groups[int(idx)])
                    acc, iou, confmat = client.evaluate(self.global_model)
                    logger.debug("Local test client {} confusion matrix:\n{}", int(idx), confmat)
                    list_acc.append(acc)
                    list_iou.append(iou)
                self.local_test_accuracy.append(sum(list_acc) / len(list_acc))
                self.local_test_iou.append(sum(list_iou) / len(list_iou))

                logger.info(
                    "Local test after {} rounds: train_loss_avg={:.6f}, acc={:.2f}%, miou={:.2f}%, time={:.2f}s",
                    epoch + 1,
                    np.mean(np.array(self.train_loss)),
                    self.local_test_accuracy[-1],
                    self.local_test_iou[-1],
                    time.time() - local_test_start_time,
                )
                self._log({"train_acc": self.local_test_accuracy[-1], "train_MIOU": self.local_test_iou[-1]}, epoch + 1)

            if not self.args.train_only and (epoch + 1) % self.args.global_test_frequency == 0:
                logger.info("Evaluate global model on global test dataset")
                eval_result = self._global_eval(
                    batch_size=self.args.eval_batch_size,
                    bucket_align=32,
                    log_prefix="Round {} fast global test".format(epoch + 1),
                    log_runtime=self._profile_runtime,
                )
                logger.debug("Global test confusion matrix:\n{}", eval_result["confmat"])
                logger.info(
                    "Global test after {} rounds: acc={:.2f}%, miou={:.2f}%, total_time={:.2f}min",
                    epoch + 1,
                    eval_result["acc"],
                    eval_result["iou"],
                    (time.time() - self.start_time) / 60,
                )
                self.global_test_acc.append(eval_result["acc"])
                self.global_test_iou.append(eval_result["iou"])
                self._log({"test_acc": eval_result["acc"], "test_MIOU": eval_result["iou"]}, epoch + 1)

            if self.wandb is not None:
                self.wandb.log({}, commit=True)
                logger.info("wandb commit at epoch {}", epoch + 1)

        if self.global_test_acc:
            window_acc = self.global_test_acc[-5:]
            window_iou = self.global_test_iou[-5:]
            logger.info("Average results of final {} evaluated epochs", len(window_acc))
            logger.info("Global Test Accuracy: {:.2f}%", sum(window_acc) / len(window_acc))
            logger.info("Global Test IoU: {:.2f}%", sum(window_iou) / len(window_iou))

        if not self.args.train_only and self.args.final_eval_precise:
            logger.info(
                "Run final precise global evaluation after training with batch_size={}",
                self.args.final_eval_batch_size,
            )
            final_eval = self._global_eval(
                batch_size=self.args.final_eval_batch_size,
                bucket_align=1,
                log_prefix="Final precise global test",
                log_runtime=self._profile_runtime,
            )
            logger.info("Final precise Global Test Accuracy: {:.2f}%", final_eval["acc"])
            logger.info("Final precise Global Test IoU: {:.2f}%", final_eval["iou"])
            logger.info("Final precise Eval Time: {:.2f}s", final_eval["runtime"]["total"])
            logger.debug("Final precise confusion matrix:\n{}", final_eval["confmat"])


def main():
    args = args_parser()
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/federated_main", log_name="train")
    set_seed(args.seed)
    exp_details(args)

    trainer = FederatedTrainer(args)
    logger.info("Device: {}", trainer.device)
    logger.info("Experiment name: {}", trainer.exp_name)
    trainer.run()


if __name__ == "__main__":
    main()

import copy
import json
import os
import time
import warnings

from options import args_parser
from runtime_utils import (
    configure_tensorflow_env,
    configure_tensorflow_runtime,
    install_tensorflow_stderr_filter,
)

_BOOTSTRAP_ARGS = args_parser(allow_unknown=True) if __name__ == "__main__" else None

configure_tensorflow_env(gpu=_BOOTSTRAP_ARGS.gpu if _BOOTSTRAP_ARGS is not None else None)
install_tensorflow_stderr_filter()

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from client import Client, test_inference
from eval_utils import build_tfdata_shape_batched_eval_loader, evaluate_fast_shape_bucket, parse_eval_buckets
from logging_utils import logger, setup_logger
from myseg.bisenet_utils import set_model_bisenetv2
from myseg.datasplit import get_dataset_ade20k, get_dataset_camvid, get_dataset_cityscapes
from myseg.magic import create_tf_dataloader_from_custom_dataset_test
from runtime_utils import should_disable_tqdm
from seed_utils import seed_everything
from tf2_tools import assign_model_weights, build_fast_tf_bisenetv2_from_model, snapshot_model_weights
from utils import EMA, average_weights, exp_details, weighted_average_weights

configure_tensorflow_runtime(tf)

warnings.filterwarnings("ignore")


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


def _checkpoint_metadata_path(checkpoint_path):
    if checkpoint_path.endswith(".weights.h5"):
        return checkpoint_path[: -len(".weights.h5")] + ".meta.json"
    return checkpoint_path + ".meta.json"


def _is_fast_eval_enabled(args):
    return bool(getattr(args, "eval_fast_mode", False))


def _is_tfdata_eval_enabled(args):
    return bool(getattr(args, "eval_tfdata_batch", False))


def _use_voc_special_eval_loader(args):
    return args.dataset == "voc" and _is_tfdata_eval_enabled(args)


def _configure_fast_eval_defaults(args):
    if getattr(args, "eval_fast_mode", False):
        args.eval_bs = max(1, int(getattr(args, "eval_bs", 1)))
        args.eval_tfdata_batch = True
        if not getattr(args, "eval_buckets", ""):
            args.eval_buckets = "384x512,512x512"


def _default_eval_only_checkpoint(args):
    if getattr(args, "eval_only", False) and args.checkpoint == "":
        args.checkpoint = "FedSeg.weights.h5"
    if getattr(args, "eval_only", False):
        args.rand_init = True


def _build_voc_fast_dataset_and_loader(args):
    from eval_voc import build_voc_eval_dataset, build_voc_eval_loader, resolve_path

    project_root = resolve_path(os.getcwd(), args.root)
    dataset = build_voc_eval_dataset(args, project_root)
    return dataset, build_voc_eval_loader(args, dataset)


def _make_voc_exact_eval_args(args):
    exact_args = copy.copy(args)
    exact_args.eval_fast_mode = False
    exact_args.eval_buckets = ""
    return exact_args


class FederatedTrainer:
    def __init__(self, args):
        self.args = args
        _configure_fast_eval_defaults(self.args)
        _default_eval_only_checkpoint(self.args)
        self.device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
        self._configure_gpu()
        self.train_dataset, self.user_groups = None, {}
        self.clients = {}
        self.test_dataset, self.test_loader = None, None
        self.fast_test_dataset, self.fast_test_loader = None, None
        self.exact_voc_dataset, self.exact_voc_loader = None, None
        if getattr(self.args, "eval_only", False) and _is_fast_eval_enabled(self.args) and _use_voc_special_eval_loader(self.args):
            self.fast_test_dataset, self.fast_test_loader = _build_voc_fast_dataset_and_loader(self.args)
            self.test_dataset, self.test_loader = self.fast_test_dataset, self.fast_test_loader
        elif getattr(self.args, "eval_only", False) and _is_fast_eval_enabled(self.args):
            self.train_dataset, self.test_dataset, self.user_groups = load_datasets(args)
            self.fast_test_dataset, self.fast_test_loader = self._build_fast_test_loader()
            self.test_loader = self.fast_test_loader
        elif getattr(self.args, "eval_only", False) and self.args.dataset == "voc":
            self.exact_voc_dataset, self.exact_voc_loader = self._build_exact_voc_test_loader()
            self.test_dataset, self.test_loader = self.exact_voc_dataset, self.exact_voc_loader
        else:
            self.train_dataset, self.test_dataset, self.user_groups = load_datasets(args)
            if not getattr(self.args, "eval_only", False):
                self.clients = self._build_clients()
            self.test_loader = self._build_exact_test_loader()

        self.global_model = make_model(args)
        sample_image = self._sample_model_input()
        _ = self.global_model(sample_image[None, ...], training=False)
        self.start_epoch = 0
        self.wandb_id = None
        self.exp_name = get_exp_name(args)
        self.global_weights = snapshot_model_weights(self.global_model)
        self._load_checkpoint()

        self.wandb = init_wandb(args, self.wandb_id, project_name="Fedavg_seg") if args.USE_WANDB else None
        if self.wandb is not None:
            try:
                self.wandb_id = self.wandb.run.id
            except Exception:
                self.wandb_id = None

        self.ema = None
        if args.globalema:
            self.ema = EMA(self.global_model, args.momentum)
            self.ema.register()

        self.train_loss = []
        self.local_test_accuracy = []
        self.local_test_iou = []
        self.global_test_acc = []
        self.global_test_iou = []
        self.start_time = time.time()
        self._profile_runtime = bool(getattr(args, "profile_runtime", False))
        self._log_clients()

    def _build_exact_test_loader(self):
        return create_tf_dataloader_from_custom_dataset_test(self.test_dataset)

    def _build_fast_test_loader(self):
        if _is_fast_eval_enabled(self.args) and _use_voc_special_eval_loader(self.args):
            return _build_voc_fast_dataset_and_loader(self.args)
        if _is_fast_eval_enabled(self.args) and _is_tfdata_eval_enabled(self.args):
            eval_buckets = parse_eval_buckets(getattr(self.args, "eval_buckets", ""))
            return self.test_dataset, build_tfdata_shape_batched_eval_loader(
                self.test_dataset,
                batch_size=max(1, int(getattr(self.args, "eval_bs", 1))),
                eval_buckets=eval_buckets,
                num_parallel_calls=max(1, int(getattr(self.args, "num_workers", 1))),
            )
        return self.test_dataset, self.test_loader

    def _build_exact_voc_test_loader(self):
        exact_args = _make_voc_exact_eval_args(self.args)
        return _build_voc_fast_dataset_and_loader(exact_args)

    def _sample_model_input(self):
        if self.train_dataset is not None:
            return self.train_dataset[0][0]
        if self.test_dataset is not None:
            return self.test_dataset[0][0]
        if self.fast_test_dataset is not None:
            return self.fast_test_dataset[0][0]
        return tf.zeros([3, 64, 64], dtype=tf.float32)

    def _configure_gpu(self):
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if not gpus:
            logger.warning(
                "TensorFlow did not detect a usable GPU. requested_gpu={} CUDA_VISIBLE_DEVICES={} running_on_cpu=True",
                self.args.gpu,
                os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            )
            return
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(
                "Enabled TensorFlow memory growth for {} GPU(s): requested_gpu={} visible_gpus={}",
                len(gpus),
                self.args.gpu,
                [gpu.name for gpu in gpus],
            )
        except RuntimeError as exc:
            logger.warning("Failed to configure GPU memory growth: {}", exc)

    def _load_checkpoint(self):
        if self.args.checkpoint == "":
            return
        if os.path.isabs(self.args.checkpoint):
            checkpoint_path = self.args.checkpoint
        else:
            checkpoint_path = os.path.join(self.args.root, "save/checkpoints", self.args.checkpoint)
        self.global_model.load_weights(checkpoint_path)
        metadata_path = _checkpoint_metadata_path(checkpoint_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            self.start_epoch = int(metadata.get("epoch", -1)) + 1
            self.wandb_id = metadata.get("wandb_id")
            if metadata.get("exp_name"):
                self.exp_name = metadata["exp_name"]
        logger.info("Resume from checkpoint: {}", self.args.checkpoint)

    def _build_eval_model(self, fast=False):
        eval_model = self.global_model
        if fast and getattr(self.args, "fast_nhwc", False):
            eval_model = build_fast_tf_bisenetv2_from_model(self.global_model)
            logger.info("Using TensorFlow-native NHWC inference model for fast global eval")
        return eval_model

    def _evaluate_global(self, fast=None):
        fast = _is_fast_eval_enabled(self.args) if fast is None else bool(fast)
        eval_model = self._build_eval_model(fast=fast)
        if fast and _use_voc_special_eval_loader(self.args):
            from eval_voc import run_eval_inference_fast

            if self.fast_test_dataset is None or self.fast_test_loader is None:
                self.fast_test_dataset, self.fast_test_loader = self._build_fast_test_loader()
            return run_eval_inference_fast(self.args, eval_model, self.fast_test_dataset, self.fast_test_loader)
        if fast:
            if self.fast_test_dataset is None or self.fast_test_loader is None:
                self.fast_test_dataset, self.fast_test_loader = self._build_fast_test_loader()
            return evaluate_fast_shape_bucket(
                eval_model,
                self.fast_test_loader,
                self.args.num_classes,
                batch_size=max(1, int(getattr(self.args, "eval_bs", 1))),
                dataset_size=len(self.fast_test_dataset) if hasattr(self.fast_test_dataset, "__len__") else None,
                eval_buckets=parse_eval_buckets(getattr(self.args, "eval_buckets", "")),
                profile_runtime=bool(getattr(self.args, "profile_runtime", False)),
                tfdata_batch=_is_tfdata_eval_enabled(self.args),
                desc="Evaluating global fast",
            )
        if self.args.dataset == "voc":
            from eval_voc import run_eval_inference_fast

            if self.exact_voc_dataset is None or self.exact_voc_loader is None:
                self.exact_voc_dataset, self.exact_voc_loader = self._build_exact_voc_test_loader()
            exact_args = _make_voc_exact_eval_args(self.args)
            logger.info("Using VOC exact global eval with eval_buckets disabled")
            return run_eval_inference_fast(exact_args, eval_model, self.exact_voc_dataset, self.exact_voc_loader)
        return test_inference(self.args, eval_model, self.test_loader)

    def _build_clients(self):
        return {
            idx: Client(self.args, self.train_dataset, self.user_groups[idx], client_id=idx)
            for idx in range(self.args.num_users)
        }

    def _log_clients(self):
        if not self.clients:
            logger.info("No federated clients were built (eval_only={})", getattr(self.args, "eval_only", False))
            return
        sample_counts = [client.num_samples for client in self.clients.values()]
        preview_client_ids = sorted(self.clients)[:2]
        preview_count = 8
        logger.info(
            "Prepared {} clients: total_samples={}, min={}, max={}, mean={:.2f}, selected_per_round={}",
            len(self.clients),
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
            client = self.clients[client_id]
            logger.info(
                "Client {}: train_samples={}, first_{}_indices={}",
                client_id,
                client.num_samples,
                preview_count,
                client.preview_indices(preview_count),
            )
        for client_id, client in self.clients.items():
            logger.debug(
                "Client {}: train_samples={}, local_eval_samples={}",
                client_id,
                client.num_samples,
                client.num_eval_samples,
            )

    def _select_clients(self, epoch):
        rng = np.random.default_rng(self.args.seed + epoch)
        return rng.choice(range(self.args.num_users), int(self.args.frac_num), replace=False)

    def _make_client(self, idx):
        return self.clients[int(idx)]

    def _prepare_prototypes(self, client, epoch):
        if not self.args.is_proto or not self.args.localmem or epoch < self.args.proto_start_epoch:
            return None, None

        logger.info("Extracting prototypes for client {}", client.client_id)
        proto_tmp, _, label_mask_ = client.extract_prototypes(self.global_model, epoch)
        if self.args.kmean_num > 0:
            proto_tmp = proto_tmp / (tf.norm(proto_tmp, axis=2, keepdims=True) + 1e-8)
        else:
            proto_tmp = tf.reduce_mean(proto_tmp, axis=0)
            proto_tmp = proto_tmp / (tf.norm(proto_tmp, axis=1, keepdims=True) + 1e-8)
            label_mask_ = tf.reduce_sum(tf.cast(label_mask_, tf.int32), axis=0) > 0
        return proto_tmp, label_mask_

    def _aggregate(self, local_weights, client_dataset_len):
        if self.args.iid:
            logger.info("Aggregating with average_weights")
            return average_weights(local_weights)
        logger.info("Aggregating with weighted_average_weights")
        return weighted_average_weights(local_weights, client_dataset_len)

    def _init_aggregate_accumulator(self):
        return [tf.zeros_like(var) for var in self.global_model.weights]

    def _accumulate_client_weights(self, accumulator, client_weights, client_weight):
        scale = 1.0 if self.args.iid else float(client_weight)
        for idx, variable in enumerate(client_weights):
            accumulator[idx] = accumulator[idx] + tf.cast(scale, variable.dtype) * tf.identity(variable)

    def _finalize_accumulator(self, accumulator, client_dataset_len, num_clients):
        if self.args.iid:
            denom = float(num_clients)
        else:
            denom = float(sum(client_dataset_len))
        return [value / denom for value in accumulator]

    def _save_checkpoint(self, epoch):
        os.makedirs(os.path.join(self.args.root, "save/checkpoints"), exist_ok=True)
        checkpoint_path = os.path.join(self.args.root, "save/checkpoints", self.exp_name + ".weights.h5")
        self.global_model.save_weights(checkpoint_path)
        metadata = {
            "epoch": int(epoch),
            "exp_name": self.exp_name,
            "wandb_id": self.wandb_id,
        }
        with open(_checkpoint_metadata_path(checkpoint_path), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=True, indent=2)
        logger.info("Global model weights saved to checkpoint: {}", os.path.basename(checkpoint_path))

    def _log(self, payload, step, commit=False):
        if self.wandb is not None:
            self.wandb.log(payload, commit=commit, step=step)

    def run(self):
        if getattr(self.args, "eval_only", False):
            logger.info("Eval-only mode enabled; skipping federated training")
            start_time = time.time()
            test_acc, test_iou, confmat = self._evaluate_global()
            logger.debug("Eval-only confusion matrix:\n{}", confmat)
            logger.info("Eval-only Global Test Accuracy: {:.2f}%", test_acc)
            logger.info("Eval-only Global Test IoU: {:.2f}%", test_iou)
            logger.info("Eval-only Time: {:.2f}s", time.time() - start_time)
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
            disable=should_disable_tqdm(),
        )
        for epoch in global_bar:
            round_start_time = time.time()
            local_losses = []
            client_dataset_len = []
            aggregated_weights = self._init_aggregate_accumulator()
            logger.info("Global training round {}", epoch)

            if ema is not None:
                ema.apply_shadow()

            idxs_users = self._select_clients(epoch)
            logger.info("Selected clients: {}", [int(idx) for idx in idxs_users])
            client_bar = tqdm(
                idxs_users,
                desc=f"Round {epoch} clients",
                leave=False,
                dynamic_ncols=True,
                disable=should_disable_tqdm(),
            )
            for idx in client_bar:
                client_bar.set_postfix(client=int(idx))
                client = self._make_client(idx)
                logger.debug("Training client {} with {} samples", client.client_id, client.num_samples)
                stage_start = time.perf_counter()
                local_mem, local_mask = self._prepare_prototypes(client, epoch)
                prototype_time = time.perf_counter() - stage_start
                self.global_model.trainable = False
                stage_start = time.perf_counter()
                w, loss = client.train(
                    model=self.global_model,
                    global_round=epoch,
                    prototypes=local_mem,
                    proto_mask=local_mask,
                )
                train_time = time.perf_counter() - stage_start
                self.global_model.trainable = True
                local_losses.append(copy.deepcopy(loss))
                client_dataset_len.append(client.num_samples)
                self._accumulate_client_weights(aggregated_weights, w, client.num_samples)
                if self._profile_runtime:
                    logger.info(
                        "Runtime profile | round={} client={} prototypes={:.3f}s train={:.3f}s samples={}",
                        epoch,
                        client.client_id,
                        prototype_time,
                        train_time,
                        client.num_samples,
                    )

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)
            global_bar.set_postfix(loss=f"{loss_avg:.4f}")
            logger.info("Global round {} summary: local_train_loss_avg={:.6f}", epoch, loss_avg)

            logger.info("Weight averaging")
            self.global_weights = self._finalize_accumulator(aggregated_weights, client_dataset_len, len(client_dataset_len))
            assign_model_weights(self.global_model, self.global_weights)
            if ema is not None:
                ema.update()

            if (epoch + 1) % self.args.save_frequency == 0 or epoch == self.args.epochs - 1:
                self._save_checkpoint(epoch)

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
                    disable=should_disable_tqdm(),
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
                test_acc, test_iou, confmat = self._evaluate_global()
                logger.debug("Global test confusion matrix:\n{}", confmat)
                logger.info(
                    "Global test after {} rounds: acc={:.2f}%, miou={:.2f}%, total_time={:.2f}min",
                    epoch + 1,
                    test_acc,
                    test_iou,
                    (time.time() - self.start_time) / 60,
                )
                self.global_test_acc.append(test_acc)
                self.global_test_iou.append(test_iou)
                self._log({"test_acc": test_acc, "test_MIOU": test_iou}, epoch + 1)

            round_time = time.time() - round_start_time
            logger.info("Global round {} train time: {:.2f}s", epoch, round_time)
            self._log({"train_loss": loss_avg, "epoch_time (s)": round_time}, epoch + 1)

            if self.wandb is not None:
                self.wandb.log({}, commit=True)
                logger.info("wandb commit at epoch {}", epoch + 1)

        if self.global_test_acc:
            window_acc = self.global_test_acc[-5:]
            window_iou = self.global_test_iou[-5:]
            logger.info("Average results of final {} evaluated epochs", len(window_acc))
            logger.info("Global Test Accuracy: {:.2f}%", sum(window_acc) / len(window_acc))
            logger.info("Global Test IoU: {:.2f}%", sum(window_iou) / len(window_iou))

        if not self.args.train_only:
            final_eval_start = time.time()
            logger.info("Final exact global evaluation after training finished")
            test_acc, test_iou, confmat = self._evaluate_global(fast=False)
            logger.debug("Final exact global test confusion matrix:\n{}", confmat)
            logger.info("Final Exact Global Test Accuracy: {:.2f}%", test_acc)
            logger.info("Final Exact Global Test IoU: {:.2f}%", test_iou)
            logger.info("Final Exact Eval Time: {:.2f}s", time.time() - final_eval_start)
            self._log({"final_exact_test_acc": test_acc, "final_exact_test_MIOU": test_iou}, self.args.epochs, commit=True)


def main():
    args = args_parser()
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/federated_main", log_name="train")
    set_seed(args.seed)
    exp_details(args)

    trainer = FederatedTrainer(args)
    logger.info(
        "Device: {} | requested_gpu={} | CUDA_VISIBLE_DEVICES={}",
        trainer.device,
        args.gpu,
        os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    )
    logger.info("Experiment name: {}", trainer.exp_name)
    trainer.run()


if __name__ == "__main__":
    main()

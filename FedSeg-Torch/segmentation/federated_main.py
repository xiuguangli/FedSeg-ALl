import copy
import os
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from client import Client, test_inference
from logging_utils import logger, setup_logger
from myseg.bisenet_utils import set_model_bisenetv2
from myseg.datasplit import get_dataset_ade20k, get_dataset_camvid, get_dataset_cityscapes
from options import args_parser
from seed_utils import seed_everything
from utils import EMA, average_weights, exp_details, weighted_average_weights

warnings.filterwarnings("ignore")


def set_seed(seed):
    seed_everything(seed)


def configure_torch_runtime(args):
    if not getattr(args, "torch_disable_tf32", False):
        tf32_disabled = False
    else:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.allow_tf32 = False
        tf32_disabled = True

    strict_deterministic = bool(getattr(args, "torch_strict_deterministic", False))
    if strict_deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)

    if tf32_disabled or strict_deterministic:
        logger.info(
            "Torch runtime overrides: disable_tf32={} strict_deterministic={}",
            tf32_disabled,
            strict_deterministic,
        )


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


class FederatedTrainer:
    # 联邦训练总控：负责客户端组织、全局聚合、评测以及断点恢复。
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(args.gpu))

        self.train_dataset, self.test_dataset, self.user_groups = load_datasets(args)
        # 客户端先统一创建好，便于集中打印划分信息；真正的 DataLoader 会在本地训练/评估时再延迟构造。
        self.clients = self._build_clients()
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        self.global_model = make_model(args).to(self.device)
        self.global_model.train()
        self.start_epoch = 0
        self.wandb_id = None
        self.exp_name = get_exp_name(args)
        self.global_weights = copy.deepcopy(self.global_model.state_dict())
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

    def _load_checkpoint(self):
        if self.args.checkpoint == "":
            return
        checkpoint = torch.load(os.path.join(self.args.root, "save/checkpoints", self.args.checkpoint), map_location=self.device)
        self.global_model.load_state_dict(checkpoint["model"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.wandb_id = checkpoint["wandb_id"]
        self.global_weights = copy.deepcopy(self.global_model.state_dict())
        logger.info("Resume from checkpoint: {}", self.args.checkpoint)

    def _build_clients(self):
        return {
            idx: Client(self.args, self.train_dataset, self.user_groups[idx], client_id=idx)
            for idx in range(self.args.num_users)
        }

    def _log_clients(self):
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
        # 只展示前两个客户端的前几个索引，便于快速核对数据划分和随机种子是否稳定复现。
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
        # 每一轮都基于固定 seed + round 采样，确保同参重跑时客户端选择顺序完全一致。
        rng = np.random.default_rng(self.args.seed + epoch)
        return rng.choice(range(self.args.num_users), int(self.args.frac_num), replace=False)

    def _make_client(self, idx):
        return self.clients[int(idx)]

    def _prepare_prototypes(self, client, epoch):
        if not self.args.is_proto or not self.args.localmem or epoch < self.args.proto_start_epoch:
            return None, None

        # prototype 只在相关分支真正启用后提取，避免平时增加额外前向和通信心智负担。
        logger.info("Extracting prototypes for client {}", client.client_id)
        proto_tmp, _, label_mask_ = client.extract_prototypes(copy.deepcopy(self.global_model), epoch)
        if self.args.kmean_num > 0:
            proto_tmp = F.normalize(proto_tmp, dim=2)
        else:
            proto_tmp = F.normalize(proto_tmp.mean(0), dim=1)
            label_mask_ = label_mask_.sum(0) > 0
        return proto_tmp, label_mask_

    def _aggregate(self, local_weights, client_dataset_len):
        # IID 设置下直接平均；非 IID 时按样本量加权，避免小客户端对全局模型影响过大。
        if self.args.iid:
            logger.info("Aggregating with average_weights")
            return average_weights(local_weights)
        logger.info("Aggregating with weighted_average_weights")
        return weighted_average_weights(local_weights, client_dataset_len)

    def _save_checkpoint(self, epoch):
        torch.save(
            {
                "model": self.global_model.state_dict(),
                "epoch": epoch,
                "exp_name": self.exp_name,
                "wandb_id": self.wandb_id,
            },
            os.path.join(self.args.root, "save/checkpoints", self.exp_name + ".pth"),
        )
        logger.info("Global model weights saved to checkpoint: {}", self.exp_name + ".pth")

    def _log(self, payload, step, commit=False):
        if self.wandb is not None:
            self.wandb.log(payload, commit=commit, step=step)

    def run(self):
        logger.info(
            "Training global model on {} of {} users locally for {} epochs",
            self.args.frac_num,
            self.args.num_users,
            self.args.epochs,
        )

        if self.ema is not None:
            ema = self.ema
        else:
            ema = None

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
            self.global_model.train()

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
                local_mem, local_mask = self._prepare_prototypes(client, epoch)
                prototype_time = time.perf_counter() - stage_start
                stage_start = time.perf_counter()
                w, loss = client.train(
                    model=copy.deepcopy(self.global_model),
                    global_round=epoch,
                    prototypes=local_mem,
                    proto_mask=local_mask,
                )
                train_time = time.perf_counter() - stage_start
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))
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

            loss_avg = sum(local_losses) / len(local_losses)
            self.train_loss.append(loss_avg)
            global_bar.set_postfix(loss=f"{loss_avg:.4f}")
            logger.info("Global round {} summary: local_train_loss_avg={:.6f}", epoch, loss_avg)
            self._log({"train_loss": loss_avg, "epoch_time (s)": time.time() - round_start_time}, epoch + 1)

            logger.info("Weight averaging")
            self.global_weights = self._aggregate(local_weights, client_dataset_len)
            if ema is not None:
                ema.model.load_state_dict(self.global_weights)
                ema.update()
            else:
                self.global_model.load_state_dict(self.global_weights)

            if (epoch + 1) % self.args.save_frequency == 0 or epoch == self.args.epochs - 1:
                self._save_checkpoint(epoch)

            self.global_model.eval()
            if (epoch + 1) % self.args.local_test_frequency == 0:
                local_test_start_time = time.time()
                logger.info(
                    "Testing global model on 50% train data from {} selected clients after {} epochs",
                    len(idxs_users),
                    epoch + 1,
                )
                list_acc, list_iou = [], []
                # 本地验证只在当前轮被选中的客户端上做，延续原始实现里“快评估、低开销”的节奏。
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
                test_acc, test_iou, confmat = test_inference(self.args, self.global_model, self.test_loader)
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

            if self.wandb is not None:
                self.wandb.log({}, commit=True)
                logger.info("wandb commit at epoch {}", epoch + 1)

        if self.global_test_acc:
            window_acc = self.global_test_acc[-5:]
            window_iou = self.global_test_iou[-5:]
            logger.info("Average results of final {} evaluated epochs", len(window_acc))
            logger.info("Global Test Accuracy: {:.2f}%", sum(window_acc) / len(window_acc))
            logger.info("Global Test IoU: {:.2f}%", sum(window_iou) / len(window_iou))


def main():
    args = args_parser()
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/federated_main", log_name="train")
    set_seed(args.seed)
    configure_torch_runtime(args)
    exp_details(args)

    trainer = FederatedTrainer(args)
    logger.info("Device: {}", trainer.device)
    logger.info("Experiment name: {}", trainer.exp_name)
    trainer.run()


if __name__ == "__main__":
    main()

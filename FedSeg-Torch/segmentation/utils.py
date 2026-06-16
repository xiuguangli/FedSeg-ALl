import copy
import torch
from logging_utils import logger



class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# 初始化
#ema = EMA(model, 0.999)
#ema.register()

# 训练过程中，更新完参数后，同步update shadow weights
#def train():
#    optimizer.step()
#    ema.update()

# eval前，apply shadow weights；eval之后，恢复原来模型的参数
#def evaluate():
#    ema.apply_shadow()
    # evaluate
#    ema.restore()




def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(w, client_dataset_len):
    """
    Returns the weighted average of the weights.

    client_dataset_len: a list of the length of the client dataset
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = torch.mul(w_avg[key], client_dataset_len[0])  # w[0][key] * client_dataset_len[0]
        for i in range(1, len(w)):
            w_avg[key] += torch.mul((w[i][key]), client_dataset_len[i])  # w[i][key] * client_dataset_len[i]
        w_avg[key] = torch.div(w_avg[key], sum(client_dataset_len))
    return w_avg


def exp_details(args):
    logger.info("Experimental details")
    logger.info(
        "Dataset={}, root_dir={}, erase_data={}, classes={}, split={}, model={}, checkpoint={}",
        args.dataset,
        args.root_dir,
        args.USE_ERASE_DATA,
        args.num_classes,
        args.data,
        args.model,
        args.checkpoint,
    )
    logger.info(
        "Optimizer={}, scheduler={}, lr={}, momentum={}, weight_decay={}, global_rounds={}",
        args.optimizer,
        args.lr_scheduler,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.epochs,
    )
    logger.info(
        "Federated: split={}, users={}, selected_per_round={}, local_epochs={}, local_batch_size={}",
        "IID" if args.iid else "Non-IID",
        args.num_users,
        args.frac_num,
        args.local_ep,
        args.local_bs,
    )
    logger.info(
        "Logging: save_frequency={}, local_test_frequency={}, global_test_frequency={}, use_wandb={}",
        args.save_frequency,
        args.local_test_frequency,
        args.global_test_frequency,
        args.USE_WANDB,
    )
    logger.info(
        "Torch runtime: disable_tf32={}, strict_deterministic={}",
        getattr(args, "torch_disable_tf32", False),
        getattr(args, "torch_strict_deterministic", False),
    )
    logger.info(
        "Debug flags: disable_train_aug={}, disable_dropout={}, deterministic_dropout={}, deterministic_contrast={}, freeze_bn_stats={}",
        getattr(args, "debug_disable_train_aug", False),
        getattr(args, "debug_disable_dropout", False),
        getattr(args, "debug_deterministic_dropout", False),
        getattr(args, "debug_deterministic_contrast", False),
        getattr(args, "debug_freeze_bn_stats", False),
    )

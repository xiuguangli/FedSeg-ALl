import copy
# import torch

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Parameter





class EMA0():
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

class EMA():
    def __init__(self, model: nn.Cell, decay: float):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """
        创建 shadow parameters，初始化为当前模型权重。
        """
        # 使用 .parameters_and_names() 遍历模型所有参数
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                # .data 获取底层 Tensor, .clone() 创建副本
                self.shadow[name] = param.data.clone()

    def update(self):
        """
        根据 EMA 公式更新 shadow parameters。
        """
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                assert name in self.shadow
                # Tensor 运算逻辑与 PyTorch 相同
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                # 更新 shadow 字典中的 Tensor
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """
        将 shadow parameters 应用到模型上，并备份原始参数。
        通常在评估 (evaluation) 前调用。
        """
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                assert name in self.shadow
                # 备份原始参数
                self.backup[name] = param.data.clone() # 克隆一份以防意外修改
                # 关键：使用 .set_data() 更新模型参数的值
                param.set_data(self.shadow[name])

    def restore(self):
        """
        从备份中恢复模型的原始参数。
        通常在评估 (evaluation) 后调用。
        """
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                assert name in self.backup
                # 关键：使用 .set_data() 更新模型参数的值
                param.set_data(self.backup[name])
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
        # w_avg[key] = torch.div(w_avg[key], len(w))
        w_avg[key] = ops.div(w_avg[key], len(w))
    return w_avg


def weighted_average_weights(w, client_dataset_len):
    """
    Returns the weighted average of the weights.

    client_dataset_len: a list of the length of the client dataset
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # w_avg[key] = torch.mul(w_avg[key], client_dataset_len[0])  # w[0][key] * client_dataset_len[0]
        w_avg[key] = ops.mul(w_avg[key], client_dataset_len[0])  # w[0][key] * client_dataset_len[0]
        for i in range(1, len(w)):
            # w_avg[key] += torch.mul((w[i][key]), client_dataset_len[i])  # w[i][key] * client_dataset_len[i]
            w_avg[key] += ops.mul((w[i][key]), client_dataset_len[i])  # w[i][key] * client_dataset_len[i]
        # w_avg[key] = torch.div(w_avg[key], sum(client_dataset_len))
        w_avg[key] = ops.div(w_avg[key], sum(client_dataset_len))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Dataset                 : {args.dataset}')
    print(f'    Dataset root_dir        : {args.root_dir}')
    print(f'    USE_ERASE_DATA          : {args.USE_ERASE_DATA}')
    print(f'    Number of classes       : {args.num_classes}')
    print(f'    Split data (train data) : {args.data}')
    print(f'    Model                   : {args.model}')
    print(f'    resume from Checkpoint  : {args.checkpoint}')

    print(f'    Optimizer               : {args.optimizer}')
    print(f'    Scheduler               : {args.lr_scheduler}')
    print(f'    Learning rate           : {args.lr}')
    print(f'    Momentum                : {args.momentum}')
    print(f'    weight decay            : {args.weight_decay}')
    print(f'    Global Rounds           : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Number of global users  : {args.num_users}')
    # print(f'    Fraction of users  : {args.frac}')
    # print(f'    Number of Fraction local users : {max(int(args.frac * args.num_users), 1)}')
    print(f'    Fraction num of users   : {args.frac_num}')
    print(f'    Local Epochs            : {args.local_ep}')
    print(f'    Local Batch size        : {args.local_bs}\n')

    print('    Logging parameters:')
    print(f'    save_frequency          : {args.save_frequency}')
    print(f'    local_test_frequency    : {args.local_test_frequency}')
    print(f'    global_test_frequency   : {args.global_test_frequency}')
    print(f'    USE_WANDB               : {args.USE_WANDB}\n')
    return

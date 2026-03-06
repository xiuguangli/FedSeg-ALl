import copy
import numpy as np
# import torch
import tensorflow as tf



class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for var in self.model.trainable_variables:
            self.shadow[var.name] = var.numpy().copy()

    def update(self):
        for var in self.model.trainable_variables:
            name = var.name
            if name in self.shadow:
                new_average = (1.0 - self.decay) * var.numpy() + self.decay * self.shadow[name]
                self.shadow[name] = new_average.copy()

    def apply_shadow(self):
        for var in self.model.trainable_variables:
            name = var.name
            if name in self.shadow:
                self.backup[name] = var.numpy().copy()
                var.assign(self.shadow[name])

    def restore(self):
        for var in self.model.trainable_variables:
            name = var.name
            if name in self.backup:
                var.assign(self.backup[name])
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




def average_weights0(w):
    """
    Returns the average of the weights.
    w: list of dicts of numpy arrays or tf.Tensor
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = w_avg[key] / len(w)
    return w_avg

def average_weights(w):
    """
    Returns the average of the weights.
    w: list of dicts of numpy arrays or tf.Tensor
    """
    w_avg = copy.deepcopy(w[0])
    for i in range(len(w_avg)):
        # 1. 获取所有客户端在第 i 层的权重
        layer_weights_across_clients = [client_w[i] for client_w in w]
        
        # 2. 计算平均值 (axis=0 表示在客户端维度上平均)
        # np.mean 可以直接对堆叠的数组求平均
        w_avg[i] = np.mean(layer_weights_across_clients, axis=0)
        
    return w_avg

def get_weights_dict(model):
    # model.trainable_variables 包含名字和值
    return {v.name: v.numpy() for v in model.trainable_variables}

def weighted_average_weights(w, client_dataset_len):
    """
    Returns the weighted average of the weights.

    client_dataset_len: a list of the length of the client dataset
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] * client_dataset_len[0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] * client_dataset_len[i]
        w_avg[key] = w_avg[key] / sum(client_dataset_len)
    return w_avg


def weighted_average_weights(w, client_dataset_len):
    """
    Returns the weighted average of the weights.

    client_dataset_len: a list of the length of the client dataset
    """
    w_avg = copy.deepcopy(w[0])
    for i in range(len(w_avg)):
        w_avg[i] = w_avg[i] * client_dataset_len[0]
        for j in range(1, len(w)):
            w_avg[i] += w[j][i] * client_dataset_len[j]
        w_avg[i] = w_avg[i] / sum(client_dataset_len)
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

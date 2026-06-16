import copy

import numpy as np
import tensorflow as tf

from logging_utils import logger


class EMA:
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
                self.shadow[name] = (1.0 - self.decay) * var.numpy() + self.decay * self.shadow[name]

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


def average_weights(weights):
    if weights and isinstance(weights[0][0], tf.Tensor):
        return [tf.add_n([client_weights[layer_idx] for client_weights in weights]) / float(len(weights)) for layer_idx in range(len(weights[0]))]
    averaged = copy.deepcopy(weights[0])
    for layer_idx in range(len(averaged)):
        averaged[layer_idx] = np.mean([client_weights[layer_idx] for client_weights in weights], axis=0)
    return averaged


def weighted_average_weights(weights, client_dataset_len):
    if weights and isinstance(weights[0][0], tf.Tensor):
        total = float(sum(client_dataset_len))
        averaged = []
        for layer_idx in range(len(weights[0])):
            weighted = [client_weights[layer_idx] * float(client_dataset_len[client_idx]) for client_idx, client_weights in enumerate(weights)]
            averaged.append(tf.add_n(weighted) / total)
        return averaged
    averaged = copy.deepcopy(weights[0])
    total = float(sum(client_dataset_len))
    for layer_idx in range(len(averaged)):
        averaged[layer_idx] = averaged[layer_idx] * client_dataset_len[0]
        for client_idx in range(1, len(weights)):
            averaged[layer_idx] += weights[client_idx][layer_idx] * client_dataset_len[client_idx]
        averaged[layer_idx] = averaged[layer_idx] / total
    return averaged


def get_weights_dict(model):
    return {var.name: var.numpy() for var in model.trainable_variables}


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

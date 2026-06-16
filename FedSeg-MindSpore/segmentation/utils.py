import mindspore as ms
import numpy as np
import mindspore.ops as ops

from logging_utils import logger


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    @staticmethod
    def _materialize_tensor(value, dtype=None):
        if isinstance(value, ms.Parameter):
            value = value.data
        if isinstance(value, ms.Tensor):
            tensor = value
        elif isinstance(value, np.ndarray):
            tensor = ms.Tensor(np.array(value, copy=True))
        else:
            tensor = ms.Tensor(np.array(value))
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.astype(dtype)
        return tensor

    def register(self):
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def load_shadow(self, shadow_state):
        loaded_shadow = {}
        for name, param in self.model.parameters_and_names():
            if not param.requires_grad or name not in shadow_state:
                continue
            # Avoid Tensor.clone() on checkpoint-restored host arrays here:
            # some MindSpore GPU builds materialize that eager Clone via a
            # kernel path that later fails inside Parameter.set_data().
            loaded_shadow[name] = self._materialize_tensor(shadow_state[name], dtype=param.dtype)
        self.shadow = loaded_shadow

    def update(self):
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                assert name in self.shadow
                one_minus_decay = ms.Tensor(1.0 - self.decay, param.dtype)
                decay = ms.Tensor(self.decay, param.dtype)
                # MindSpore GPU may fail to materialize Clone for an eager EMA
                # expression result, so keep the computed tensor directly.
                self.shadow[name] = ops.add(
                    one_minus_decay * param.data,
                    decay * self.shadow[name].astype(param.dtype),
                )

    def apply_shadow(self):
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.set_data(self.shadow[name])

    def restore(self):
        for name, param in self.model.parameters_and_names():
            if param.requires_grad:
                assert name in self.backup
                param.set_data(self.backup[name])
        self.backup = {}


def _clone_value(value):
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    return np.array(value, copy=True)


def _clone_tensor_dict(weights):
    return {key: _clone_value(value) for key, value in weights.items()}


def average_weights(weights_list):
    averaged = _clone_tensor_dict(weights_list[0])
    num_models = len(weights_list)
    for key in averaged:
        for idx in range(1, num_models):
            averaged[key] = averaged[key] + weights_list[idx][key]
        averaged[key] = averaged[key] / num_models
    return averaged


def weighted_average_weights(weights_list, client_dataset_len):
    averaged = _clone_tensor_dict(weights_list[0])
    total_size = float(sum(client_dataset_len))
    for key in averaged:
        averaged[key] = averaged[key] * float(client_dataset_len[0])
        for idx in range(1, len(weights_list)):
            averaged[key] = averaged[key] + weights_list[idx][key] * float(client_dataset_len[idx])
        averaged[key] = averaged[key] / total_size
    return averaged


def _log_voc_bisenetv2_reference_recipe(args):
    if getattr(args, "dataset", "") != "voc" or getattr(args, "model", "") != "bisenetv2":
        return

    expected = {
        "losstype": "back",
        "label_online_gen": False,
        "is_proto": True,
        "localmem": True,
        "pseudo_label": True,
        "globalema": False,
        "proto_start_epoch": 1,
        "pseudo_label_start_epoch": 1,
        "max_anchor": 4096,
        "kmean_num": 2,
        "con_lamb": 0.1,
        "con_lamb_local": 1.0,
        "temperature": 0.07,
        "temp_dist": 0.1,
        "warmstep": 20,
        "mixlabel": True,
        "rand_init": False,
        "auto_align_torch_init": True,
        "ms_deterministic": True,
        "ms_disable_tf32": True,
        "reuse_local_trainer": True,
        "aggregate_bn_stats": True,
    }
    mismatches = []
    for field, expected_value in expected.items():
        current_value = getattr(args, field, None)
        if current_value != expected_value:
            mismatches.append(
                "{}={} (ref {})".format(field, repr(current_value), repr(expected_value))
            )

    if mismatches:
        logger.warning(
            "VOC BiSeNetV2 run differs from the validated torch-aligned recipe: {}",
            ", ".join(mismatches),
        )
    else:
        logger.info("VOC BiSeNetV2 run matches the validated torch-aligned recipe")


def exp_details(args):
    logger.info("Experimental details")
    logger.info(
        "Dataset={}, root_dir={}, erase_data={}, classes={}, split={}, model={}, checkpoint={}, init_checkpoint={}",
        args.dataset,
        args.root_dir,
        args.USE_ERASE_DATA,
        args.num_classes,
        args.data,
        args.model,
        args.checkpoint,
        getattr(args, "init_checkpoint", ""),
    )
    logger.info(
        "Optimizer={}, scheduler={}, lr={}, momentum={}, weight_decay={}, global_rounds={}, globalema={}",
        args.optimizer,
        args.lr_scheduler,
        args.lr,
        args.momentum,
        args.weight_decay,
        args.epochs,
        getattr(args, "globalema", False),
    )
    logger.info(
        "Loss/proto: losstype={}, label_online_gen={}, proto={}, localmem={}, pseudo_label={}, proto_start_epoch={}, pseudo_label_start_epoch={}, max_anchor={}, kmean_num={}, con_lamb={}, con_lamb_local={}, temp={}, temp_dist={}, warmstep={}, mixlabel={}, rand_init={}",
        getattr(args, "losstype", ""),
        getattr(args, "label_online_gen", False),
        getattr(args, "is_proto", False),
        getattr(args, "localmem", False),
        getattr(args, "pseudo_label", False),
        getattr(args, "proto_start_epoch", 0),
        getattr(args, "pseudo_label_start_epoch", 0),
        getattr(args, "max_anchor", 0),
        getattr(args, "kmean_num", 0),
        getattr(args, "con_lamb", 0.0),
        getattr(args, "con_lamb_local", 0.0),
        getattr(args, "temperature", 0.0),
        getattr(args, "temp_dist", 0.0),
        getattr(args, "warmstep", 0),
        getattr(args, "mixlabel", False),
        getattr(args, "rand_init", False),
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
        "Evaluation: fast_eval_batch_size={}, final_eval_precise={}, final_eval_batch_size={}",
        args.eval_batch_size,
        args.final_eval_precise,
        args.final_eval_batch_size,
    )
    logger.info(
        "MindSpore runtime: mode={}, deterministic={}, disable_tf32={}, conv_fprop_algo={}, conv_dgrad_algo={}, conv_wgrad_algo={}",
        getattr(args, "ms_mode", "pynative"),
        getattr(args, "ms_deterministic", False),
        getattr(args, "ms_disable_tf32", False),
        getattr(args, "ms_conv_fprop_algo", ""),
        getattr(args, "ms_conv_dgrad_algo", ""),
        getattr(args, "ms_conv_wgrad_algo", ""),
    )
    logger.info(
        "Debug flags: disable_train_aug={}, disable_dropout={}, deterministic_dropout={}, deterministic_contrast={}, freeze_bn_stats={}, emulate_torch_worker_rng={}, reuse_local_trainer={}, aggregate_bn_stats={}",
        getattr(args, "debug_disable_train_aug", False),
        getattr(args, "debug_disable_dropout", False),
        getattr(args, "debug_deterministic_dropout", False),
        getattr(args, "debug_deterministic_contrast", False),
        getattr(args, "debug_freeze_bn_stats", False),
        getattr(args, "debug_emulate_torch_worker_rng", False),
        getattr(args, "reuse_local_trainer", False),
        getattr(args, "aggregate_bn_stats", True),
    )
    _log_voc_bisenetv2_reference_recipe(args)

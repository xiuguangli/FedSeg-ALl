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

import tensorflow as tf

from federated_main import load_datasets, make_model
from logging_utils import logger, setup_logger
from myseg.magic import create_tf_dataloader_from_custom_dataset_test
from update import test_inference

configure_tensorflow_runtime(tf)

warnings.filterwarnings("ignore")


def _normalize_checkpoint_name(checkpoint_name: str) -> str:
    if checkpoint_name.endswith(".pth"):
        return checkpoint_name[: -len(".pth")] + ".weights.h5"
    return checkpoint_name


def build_eval_dataset(eval_args):
    if eval_args.dataset == "cityscapes":
        if eval_args.data not in {"train", "val"}:
            raise ValueError("cityscapes only supports train/val in TF2 eval")
        _, test_dataset, _ = load_datasets(eval_args)
        logger.info("args.data: {}", eval_args.data)
        return test_dataset

    if eval_args.dataset in {"camvid", "ade20k", "voc"}:
        _, test_dataset, _ = load_datasets(eval_args)
        logger.info("args.data: {}", eval_args.data)
        return test_dataset

    raise ValueError("unrecognized dataset")


def main():
    eval_args = args_parser()
    setup_logger(verbose=False, logs_dir="logs/eval", log_name="eval")

    start_time = time.time()
    device = "cuda" if tf.config.list_physical_devices("GPU") else "cpu"
    logger.info("device: {}", device)

    test_dataset = build_eval_dataset(eval_args)
    test_loader = create_tf_dataloader_from_custom_dataset_test(
        test_dataset,
        batch_size=4,
        shuffle=False,
    )

    global_model = make_model(eval_args)
    _ = global_model(test_dataset[0][0][None, ...], training=False)

    if eval_args.checkpoint == "":
        raise ValueError("args.checkpoint is empty")

    checkpoint_path = os.path.join(eval_args.root, "save/checkpoints", _normalize_checkpoint_name(eval_args.checkpoint))
    global_model.load_weights(checkpoint_path)
    logger.info("resume from: {}", os.path.basename(checkpoint_path))

    logger.info("Evaluate global model on global Test dataset")
    test_acc, test_iou, confmat = test_inference(eval_args, global_model, test_loader)
    logger.debug("Confusion matrix:\n{}", confmat)
    logger.info("Global Test Accuracy: {:.2f}%", test_acc)
    logger.info("Global Test IoU: {:.2f}%", test_iou)
    logger.info("Total Run Time: {:.2f}s", time.time() - start_time)


if __name__ == "__main__":
    main()

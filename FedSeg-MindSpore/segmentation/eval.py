import argparse
import os
import time
import warnings

import mindspore as ms

from batch_utils import build_batches
from checkpoint_utils import load_training_checkpoint
from logging_utils import logger, setup_logger
from mindspore_runtime import setup_mindspore_device
from myseg.dataloader import Cityscapes_Dataset
from update import test_inference
from federated_main import make_model

warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="index of GPU to use")
    parser.add_argument("--num_workers", type=int, default=1, help="kept for compatibility")
    parser.add_argument("--model", type=str, default="bisenetv2", choices=["lraspp_mobilenetv3", "bisenetv2"])
    parser.add_argument("--num_classes", type=int, default=21)
    parser.add_argument("--checkpoint", type=str, default="", help="full file name of the checkpoint")
    parser.add_argument("--dataset", type=str, default="cityscapes", help="dataset name")
    parser.add_argument("--root_dir", type=str, default="/home/data/cityscapes/", help="root of dataset")
    parser.add_argument("--root", type=str, default="./", help="home directory")
    parser.add_argument("--data", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--USE_ERASE_DATA", type=str2bool)
    parser.add_argument("--proj_dim", type=int, default=256)
    parser.add_argument("--rand_init", type=str2bool, default=False)
    return parser.parse_args()


def _setup_device(args):
    return setup_mindspore_device(args.gpu, mode=ms.PYNATIVE_MODE)


def main():
    args = args_parser()
    setup_logger(verbose=False, logs_dir="logs/eval", log_name="eval")
    start_time = time.time()
    device = _setup_device(args)
    logger.info("device: {}", device)

    if args.dataset != "cityscapes":
        raise ValueError("Error: unrecognized dataset")

    test_dataset = Cityscapes_Dataset(args.root_dir, args.data, args.USE_ERASE_DATA)
    logger.info("args.data: {}", args.data)
    test_loader = build_batches(
        test_dataset,
        batch_size=4,
        shuffle=False,
        drop_last=False,
        pad_to_max_shape=True,
    )

    global_model = make_model(args)
    if args.checkpoint == "":
        raise ValueError("Error: args.checkpoint is empty")

    checkpoint_path = os.path.join(args.root, "save/checkpoints", args.checkpoint)
    checkpoint_info = load_training_checkpoint(global_model, checkpoint_path, strict=False)
    start_ep = int(checkpoint_info["meta"].get("epoch", -1)) + 1
    logger.info("resume from: {}", args.checkpoint)

    global_model.set_train(False)
    logger.info("Evaluate global model on global Test dataset")
    test_acc, test_iou, confmat = test_inference(args, global_model, test_loader)
    logger.debug("Confusion matrix:\n{}", confmat)
    logger.info("Results after {} global rounds of training", start_ep)
    logger.info("Global Test Accuracy: {:.2f}%", test_acc)
    logger.info("Global Test IoU: {:.2f}%", test_iou)
    logger.info("Total Run Time: {:.2f}s", time.time() - start_time)


if __name__ == "__main__":
    main()

import sys

sys.path.append("/home/pjl/project/FedSeg/paddle_project")
import argparse
import os
import time
import warnings

import paddle
from federated_main import make_model
from myseg.dataloader import Cityscapes_Dataset
from myseg.datasplit import get_dataset_cityscapes
from paddle_utils import *
from update import test_inference
from utils import exp_details


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0", help="index of GPU to use")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="test colab gpu num_workers=1 is faster",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="bisenetv2",
        choices=["lraspp_mobilenetv3", "bisenetv2"],
        help="model name",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=21,
        help="number of classes max is 81, pretrained is 21",
    )
    parser.add_argument(
        "--checkpoint", type=str, default="", help="full file name of the checkpoint"
    )
    parser.add_argument(
        "--dataset", type=str, default="cityscapes", help="name of dataset"
    )
    parser.add_argument(
        "--root_dir", type=str, default="/home/data/cityscapes/", help="root of dataset"
    )
    parser.add_argument("--root", type=str, default="./", help="home directory")
    parser.add_argument(
        "--data",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="cityscapes train or val or test",
    )
    parser.add_argument("--USE_ERASE_DATA", type=str2bool, help="name of dataset")
    parser.add_argument("--proj_dim", type=int, default=256, help="name of dataset")
    parser.add_argument(
        "--rand_init", type=str2bool, default=False, help="name of dataset"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = args_parser()
    start_time = time.time()
    paddle.device.set_device(device=device2str(int(args.gpu)))
    device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"
    print("device: " + device)
    if args.dataset == "cityscapes":
        test_dataset = Cityscapes_Dataset(args.root_dir, args.data, args.USE_ERASE_DATA)
        print("args.data: ", args.data)
    else:
        exit("Error: unrecognized dataset")
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset, batch_size=4, num_workers=args.num_workers, shuffle=False
    )
    global_model = make_model(args)
    global_model.to(device)
    if args.checkpoint != "":
        checkpoint = paddle.load(
            path=str(os.path.join(args.root, "save/checkpoints", args.checkpoint))
        )
        global_model.set_state_dict(state_dict=checkpoint["model"])
        start_ep = checkpoint["epoch"] + 1
        print("resume from: ", args.checkpoint)
    else:
        exit("Error: args.checkpoint is empty")
    global_model.eval()
    print("\n*******************************************")
    print("Evaluate global model on global Test dataset")
    test_acc, test_iou, confmat = test_inference(args, global_model, test_loader)
    print(confmat)
    print("\nResults after {} global rounds of training:".format(start_ep))
    print("|---- Global Test Accuracy: {:.2f}%".format(test_acc))
    print("|---- Global Test IoU: {:.2f}%".format(test_iou))
    print("\nTotal Run Time: {:.2f}s".format(time.time() - start_time))
    print("*******************************************")

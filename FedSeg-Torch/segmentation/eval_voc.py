import argparse
import os
import sys
import time
import warnings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTATION_DIR = os.path.dirname(os.path.abspath(__file__))

if SEGMENTATION_DIR not in sys.path:
    sys.path.insert(0, SEGMENTATION_DIR)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from myseg.bisenet_utils import set_model_bisenetv2
from myseg.dataloader_camvid import CamVid_Dataset
from torch_runtime import require_torch_device
from update import test_inference


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
    parser = argparse.ArgumentParser(
        description="Evaluate one or more VOC checkpoints."
    )

    parser.add_argument("--gpu", type=str, default="0", help="index of GPU to use")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of dataloader workers",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="batch size for evaluation",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="bisenetv2",
        choices=["bisenetv2"],
        help="model name",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=20,
        help="number of VOC foreground classes used by this repo",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="voc",
        choices=["voc"],
        help="dataset name",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/voc",
        help="root directory of the VOC split used by this repo",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="val",
        choices=["train", "val"],
        help="dataset split to evaluate",
    )
    parser.add_argument(
        "--USE_ERASE_DATA",
        type=str2bool,
        default=True,
        help="kept for compatibility with existing args",
    )

    parser.add_argument(
        "--root",
        type=str,
        default=PROJECT_ROOT,
        help="project root, used to resolve relative paths",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="save/checkpoints",
        help="directory that stores checkpoints",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="checkpoint filenames or paths; if omitted, evaluate all .pth files in checkpoint_dir",
    )

    parser.add_argument(
        "--proj_dim",
        type=int,
        default=256,
        help="projection head dimension for BiSeNetV2",
    )
    parser.add_argument(
        "--rand_init",
        type=str2bool,
        default=True,
        help="skip loading backbone pretrain before loading the full checkpoint",
    )

    return parser.parse_args()


def build_model(args):
    if args.model != "bisenetv2":
        raise ValueError("Only bisenetv2 is supported in eval_voc.py")
    return set_model_bisenetv2(args=args, num_classes=args.num_classes)


def pad_collate_fn(batch):
    images, labels = zip(*batch)

    max_h = max(image.shape[1] for image in images)
    max_w = max(image.shape[2] for image in images)

    padded_images = []
    padded_labels = []
    for image, label in zip(images, labels):
        pad_h = max_h - image.shape[1]
        pad_w = max_w - image.shape[2]

        padded_images.append(F.pad(image, (0, pad_w, 0, pad_h), value=0.0))
        padded_labels.append(F.pad(label, (0, pad_w, 0, pad_h), value=255))

    return torch.stack(padded_images, dim=0), torch.stack(padded_labels, dim=0)


def resolve_path(root_dir, path_value):
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(root_dir, path_value))


def _unique_existing_paths(paths):
    seen = set()
    unique_paths = []
    for path in paths:
        norm_path = os.path.normpath(path)
        if norm_path in seen:
            continue
        seen.add(norm_path)
        unique_paths.append(norm_path)
    return unique_paths


def _case_insensitive_match(directory, target_name):
    if not os.path.isdir(directory):
        return None

    target_name_lower = target_name.lower()
    matches = [
        os.path.join(directory, entry)
        for entry in os.listdir(directory)
        if entry.lower() == target_name_lower
    ]
    if len(matches) == 1:
        return os.path.normpath(matches[0])
    return None


def resolve_checkpoint_path(root_dir, checkpoint_dir, checkpoint):
    if os.path.isabs(checkpoint):
        base_path = checkpoint
    elif os.path.dirname(checkpoint):
        base_path = os.path.join(root_dir, checkpoint)
    else:
        base_path = os.path.join(checkpoint_dir, checkpoint)

    candidate_paths = [base_path]
    if not os.path.splitext(base_path)[1]:
        candidate_paths.append(base_path + ".pth")

    candidate_paths = _unique_existing_paths(candidate_paths)
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    for candidate_path in candidate_paths:
        matched_path = _case_insensitive_match(
            os.path.dirname(candidate_path),
            os.path.basename(candidate_path),
        )
        if matched_path is not None:
            return matched_path

    available_checkpoints = []
    if os.path.isdir(checkpoint_dir):
        available_checkpoints = sorted(
            filename
            for filename in os.listdir(checkpoint_dir)
            if filename.endswith(".pth")
        )

    searched_paths = ", ".join(candidate_paths)
    message = "Checkpoint not found for '{}'. Searched: {}".format(
        checkpoint,
        searched_paths,
    )
    if available_checkpoints:
        message += ". Available checkpoints: {}".format(
            ", ".join(available_checkpoints)
        )
    raise FileNotFoundError(message)


def resolve_checkpoint_paths(args):
    checkpoint_dir = resolve_path(args.root, args.checkpoint_dir)

    if args.checkpoints:
        return [
            resolve_checkpoint_path(args.root, checkpoint_dir, checkpoint)
            for checkpoint in args.checkpoints
        ]

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            "Checkpoint directory not found: {}".format(checkpoint_dir)
        )

    discovered = sorted(
        os.path.join(checkpoint_dir, filename)
        for filename in os.listdir(checkpoint_dir)
        if filename.endswith(".pth")
    )
    if not discovered:
        raise FileNotFoundError(
            "No .pth checkpoints found in {}".format(checkpoint_dir)
        )
    return discovered


def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            "Checkpoint {} does not contain a 'model' state_dict".format(
                checkpoint_path
            )
        )
    return checkpoint


def evaluate_checkpoint(args, checkpoint_path, test_loader, device):
    print("\n" + "=" * 100)
    print("Evaluating checkpoint: {}".format(checkpoint_path))

    model = build_model(args)
    model.to(device)

    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    start_time = time.time()
    test_acc, test_iou, confmat = test_inference(args, model, test_loader)

    epoch = checkpoint.get("epoch")
    exp_name = checkpoint.get("exp_name", os.path.basename(checkpoint_path))
    rounds = epoch + 1 if isinstance(epoch, int) else "unknown"

    print("exp_name: {}".format(exp_name))
    print("saved_epoch: {}".format(epoch))
    print("effective_rounds: {}".format(rounds))
    print(confmat)
    print("|---- Global Test Accuracy: {:.2f}%".format(test_acc))
    print("|---- Global Test IoU: {:.2f}%".format(test_iou))
    print("|---- Eval Time: {:.2f}s".format(time.time() - start_time))

    return {
        "checkpoint": checkpoint_path,
        "exp_name": exp_name,
        "epoch": epoch,
        "acc": test_acc,
        "iou": test_iou,
    }


def main():
    args = args_parser()

    device = require_torch_device(args.gpu)
    print("device: {}".format(device))

    checkpoint_paths = resolve_checkpoint_paths(args)
    print("checkpoints to evaluate:")
    for checkpoint_path in checkpoint_paths:
        print("  - {}".format(checkpoint_path))

    dataset_root = resolve_path(args.root, args.root_dir)
    test_dataset = CamVid_Dataset(args, dataset_root, args.data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        collate_fn=pad_collate_fn,
    )

    results = []
    for checkpoint_path in checkpoint_paths:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))
        results.append(evaluate_checkpoint(args, checkpoint_path, test_loader, device))

    print("\n" + "=" * 100)
    print("Summary")
    for result in results:
        print(
            "{} | epoch={} | Acc={:.2f}% | mIoU={:.2f}%".format(
                os.path.basename(result["checkpoint"]),
                result["epoch"],
                result["acc"],
                result["iou"],
            )
        )


if __name__ == "__main__":
    main()

# Example commands from the project root:
# python segmentation/eval_voc.py
# python segmentation/eval_voc.py --checkpoints fedseg-torch.pth
# python segmentation/eval_voc.py --checkpoints save/checkpoints/fedseg-torch.pth
# python segmentation/eval_voc.py --gpu 0 --batch_size 4 --num_workers 4
# python segmentation/eval_voc.py --root_dir data/voc --data val

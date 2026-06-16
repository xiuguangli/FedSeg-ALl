import argparse
import os
import sys
import time
import warnings

import mindspore as ms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTATION_DIR = os.path.dirname(os.path.abspath(__file__))

if SEGMENTATION_DIR not in sys.path:
    sys.path.insert(0, SEGMENTATION_DIR)

from checkpoint_utils import load_training_checkpoint
from fast_eval import checkpoint_label, evaluate_grouped_dataset, format_runtime_detail, format_runtime_profile
from mindspore_runtime import setup_mindspore_device
from myseg.bisenet_utils import set_model_bisenetv2

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
    parser = argparse.ArgumentParser(description="Evaluate one or more MindSpore segmentation checkpoints.")
    parser.add_argument("--gpu", type=str, default="0", help="index of GPU to use")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of worker threads used to preload evaluation samples",
    )
    parser.add_argument("--batch_size", type=int, default=24, help="batch size for evaluation")
    parser.add_argument("--model", type=str, default="bisenetv2", choices=["bisenetv2"], help="model name")
    parser.add_argument("--num_classes", type=int, default=20, help="number of segmentation classes")
    parser.add_argument("--dataset", type=str, default="voc", choices=["cityscapes", "camvid", "ade20k", "voc"], help="dataset name")
    parser.add_argument("--root_dir", type=str, default="data/voc", help="root directory of the dataset split")
    parser.add_argument("--data", type=str, default="val", choices=["train", "val", "test"], help="dataset split to evaluate")
    parser.add_argument("--USE_ERASE_DATA", type=str2bool, default=True, help="kept for compatibility")
    parser.add_argument("--root", type=str, default=PROJECT_ROOT, help="project root")
    parser.add_argument("--checkpoint_dir", type=str, default="save/checkpoints", help="checkpoint directory")
    parser.add_argument(
        "--checkpoints",
        nargs="*",
        default=None,
        help="checkpoint filenames or paths; if omitted, evaluate all .ckpt files in checkpoint_dir",
    )
    parser.add_argument("--proj_dim", type=int, default=256, help="projection head dimension")
    parser.add_argument("--rand_init", type=str2bool, default=True, help="ignored here; checkpoint should provide weights")
    parser.add_argument(
        "--profile_runtime",
        type=str2bool,
        default=False,
        help="print lightweight runtime breakdown for eval stages",
    )
    return parser.parse_args()


def _setup_device(args):
    return setup_mindspore_device(args.gpu, mode=ms.PYNATIVE_MODE)


def build_model(args):
    if args.model != "bisenetv2":
        raise ValueError("Only bisenetv2 is supported in eval_voc.py")
    return set_model_bisenetv2(args=args, num_classes=args.num_classes)


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
        candidate_paths.append(base_path + ".ckpt")

    candidate_paths = _unique_existing_paths(candidate_paths)
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path

    for candidate_path in candidate_paths:
        matched_path = _case_insensitive_match(os.path.dirname(candidate_path), os.path.basename(candidate_path))
        if matched_path is not None:
            return matched_path

    available_checkpoints = []
    if os.path.isdir(checkpoint_dir):
        available_checkpoints = sorted(
            filename
            for filename in os.listdir(checkpoint_dir)
            if filename.endswith(".ckpt")
        )
    searched_paths = ", ".join(candidate_paths)
    message = "Checkpoint not found for '{}'. Searched: {}".format(checkpoint, searched_paths)
    if available_checkpoints:
        message += ". Available checkpoints: {}".format(", ".join(available_checkpoints))
    raise FileNotFoundError(message)


def resolve_checkpoint_paths(args):
    checkpoint_dir = resolve_path(args.root, args.checkpoint_dir)
    if args.checkpoints:
        return [
            resolve_checkpoint_path(args.root, checkpoint_dir, checkpoint)
            for checkpoint in args.checkpoints
        ]

    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError("Checkpoint directory not found: {}".format(checkpoint_dir))

    discovered = sorted(
        os.path.join(checkpoint_dir, filename)
        for filename in os.listdir(checkpoint_dir)
        if filename.endswith(".ckpt")
    )
    if not discovered:
        raise FileNotFoundError("No checkpoints found in {}".format(checkpoint_dir))
    return discovered


def load_checkpoint(model, checkpoint_path):
    suffix = os.path.splitext(checkpoint_path)[1].lower()
    if suffix == ".ckpt":
        checkpoint_info = load_training_checkpoint(model, checkpoint_path, strict=False)
        epoch = checkpoint_info["meta"].get("epoch")
        exp_name = checkpoint_info["meta"].get("exp_name", os.path.basename(checkpoint_path))
        return {
            "epoch": epoch,
            "exp_name": exp_name,
            "path": checkpoint_path,
        }
    raise ValueError("Only MindSpore .ckpt checkpoints are supported: {}".format(checkpoint_path))

def evaluate_checkpoint(args, checkpoint_path):
    overall_start = time.perf_counter()
    model = build_model(args)
    model_build_time = time.perf_counter() - overall_start

    checkpoint_start = time.perf_counter()
    checkpoint = load_checkpoint(model, checkpoint_path)
    checkpoint_load_time = time.perf_counter() - checkpoint_start
    model.set_train(False)

    eval_result = evaluate_grouped_dataset(
        args,
        model,
        split=args.data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bucket_align=32,
        profile_runtime=args.profile_runtime,
        progress_desc=checkpoint_label(checkpoint_path),
    )
    total_time = time.perf_counter() - overall_start
    runtime = dict(eval_result["runtime"])
    runtime.update(
        {
            "build_model": model_build_time,
            "load_checkpoint": checkpoint_load_time,
            "total": total_time,
        }
    )
    return {
        "checkpoint": checkpoint["path"],
        "epoch": checkpoint.get("epoch"),
        "exp_name": checkpoint.get("exp_name"),
        "num_samples": eval_result["num_samples"],
        "acc": eval_result["acc"],
        "iou": eval_result["iou"],
        "confmat": eval_result["confmat"],
        "runtime": runtime,
    }


def main():
    args = args_parser()
    device = _setup_device(args)
    checkpoint_paths = resolve_checkpoint_paths(args)

    print("device: {}".format(device))
    print("checkpoints to evaluate:")
    for checkpoint_path in checkpoint_paths:
        print("  - {}".format(checkpoint_path))

    results = []
    for checkpoint_path in checkpoint_paths:
        print("\n" + "=" * 100)
        print("Evaluating checkpoint: {}".format(checkpoint_path))
        result = evaluate_checkpoint(args, checkpoint_path)
        results.append(result)
        print("exp_name: {}".format(result["exp_name"]))
        print("saved_epoch: {}".format(result["epoch"]))
        rounds = result["epoch"] + 1 if isinstance(result["epoch"], int) else "unknown"
        print("effective_rounds: {}".format(rounds))
        print("num_samples: {}".format(result["num_samples"]))
        print(result["confmat"])
        print("|---- Global Test Accuracy: {:.2f}%".format(result["acc"]))
        print("|---- Global Test IoU: {:.2f}%".format(result["iou"]))
        print("|---- Eval Time: {:.2f}s".format(result["runtime"]["total"]))
        if args.profile_runtime:
            runtime = result["runtime"]
            print(
                "Runtime profile: build_model={:.2f}s load_checkpoint={:.2f}s {}".format(
                    runtime["build_model"],
                    runtime["load_checkpoint"],
                    format_runtime_profile(runtime).replace("Runtime profile: ", ""),
                )
            )
            print(format_runtime_detail(runtime))

    print("\n" + "=" * 100)
    print("Summary")
    for result in results:
        print(
            "{} | epoch={} | Acc={:.2f}% | mIoU={:.2f}% | EvalTime={:.2f}s".format(
                os.path.basename(result["checkpoint"]),
                result["epoch"],
                result["acc"],
                result["iou"],
                result["runtime"]["total"],
            )
        )


if __name__ == "__main__":
    main()

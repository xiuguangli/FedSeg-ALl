import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from federated_main import FederatedTrainer
from logging_utils import setup_logger
from options import args_parser
from seed_utils import seed_everything


DEFAULT_MODULES = [
    "detail.S1.0.conv",
    "detail.S1.0.bn",
    "detail.S1.0.relu",
    "segment.S1S2.conv.conv",
    "segment.S1S2.conv.bn",
    "segment.S1S2.conv.relu",
    "segment.S1S2.left.0.conv",
    "segment.S1S2.left.0.bn",
    "segment.S1S2.left.0.relu",
]


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--global_round", type=int, required=True)
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state_path", type=Path, default=None)
    parser.add_argument("--module", action="append", dest="modules", default=None)
    export_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return export_args, train_args


def sanitize_name(name):
    return name.replace(".", "_dot_").replace("/", "_slash_")


def install_activation_trace(model, target_modules):
    traces = {}
    hooks = []

    for name, module in model.named_modules():
        if name not in target_modules:
            continue

        def hook(_module, _inputs, output, _name=name):
            if isinstance(output, torch.Tensor):
                traces[_name] = output.detach().cpu().numpy().astype(np.float32, copy=True)

        hooks.append(module.register_forward_hook(hook))
    return traces, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def save_trace(output_path, traces, metadata):
    arrays = {}
    module_map = {}
    for module_name, value in traces.items():
        safe = sanitize_name(module_name)
        module_map[safe] = module_name
        arrays[safe] = value

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **arrays)
    with output_path.with_suffix(output_path.suffix + ".json").open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "metadata": metadata,
                "module_map": module_map,
            },
            fout,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )


def main():
    export_args, args = parse_args()
    args.USE_WANDB = False
    args.globalema = False
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_activation_trace_torch")
    seed_everything(args.seed)

    trainer = FederatedTrainer(args)
    if export_args.state_path is not None:
        checkpoint = torch.load(str(export_args.state_path), map_location=trainer.device)
        trainer.global_model.load_state_dict(checkpoint["model"], strict=False)
        trainer.global_weights = copy.deepcopy(trainer.global_model.state_dict())
    client = trainer._make_client(export_args.client_id)

    model = copy.deepcopy(trainer.global_model).to(trainer.device)
    model.train()
    trainloader = client._build_trainloader(export_args.global_round)
    images = None
    labels = None
    for local_epoch in range(int(args.local_ep)):
        for batch_index, batch in enumerate(trainloader):
            if local_epoch == int(export_args.epoch) and batch_index == int(export_args.batch_index):
                images, labels = batch
                break
        if images is not None:
            break
    if images is None:
        raise ValueError("batch_index {} out of range".format(export_args.batch_index))

    images = images.to(trainer.device)
    target_modules = list(export_args.modules or DEFAULT_MODULES)
    traces, hooks = install_activation_trace(model, target_modules)
    try:
        with torch.no_grad():
            _ = model(images)
    finally:
        remove_hooks(hooks)

    metadata = {
        "framework": "torch",
        "checkpoint": str(export_args.state_path) if export_args.state_path is not None else "",
        "client_id": int(export_args.client_id),
        "global_round": int(export_args.global_round),
        "epoch": int(export_args.epoch),
        "batch_index": int(export_args.batch_index),
        "modules": target_modules,
        "input_shape": list(images.shape),
        "label_shape": list(labels.shape),
    }
    save_trace(export_args.output, traces, metadata)
    print(
        "EXPORT_ACTIVATION_TRACE "
        + json.dumps(
            {
                "output": str(export_args.output),
                "metadata": metadata,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

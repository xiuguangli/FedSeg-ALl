import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from federated_main import FederatedTrainer
from logging_utils import setup_logger
from myseg.bisenet_utils import ContrastLoss
from options import args_parser
from seed_utils import seed_everything


DEFAULT_LAYERS = [
    "bga.conv.1",
    "segment.S1S2.fuse.bn",
    "segment.S4.0.conv1.bn",
    "aux5_4.conv.bn",
    "head.conv.bn",
]
TRACE_SEP = "__TRACE__"


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--global_round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state_path", type=Path, default=None)
    parser.add_argument("--deterministic_contrast", action="store_true")
    parser.add_argument("--layer", action="append", dest="layers", default=None)
    export_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return export_args, train_args


def install_deterministic_anchor_sampling():
    original = ContrastLoss._anchor_sampling

    def deterministic_anchor_sampling(self, embs, labels):
        device = embs.device
        _, channels, _, _ = embs.size()
        embs_flat = embs.permute(0, 2, 3, 1).reshape(-1, channels)
        labels_flat = labels.view(-1)
        index_all = torch.arange(len(labels_flat), device=device)
        class_u = torch.unique(labels_flat)
        class_u = class_u[class_u != 255]

        if len(class_u) == 0:
            return None, None

        num_per_class = max(1, int(self.max_anchor) // int(len(class_u)))
        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
            selected_index = torch.masked_select(index_all, labels_flat == cls_)
            if len(selected_index) > num_per_class:
                selected_index = selected_index[:num_per_class]
            sampled_list.append(embs_flat[selected_index])
            sampled_label_list.append(torch.ones(len(selected_index), device=device) * cls_)

        return torch.cat(sampled_list, 0), torch.cat(sampled_label_list, 0)

    ContrastLoss._anchor_sampling = deterministic_anchor_sampling
    return original


def sanitize_name(name):
    return name.replace(".", "_dot_").replace("/", "_slash_")


def install_bn_trace(model, target_layers):
    traces = {
        name: {
            "input_mean": [],
            "input_var": [],
            "pre_mean": [],
            "pre_var": [],
            "post_mean": [],
            "post_var": [],
        }
        for name in target_layers
    }
    hooks = []

    for name, module in model.named_modules():
        if name not in target_layers:
            continue
        expected_module_id = id(module)

        def pre_hook(mod, inputs, _name=name, _expected_module_id=expected_module_id):
            if id(mod) != _expected_module_id:
                return
            x = inputs[0].detach()
            traces[_name]["input_mean"].append(x.mean(dim=(0, 2, 3)).cpu().numpy().astype(np.float32))
            traces[_name]["input_var"].append(x.var(dim=(0, 2, 3), unbiased=False).cpu().numpy().astype(np.float32))
            traces[_name]["pre_mean"].append(mod.running_mean.detach().cpu().numpy().copy().astype(np.float32))
            traces[_name]["pre_var"].append(mod.running_var.detach().cpu().numpy().copy().astype(np.float32))

        def post_hook(mod, inputs, output, _name=name, _expected_module_id=expected_module_id):
            if id(mod) != _expected_module_id:
                return
            traces[_name]["post_mean"].append(mod.running_mean.detach().cpu().numpy().copy().astype(np.float32))
            traces[_name]["post_var"].append(mod.running_var.detach().cpu().numpy().copy().astype(np.float32))

        hooks.append(module.register_forward_pre_hook(pre_hook))
        hooks.append(module.register_forward_hook(post_hook))

    return traces, hooks


def remove_hooks(hooks):
    for hook in hooks:
        hook.remove()


def save_trace(output_path, traces, metadata):
    arrays = {}
    layer_map = {}
    for layer_name, stats in traces.items():
        safe_name = sanitize_name(layer_name)
        layer_map[safe_name] = layer_name
        for stat_name, values in stats.items():
            key = safe_name + TRACE_SEP + stat_name
            arrays[key] = np.stack(values, axis=0) if values else np.zeros((0,), dtype=np.float32)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **arrays)
    with output_path.with_suffix(output_path.suffix + ".json").open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "metadata": metadata,
                "layer_map": layer_map,
                "trace_sep": TRACE_SEP,
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
    args.profile_runtime = False
    args.checkpoint = ""
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_bn_trace_torch")
    seed_everything(args.seed)

    target_layers = list(export_args.layers or DEFAULT_LAYERS)
    restore_anchor_sampling = None
    if export_args.deterministic_contrast:
        restore_anchor_sampling = install_deterministic_anchor_sampling()

    try:
        trainer = FederatedTrainer(args)
        if export_args.state_path is not None:
            checkpoint = torch.load(str(export_args.state_path), map_location=trainer.device)
            trainer.global_model.load_state_dict(checkpoint["model"], strict=False)
            trainer.global_weights = copy.deepcopy(trainer.global_model.state_dict())

        client = trainer._make_client(export_args.client_id)
        local_mem, local_mask = trainer._prepare_prototypes(client, export_args.global_round)
        model = copy.deepcopy(trainer.global_model).to(trainer.device)
        model.train()
        traces, hooks = install_bn_trace(model, target_layers)
        try:
            _, loss = client.train(
                model=model,
                global_round=export_args.global_round,
                prototypes=local_mem,
                proto_mask=local_mask,
            )
        finally:
            remove_hooks(hooks)

        num_steps = 0
        if target_layers:
            num_steps = len(traces[target_layers[0]]["input_mean"])
        metadata = {
            "framework": "torch",
            "checkpoint": str(export_args.state_path) if export_args.state_path is not None else "",
            "client_id": export_args.client_id,
            "global_round": export_args.global_round,
            "deterministic_contrast": bool(export_args.deterministic_contrast),
            "loss": float(loss),
            "num_steps": int(num_steps),
            "layers": target_layers,
        }
        save_trace(export_args.output, traces, metadata)
        print(
            "EXPORT_BN_TRACE "
            + json.dumps(
                {
                    "output": str(export_args.output),
                    "metadata": metadata,
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    finally:
        if restore_anchor_sampling is not None:
            ContrastLoss._anchor_sampling = restore_anchor_sampling


if __name__ == "__main__":
    main()

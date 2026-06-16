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


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--global_round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state_path", type=Path, default=None)
    parser.add_argument("--deterministic_contrast", action="store_true")
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


def save_export(output_path, state_dict, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(state_dict.keys())
    arrays = {
        "arr_{:04d}".format(index): state_dict[name]
        for index, name in enumerate(names)
    }
    np.savez_compressed(str(output_path), **arrays)
    meta_path = output_path.with_suffix(output_path.suffix + ".json")
    with meta_path.open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "names": names,
                "metadata": metadata,
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
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_local_update_torch")
    seed_everything(args.seed)

    restore_anchor_sampling = None
    if export_args.deterministic_contrast:
        restore_anchor_sampling = install_deterministic_anchor_sampling()

    try:
        trainer = FederatedTrainer(args)
        if export_args.state_path is not None:
            checkpoint = torch.load(str(export_args.state_path), map_location=trainer.device)
            load_result = trainer.global_model.load_state_dict(checkpoint["model"], strict=False)
            trainer.global_weights = copy.deepcopy(trainer.global_model.state_dict())
        else:
            load_result = None

        client = trainer._make_client(export_args.client_id)
        local_mem, local_mask = trainer._prepare_prototypes(client, export_args.global_round)
        updated_state, loss = client.train(
            model=copy.deepcopy(trainer.global_model),
            global_round=export_args.global_round,
            prototypes=local_mem,
            proto_mask=local_mask,
        )
        host_state = {name: tensor.detach().cpu().numpy().copy() for name, tensor in updated_state.items()}
        metadata = {
            "framework": "torch",
            "checkpoint": str(export_args.state_path) if export_args.state_path is not None else "",
            "client_id": export_args.client_id,
            "global_round": export_args.global_round,
            "loss": float(loss),
            "deterministic_contrast": bool(export_args.deterministic_contrast),
            "load_missing": [] if load_result is None else list(load_result.missing_keys),
            "load_unexpected": [] if load_result is None else list(load_result.unexpected_keys),
        }
        save_export(export_args.output, host_state, metadata)
        print(
            "EXPORT_LOCAL_UPDATE "
            + json.dumps(
                {
                    "output": str(export_args.output),
                    "metadata": metadata,
                    "num_tensors": len(host_state),
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

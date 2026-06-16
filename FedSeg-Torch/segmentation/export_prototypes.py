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


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--global_round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state_path", type=Path, default=None)
    export_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return export_args, train_args


def save_export(output_path, arrays, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output_path), **arrays)
    with output_path.with_suffix(output_path.suffix + ".json").open("w", encoding="utf-8") as fout:
        json.dump(
            {"metadata": metadata},
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
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_prototypes_torch")
    seed_everything(args.seed)

    trainer = FederatedTrainer(args)
    if export_args.state_path is not None:
        checkpoint = torch.load(str(export_args.state_path), map_location=trainer.device)
        trainer.global_model.load_state_dict(checkpoint["model"], strict=False)
        trainer.global_weights = copy.deepcopy(trainer.global_model.state_dict())

    client = trainer._make_client(export_args.client_id)
    prototypes, proto_mask = trainer._prepare_prototypes(client, export_args.global_round)

    arrays = {
        "proto": prototypes.detach().cpu().numpy().copy(),
        "mask": proto_mask.detach().cpu().numpy().copy(),
        "eval_indices": np.asarray(client.idxs, dtype=np.int32),
    }
    metadata = {
        "framework": "torch",
        "checkpoint": str(export_args.state_path) if export_args.state_path is not None else "",
        "client_id": export_args.client_id,
        "global_round": export_args.global_round,
        "proto_shape": list(prototypes.shape),
        "mask_shape": list(proto_mask.shape),
    }
    save_export(export_args.output, arrays, metadata)
    print(
        "EXPORT_PROTOTYPES "
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

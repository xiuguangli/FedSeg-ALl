import argparse
import json
from pathlib import Path

import numpy as np

from federated_main import FederatedTrainer
from logging_utils import setup_logger
from options import args_parser
from seed_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--global_round", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    export_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return export_args, train_args


def save_batches(output_path, batch_records, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    json_batches = []
    for batch_index, record in enumerate(batch_records):
        arrays[f"images_{batch_index:03d}"] = record["images"]
        arrays[f"labels_{batch_index:03d}"] = record["labels"]
        arrays[f"indices_{batch_index:03d}"] = np.asarray(record["indices"], dtype=np.int32)
        json_batches.append(
            {
                "batch_index": batch_index,
                "epoch": record["epoch"],
                "indices": [int(idx) for idx in record["indices"]],
                "shape_images": list(record["images"].shape),
                "shape_labels": list(record["labels"].shape),
                "sum_images": float(record["images"].sum()),
                "sum_labels": float(record["labels"].sum()),
            }
        )
    np.savez_compressed(str(output_path), **arrays)
    with output_path.with_suffix(output_path.suffix + ".json").open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "metadata": metadata,
                "batches": json_batches,
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
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_batch_sequence_torch")
    seed_everything(args.seed)

    trainer = FederatedTrainer(args)
    client = trainer._make_client(export_args.client_id)

    trainloader = client._build_trainloader(export_args.global_round)
    batch_records = []
    for local_epoch in range(args.local_ep):
        epoch_absolute_indices = client.debug_train_epoch_absolute_indices(export_args.global_round, local_epoch)
        batch_size = int(args.local_bs)
        for batch_index, (images, labels) in enumerate(trainloader):
            start = batch_index * batch_size
            stop = start + batch_size
            batch_records.append(
                {
                    "epoch": int(local_epoch),
                    "images": images.detach().cpu().numpy().copy(),
                    "labels": labels.detach().cpu().numpy().copy(),
                    "indices": epoch_absolute_indices[start:stop],
                }
            )

    metadata = {
        "framework": "torch",
        "client_id": export_args.client_id,
        "global_round": export_args.global_round,
        "local_ep": args.local_ep,
        "local_bs": args.local_bs,
        "num_batches": len(batch_records),
        "relative_epoch_orders": [
            client.debug_train_epoch_relative_indices(export_args.global_round, local_epoch)
            for local_epoch in range(args.local_ep)
        ],
    }
    save_batches(export_args.output, batch_records, metadata)
    print(
        "EXPORT_BATCH_SEQUENCE "
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

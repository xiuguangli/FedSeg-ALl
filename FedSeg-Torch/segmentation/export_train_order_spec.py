import argparse
import json
from pathlib import Path

from federated_main import FederatedTrainer
from logging_utils import setup_logger
from options import args_parser
from seed_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--round_start", type=int, required=True)
    parser.add_argument("--round_end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    export_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return export_args, train_args


def main():
    export_args, args = parse_args()
    args.USE_WANDB = False
    args.globalema = False
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_train_order_spec_torch")
    seed_everything(args.seed)

    trainer = FederatedTrainer(args)
    round_client_orders = {}
    selected_clients = {}
    for global_round in range(int(export_args.round_start), int(export_args.round_end) + 1):
        selected = [int(idx) for idx in trainer._select_clients(global_round)]
        selected_clients[str(global_round)] = selected
        client_orders = {}
        for client_id in range(int(args.num_users)):
            client = trainer._make_client(client_id)
            client_orders[str(client_id)] = {
                "client_id": int(client_id),
                "global_round": int(global_round),
                "relative_epoch_orders": [
                    client.debug_train_epoch_relative_indices(global_round, local_epoch)
                    for local_epoch in range(int(args.local_ep))
                ],
            }
        round_client_orders[str(global_round)] = client_orders

    payload = {
        "metadata": {
            "framework": "torch",
            "round_start": int(export_args.round_start),
            "round_end": int(export_args.round_end),
            "num_users": int(args.num_users),
            "local_ep": int(args.local_ep),
            "local_bs": int(args.local_bs),
            "seed": int(args.seed),
        },
        "selected_clients": selected_clients,
        "round_client_orders": round_client_orders,
    }
    export_args.output.parent.mkdir(parents=True, exist_ok=True)
    with export_args.output.open("w", encoding="utf-8") as fout:
        json.dump(payload, fout, ensure_ascii=True, indent=2, sort_keys=True)
    print(
        "EXPORT_TRAIN_ORDER_SPEC "
        + json.dumps(
            {
                "output": str(export_args.output),
                "round_start": int(export_args.round_start),
                "round_end": int(export_args.round_end),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

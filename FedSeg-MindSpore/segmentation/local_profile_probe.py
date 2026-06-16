import argparse
import json
import time

import mindspore as ms

from checkpoint_utils import clone_state_dict
from federated_main import FederatedTrainer
from logging_utils import logger, setup_logger
from options import args_parser
from seed_utils import seed_everything


def parse_probe_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, default=2)
    parser.add_argument("--global_round", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    probe_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return probe_args, train_args


def sync_device():
    runtime_api = getattr(ms, "runtime", None)
    hal_api = getattr(ms, "hal", None)
    sync_fn = getattr(runtime_api, "synchronize", None)
    if sync_fn is None:
        sync_fn = getattr(hal_api, "synchronize", None)
    if sync_fn is None:
        return
    try:
        sync_fn()
    except Exception:
        return


def mean_metric(records, key):
    values = [record[key] for record in records]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def main():
    probe_args, args = parse_probe_args()
    args.profile_runtime = True
    args.profile_runtime_detail = bool(getattr(args, "profile_runtime_detail", False))
    args.profile_runtime_sync = bool(getattr(args, "profile_runtime_sync", False))
    args.USE_WANDB = False
    args.globalema = False
    args.final_eval_precise = False
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="mindspore_local_probe")
    seed_everything(args.seed)

    trainer = FederatedTrainer(args)
    client = trainer._make_client(probe_args.client_id)
    num_batches = len(client._build_trainloader(probe_args.global_round))

    logger.info(
        "MindSpore local profile probe | mode={} client={} round={} samples={} batches={} warmup={} repeat={}",
        args.ms_mode,
        client.client_id,
        probe_args.global_round,
        client.num_samples,
        num_batches,
        probe_args.warmup,
        probe_args.repeat,
    )

    all_runs = []
    total_runs = probe_args.warmup + probe_args.repeat
    round_model_state = clone_state_dict(trainer.global_model.parameters_dict(), host=True)
    for run_idx in range(total_runs):
        sync_device()
        start = time.perf_counter()
        local_mem, local_mask = trainer._prepare_prototypes(
            client,
            probe_args.global_round,
            round_model_state,
        )
        sync_device()
        proto_time = time.perf_counter() - start

        sync_device()
        start = time.perf_counter()
        _, loss = trainer.local_trainer.train(
            client=client,
            model_state=round_model_state,
            global_round=probe_args.global_round,
            prototypes=local_mem,
            proto_mask=local_mask,
        )
        sync_device()
        train_time = time.perf_counter() - start

        record = {
            "run": run_idx,
            "warmup": run_idx < probe_args.warmup,
            "proto_time_s": proto_time,
            "train_time_s": train_time,
            "total_time_s": proto_time + train_time,
            "loss": float(loss),
        }
        all_runs.append(record)
        logger.info(
            "MindSpore local profile probe run={} warmup={} proto={:.3f}s train={:.3f}s total={:.3f}s loss={:.6f}",
            run_idx,
            record["warmup"],
            proto_time,
            train_time,
            proto_time + train_time,
            float(loss),
        )

    measured = [record for record in all_runs if not record["warmup"]]
    summary = {
        "framework": "mindspore",
        "mode": args.ms_mode,
        "client_id": client.client_id,
        "global_round": probe_args.global_round,
        "samples": client.num_samples,
        "batches": num_batches,
        "warmup": probe_args.warmup,
        "repeat": probe_args.repeat,
        "avg_proto_time_s": mean_metric(measured, "proto_time_s"),
        "avg_train_time_s": mean_metric(measured, "train_time_s"),
        "avg_total_time_s": mean_metric(measured, "total_time_s"),
        "avg_loss": mean_metric(measured, "loss"),
        "runs": all_runs,
    }
    print("BENCHMARK_SUMMARY " + json.dumps(summary, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()

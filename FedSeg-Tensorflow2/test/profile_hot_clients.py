from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile hot client runtime for FedSeg TF2/Torch.")
    parser.add_argument("--backend", choices=["tf2", "torch", "compare"], required=True)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--dataset", type=str, default="voc")
    parser.add_argument("--root-dir", type=str, default="data/voc")
    parser.add_argument("--tf-root-dir", type=str, default=None)
    parser.add_argument("--torch-root-dir", type=str, default=None)
    parser.add_argument("--warm-round", type=int, default=0)
    parser.add_argument("--target-round", type=int, default=1)
    parser.add_argument("--warm-client", type=int, default=None)
    parser.add_argument("--clients", type=str, default="")
    parser.add_argument("--frac-num", type=int, default=5)
    parser.add_argument("--local-ep", type=int, default=2)
    parser.add_argument("--local-bs", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--profile-runtime", action="store_true")
    parser.add_argument("--json-output", type=str, default=None)
    return parser.parse_args()


def build_common_argv(args: argparse.Namespace) -> list[str]:
    dataset_to_classes = {
        "voc": 20,
        "ade20k": 150,
        "camvid": 11,
        "cityscapes": 19,
    }
    num_classes = dataset_to_classes[args.dataset]
    return [
        "--gpu",
        args.gpu,
        "--dataset",
        args.dataset,
        "--root_dir",
        args.root_dir,
        "--USE_ERASE_DATA",
        "True",
        "--num_classes",
        str(num_classes),
        "--data",
        "train",
        "--num_workers",
        str(args.num_workers),
        "--model",
        "bisenetv2",
        "--checkpoint",
        "",
        "--lr",
        "0.05",
        "--lr_scheduler",
        "step",
        "--iid",
        "False",
        "--num_users",
        "60",
        "--frac_num",
        str(args.frac_num),
        "--epochs",
        str(max(args.target_round + 1, 2)),
        "--local_ep",
        str(args.local_ep),
        "--local_bs",
        str(args.local_bs),
        "--is_proto",
        "True",
        "--losstype",
        "back",
        "--fedprox_mu",
        "0",
        "--label_online_gen",
        "False",
        "--distill",
        "False",
        "--distill_lamb_pi",
        "0.1",
        "--distill_lamb_pa",
        "0",
        "--rand_init",
        "False",
        "--warmstep",
        "20",
        "--globalema",
        "False",
        "--temp_dist",
        "0.1",
        "--mixlabel",
        "True",
        "--proto_start_epoch",
        "1",
        "--con_lamb",
        "0.1",
        "--con_lamb_local",
        "1",
        "--momentum",
        "0.99",
        "--temperature",
        "0.07",
        "--max_anchor",
        "4096",
        "--kmean_num",
        "2",
        "--pseudo_label",
        "True",
        "--pseudo_label_start_epoch",
        "1",
        "--localmem",
        "True",
        "--mom_update",
        "False",
        "--save_frequency",
        "9999",
        "--local_test_frequency",
        "9999",
        "--global_test_frequency",
        "9999",
        "--USE_WANDB",
        "0",
        "--seed",
        str(args.seed),
        "--date_now",
        f"profile_{args.backend}",
        "--profile_runtime",
        "True" if args.profile_runtime else "False",
    ]


def resolve_root_dir(repo_dir: Path, root_dir: str) -> str:
    candidate = Path(root_dir)
    if candidate.is_absolute():
        return str(candidate)

    if candidate.parts and candidate.parts[0] == repo_dir.name:
        stripped = Path(*candidate.parts[1:])
        resolved = (repo_dir / stripped).resolve()
        if resolved.exists():
            return str(resolved)

    search_bases = [Path.cwd(), repo_dir, REPO_ROOT]
    for base in search_bases:
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return str(resolved)

    return str(candidate)


def _prepare_import_paths(repo_dir: Path) -> None:
    seg_dir = repo_dir / "segmentation"
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    if str(seg_dir) not in sys.path:
        sys.path.insert(0, str(seg_dir))


def load_backend(args: argparse.Namespace):
    repo_dir = REPO_ROOT / ("FedSeg-tf2" if args.backend == "tf2" else "FedSeg-torch")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("FEDSEG_DISABLE_TQDM", "1")
    os.chdir(repo_dir)
    _prepare_import_paths(repo_dir)
    args.root_dir = resolve_root_dir(repo_dir, args.root_dir)

    if args.backend == "tf2":
        from segmentation.options import args_parser
        from segmentation.federated_main import FederatedTrainer, set_seed
        from segmentation.logging_utils import setup_logger

        backend_args = args_parser(build_common_argv(args))
    else:
        from segmentation.options import args_parser
        from segmentation.federated_main import FederatedTrainer, set_seed
        from segmentation.logging_utils import setup_logger

        backend_args = args_parser(build_common_argv(args))

    setup_logger(
        verbose=False,
        logs_dir=str(repo_dir / "logs" / "profile_hot_clients"),
        log_name=f"{args.backend}_{args.dataset}",
    )
    set_seed(backend_args.seed)
    trainer = FederatedTrainer(backend_args)
    return trainer


def summarize_results(results: list[dict[str, float | int]]) -> dict[str, float]:
    return {
        "prototype_mean_sec": sum(item["prototype_sec"] for item in results) / len(results),
        "train_mean_sec": sum(item["train_sec"] for item in results) / len(results),
        "total_mean_sec": sum(item["total_sec"] for item in results) / len(results),
    }


def run_single_client(trainer, backend: str, round_idx: int, client_id: int) -> dict[str, float | int]:
    client = trainer._make_client(client_id)
    start = time.perf_counter()
    prototypes, proto_mask = trainer._prepare_prototypes(client, round_idx)
    prototype_time = time.perf_counter() - start

    if backend == "tf2":
        trainer.global_model.trainable = False
        start = time.perf_counter()
        _weights, loss = client.train(
            model=trainer.global_model,
            global_round=round_idx,
            prototypes=prototypes,
            proto_mask=proto_mask,
        )
        train_time = time.perf_counter() - start
        trainer.global_model.trainable = True
    else:
        start = time.perf_counter()
        _weights, loss = client.train(
            model=copy.deepcopy(trainer.global_model),
            global_round=round_idx,
            prototypes=prototypes,
            proto_mask=proto_mask,
        )
        train_time = time.perf_counter() - start

    return {
        "client_id": int(client_id),
        "samples": int(client.num_samples),
        "prototype_sec": float(prototype_time),
        "train_sec": float(train_time),
        "total_sec": float(prototype_time + train_time),
        "loss": float(loss),
    }


def resolve_clients(args: argparse.Namespace, trainer) -> tuple[int, list[int], list[int]]:
    warm_candidates = [int(idx) for idx in trainer._select_clients(args.warm_round)]
    target_candidates = [int(idx) for idx in trainer._select_clients(args.target_round)]

    warm_client = args.warm_client if args.warm_client is not None else warm_candidates[0]
    if args.clients.strip():
        target_clients = [int(part.strip()) for part in args.clients.split(",") if part.strip()]
    else:
        target_clients = target_candidates
    return int(warm_client), warm_candidates, target_clients


def extract_json_object(text: str) -> dict[str, object]:
    decoder = json.JSONDecoder()
    for start_idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _end_idx = decoder.raw_decode(text[start_idx:])
            return payload
        except json.JSONDecodeError:
            continue
    raise ValueError("No JSON payload found in command output")


def _comparison_entry(tf_value: float, torch_value: float) -> dict[str, float | None]:
    ratio = None if torch_value == 0 else tf_value / torch_value
    return {
        "tf2": tf_value,
        "torch": torch_value,
        "delta_sec": tf_value - torch_value,
        "ratio": ratio,
    }


def build_comparison_payload(tf_payload: dict[str, object], torch_payload: dict[str, object]) -> dict[str, object]:
    tf_summary = tf_payload.get("summary", {})
    torch_summary = torch_payload.get("summary", {})
    summary = {
        "prototype_mean_sec": _comparison_entry(
            float(tf_summary.get("prototype_mean_sec", 0.0)),
            float(torch_summary.get("prototype_mean_sec", 0.0)),
        ),
        "train_mean_sec": _comparison_entry(
            float(tf_summary.get("train_mean_sec", 0.0)),
            float(torch_summary.get("train_mean_sec", 0.0)),
        ),
        "total_mean_sec": _comparison_entry(
            float(tf_summary.get("total_mean_sec", 0.0)),
            float(torch_summary.get("total_mean_sec", 0.0)),
        ),
    }

    torch_results = {
        int(item["client_id"]): item
        for item in torch_payload.get("results", [])
    }
    per_client = []
    for tf_item in tf_payload.get("results", []):
        client_id = int(tf_item["client_id"])
        torch_item = torch_results.get(client_id)
        if torch_item is None:
            continue
        per_client.append(
            {
                "client_id": client_id,
                "samples": int(tf_item["samples"]),
                "prototype_sec": _comparison_entry(float(tf_item["prototype_sec"]), float(torch_item["prototype_sec"])),
                "train_sec": _comparison_entry(float(tf_item["train_sec"]), float(torch_item["train_sec"])),
                "total_sec": _comparison_entry(float(tf_item["total_sec"]), float(torch_item["total_sec"])),
            }
        )

    return {
        "dataset": tf_payload.get("dataset"),
        "gpu": tf_payload.get("gpu"),
        "warm_round": tf_payload.get("warm_round"),
        "target_round": tf_payload.get("target_round"),
        "warm_client": tf_payload.get("warm_client"),
        "target_clients": [int(item["client_id"]) for item in tf_payload.get("results", [])],
        "summary": summary,
        "per_client": per_client,
    }


def emit_payload(payload: dict[str, object], json_output: str | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=True, indent=2)
    print(rendered)
    if json_output:
        output_path = Path(json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


def run_single_backend_profile(args: argparse.Namespace) -> dict[str, object]:
    trainer = load_backend(args)
    warm_client, warm_candidates, target_clients = resolve_clients(args, trainer)

    payload: dict[str, object] = {
        "backend": args.backend,
        "dataset": args.dataset,
        "gpu": args.gpu,
        "warm_round": args.warm_round,
        "target_round": args.target_round,
        "warm_round_selected_clients": warm_candidates,
        "target_round_selected_clients": [int(idx) for idx in trainer._select_clients(args.target_round)],
        "warm_client": warm_client,
        "results": [],
    }

    warmup_result = run_single_client(trainer, args.backend, args.warm_round, warm_client)
    payload["warmup"] = warmup_result

    results = [run_single_client(trainer, args.backend, args.target_round, client_id) for client_id in target_clients]
    payload["results"] = results
    if results:
        payload["summary"] = summarize_results(results)
    return payload


def _build_subprocess_args(base_args: argparse.Namespace, backend: str, root_dir: str | None) -> list[str]:
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--backend",
        backend,
        "--gpu",
        base_args.gpu,
        "--dataset",
        base_args.dataset,
        "--root-dir",
        root_dir or base_args.root_dir,
        "--warm-round",
        str(base_args.warm_round),
        "--target-round",
        str(base_args.target_round),
        "--frac-num",
        str(base_args.frac_num),
        "--local-ep",
        str(base_args.local_ep),
        "--local-bs",
        str(base_args.local_bs),
        "--num-workers",
        str(base_args.num_workers),
        "--seed",
        str(base_args.seed),
    ]
    if base_args.warm_client is not None:
        argv.extend(["--warm-client", str(base_args.warm_client)])
    if base_args.clients.strip():
        argv.extend(["--clients", base_args.clients])
    if base_args.profile_runtime:
        argv.append("--profile-runtime")
    return argv


def run_compare_mode(args: argparse.Namespace) -> dict[str, object]:
    tf_proc = subprocess.run(
        _build_subprocess_args(args, "tf2", args.tf_root_dir),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    torch_proc = subprocess.run(
        _build_subprocess_args(args, "torch", args.torch_root_dir),
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )

    tf_payload = extract_json_object(tf_proc.stdout)
    torch_payload = extract_json_object(torch_proc.stdout)
    return {
        "mode": "compare",
        "tf2": tf_payload,
        "torch": torch_payload,
        "comparison": build_comparison_payload(tf_payload, torch_payload),
    }


def main() -> None:
    args = parse_args()
    payload = run_compare_mode(args) if args.backend == "compare" else run_single_backend_profile(args)
    emit_payload(payload, args.json_output)


if __name__ == "__main__":
    main()

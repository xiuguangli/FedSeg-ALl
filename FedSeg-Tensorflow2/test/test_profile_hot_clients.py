from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path


def load_profile_hot_clients_module():
    module_path = Path(__file__).resolve().with_name("profile_hot_clients.py")
    spec = importlib.util.spec_from_file_location("fedseg_profile_hot_clients", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


profile_hot_clients = load_profile_hot_clients_module()


def test_extract_json_object_skips_log_prefix_lines():
    text = "log line\nanother line\n{\n  \"backend\": \"tf2\",\n  \"summary\": {\"total_mean_sec\": 1.23}\n}\ntrailing"

    payload = profile_hot_clients.extract_json_object(text)

    assert payload["backend"] == "tf2"
    assert payload["summary"]["total_mean_sec"] == 1.23


def test_build_comparison_payload_computes_ratios_and_deltas():
    tf_payload = {
        "dataset": "voc",
        "gpu": "0",
        "warm_round": 1,
        "target_round": 1,
        "warm_client": 17,
        "results": [
            {"client_id": 6, "samples": 31, "prototype_sec": 0.5, "train_sec": 1.0, "total_sec": 1.5},
        ],
        "summary": {
            "prototype_mean_sec": 0.5,
            "train_mean_sec": 1.0,
            "total_mean_sec": 1.5,
        },
    }
    torch_payload = {
        "results": [
            {"client_id": 6, "samples": 31, "prototype_sec": 1.0, "train_sec": 2.0, "total_sec": 3.0},
        ],
        "summary": {
            "prototype_mean_sec": 1.0,
            "train_mean_sec": 2.0,
            "total_mean_sec": 3.0,
        },
    }

    payload = profile_hot_clients.build_comparison_payload(tf_payload, torch_payload)

    assert payload["summary"]["total_mean_sec"]["delta_sec"] == -1.5
    assert payload["summary"]["total_mean_sec"]["ratio"] == 0.5
    assert payload["per_client"][0]["train_sec"]["delta_sec"] == -1.0
    assert payload["per_client"][0]["prototype_sec"]["ratio"] == 0.5


def test_resolve_root_dir_accepts_repo_prefixed_relative_paths(tmp_path: Path):
    repo_dir = tmp_path / "FedSeg-tf2"
    dataset_dir = repo_dir / "data" / "voc"
    dataset_dir.mkdir(parents=True)

    resolved = profile_hot_clients.resolve_root_dir(repo_dir, "FedSeg-tf2/data/voc")

    assert Path(resolved) == dataset_dir.resolve()


def test_build_subprocess_args_passes_backend_specific_root_dir():
    args = Namespace(
        gpu="0",
        dataset="voc",
        root_dir="data/voc",
        tf_root_dir="FedSeg-tf2/data/voc",
        torch_root_dir="FedSeg-torch/data/voc",
        warm_round=1,
        target_round=1,
        warm_client=17,
        clients="6,46",
        frac_num=5,
        local_ep=2,
        local_bs=8,
        num_workers=4,
        seed=1,
        profile_runtime=True,
    )

    argv = profile_hot_clients._build_subprocess_args(args, "tf2", args.tf_root_dir)

    assert "--backend" in argv and "tf2" in argv
    root_idx = argv.index("--root-dir")
    assert argv[root_idx + 1] == "FedSeg-tf2/data/voc"
    assert "--profile-runtime" in argv

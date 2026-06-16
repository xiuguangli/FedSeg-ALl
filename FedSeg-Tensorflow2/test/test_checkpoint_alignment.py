from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from pathlib import Path

from segmentation.eval import _normalize_checkpoint_name
from segmentation.federated_main import FederatedTrainer, _checkpoint_metadata_path
from segmentation.myseg.bisenetv2 import BiSeNetV2
from segmentation.tf2_tools import infer_torch_checkpoint_for_tf_weights, load_tf_weights_with_torch_fallback, metadata_output_path


class DummyDataset:
    def __init__(self, count: int = 2):
        self.samples = []
        for idx in range(count):
            image = np.full((3, 32, 32), idx, dtype=np.float32)
            label = np.zeros((32, 32), dtype=np.int64)
            self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_args(root, checkpoint=""):
    return SimpleNamespace(
        gpu="0",
        dataset="camvid",
        data="train",
        root=str(root),
        root_dir="unused",
        USE_ERASE_DATA=False,
        num_classes=3,
        num_workers=0,
        model="bisenetv2",
        checkpoint=checkpoint,
        lr=0.01,
        lr_scheduler="step",
        iid=True,
        num_users=2,
        frac_num=1,
        epochs=3,
        local_ep=1,
        local_bs=1,
        is_proto=False,
        losstype="ce",
        fedprox_mu=0.0,
        label_online_gen=False,
        distill=False,
        distill_lamb_pi=0.0,
        distill_lamb_pa=0.0,
        rand_init=True,
        warmstep=1,
        globalema=False,
        temp_dist=0.1,
        mixlabel=False,
        proto_start_epoch=1,
        con_lamb=0.0,
        con_lamb_local=0.0,
        momentum=0.9,
        temperature=0.07,
        max_anchor=32,
        kmean_num=0,
        pseudo_label=False,
        pseudo_label_start_epoch=1,
        localmem=False,
        mom_update=False,
        save_frequency=1,
        local_test_frequency=9999,
        global_test_frequency=9999,
        USE_WANDB=0,
        date_now="20260602_000000",
        verbose=0,
        seed=1,
        optimizer="sgd",
        weight_decay=0.0,
        proj_dim=16,
        train_only=True,
    )


def test_checkpoint_metadata_roundtrip(monkeypatch, tmp_path):
    dataset = DummyDataset()
    user_groups = {0: [0], 1: [1]}
    monkeypatch.setattr("segmentation.federated_main.load_datasets", lambda args: (dataset, dataset, user_groups))
    monkeypatch.setattr(
        "segmentation.federated_main.create_tf_dataloader_from_custom_dataset_test",
        lambda dataset, batch_size=1, shuffle=False: object(),
    )

    args = build_args(tmp_path)
    trainer = FederatedTrainer(args)
    trainer.wandb_id = "run_123"
    trainer.exp_name = "fed_unit_test"
    trainer._save_checkpoint(epoch=2)

    checkpoint_path = tmp_path / "save" / "checkpoints" / "fed_unit_test.weights.h5"
    metadata_path = checkpoint_path.parent / (_checkpoint_metadata_path(str(checkpoint_path)).split("/")[-1])

    assert checkpoint_path.exists()
    assert metadata_path.exists()

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata == {"epoch": 2, "exp_name": "fed_unit_test", "wandb_id": "run_123"}

    args_resume = build_args(tmp_path, checkpoint="fed_unit_test.weights.h5")
    resumed = FederatedTrainer(args_resume)
    assert resumed.start_epoch == 3
    assert resumed.wandb_id == "run_123"
    assert resumed.exp_name == "fed_unit_test"


def test_checkpoint_name_normalization_matches_tf_suffix():
    assert _normalize_checkpoint_name("demo.pth") == "demo.weights.h5"
    assert _normalize_checkpoint_name("demo.weights.h5") == "demo.weights.h5"


def test_checkpoint_metadata_path_matches_saved_suffix():
    path = "save/checkpoints/demo.weights.h5"
    assert _checkpoint_metadata_path(path) == "save/checkpoints/demo.meta.json"


def test_tf_checkpoint_metadata_helpers_roundtrip(tmp_path):
    torch_checkpoint = tmp_path / "demo.pth"
    tf_checkpoint = tmp_path / "demo.weights.h5"
    torch_checkpoint.write_bytes(b"placeholder")

    metadata = {"source_torch_checkpoint": str(torch_checkpoint), "epoch": 3}
    metadata_output_path(tf_checkpoint).write_text(json.dumps(metadata), encoding="utf-8")

    assert metadata_output_path(tf_checkpoint) == tmp_path / "demo.meta.json"
    assert infer_torch_checkpoint_for_tf_weights(tf_checkpoint) == torch_checkpoint.resolve()


def test_load_tf_weights_with_torch_fallback_uses_saved_tf_weights(tmp_path):
    source_model = BiSeNetV2(n_classes=3, proj_dim=16, aux_mode="eval")
    target_model = BiSeNetV2(n_classes=3, proj_dim=16, aux_mode="eval")
    sample = tf.zeros([1, 3, 32, 32], dtype=tf.float32)
    _ = source_model(sample, training=False)
    _ = target_model(sample, training=False)

    for variable in source_model.weights:
        variable.assign(tf.ones_like(variable) * 0.25)

    tf_checkpoint = tmp_path / "demo.weights.h5"
    source_model.save_weights(str(tf_checkpoint))
    metadata_output_path(tf_checkpoint).write_text(
        json.dumps({"source_torch_checkpoint": str(tmp_path / "missing_source.pth")}),
        encoding="utf-8",
    )

    loaded_path, fallback_source = load_tf_weights_with_torch_fallback(target_model, tf_checkpoint)
    assert loaded_path == tf_checkpoint.resolve()
    assert fallback_source is None

    source_weights = source_model.get_weights()
    target_weights = target_model.get_weights()
    assert len(source_weights) == len(target_weights)
    for expected, actual in zip(source_weights, target_weights):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf
import torch

from segmentation.tf2_tools import (
    build_fast_tf_bisenetv2_from_model,
    build_tf_bisenetv2,
    build_torch_bisenetv2,
    load_tf_backbone_into_tf,
    load_torch_backbone_into_tf,
    load_torch_state_into_tf,
    save_torch_backbone_as_tf_weights,
)


def test_bisenetv2_forward_alignment_with_torch_weights():
    torch.manual_seed(0)
    tf.random.set_seed(0)
    torch_model = build_torch_bisenetv2(num_classes=20, proj_dim=256, aux_mode="train")
    tf_model = build_tf_bisenetv2(num_classes=20, proj_dim=256, aux_mode="train")
    load_torch_state_into_tf(tf_model, torch_model.state_dict())

    np.random.seed(0)
    inputs = np.random.randn(2, 3, 64, 64).astype(np.float32)

    with torch.no_grad():
        torch_outputs = torch_model(torch.from_numpy(inputs))
    tf_outputs = tf_model(tf.convert_to_tensor(inputs), training=False)

    # Main logits and embedding are the most important invariants to hold tightly.
    main_logits_torch = torch_outputs[0].detach().cpu().numpy()
    main_logits_tf = tf_outputs[0].numpy()
    emb_torch = torch_outputs[1].detach().cpu().numpy()
    emb_tf = tf_outputs[1].numpy()

    main_diff = np.abs(main_logits_tf - main_logits_torch)
    emb_diff = np.abs(emb_tf - emb_torch)
    assert float(main_diff.mean()) < 0.13
    assert float(main_diff.max()) < 0.7
    assert float(emb_diff.mean()) < 0.05
    assert float(emb_diff.max()) < 0.2

    # Keep visibility on auxiliary heads too, with slightly looser tolerances for now.
    for torch_aux, tf_aux in zip(torch_outputs[2:], tf_outputs[2:]):
        aux_diff = np.abs(tf_aux.numpy() - torch_aux.detach().cpu().numpy())
        assert float(aux_diff.mean()) < 0.5
        assert float(aux_diff.max()) < 4.0


def test_torch_backbone_checkpoint_loads_into_tf_model():
    tf_model = build_tf_bisenetv2(num_classes=20, proj_dim=256, aux_mode="train")
    checkpoint = Path(__file__).resolve().parents[1] / "segmentation" / "myseg" / "backbone_v2.pth"
    load_torch_backbone_into_tf(tf_model, checkpoint)
    outputs = tf_model(tf.random.normal([1, 3, 64, 64]), training=False)
    assert outputs[0].shape == (1, 20, 64, 64)


def test_torch_backbone_checkpoint_converts_to_tf_weights(tmp_path):
    torch_checkpoint = Path(__file__).resolve().parents[1] / "segmentation" / "myseg" / "backbone_v2.pth"
    tf_checkpoint = tmp_path / "backbone_v2.weights.h5"

    saved_path = save_torch_backbone_as_tf_weights(torch_checkpoint, tf_checkpoint)
    assert saved_path == tf_checkpoint
    assert tf_checkpoint.exists()

    reference_model = build_tf_bisenetv2(num_classes=20, proj_dim=256, aux_mode="train")
    converted_model = build_tf_bisenetv2(num_classes=20, proj_dim=256, aux_mode="train")
    load_torch_backbone_into_tf(reference_model, torch_checkpoint)
    load_tf_backbone_into_tf(converted_model, tf_checkpoint)

    for expected, actual in zip(reference_model.detail.get_weights(), converted_model.detail.get_weights()):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for expected, actual in zip(reference_model.segment.get_weights(), converted_model.segment.get_weights()):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    for expected, actual in zip(reference_model.bga.get_weights(), converted_model.bga.get_weights()):
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_fast_bisenetv2_forward_matches_nchw_reference():
    torch.manual_seed(1)
    tf.random.set_seed(1)
    torch_model = build_torch_bisenetv2(num_classes=20, proj_dim=256, aux_mode="eval")
    reference_model = build_tf_bisenetv2(num_classes=20, proj_dim=256, aux_mode="eval")
    load_torch_state_into_tf(reference_model, torch_model.state_dict())
    fast_model = build_fast_tf_bisenetv2_from_model(reference_model)

    np.random.seed(1)
    inputs = np.random.randn(1, 3, 64, 96).astype(np.float32)
    reference_logits = reference_model(tf.convert_to_tensor(inputs), training=False)[0].numpy()
    fast_logits = fast_model(tf.convert_to_tensor(inputs), training=False)[0].numpy()

    diff = np.abs(fast_logits - reference_logits)
    assert float(diff.mean()) < 1e-4
    assert float(diff.max()) < 1e-3

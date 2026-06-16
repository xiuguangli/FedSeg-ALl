from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf

from myseg.bisenetv2 import BGALayer, BiSeNetV2, DetailBranch, SegmentBranch
from myseg.bisenetv2_fast import FastBiSeNetV2, copy_nchw_weights_to_fast_model


def _load_torch():
    import torch

    return torch


def build_tf_bisenetv2(num_classes: int, proj_dim: int = 256, aux_mode: str = "train") -> BiSeNetV2:
    model = BiSeNetV2(n_classes=num_classes, proj_dim=proj_dim, aux_mode=aux_mode)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)
    return model


def build_fast_tf_bisenetv2(num_classes: int, proj_dim: int = 256, aux_mode: str = "eval") -> FastBiSeNetV2:
    model = FastBiSeNetV2(n_classes=num_classes, proj_dim=proj_dim, aux_mode=aux_mode)
    _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)
    return model


def build_fast_tf_bisenetv2_from_model(model: BiSeNetV2) -> FastBiSeNetV2:
    fast_model = build_fast_tf_bisenetv2(
        num_classes=model.n_classes,
        proj_dim=model.proj_dim,
        aux_mode=model.aux_mode,
    )
    return copy_nchw_weights_to_fast_model(model, fast_model)


class BiSeNetV2Backbone(tf.keras.Model):
    def __init__(self):
        super().__init__(name="bisenetv2_backbone")
        self.detail = DetailBranch(name="detail")
        self.segment = SegmentBranch(name="segment")
        self.bga = BGALayer(name="bga")

    def call(self, x, training=False):
        feat_d = self.detail(x, training=training)
        _feat2, _feat3, _feat4, _feat5_4, feat_s = self.segment(x, training=training)
        return self.bga(feat_d, feat_s, training=training)


def build_tf_bisenetv2_backbone() -> BiSeNetV2Backbone:
    backbone = BiSeNetV2Backbone()
    _ = backbone(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)
    return backbone


def clone_tf_model(model: BiSeNetV2, input_shape: tuple[int, ...]) -> BiSeNetV2:
    clone = type(model).from_config(model.get_config())
    _ = clone(tf.zeros([1, *input_shape], dtype=tf.float32), training=False)
    clone.set_weights(model.get_weights())
    clone.trainable = model.trainable
    return clone


def normalize_tf_checkpoint_path(checkpoint_path: str | Path) -> Path:
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix == ".pth":
        return checkpoint_path.with_suffix(".weights.h5")
    return checkpoint_path


def normalize_tf_backbone_path(backbone_path: str | Path) -> Path:
    return normalize_tf_checkpoint_path(backbone_path)


def metadata_output_path(tf_checkpoint_path: str | Path) -> Path:
    tf_checkpoint_path = normalize_tf_checkpoint_path(tf_checkpoint_path)
    if tf_checkpoint_path.name.endswith(".weights.h5"):
        return tf_checkpoint_path.with_name(tf_checkpoint_path.name[: -len(".weights.h5")] + ".meta.json")
    return tf_checkpoint_path.with_suffix(tf_checkpoint_path.suffix + ".meta.json")


def infer_torch_checkpoint_for_tf_weights(tf_checkpoint_path: str | Path) -> Path | None:
    tf_checkpoint_path = normalize_tf_checkpoint_path(tf_checkpoint_path).resolve()
    metadata_path = metadata_output_path(tf_checkpoint_path)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            source_path = metadata.get("source_torch_checkpoint")
            if source_path:
                candidate = Path(source_path)
                if candidate.exists():
                    return candidate.resolve()
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    stem = tf_checkpoint_path.name
    if stem.endswith(".weights.h5"):
        candidate = tf_checkpoint_path.with_name(stem[: -len(".weights.h5")] + ".pth")
        if candidate.exists():
            return candidate.resolve()
    return None


def snapshot_model_weights(model: BiSeNetV2) -> list[tf.Tensor]:
    return [tf.identity(var) for var in model.weights]


def assign_model_weights(model: BiSeNetV2, weights) -> None:
    if len(model.weights) != len(weights):
        raise ValueError(f"weights length mismatch: expected {len(model.weights)}, got {len(weights)}")
    for variable, value in zip(model.weights, weights):
        variable.assign(value)


def build_torch_bisenetv2(num_classes: int, proj_dim: int = 256, aux_mode: str = "train"):
    _load_torch()
    repo_root = Path(__file__).resolve().parents[2]
    torch_seg_dir = repo_root / "FedSeg-torch" / "segmentation"
    module_path = torch_seg_dir / "myseg" / "bisenetv2.py"
    spec = importlib.util.spec_from_file_location("fedseg_torch_bisenetv2", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    TorchBiSeNetV2 = module.BiSeNetV2

    args = SimpleNamespace(proj_dim=proj_dim, rand_init=True)
    model = TorchBiSeNetV2(args, n_classes=num_classes, aux_mode=aux_mode)
    model.eval()
    return model


def _to_tf_conv_weight(weight: np.ndarray) -> np.ndarray:
    return np.transpose(weight, (2, 3, 1, 0))


def _to_tf_depthwise_weight(weight: np.ndarray) -> np.ndarray:
    return np.transpose(weight, (2, 3, 1, 0))


def _set_bn(module, state, prefix: str):
    module.set_weights(
        [
            state[f"{prefix}.weight"].cpu().numpy(),
            state[f"{prefix}.bias"].cpu().numpy(),
            state[f"{prefix}.running_mean"].cpu().numpy(),
            state[f"{prefix}.running_var"].cpu().numpy(),
        ]
    )


def _set_conv(conv, state, prefix: str, bias: bool = False, depthwise: bool = False):
    weight = state[f"{prefix}.weight"].cpu().numpy()
    weight = _to_tf_depthwise_weight(weight) if depthwise else _to_tf_conv_weight(weight)
    weights = [weight]
    if bias:
        weights.append(state[f"{prefix}.bias"].cpu().numpy())
    conv.set_weights(weights)


def _set_conv_bn_relu(module, state, prefix: str):
    _set_conv(module.conv, state, f"{prefix}.conv", bias=False)
    _set_bn(module.bn, state, f"{prefix}.bn")


def _set_detail(detail, state):
    for stage_name in ["S1", "S2", "S3"]:
        stage = getattr(detail, stage_name)
        for idx, block in enumerate(stage.layers):
            _set_conv_bn_relu(block, state, f"detail.{stage_name}.{idx}")


def _set_stem(stem, state, prefix: str):
    _set_conv_bn_relu(stem.conv, state, f"{prefix}.conv")
    for idx, block in enumerate(stem.left.layers):
        _set_conv_bn_relu(block, state, f"{prefix}.left.{idx}")
    _set_conv_bn_relu(stem.fuse, state, f"{prefix}.fuse")


def _set_ge_s1(module, state, prefix: str):
    _set_conv_bn_relu(module.conv1, state, f"{prefix}.conv1")
    _set_conv(module.dwconv.layers[0], state, f"{prefix}.dwconv.0", bias=False, depthwise=True)
    _set_bn(module.dwconv.layers[1], state, f"{prefix}.dwconv.1")
    _set_conv(module.conv2_conv, state, f"{prefix}.conv2.0", bias=False)
    _set_bn(module.conv2_bn, state, f"{prefix}.conv2.1")


def _set_ge_s2(module, state, prefix: str):
    _set_conv_bn_relu(module.conv1, state, f"{prefix}.conv1")
    _set_conv(module.dwconv1.layers[0], state, f"{prefix}.dwconv1.0", bias=False, depthwise=True)
    _set_bn(module.dwconv1.layers[1], state, f"{prefix}.dwconv1.1")
    _set_conv(module.dwconv2.layers[0], state, f"{prefix}.dwconv2.0", bias=False, depthwise=True)
    _set_bn(module.dwconv2.layers[1], state, f"{prefix}.dwconv2.1")
    _set_conv(module.conv2_conv, state, f"{prefix}.conv2.0", bias=False)
    _set_bn(module.conv2_bn, state, f"{prefix}.conv2.1")
    _set_conv(module.shortcut.layers[0], state, f"{prefix}.shortcut.0", bias=False, depthwise=True)
    _set_bn(module.shortcut.layers[1], state, f"{prefix}.shortcut.1")
    _set_conv(module.shortcut.layers[2], state, f"{prefix}.shortcut.2", bias=False)
    _set_bn(module.shortcut.layers[3], state, f"{prefix}.shortcut.3")


def _set_ce(module, state, prefix: str):
    _set_bn(module.bn, state, f"{prefix}.bn")
    _set_conv_bn_relu(module.conv_gap, state, f"{prefix}.conv_gap")
    _set_conv_bn_relu(module.conv_last, state, f"{prefix}.conv_last")


def _set_segment(segment, state):
    _set_stem(segment.S1S2, state, "segment.S1S2")
    _set_ge_s2(segment.S3.layers[0], state, "segment.S3.0")
    _set_ge_s1(segment.S3.layers[1], state, "segment.S3.1")
    _set_ge_s2(segment.S4.layers[0], state, "segment.S4.0")
    _set_ge_s1(segment.S4.layers[1], state, "segment.S4.1")
    _set_ge_s2(segment.S5_4.layers[0], state, "segment.S5_4.0")
    _set_ge_s1(segment.S5_4.layers[1], state, "segment.S5_4.1")
    _set_ge_s1(segment.S5_4.layers[2], state, "segment.S5_4.2")
    _set_ge_s1(segment.S5_4.layers[3], state, "segment.S5_4.3")
    _set_ce(segment.S5_5, state, "segment.S5_5")


def _set_bga(bga, state):
    _set_conv(bga.left1.layers[0], state, "bga.left1.0", bias=False, depthwise=True)
    _set_bn(bga.left1.layers[1], state, "bga.left1.1")
    _set_conv(bga.left1.layers[2], state, "bga.left1.2", bias=False)
    _set_conv(bga.left2.layers[0], state, "bga.left2.0", bias=False)
    _set_bn(bga.left2.layers[1], state, "bga.left2.1")
    _set_conv(bga.right1.layers[0], state, "bga.right1.0", bias=False)
    _set_bn(bga.right1.layers[1], state, "bga.right1.1")
    _set_conv(bga.right2.layers[0], state, "bga.right2.0", bias=False, depthwise=True)
    _set_bn(bga.right2.layers[1], state, "bga.right2.1")
    _set_conv(bga.right2.layers[2], state, "bga.right2.2", bias=False)
    _set_conv(bga.conv.layers[0], state, "bga.conv.0", bias=False)
    _set_bn(bga.conv.layers[1], state, "bga.conv.1")


def _set_segment_head(head, state, prefix: str, aux: bool):
    _set_conv_bn_relu(head.conv, state, f"{prefix}.conv")
    if aux:
        _set_conv_bn_relu(head.aux_pre, state, f"{prefix}.conv_out.0.1")
    conv_out_prefix = f"{prefix}.conv_out.1"
    _set_conv(head.conv_out, state, conv_out_prefix, bias=True)


def _set_proj_head(proj_head, state, prefix: str):
    seq = proj_head.proj
    _set_conv(seq.layers[0], state, f"{prefix}.proj.0", bias=True)
    _set_bn(seq.layers[1], state, f"{prefix}.proj.1")
    _set_conv(seq.layers[3], state, f"{prefix}.proj.3", bias=True)


def load_torch_state_into_tf(model: BiSeNetV2, torch_state: dict) -> BiSeNetV2:
    _set_detail(model.detail, torch_state)
    _set_segment(model.segment, torch_state)
    _set_bga(model.bga, torch_state)
    _set_segment_head(model.head, torch_state, "head", aux=False)
    if model.aux_mode == "train":
        _set_segment_head(model.aux2, torch_state, "aux2", aux=True)
        _set_segment_head(model.aux3, torch_state, "aux3", aux=True)
        _set_segment_head(model.aux4, torch_state, "aux4", aux=True)
        _set_segment_head(model.aux5_4, torch_state, "aux5_4", aux=True)
    _set_proj_head(model.proj_head, torch_state, "proj_head")
    return model


def load_torch_checkpoint_into_tf(model: BiSeNetV2, checkpoint_path: str | Path) -> BiSeNetV2:
    torch = _load_torch()
    state = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    return load_torch_state_into_tf(model, state)


def save_torch_checkpoint_as_tf_weights(
    model: BiSeNetV2,
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = normalize_tf_checkpoint_path(output_path or checkpoint_path)
    load_torch_checkpoint_into_tf(model, checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(output_path))
    return output_path


def load_tf_weights_with_torch_fallback(model: BiSeNetV2, checkpoint_path: str | Path) -> tuple[Path, Path | None]:
    checkpoint_path = normalize_tf_checkpoint_path(checkpoint_path).resolve()
    try:
        model.load_weights(str(checkpoint_path))
        return checkpoint_path, None
    except ValueError:
        source_torch_checkpoint = infer_torch_checkpoint_for_tf_weights(checkpoint_path)
        if source_torch_checkpoint is None:
            raise
        save_torch_checkpoint_as_tf_weights(model, source_torch_checkpoint, checkpoint_path)
        return checkpoint_path, source_torch_checkpoint


def load_torch_backbone_into_tf(model, checkpoint_path: str | Path):
    torch = _load_torch()
    state = torch.load(str(checkpoint_path), map_location="cpu")
    for prefix in ("detail", "segment", "bga"):
        sub_state = state[prefix]
        flat = {f"{prefix}.{k}": v for k, v in sub_state.items()}
        if prefix == "detail":
            _set_detail(model.detail, flat)
        elif prefix == "segment":
            _set_segment(model.segment, flat)
        else:
            _set_bga(model.bga, flat)
    return model


def _copy_backbone_weights(target, source) -> None:
    target.detail.set_weights(source.detail.get_weights())
    target.segment.set_weights(source.segment.get_weights())
    target.bga.set_weights(source.bga.get_weights())


def save_torch_backbone_as_tf_weights(
    checkpoint_path: str | Path,
    output_path: str | Path | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path)
    output_path = normalize_tf_backbone_path(output_path or checkpoint_path)
    backbone = build_tf_bisenetv2_backbone()
    load_torch_backbone_into_tf(backbone, checkpoint_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    backbone.save_weights(str(output_path))
    return output_path


def load_tf_backbone_into_tf(model: BiSeNetV2, checkpoint_path: str | Path) -> BiSeNetV2:
    checkpoint_path = normalize_tf_backbone_path(checkpoint_path)
    if not model.weights:
        _ = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)
    backbone = build_tf_bisenetv2_backbone()
    backbone.load_weights(str(checkpoint_path))
    _copy_backbone_weights(model, backbone)
    return model

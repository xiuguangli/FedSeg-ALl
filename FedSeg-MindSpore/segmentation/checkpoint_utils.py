import json
import os
import shutil

import mindspore as ms
import numpy as np
from mindspore import Parameter, Tensor


def _canonical_param_name(name):
    while name.startswith("model."):
        name = name[len("model.") :]
    return name


def _align_state_dict_keys(state_dict, target_names):
    if not target_names:
        return dict(state_dict)

    target_name_set = set(target_names)
    canonical_target_map = {}
    for target_name in target_names:
        canonical_target_map.setdefault(_canonical_param_name(target_name), target_name)

    aligned = {}
    for name, value in state_dict.items():
        if name in target_name_set and name not in aligned:
            aligned[name] = value
            continue

        canonical_name = _canonical_param_name(name)
        matched_name = canonical_target_map.get(canonical_name)
        if matched_name is not None and matched_name not in aligned:
            aligned[matched_name] = value
    return aligned


def _to_host_array(value):
    if isinstance(value, Parameter):
        value = value.data
    if isinstance(value, Tensor):
        return value.asnumpy().copy()
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return np.array(value, copy=True)


def _to_ms_tensor(value):
    if isinstance(value, Parameter):
        return value.data
    if isinstance(value, Tensor):
        return value
    if isinstance(value, np.ndarray):
        return Tensor(value)
    return Tensor(np.array(value))


def clone_state_dict(source, host=False):
    state_dict = {}
    for name, value in source.items():
        canonical_name = _canonical_param_name(name)
        if host:
            state_dict[canonical_name] = _to_host_array(value)
            continue
        tensor = _to_ms_tensor(value)
        state_dict[canonical_name] = tensor.clone()
    return state_dict


def load_state_into_net(model, state_dict, strict=False):
    target_params = model.parameters_dict()
    target_names = tuple(target_params.keys())
    aligned_state = _align_state_dict_keys(state_dict, target_names)

    missing = []
    for name in target_names:
        value = aligned_state.get(name)
        if value is None:
            if strict:
                missing.append(name)
            continue
        target_params[name].set_data(_to_ms_tensor(value))

    unexpected = [name for name in aligned_state.keys() if name not in target_params]
    return missing, unexpected


def _meta_path(checkpoint_path):
    stem, _ = os.path.splitext(checkpoint_path)
    return stem + ".meta.json"


def _ema_shadow_path(checkpoint_path):
    stem, _ = os.path.splitext(checkpoint_path)
    return stem + ".ema.ckpt"


def _normalize_meta_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_meta_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_meta_value(item) for key, item in value.items()}
    return str(value)


def save_training_checkpoint(
    model,
    checkpoint_path,
    epoch,
    exp_name,
    wandb_id=None,
    ema_shadow=None,
    config_snapshot=None,
):
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    ms.save_checkpoint(model, checkpoint_path)
    ema_shadow_path = _ema_shadow_path(checkpoint_path)
    if ema_shadow is not None:
        ema_shadow_params = []
        for name, value in ema_shadow.items():
            ema_shadow_params.append(
                {
                    "name": _canonical_param_name(name),
                    "data": _to_ms_tensor(value),
                }
            )
        ms.save_checkpoint(ema_shadow_params, ema_shadow_path)
    elif os.path.exists(ema_shadow_path):
        os.remove(ema_shadow_path)
    meta = {
        "epoch": int(epoch),
        "exp_name": exp_name,
        "wandb_id": wandb_id,
    }
    if config_snapshot:
        meta["config_snapshot"] = _normalize_meta_value(config_snapshot)
    with open(_meta_path(checkpoint_path), "w", encoding="utf-8") as fout:
        json.dump(meta, fout, ensure_ascii=True, indent=2)


def load_training_checkpoint(model, checkpoint_path, strict=False):
    param_dict = ms.load_checkpoint(checkpoint_path)
    filtered = _align_state_dict_keys(param_dict, tuple(model.parameters_dict().keys()))
    missing, unexpected = load_state_into_net(model, filtered, strict=strict)
    ema_shadow = None
    ema_shadow_path = _ema_shadow_path(checkpoint_path)
    if os.path.exists(ema_shadow_path):
        ema_shadow_raw = ms.load_checkpoint(ema_shadow_path)
        ema_shadow = clone_state_dict(
            _align_state_dict_keys(ema_shadow_raw, tuple(model.parameters_dict().keys())),
            host=True,
        )

    meta = {}
    meta_path = _meta_path(checkpoint_path)
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as fin:
            meta = json.load(fin)

    aligned_input_keys = set(filtered.keys())
    ignored = sorted(
        name
        for name in param_dict.keys()
        if _canonical_param_name(name) not in {_canonical_param_name(key) for key in aligned_input_keys}
    )
    return {
        "missing": missing,
        "unexpected": unexpected,
        "ignored": ignored,
        "meta": meta,
        "ema_shadow": ema_shadow,
    }


def copy_training_checkpoint(source_checkpoint_path, target_checkpoint_path):
    os.makedirs(os.path.dirname(target_checkpoint_path), exist_ok=True)
    shutil.copy2(source_checkpoint_path, target_checkpoint_path)

    source_meta_path = _meta_path(source_checkpoint_path)
    target_meta_path = _meta_path(target_checkpoint_path)
    if os.path.exists(source_meta_path):
        shutil.copy2(source_meta_path, target_meta_path)
    elif os.path.exists(target_meta_path):
        os.remove(target_meta_path)

    source_ema_shadow_path = _ema_shadow_path(source_checkpoint_path)
    target_ema_shadow_path = _ema_shadow_path(target_checkpoint_path)
    if os.path.exists(source_ema_shadow_path):
        shutil.copy2(source_ema_shadow_path, target_ema_shadow_path)
    elif os.path.exists(target_ema_shadow_path):
        os.remove(target_ema_shadow_path)

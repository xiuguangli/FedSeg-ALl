import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from federated_main import FederatedTrainer
from logging_utils import setup_logger
from myseg.bisenet_utils import ContrastLoss
from options import args_parser
from seed_utils import seed_everything


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--global_round", type=int, required=True)
    parser.add_argument("--batch_npz", type=Path, required=True)
    parser.add_argument("--batch_index", type=int, default=0)
    parser.add_argument("--proto_npz", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state_path", type=Path, default=None)
    parser.add_argument("--deterministic_contrast", action="store_true")
    export_args, remaining = parser.parse_known_args()
    train_args = args_parser(remaining)
    return export_args, train_args


def install_deterministic_anchor_sampling():
    original = ContrastLoss._anchor_sampling

    def deterministic_anchor_sampling(self, embs, labels):
        device = embs.device
        _, channels, _, _ = embs.size()
        embs_flat = embs.permute(0, 2, 3, 1).reshape(-1, channels)
        labels_flat = labels.view(-1)
        index_all = torch.arange(len(labels_flat), device=device)
        class_u = torch.unique(labels_flat)
        class_u = class_u[class_u != 255]

        if len(class_u) == 0:
            return None, None

        num_per_class = max(1, int(self.max_anchor) // int(len(class_u)))
        sampled_list = []
        sampled_label_list = []
        for cls_ in class_u:
            selected_index = torch.masked_select(index_all, labels_flat == cls_)
            if len(selected_index) > num_per_class:
                selected_index = selected_index[:num_per_class]
            sampled_list.append(embs_flat[selected_index])
            sampled_label_list.append(torch.ones(len(selected_index), device=device) * cls_)

        return torch.cat(sampled_list, 0), torch.cat(sampled_label_list, 0)

    ContrastLoss._anchor_sampling = deterministic_anchor_sampling
    return original


def load_batch(batch_npz, batch_index):
    data = np.load(str(batch_npz), allow_pickle=False)
    return (
        np.asarray(data[f"images_{batch_index:03d}"], dtype=np.float32),
        np.asarray(data[f"labels_{batch_index:03d}"], dtype=np.int64),
        np.asarray(data[f"indices_{batch_index:03d}"], dtype=np.int32),
    )


def load_prototypes(proto_npz):
    data = np.load(str(proto_npz), allow_pickle=False)
    return (
        np.asarray(data["proto"], dtype=np.float32),
        np.asarray(data["mask"], dtype=np.float32),
    )


def save_probe(output_path, arrays, metadata):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = sorted(arrays.keys())
    np.savez_compressed(
        str(output_path),
        **{"arr_{:04d}".format(index): np.asarray(arrays[name]) for index, name in enumerate(names)},
    )
    with output_path.with_suffix(output_path.suffix + ".json").open("w", encoding="utf-8") as fout:
        json.dump(
            {
                "metadata": metadata,
                "names": names,
            },
            fout,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )


def to_numpy(tensor):
    if tensor is None:
        return None
    return tensor.detach().cpu().numpy().copy()


def main():
    export_args, args = parse_args()
    args.USE_WANDB = False
    args.globalema = False
    setup_logger(verbose=bool(args.verbose), logs_dir="logs/profile_hot_clients", log_name="export_contrast_probe_torch")
    seed_everything(args.seed)

    restore_anchor_sampling = None
    if export_args.deterministic_contrast:
        restore_anchor_sampling = install_deterministic_anchor_sampling()

    try:
        trainer = FederatedTrainer(args)
        if export_args.state_path is not None:
            checkpoint = torch.load(str(export_args.state_path), map_location=trainer.device)
            trainer.global_model.load_state_dict(checkpoint["model"], strict=False)
            trainer.global_weights = copy.deepcopy(trainer.global_model.state_dict())

        client = trainer._make_client(export_args.client_id)
        model = copy.deepcopy(trainer.global_model).to(trainer.device)
        model.train()

        batch_images, batch_labels, batch_indices = load_batch(export_args.batch_npz, export_args.batch_index)
        proto_np, proto_mask_np = load_prototypes(export_args.proto_npz)

        images = torch.from_numpy(batch_images).to(trainer.device)
        labels = torch.from_numpy(batch_labels).to(trainer.device)
        prototypes = torch.from_numpy(proto_np).to(trainer.device)
        proto_mask = torch.from_numpy(proto_mask_np).to(trainer.device)

        seg_logits, feat_head, logits_aux = client._forward(model, images)
        _, _, height, width = feat_head.size()
        labels_1 = F.interpolate(labels.unsqueeze(1).float(), size=(height, width), mode="nearest").squeeze(1)
        if args.kmean_num > 0:
            proto_mask_tmp = proto_mask.sum(1) < 1
        else:
            proto_mask_tmp = proto_mask < 1
        for class_idx, missing in enumerate(proto_mask_tmp):
            if bool(missing):
                labels_1[labels_1 == class_idx] = 255

        criteria = ContrastLoss(args)
        anchors, anchor_labels = criteria._anchor_sampling(feat_head, labels_1)
        if args.kmean_num > 0:
            classes, proto_slots, channels = prototypes.size()
            proto_labels = torch.arange(classes, device=trainer.device).unsqueeze(1).repeat(1, proto_slots)
            proto_mem_flat = prototypes.reshape(-1, channels)
            proto_labels = proto_labels.view(-1)
            proto_mask_flat = proto_mask.view(-1)
            proto_idx = torch.arange(len(proto_mask_flat), device=trainer.device)
            sel_idx = torch.masked_select(proto_idx, proto_mask_flat.bool())
            proto_mem_flat = proto_mem_flat[sel_idx]
            proto_labels = proto_labels[sel_idx]
        else:
            classes, channels = prototypes.size()
            proto_labels = torch.arange(classes, device=trainer.device)
            proto_idx = torch.arange(len(proto_mask), device=trainer.device)
            sel_idx = torch.masked_select(proto_idx, proto_mask.bool())
            proto_mem_flat = prototypes[sel_idx]
            proto_labels = proto_labels[sel_idx]

        anchor_dot = torch.div(torch.matmul(anchors, proto_mem_flat.T), criteria.temperature)
        pos_mask = (anchor_labels.unsqueeze(1) == proto_labels.unsqueeze(0)).float()
        neg_mask = 1.0 - pos_mask
        logits_max, _ = torch.max(anchor_dot, dim=1, keepdim=True)
        contrast_logits = anchor_dot - logits_max.detach()
        exp_logits = torch.exp(contrast_logits)
        neg_logits = (exp_logits * neg_mask).sum(1, keepdim=True)
        pos_exp_logits = exp_logits * pos_mask
        log_prob = contrast_logits - torch.log(pos_exp_logits + neg_logits)
        pos_count = pos_mask.sum(1)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_count
        loss = (-mean_log_prob_pos).mean()

        probe = {
            "batch_images": batch_images,
            "batch_labels": batch_labels,
            "batch_indices": batch_indices,
            "seg_logits": to_numpy(seg_logits),
            "feat_head": to_numpy(feat_head),
            "labels_1": to_numpy(labels_1),
            "anchors": to_numpy(anchors),
            "anchor_labels": to_numpy(anchor_labels),
            "proto_mem_flat": to_numpy(proto_mem_flat),
            "proto_labels": to_numpy(proto_labels),
            "anchor_dot": to_numpy(anchor_dot),
            "pos_mask": to_numpy(pos_mask),
            "neg_mask": to_numpy(neg_mask),
            "logits_max": to_numpy(logits_max),
            "contrast_logits": to_numpy(contrast_logits),
            "exp_logits": to_numpy(exp_logits),
            "neg_logits": to_numpy(neg_logits),
            "pos_exp_logits": to_numpy(pos_exp_logits),
            "log_prob": to_numpy(log_prob),
            "pos_count": to_numpy(pos_count),
            "mean_log_prob_pos": to_numpy(mean_log_prob_pos),
            "loss": np.asarray([float(loss.item())], dtype=np.float32),
            "aux0": to_numpy(logits_aux[0]) if logits_aux else np.zeros((0,), dtype=np.float32),
        }
        metadata = {
            "framework": "torch",
            "checkpoint": str(export_args.state_path) if export_args.state_path is not None else "",
            "client_id": export_args.client_id,
            "global_round": export_args.global_round,
            "batch_index": int(export_args.batch_index),
            "deterministic_contrast": bool(export_args.deterministic_contrast),
            "loss": float(loss.item()),
        }
        save_probe(export_args.output, probe, metadata)
        print(
            "EXPORT_CONTRAST_PROBE "
            + json.dumps(
                {
                    "output": str(export_args.output),
                    "metadata": metadata,
                    "num_arrays": len(probe),
                },
                ensure_ascii=True,
                sort_keys=True,
            )
        )
    finally:
        if restore_anchor_sampling is not None:
            ContrastLoss._anchor_sampling = restore_anchor_sampling


if __name__ == "__main__":
    main()

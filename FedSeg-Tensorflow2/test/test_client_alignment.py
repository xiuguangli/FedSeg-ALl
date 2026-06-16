from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
import torch

from segmentation.client import Client
from segmentation.myseg.bisenet_utils import ContrastLoss, set_model_bisenetv2, set_optimizer
from segmentation.tf2_tools import build_tf_bisenetv2, build_torch_bisenetv2, load_torch_state_into_tf


class OrderedDataset:
    def __init__(self, size: int = 6, spatial_size: int = 8):
        self.samples = []
        for idx in range(size):
            image = np.full((3, spatial_size, spatial_size), idx, dtype=np.float32)
            label = np.full((spatial_size, spatial_size), idx, dtype=np.int64)
            self.samples.append((image, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def build_args(**overrides):
    base = dict(
        seed=7,
        local_bs=2,
        num_workers=0,
        losstype="ce",
        temp_dist=0.07,
        max_anchor=32,
        temperature=0.07,
        num_classes=3,
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        model="bisenetv2",
        local_ep=1,
        is_proto=False,
        proto_start_epoch=1,
        kmean_num=0,
        pseudo_label=False,
        pseudo_label_start_epoch=1,
        con_lamb=0.1,
        con_lamb_local=0.1,
        fedprox_mu=0.0,
        distill=False,
        distill_lamb_pi=0.0,
        distill_lamb_pa=0.0,
        verbose=0,
        proj_dim=16,
        lr_scheduler="poly",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def load_torch_client_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "FedSeg-torch" / "segmentation" / "client.py"
    spec = importlib.util.spec_from_file_location("fedseg_torch_client", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tf_client_local_eval_split_matches_torch_convention():
    args = build_args()
    dataset = OrderedDataset(size=6)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)

    assert client.preview_indices() == [0, 1, 2, 3, 4, 5]
    assert client.num_samples == 6
    assert client.num_eval_samples == 3
    assert client._test_split == [0, 1, 2]


def test_tf_test_loader_respects_shuffle_false_order():
    from segmentation.myseg.magic import create_tf_dataloader_from_custom_dataset_train

    dataset = OrderedDataset(size=5)
    loader = create_tf_dataloader_from_custom_dataset_train(
        dataset,
        batch_size=1,
        shuffle=False,
        repeat=False,
        drop_last=False,
        output_img_shape=(3, 8, 8),
        output_lbl_shape=(8, 8),
    )

    seen = []
    for image, _label in loader:
        seen.append(int(image.numpy()[0, 0, 0, 0]))

    assert seen == [0, 1, 2, 3, 4]


def test_tf_optimizer_parameter_groups_match_torch_structure():
    args = build_args(num_classes=3, proj_dim=16, lr=0.05, momentum=0.99, weight_decay=0.0005)
    tf_model = set_model_bisenetv2(args, num_classes=args.num_classes)
    _ = tf_model(tf.zeros([1, 3, 32, 32], dtype=tf.float32), training=False)

    optimizer = set_optimizer(tf_model, args)
    multipliers = optimizer._fedseg_var_lr_multipliers
    decays = optimizer._fedseg_var_weight_decays

    head_vars = [var for var in tf_model.head.trainable_variables if len(var.shape) in {1, 4}]
    aux_vars = [var for name in ["aux2", "aux3", "aux4", "aux5_4"] for var in getattr(tf_model, name).trainable_variables if len(var.shape) in {1, 4}]
    body_vars = [var for var in tf_model.detail.trainable_variables if len(var.shape) in {1, 4}]

    assert head_vars
    assert aux_vars
    assert body_vars

    for var in body_vars:
        assert multipliers[id(var)] == 1.0
    for var in head_vars + aux_vars:
        assert multipliers[id(var)] == 10.0

    for var in tf_model.detail.trainable_variables + tf_model.segment.trainable_variables + tf_model.bga.trainable_variables:
        if len(var.shape) == 1:
            assert decays[id(var)] == 0.0
        elif len(var.shape) in {2, 4}:
            assert decays[id(var)] == args.weight_decay


def test_set_model_bisenetv2_loads_tf_backbone_when_not_rand_init(tmp_path):
    source = Path(__file__).resolve().parents[1] / "segmentation" / "myseg" / "backbone_v2.weights.h5"
    target = tmp_path / "backbone_v2.weights.h5"
    shutil.copy2(source, target)

    args = build_args(num_classes=3, proj_dim=16, rand_init=False, backbone_checkpoint=str(target))
    model = set_model_bisenetv2(args, num_classes=20)
    outputs = model(tf.zeros([1, 3, 64, 64], dtype=tf.float32), training=False)

    assert outputs[0].shape == (1, 20, 64, 64)


def test_tf_scheduler_values_match_torch_reference():
    args = build_args(local_bs=2, local_ep=2, lr=0.05)
    dataset = OrderedDataset(size=8)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)
    trainloader = client._build_trainloader(global_round=0)
    tf_schedule = client._build_scheduler(None, trainloader, global_round=0)

    values = [tf_schedule(step) for step in range(4)]
    expected = [0.05, 0.044338118051951836, 0.038594475336178526, 0.03275382467090493]

    np.testing.assert_allclose(values, expected, rtol=1e-6, atol=1e-8)


def test_step_scheduler_matches_torch_reference_after_round_1000():
    args = build_args(lr=0.05, lr_scheduler="step")
    dataset = OrderedDataset(size=8)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)
    trainloader = client._build_trainloader(global_round=0)
    tf_schedule = client._build_scheduler(None, trainloader, global_round=1001)

    values = [tf_schedule(step) for step in range(3)]
    np.testing.assert_allclose(values, [0.005, 0.005, 0.005], rtol=1e-9, atol=1e-9)


def test_tf_trainloader_length_matches_torch_multiepoch_batch_semantics():
    args = build_args(local_bs=2, local_ep=1)
    dataset = OrderedDataset(size=6)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)

    trainloader = client._build_trainloader(global_round=0)
    trainloader_eval = client._build_trainloader_eval(global_round=0)

    assert len(trainloader) == 3
    assert len(trainloader_eval) == 3


def test_tf_trainloader_repeats_full_batches_without_crossing_epoch_boundaries():
    args = build_args(local_bs=2, local_ep=2, num_workers=1)
    dataset = OrderedDataset(size=5)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)

    trainloader = client._build_loader(
        client._train_split,
        batch_size=args.local_bs,
        shuffle=False,
        drop_last=True,
        seed_offset=0,
        repeat=True,
        num_batches=client._num_batches(client.num_samples, args.local_bs, drop_last=True),
        persistent_iterator=True,
    )
    first_epoch = [tuple(int(v) for v in images.numpy()[:, 0, 0, 0]) for images, _labels in trainloader]
    second_epoch = [tuple(int(v) for v in images.numpy()[:, 0, 0, 0]) for images, _labels in trainloader]

    assert len(first_epoch) == 2
    assert len(second_epoch) == 2
    assert first_epoch == second_epoch
    assert all(len(batch) == 2 for batch in first_epoch + second_epoch)
    assert all(len(set(batch)) == 2 for batch in first_epoch + second_epoch)


class SeedTrackingDataset:
    def __init__(self, size: int = 2):
        self.size = size
        self.seeds = []

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        import myseg.cv2_transform as cv2_transforms

        rng = cv2_transforms._active_numpy_rng()
        marker = -1 if rng is None else int(rng.randint(0, 2**31 - 1))
        self.seeds.append((int(idx), marker))
        image = np.full((3, 4, 4), idx, dtype=np.float32)
        label = np.full((4, 4), idx, dtype=np.int64)
        return image, label


def test_tf_parallel_loader_advances_numpy_seed_across_repeated_accesses():
    from segmentation.myseg.magic import create_tf_dataloader_from_custom_dataset_train

    dataset = SeedTrackingDataset(size=2)
    loader = create_tf_dataloader_from_custom_dataset_train(
        dataset,
        batch_size=1,
        shuffle=False,
        repeat=True,
        drop_last=False,
        output_img_shape=(3, 4, 4),
        output_lbl_shape=(4, 4),
        seed=11,
        num_parallel_calls=1,
        private_threadpool_size=1,
    )

    for step_idx, (_image, _label) in enumerate(loader):
        if step_idx == 3:
            break

    idx0_markers = [marker for idx, marker in dataset.seeds if idx == 0]
    idx1_markers = [marker for idx, marker in dataset.seeds if idx == 1]

    assert len(idx0_markers) >= 2
    assert len(idx1_markers) >= 2
    assert idx0_markers[0] != idx0_markers[1]
    assert idx1_markers[0] != idx1_markers[1]


def test_prototype_forward_batch_aggregation_matches_samplewise_results():
    torch.manual_seed(0)
    tf.random.set_seed(0)
    np.random.seed(0)

    args = build_args(
        num_classes=3,
        proj_dim=16,
        local_bs=2,
        local_ep=1,
        is_proto=True,
        pseudo_label=False,
    )
    dataset = OrderedDataset(size=2, spatial_size=32)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)

    torch_model = build_torch_bisenetv2(num_classes=args.num_classes, proj_dim=args.proj_dim, aux_mode="train")
    tf_model = build_tf_bisenetv2(num_classes=args.num_classes, proj_dim=args.proj_dim, aux_mode="train")
    load_torch_state_into_tf(tf_model, torch_model.state_dict())
    workspace = client._ensure_local_models(tf_model, global_round=0, require_train=False, require_reference=True)

    inputs_np = np.stack([dataset[idx][0] for idx in range(len(dataset))], axis=0).astype(np.float32)
    labels_np = np.stack([dataset[idx][1] for idx in range(len(dataset))], axis=0).astype(np.int64)
    inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
    labels = tf.convert_to_tensor(labels_np, dtype=tf.int64)

    batch_proto, batch_mask = workspace.prototype_forward(inputs, labels)

    sample_protos = []
    sample_masks = []
    for idx in range(len(dataset)):
        proto_i, mask_i = workspace.prototype_forward(inputs[idx : idx + 1], labels[idx : idx + 1])
        sample_protos.append(proto_i)
        sample_masks.append(mask_i)

    sample_proto = tf.concat(sample_protos, axis=0).numpy()
    sample_mask = tf.concat(sample_masks, axis=0).numpy()

    np.testing.assert_allclose(
        batch_proto.numpy(),
        sample_proto,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(batch_mask.numpy(), sample_mask)


def test_contrast_loss_preprocessed_proto_matches_direct_path():
    tf.random.set_seed(0)
    np.random.seed(0)

    args = build_args(num_classes=3, max_anchor=16, temperature=0.07, kmean_num=0)
    criterion = ContrastLoss(args)

    embs = tf.random.normal([2, 4, 5, 5], dtype=tf.float32)
    labels = tf.constant(
        [
            [
                [0, 0, 1, 1, 255],
                [0, 1, 1, 2, 2],
                [0, 2, 2, 2, 2],
                [1, 1, 2, 2, 2],
                [255, 1, 1, 2, 2],
            ],
            [
                [2, 2, 1, 1, 255],
                [2, 1, 1, 0, 0],
                [2, 0, 0, 0, 0],
                [1, 1, 0, 0, 0],
                [255, 1, 1, 0, 0],
            ],
        ],
        dtype=tf.int32,
    )
    proto_mem = tf.random.normal([3, 4], dtype=tf.float32)
    proto_mask = tf.constant([True, False, True], dtype=tf.bool)

    tf.random.set_seed(123)
    direct_loss = criterion(embs, labels, proto_mem, proto_mask)
    processed_proto = criterion.preprocess_prototypes(proto_mem, proto_mask)
    tf.random.set_seed(123)
    cached_loss = criterion(embs, labels, proto_mem, proto_mask, preprocessed_proto=processed_proto)

    np.testing.assert_allclose(float(direct_loss), float(cached_loss), rtol=1e-6, atol=1e-6)


def test_pseudo_contrast_loss_matches_torch_reference():
    repo_root = Path(__file__).resolve().parents[2]
    torch_utils_path = repo_root / "FedSeg-torch" / "segmentation" / "myseg" / "bisenet_utils.py"
    spec = importlib.util.spec_from_file_location("fedseg_torch_bisenet_utils_pseudo", torch_utils_path)
    torch_utils = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(torch_utils)

    np.random.seed(7)
    torch.manual_seed(7)
    tf.random.set_seed(7)
    args = build_args(num_classes=4, max_anchor=10_000, temperature=0.07, kmean_num=2)
    batch_size, feat_dim, height, width = 2, 3, 3, 4

    feat_np = np.random.randn(batch_size, feat_dim, height, width).astype(np.float32)
    proto_np = np.random.randn(args.num_classes, args.kmean_num, feat_dim).astype(np.float32)
    proto_mask_np = np.array(
        [
            [True, True],
            [False, False],
            [True, False],
            [True, True],
        ],
        dtype=bool,
    )

    target_cls = np.array(
        [
            [[0, 1, 2, 0], [2, 1, 0, 2], [0, 2, 1, 0]],
            [[2, 0, 1, 2], [0, 2, 2, 1], [1, 0, 2, 2]],
        ],
        dtype=np.int64,
    )
    low_conf_mask = np.zeros((batch_size, height, width), dtype=bool)
    low_conf_mask[0, 0, 3] = True
    low_conf_mask[1, 2, 0] = True
    logits_t_np = np.full((batch_size, args.num_classes, height, width), -5.0, dtype=np.float32)
    for sample_idx in range(batch_size):
        for row_idx in range(height):
            for col_idx in range(width):
                if low_conf_mask[sample_idx, row_idx, col_idx]:
                    logits_t_np[sample_idx, :, row_idx, col_idx] = 0.0
                else:
                    cls_idx = target_cls[sample_idx, row_idx, col_idx]
                    logits_t_np[sample_idx, cls_idx, row_idx, col_idx] = 5.0

    torch_logits_t = torch.tensor(logits_t_np, dtype=torch.float32)
    torch_labels_2 = torch.nn.functional.interpolate(torch_logits_t.float(), size=(height, width), mode="bilinear")
    torch_labels_2 = torch.softmax(torch_labels_2, dim=1)
    torch_props, torch_labels_2 = torch.max(torch_labels_2, dim=1)
    torch_labels_2[torch_props < 0.8] = 255
    proto_mask_tmp = torch.tensor(proto_mask_np).sum(1) < 1
    for class_idx, missing in enumerate(proto_mask_tmp):
        if bool(missing):
            torch_labels_2[torch_labels_2 == class_idx] = 255

    tf_logits_t = tf.convert_to_tensor(logits_t_np, dtype=tf.float32)
    tf_labels_2 = tf.image.resize(tf.transpose(tf_logits_t, [0, 2, 3, 1]), [height, width], method="bilinear")
    tf_labels_2 = tf.transpose(tf_labels_2, [0, 3, 1, 2])
    tf_labels_2 = tf.nn.softmax(tf_labels_2, axis=1)
    tf_props = tf.reduce_max(tf_labels_2, axis=1)
    tf_labels_2_cls = tf.argmax(tf_labels_2, axis=1, output_type=tf.int32)
    tf_labels_2_cls = tf.where(tf_props < 0.8, tf.cast(255, tf.int32), tf_labels_2_cls)
    missing_class_ids = tf.cast(tf.reshape(tf.where(tf.reduce_sum(tf.cast(proto_mask_np, tf.int32), axis=1) < 1), [-1]), tf.int32)
    missing_labels = tf.reshape(missing_class_ids, [1, 1, 1, -1])
    missing_mask = tf.reduce_any(tf.equal(tf.expand_dims(tf_labels_2_cls, axis=-1), missing_labels), axis=-1)
    tf_labels_2_cls = tf.where(missing_mask, tf.fill(tf.shape(tf_labels_2_cls), tf.cast(255, tf.int32)), tf_labels_2_cls)

    np.testing.assert_array_equal(tf_labels_2_cls.numpy(), torch_labels_2.numpy().astype(np.int32))

    torch_loss = torch_utils.ContrastLoss(args)(
        torch.tensor(feat_np, dtype=torch.float32),
        torch_labels_2.long(),
        torch.tensor(proto_np, dtype=torch.float32),
        torch.tensor(proto_mask_np),
    ).item()
    tf_loss = ContrastLoss(args)(
        tf.convert_to_tensor(feat_np, dtype=tf.float32),
        tf_labels_2_cls,
        tf.convert_to_tensor(proto_np, dtype=tf.float32),
        tf.convert_to_tensor(proto_mask_np),
    ).numpy()

    np.testing.assert_allclose(tf_loss, torch_loss, rtol=1e-6, atol=1e-6)


def test_single_step_ce_eval_mode_update_stays_aligned_with_torch():
    torch.manual_seed(0)
    tf.random.set_seed(0)
    np.random.seed(0)

    args = build_args(
        num_classes=3,
        proj_dim=16,
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
        local_ep=1,
        local_bs=2,
        losstype="ce",
        lr_scheduler="step",
    )
    dataset = OrderedDataset(size=2, spatial_size=32)
    client = Client(args, dataset, idxs=range(len(dataset)), client_id=0)

    torch_model = build_torch_bisenetv2(num_classes=args.num_classes, proj_dim=args.proj_dim, aux_mode="train")
    tf_model = build_tf_bisenetv2(num_classes=args.num_classes, proj_dim=args.proj_dim, aux_mode="train")
    load_torch_state_into_tf(tf_model, torch_model.state_dict())
    inputs_np = np.stack([dataset[idx][0] for idx in range(len(dataset))], axis=0).astype(np.float32)
    labels_np = np.stack([dataset[idx][1] for idx in range(len(dataset))], axis=0).astype(np.int64)

    torch_inputs = torch.from_numpy(inputs_np)
    torch_labels = torch.from_numpy(labels_np)
    torch_criteria = torch.nn.CrossEntropyLoss(ignore_index=255, reduction="mean")
    wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = torch_model.get_params()
    torch_optimizer = torch.optim.SGD(
        [
            {"params": wd_params},
            {"params": nowd_params, "weight_decay": 0.0},
            {"params": lr_mul_wd_params, "lr": args.lr * 10},
            {"params": lr_mul_nowd_params, "weight_decay": 0.0, "lr": args.lr * 10},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    torch_model.eval()
    torch_outputs = torch_model(torch_inputs)
    torch_loss = torch_criteria(torch_outputs[0], torch_labels)
    for aux_logits in torch_outputs[2:]:
        torch_loss = torch_loss + torch_criteria(aux_logits, torch_labels)
    torch_optimizer.zero_grad()
    torch_loss.backward()
    torch_optimizer.step()

    tf_inputs = tf.convert_to_tensor(inputs_np, dtype=tf.float32)
    tf_labels = tf.convert_to_tensor(labels_np, dtype=tf.int64)
    tf_optimizer = set_optimizer(tf_model, args)
    with tf.GradientTape() as tape:
        tf_outputs = tf_model(tf_inputs, training=False)
        tf_loss = client.criteria_pre(tf_labels, tf_outputs[0])
        for crit, aux_logits in zip(client.criteria_aux, tf_outputs[2:]):
            tf_loss = tf_loss + crit(tf_labels, aux_logits)
    grads = tape.gradient(tf_loss, tf_model.trainable_variables)
    client._apply_optimizer_step(tf_optimizer, grads, tf_model.trainable_variables, args.lr)

    eval_inputs = np.random.randn(2, 3, 64, 64).astype(np.float32)
    torch_model.eval()
    tf_model.aux_mode = "eval"
    with torch.no_grad():
        torch_logits = torch_model(torch.from_numpy(eval_inputs))[0].detach().cpu().numpy()
    tf_logits = tf_model(tf.convert_to_tensor(eval_inputs), training=False)[0].numpy()

    diff = np.abs(tf_logits - torch_logits)
    assert float(diff.mean()) < 5e-3
    assert float(diff.max()) < 2e-2

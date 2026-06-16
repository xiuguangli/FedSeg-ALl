import numpy as np
import tensorflow as tf
from tqdm import tqdm
import time

from eval_utils import build_tfdata_shape_batched_eval_loader, evaluate, evaluate_fast_shape_bucket, parse_eval_buckets
from logging_utils import logger
from myseg.bisenet_utils import (
    BackCELoss,
    DiceLoss,
    ContrastLoss,
    CriterionPixelPairSeq,
    CriterionPixelRegionPair,
    FocalLoss,
    LovaszLoss,
    OhemCELoss,
    SoftBCEWithLogitsLoss,
    SparseCEIgnore,
    set_optimizer,
)
from myseg.magic import create_tf_dataloader_from_custom_dataset_train
from runtime_utils import should_disable_tqdm
from tf2_tools import clone_tf_model


class DatasetSplit:
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in sorted(idxs)]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        if isinstance(image, np.ndarray):
            image = image.astype(np.float32)
        elif isinstance(image, tf.Tensor):
            image = tf.cast(image, tf.float32)

        if isinstance(label, np.ndarray):
            label = label.copy()
        elif isinstance(label, tf.Tensor):
            label = tf.identity(label)
        return image, label


class FiniteTfLoader:
    def __init__(self, dataset, num_batches, persistent_iterator=False):
        self.dataset = dataset
        self.num_batches = int(num_batches)
        self.persistent_iterator = bool(persistent_iterator)
        self._iterator = None

    def __iter__(self):
        if not self.persistent_iterator:
            yield from self.dataset.take(self.num_batches)
            return

        if self._iterator is None:
            self._iterator = iter(self.dataset)

        for _ in range(self.num_batches):
            yield next(self._iterator)

    def __len__(self):
        return self.num_batches

    def close(self):
        iterator = self._iterator
        self._iterator = None
        if iterator is not None:
            del iterator


_SHARED_WORKSPACE_CACHE = {}


class _SharedModelWorkspace:
    def __init__(self, template_model, input_shape, num_classes, local_bs):
        self.input_shape = tuple(int(dim) for dim in input_shape)
        self.num_classes = int(num_classes)
        self.local_bs = int(local_bs)
        self.train_model = clone_tf_model(template_model, input_shape=self.input_shape)
        self.reference_model = None
        self.optimizer = None
        self._compiled_train_step_cache = {}
        self.last_train_sync_token = None
        self.last_reference_sync_token = None
        self.train_forward = self._build_train_forward()
        self.reference_forward = None
        self.prototype_forward = None
        self._warmup_train_function()

    def _build_train_forward(self):
        c, h, w = self.input_shape
        train_spec = tf.TensorSpec(shape=[None, c, h, w], dtype=tf.float32)

        @tf.function(input_signature=[train_spec])
        def train_forward(images):
            outputs = self.train_model(images, training=True)
            return outputs[0], outputs[1], outputs[2:]

        return train_forward

    def _build_reference_functions(self):
        c, h, w = self.input_shape
        train_spec = tf.TensorSpec(shape=[None, c, h, w], dtype=tf.float32)
        proto_img_spec = tf.TensorSpec(shape=[None, c, h, w], dtype=tf.float32)
        proto_lbl_spec = tf.TensorSpec(shape=[None, h, w], dtype=tf.int64)

        @tf.function(input_signature=[train_spec])
        def reference_forward(images):
            outputs = self.reference_model(images, training=False)
            return outputs[0], outputs[1], outputs[2:]

        @tf.function(input_signature=[proto_img_spec, proto_lbl_spec])
        def prototype_forward(images, labels):
            outputs = self.reference_model(images, training=False)
            logits, feat_head = outputs[0], outputs[1]
            feat_h = tf.shape(feat_head)[2]
            feat_w = tf.shape(feat_head)[3]

            logits_nhwc = tf.transpose(logits, perm=[0, 2, 3, 1])
            logits_resized_nhwc = tf.image.resize(logits_nhwc, (feat_h, feat_w), method="bilinear")
            logits_resized = tf.transpose(logits_resized_nhwc, perm=[0, 3, 1, 2])

            probs = tf.nn.softmax(logits_resized, axis=1)
            props = tf.reduce_max(probs, axis=1)
            labels_2 = tf.argmax(probs, axis=1, output_type=tf.int32)
            labels_2 = tf.where(props < 0.8, tf.cast(255, tf.int32), labels_2)

            labels_expanded = tf.expand_dims(labels, axis=-1)
            labels_resized_nhwc = tf.image.resize(labels_expanded, (feat_h, feat_w), method="nearest")
            labels_resized = tf.cast(tf.squeeze(labels_resized_nhwc, axis=-1), tf.int32)
            labels_final = tf.where(labels_resized != 255, labels_resized, labels_2)

            labels_one_hot_nhwc = tf.one_hot(labels_final, depth=self.num_classes)
            labels_one_hot = tf.transpose(labels_one_hot_nhwc, perm=[0, 3, 1, 2])
            weight_sum = tf.reduce_sum(labels_one_hot, axis=[2, 3], keepdims=True)
            weight_norm = labels_one_hot / (weight_sum + 1e-5)
            out = tf.einsum("nfhw,nchw->ncf", feat_head, weight_norm)
            class_present = tf.reduce_sum(labels_one_hot, axis=[2, 3]) > 0
            label_mask = tf.cast(class_present, tf.float32)
            return out, label_mask

        self.reference_forward = reference_forward
        self.prototype_forward = prototype_forward

    def _warmup_train_function(self):
        c, h, w = self.input_shape
        dummy_single = tf.zeros([1, c, h, w], dtype=tf.float32)
        self.train_forward(dummy_single)

    def ensure_reference_model(self, template_model):
        if self.reference_model is None:
            self.reference_model = clone_tf_model(template_model, input_shape=self.input_shape)
            self.reference_model.trainable = False
            self._build_reference_functions()
            self.last_reference_sync_token = None
        return self.reference_model

    def ensure_optimizer(self, args):
        if self.optimizer is None:
            self.optimizer = set_optimizer(self.train_model, args)
        return self.optimizer

    def reset_optimizer(self, base_lr):
        if self.optimizer is None:
            return
        for variable in self.optimizer.variables:
            variable.assign(tf.zeros_like(variable))
        self.optimizer.learning_rate.assign(base_lr)


class Client:
    def __init__(self, args, dataset, idxs, client_id=None):
        self.args = args
        self.dataset = dataset
        self.idxs = [int(i) for i in sorted(idxs)]
        self.client_id = int(client_id if client_id is not None else 0)
        self.shape = dataset[0][0].shape
        self._train_split = self.idxs[:]
        self._test_split = self.idxs[: int(0.5 * len(self.idxs))]
        self._build_losses()
        self._profile_runtime = bool(getattr(args, "profile_runtime", False))

    @property
    def num_samples(self):
        return len(self.idxs)

    @property
    def num_eval_samples(self):
        return len(self._test_split)

    def preview_indices(self, limit=8):
        return self.idxs[:limit]

    def _loader_seed(self, seed_offset=0):
        base_seed = getattr(self.args, "seed", 0)
        return base_seed + self.client_id * 1000 + seed_offset

    def _num_batches(self, num_samples, batch_size, drop_last):
        if drop_last:
            return max(1, num_samples // batch_size)
        return max(1, (num_samples + batch_size - 1) // batch_size)

    def _build_loader(
        self,
        idxs,
        batch_size,
        shuffle,
        drop_last,
        seed_offset=0,
        repeat=False,
        num_batches=None,
        persistent_iterator=False,
        num_parallel_calls=None,
        private_threadpool_size=None,
    ):
        split = DatasetSplit(self.dataset, idxs)
        sample_image, sample_label = split[0]
        dataset = create_tf_dataloader_from_custom_dataset_train(
            split,
            batch_size=batch_size,
            shuffle=shuffle,
            repeat=repeat,
            drop_last=drop_last,
            output_img_shape=sample_image.shape,
            output_lbl_shape=sample_label.shape,
            seed=self._loader_seed(seed_offset) if shuffle else None,
            num_parallel_calls=num_parallel_calls,
            private_threadpool_size=private_threadpool_size,
        )
        if num_batches is not None:
            return FiniteTfLoader(dataset, num_batches, persistent_iterator=persistent_iterator)
        return dataset

    def _build_trainloader(self, global_round):
        return self._build_loader(
            self._train_split,
            batch_size=self.args.local_bs,
            shuffle=True,
            drop_last=True,
            seed_offset=global_round,
            repeat=False,
            num_batches=self._num_batches(self.num_samples, self.args.local_bs, drop_last=True),
            persistent_iterator=False,
        )

    def _build_trainloader_eval(self, global_round):
        return self._build_loader(
            self._train_split,
            batch_size=self.args.local_bs,
            shuffle=False,
            drop_last=False,
            seed_offset=global_round,
            repeat=False,
            num_batches=self._num_batches(self.num_samples, self.args.local_bs, drop_last=False),
        )

    def _build_testloader(self):
        return self._build_loader(
            self._test_split,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

    def _build_optimizer(self, model):
        return set_optimizer(model, self.args)

    def _prepare_proto_tensors(self, prototypes, proto_mask):
        if prototypes is None or proto_mask is None:
            return None, None, None, None

        prototypes_tensor = tf.convert_to_tensor(prototypes, dtype=tf.float32)
        proto_mask_tensor = tf.cast(tf.convert_to_tensor(proto_mask), tf.bool)
        processed_proto = self.criteria_contrast.preprocess_prototypes(prototypes_tensor, proto_mask_tensor)
        if self.args.kmean_num > 0:
            missing_mask = tf.reduce_sum(tf.cast(proto_mask_tensor, tf.int32), axis=1) < 1
        else:
            missing_mask = tf.logical_not(proto_mask_tensor)

        missing_class_ids = tf.cast(tf.reshape(tf.where(missing_mask), [-1]), tf.int32)
        return prototypes_tensor, proto_mask_tensor, missing_class_ids, processed_proto

    def _mask_missing_classes(self, labels, missing_class_ids):
        if missing_class_ids is None:
            return labels

        labels = tf.cast(labels, tf.int32)
        missing_labels = tf.reshape(missing_class_ids, [1, 1, 1, -1])
        missing_mask = tf.reduce_any(tf.equal(tf.expand_dims(labels, axis=-1), missing_labels), axis=-1)
        return tf.where(missing_mask, tf.fill(tf.shape(labels), tf.cast(255, labels.dtype)), labels)

    def _supports_compiled_train_step(self, use_proto_branch):
        args = self.args
        if args.distill or args.fedprox_mu > 0:
            return False
        if args.losstype not in {"ce", "back"}:
            return False
        return bool(use_proto_branch)

    def _build_compiled_train_step(self, workspace, use_pseudo_label, profile_runtime=False):
        optimizer = workspace.ensure_optimizer(self.args)
        c, h, w = self.shape
        image_spec = tf.TensorSpec(shape=[None, c, h, w], dtype=tf.float32)
        label_spec = tf.TensorSpec(shape=[None, h, w], dtype=tf.int64)
        lr_spec = tf.TensorSpec(shape=[], dtype=tf.float32)
        prototype_spec = tf.TensorSpec(shape=[None, None, None], dtype=tf.float32)
        proto_mask_spec = tf.TensorSpec(shape=[None, None], dtype=tf.bool)
        processed_proto_spec = tf.TensorSpec(shape=[None, None], dtype=tf.float32)
        processed_label_spec = tf.TensorSpec(shape=[None], dtype=tf.int32)
        missing_class_spec = tf.TensorSpec(shape=[None], dtype=tf.int32)
        criteria_pre = self.criteria_pre
        criteria_aux = tuple(self.criteria_aux)
        criteria_contrast = self.criteria_contrast
        con_lamb = tf.constant(self.args.con_lamb, dtype=tf.float32)
        input_signature = [
            image_spec,
            label_spec,
            lr_spec,
            prototype_spec,
            proto_mask_spec,
            processed_proto_spec,
            processed_label_spec,
            missing_class_spec,
        ]

        if profile_runtime:

            @tf.function(input_signature=input_signature)
            def train_step(
                images,
                labels,
                current_lr,
                prototypes_tensor,
                proto_mask_tensor,
                processed_proto_mem,
                processed_proto_labels,
                missing_class_ids,
            ):
                optimizer.learning_rate.assign(current_lr)
                with tf.GradientTape() as tape:
                    t0 = tf.timestamp()
                    outputs = workspace.train_model(images, training=True)
                    logits, feat_head, logits_aux = outputs[0], outputs[1], outputs[2:]

                    base_loss = criteria_pre(labels, logits)
                    for crit, aux_logits in zip(criteria_aux, logits_aux):
                        base_loss += crit(labels, aux_logits)
                    with tf.control_dependencies([base_loss]):
                        t1 = tf.timestamp()

                    loss = base_loss
                    loss_con = tf.constant(0.0, dtype=tf.float32)
                    loss_con_2 = tf.constant(0.0, dtype=tf.float32)

                    feat_h = tf.shape(feat_head)[2]
                    feat_w = tf.shape(feat_head)[3]
                    labels_1 = tf.expand_dims(labels, axis=-1)
                    labels_1 = tf.image.resize(labels_1, [feat_h, feat_w], method="nearest")
                    labels_1 = self._mask_missing_classes(tf.squeeze(labels_1, axis=-1), missing_class_ids)

                    loss_con = criteria_contrast(
                        feat_head,
                        labels_1,
                        prototypes_tensor,
                        proto_mask_tensor,
                        preprocessed_proto=(processed_proto_mem, processed_proto_labels),
                    )
                    loss += con_lamb * loss_con
                    with tf.control_dependencies([loss_con]):
                        t2 = tf.timestamp()

                    if use_pseudo_label:
                        reference_outputs = workspace.reference_model(images, training=False)
                        logits_t = reference_outputs[0]
                        logits_t = tf.transpose(logits_t, [0, 2, 3, 1])
                        labels_2 = tf.image.resize(logits_t, [feat_h, feat_w], method="bilinear")
                        labels_2 = tf.transpose(labels_2, [0, 3, 1, 2])
                        labels_2 = tf.nn.softmax(labels_2, axis=1)
                        props = tf.reduce_max(labels_2, axis=1)
                        labels_2_cls = tf.argmax(labels_2, axis=1, output_type=tf.int32)
                        labels_2_cls = tf.where(props < 0.8, tf.cast(255, tf.int32), labels_2_cls)
                        labels_2_cls = self._mask_missing_classes(labels_2_cls, missing_class_ids)
                        loss_con_2 = criteria_contrast(
                            feat_head,
                            labels_2_cls,
                            prototypes_tensor,
                            proto_mask_tensor,
                            preprocessed_proto=(processed_proto_mem, processed_proto_labels),
                        )
                        loss += con_lamb * loss_con_2
                    with tf.control_dependencies([loss]):
                        t3 = tf.timestamp()

                grads = tape.gradient(loss, workspace.train_model.trainable_variables)
                grad_deps = [grad for grad in grads if grad is not None]
                with tf.control_dependencies(grad_deps):
                    t4 = tf.timestamp()
                apply_result = self._apply_optimizer_step(optimizer, grads, workspace.train_model.trainable_variables, current_lr)
                with tf.control_dependencies([apply_result]):
                    t5 = tf.timestamp()
                return (
                    loss,
                    base_loss,
                    loss_con,
                    loss_con_2,
                    t1 - t0,
                    t2 - t1,
                    t3 - t2,
                    t4 - t3,
                    t5 - t4,
                )

            return train_step

        @tf.function(input_signature=input_signature)
        def train_step(
            images,
            labels,
            current_lr,
            prototypes_tensor,
            proto_mask_tensor,
            processed_proto_mem,
            processed_proto_labels,
            missing_class_ids,
        ):
            optimizer.learning_rate.assign(current_lr)
            with tf.GradientTape() as tape:
                outputs = workspace.train_model(images, training=True)
                logits, feat_head, logits_aux = outputs[0], outputs[1], outputs[2:]

                base_loss = criteria_pre(labels, logits)
                for crit, aux_logits in zip(criteria_aux, logits_aux):
                    base_loss += crit(labels, aux_logits)

                loss = base_loss
                loss_con = tf.constant(0.0, dtype=tf.float32)
                loss_con_2 = tf.constant(0.0, dtype=tf.float32)

                feat_h = tf.shape(feat_head)[2]
                feat_w = tf.shape(feat_head)[3]
                labels_1 = tf.expand_dims(labels, axis=-1)
                labels_1 = tf.image.resize(labels_1, [feat_h, feat_w], method="nearest")
                labels_1 = self._mask_missing_classes(tf.squeeze(labels_1, axis=-1), missing_class_ids)

                loss_con = criteria_contrast(
                    feat_head,
                    labels_1,
                    prototypes_tensor,
                    proto_mask_tensor,
                    preprocessed_proto=(processed_proto_mem, processed_proto_labels),
                )
                loss += con_lamb * loss_con

                if use_pseudo_label:
                    reference_outputs = workspace.reference_model(images, training=False)
                    logits_t = reference_outputs[0]
                    logits_t = tf.transpose(logits_t, [0, 2, 3, 1])
                    labels_2 = tf.image.resize(logits_t, [feat_h, feat_w], method="bilinear")
                    labels_2 = tf.transpose(labels_2, [0, 3, 1, 2])
                    labels_2 = tf.nn.softmax(labels_2, axis=1)
                    props = tf.reduce_max(labels_2, axis=1)
                    labels_2_cls = tf.argmax(labels_2, axis=1, output_type=tf.int32)
                    labels_2_cls = tf.where(props < 0.8, tf.cast(255, tf.int32), labels_2_cls)
                    labels_2_cls = self._mask_missing_classes(labels_2_cls, missing_class_ids)
                    loss_con_2 = criteria_contrast(
                        feat_head,
                        labels_2_cls,
                        prototypes_tensor,
                        proto_mask_tensor,
                        preprocessed_proto=(processed_proto_mem, processed_proto_labels),
                    )
                    loss += con_lamb * loss_con_2

            grads = tape.gradient(loss, workspace.train_model.trainable_variables)
            self._apply_optimizer_step(optimizer, grads, workspace.train_model.trainable_variables, current_lr)
            return loss, base_loss, loss_con, loss_con_2

        return train_step

    def _get_compiled_train_step(self, workspace, use_pseudo_label):
        cache_key = ("compiled_train_step", self.args.losstype, bool(use_pseudo_label), bool(self._profile_runtime))
        train_step = workspace._compiled_train_step_cache.get(cache_key)
        if train_step is not None:
            return train_step

        train_step = self._build_compiled_train_step(
            workspace,
            use_pseudo_label,
            profile_runtime=self._profile_runtime,
        )
        c, h, w = self.shape
        dummy_images = tf.zeros([1, c, h, w], dtype=tf.float32)
        dummy_labels = tf.zeros([1, h, w], dtype=tf.int64)
        feat_dim = int(workspace.train_model(dummy_images, training=False)[1].shape[1])
        dummy_prototypes = tf.zeros([self.args.num_classes, 1, feat_dim], dtype=tf.float32)
        dummy_proto_mask = tf.zeros([self.args.num_classes, 1], dtype=tf.bool)
        dummy_processed_proto, dummy_processed_labels = self.criteria_contrast.preprocess_prototypes(
            dummy_prototypes, dummy_proto_mask
        )
        dummy_missing_classes = tf.zeros([0], dtype=tf.int32)
        workspace.reset_optimizer(self.args.lr)
        train_step(
            dummy_images,
            dummy_labels,
            tf.constant(self.args.lr, dtype=tf.float32),
            dummy_prototypes,
            dummy_proto_mask,
            dummy_processed_proto,
            dummy_processed_labels,
            dummy_missing_classes,
        )
        workspace.reset_optimizer(self.args.lr)
        workspace._compiled_train_step_cache[cache_key] = train_step
        return train_step

    def _sync_model_variables(self, target_model, source_model):
        for target_var, source_var in zip(target_model.weights, source_model.weights):
            target_var.assign(source_var)

    def _workspace_key(self, template_model):
        return (
            type(template_model).__name__,
            tuple(int(dim) for dim in self.shape),
            int(self.args.num_classes),
            int(getattr(self.args, "proj_dim", 0)),
        )

    def _get_workspace(self, template_model):
        workspace_key = self._workspace_key(template_model)
        workspace = _SHARED_WORKSPACE_CACHE.get(workspace_key)
        if workspace is None:
            workspace = _SharedModelWorkspace(
                template_model,
                input_shape=self.shape,
                num_classes=self.args.num_classes,
                local_bs=self.args.local_bs,
            )
            _SHARED_WORKSPACE_CACHE[workspace_key] = workspace
        return workspace

    def _train_sync_token(self, template_model, global_round):
        return (id(template_model), self.client_id, int(global_round))

    def _reference_sync_token(self, template_model, global_round):
        return (id(template_model), int(global_round))

    def _ensure_local_models(self, template_model, global_round, require_train=True, require_reference=True):
        workspace = self._get_workspace(template_model)
        train_sync_token = self._train_sync_token(template_model, global_round)
        reference_sync_token = self._reference_sync_token(template_model, global_round)
        if require_train and workspace.last_train_sync_token != train_sync_token:
            self._sync_model_variables(workspace.train_model, template_model)
            workspace.last_train_sync_token = train_sync_token
        if require_reference and workspace.last_reference_sync_token != reference_sync_token:
            workspace.ensure_reference_model(template_model)
            self._sync_model_variables(workspace.reference_model, template_model)
            workspace.last_reference_sync_token = reference_sync_token

        workspace.train_model.trainable = True
        if workspace.reference_model is not None:
            workspace.reference_model.trainable = False
        return workspace

    def _build_scheduler(self, optimizer, trainloader, global_round):
        del optimizer
        total_steps = max(1, len(trainloader) * max(1, self.args.local_ep))
        scheduler_name = getattr(self.args, "lr_scheduler", "poly")
        if scheduler_name == "step":
            if global_round < 1000:
                return lambda step_idx: self.args.lr
            return lambda step_idx: self.args.lr * 0.1
        return lambda step_idx: self.args.lr * (1.0 - (min(step_idx, total_steps) / total_steps)) ** 0.9

    def _apply_optimizer_step(self, optimizer, grads, variables, base_lr):
        grad_var_pairs = []
        lr_multipliers = getattr(optimizer, "_fedseg_var_lr_multipliers", {})
        weight_decays = getattr(optimizer, "_fedseg_var_weight_decays", {})
        global_weight_decay = getattr(self.args, "weight_decay", 0.0)

        for grad, var in zip(grads, variables):
            if grad is None:
                continue

            lr_multiplier = lr_multipliers.get(id(var), 1.0)
            if lr_multiplier != 1.0:
                grad = grad * lr_multiplier

            # Match torch SGD parameter groups by handling per-variable decay manually.
            var_weight_decay = weight_decays.get(id(var), global_weight_decay)
            if var_weight_decay:
                grad = grad + (lr_multiplier * var_weight_decay) * var
            grad_var_pairs.append((grad, var))

        return optimizer.apply_gradients(grad_var_pairs)

    def _build_losses(self):
        args = self.args
        self.criteria_distill_pi = CriterionPixelPairSeq(args, temperature=args.temp_dist)
        self.criteria_distill_pa = CriterionPixelRegionPair(args)
        self.criteria_contrast = ContrastLoss(args)

        if args.losstype == "ohem":
            self.criteria_pre = OhemCELoss(thresh=0.7)
            self.criteria_aux = [OhemCELoss(thresh=0.7) for _ in range(4)]
        elif args.losstype == "ce":
            self.criteria_pre = SparseCEIgnore()
            self.criteria_aux = [SparseCEIgnore() for _ in range(4)]
        elif args.losstype == "back":
            self.criteria_pre = BackCELoss(args)
            self.criteria_aux = [BackCELoss(args) for _ in range(4)]
        elif args.losstype == "lovasz":
            self.criteria_pre = LovaszLoss("multiclass", ignore_index=255)
            self.criteria_aux = [LovaszLoss("multiclass", ignore_index=255) for _ in range(4)]
        elif args.losstype == "dice":
            self.criteria_pre = DiceLoss("multiclass", ignore_index=255)
            self.criteria_aux = [DiceLoss("multiclass", ignore_index=255) for _ in range(4)]
        elif args.losstype == "focal":
            self.criteria_pre = FocalLoss("multiclass", alpha=0.25, ignore_index=255)
            self.criteria_aux = [FocalLoss("multiclass", alpha=0.25, ignore_index=255) for _ in range(4)]
        elif args.losstype == "bce":
            self.criteria_pre = SoftBCEWithLogitsLoss(ignore_index=255)
            self.criteria_aux = [SoftBCEWithLogitsLoss(ignore_index=255) for _ in range(4)]
        else:
            raise ValueError("loss type is not defined")

    def extract_prototypes(self, model, global_round):
        args = self.args
        tmp_ = []
        label_mask_batches = []
        stage_times = {}
        timer = time.perf_counter
        disable_tqdm = should_disable_tqdm()
        start = timer()
        workspace = self._ensure_local_models(model, global_round, require_train=False, require_reference=True)
        stage_times["sync_local_models"] = timer() - start
        start = timer()
        trainloader_eval = self._build_trainloader_eval(global_round)
        stage_times["build_loader"] = timer() - start

        proto_loader = tqdm(
            trainloader_eval,
            desc=f"Client {self.client_id} prototypes",
            leave=False,
            disable=disable_tqdm,
        )
        iter_end = timer()
        data_wait_time = 0.0
        compute_time = 0.0
        num_proto_batches = 0
        for images, labels in proto_loader:
            batch_start = timer()
            data_wait_time += batch_start - iter_end
            out, label_mask = workspace.prototype_forward(images, labels)
            tmp_.append(out)
            label_mask_batches.append(label_mask)
            num_proto_batches += 1
            iter_end = timer()
            compute_time += iter_end - batch_start

        start = timer()
        tmp_ = tf.transpose(tf.concat(tmp_, axis=0), perm=[1, 0, 2])
        label_mask_ = tf.transpose(tf.concat(label_mask_batches, axis=0), perm=[1, 0])
        observed_classes = tf.cast(
            tf.reshape(tf.where(tf.reduce_any(tf.cast(label_mask_, tf.bool), axis=1)), [-1]),
            tf.int32,
        ).numpy().tolist()
        stage_times["stack_outputs"] = timer() - start
        start = timer()
        if hasattr(trainloader_eval, "close"):
            trainloader_eval.close()
        stage_times["close_loader"] = timer() - start
        if self._profile_runtime:
            logger.info(
                "Runtime profile | client={} round={} prototypes loader={:.3f}s sync_models={:.3f}s data_wait={:.3f}s compute={:.3f}s stack={:.3f}s close_loader={:.3f}s batches={}",
                self.client_id,
                global_round,
                stage_times.get("build_loader", 0.0),
                stage_times.get("sync_local_models", 0.0),
                data_wait_time,
                compute_time,
                stage_times.get("stack_outputs", 0.0),
                stage_times.get("close_loader", 0.0),
                num_proto_batches,
            )
        return tmp_, observed_classes, label_mask_

    def train(self, model, global_round, prototypes=None, proto_mask=None):
        args = self.args
        epoch_loss = []
        pixel_seq = []
        stage_times = {}
        timer = time.perf_counter
        disable_tqdm = should_disable_tqdm()

        start = timer()
        trainloader = self._build_trainloader(global_round)
        stage_times["build_trainloader"] = timer() - start
        use_proto_branch = bool(
            args.is_proto and global_round >= args.proto_start_epoch and prototypes is not None and proto_mask is not None
        )
        needs_reference_model = bool(
            args.distill
            or args.fedprox_mu > 0
            or (use_proto_branch and args.pseudo_label and global_round >= args.pseudo_label_start_epoch)
        )
        start = timer()
        workspace = self._ensure_local_models(model, global_round, require_reference=needs_reference_model)
        model = workspace.train_model
        reference_model = workspace.reference_model if needs_reference_model else None
        stage_times["sync_local_models"] = timer() - start
        start = timer()
        optimizer = self._build_optimizer(model)
        stage_times["build_optimizer"] = timer() - start
        lr_schedule = self._build_scheduler(optimizer, trainloader, global_round)
        step_idx = 0

        if not (args.distill or args.fedprox_mu > 0 or (args.is_proto and args.pseudo_label)):
            reference_model = None

        model.aux_mode = "train"
        workspace.train_model.aux_mode = "train"

        prototypes_tensor = None
        proto_mask_tensor = None
        missing_class_ids = None
        processed_proto = None
        if use_proto_branch:
            prototypes_tensor, proto_mask_tensor, missing_class_ids, processed_proto = self._prepare_proto_tensors(
                prototypes, proto_mask
            )

        use_pseudo_label = bool(
            use_proto_branch
            and args.pseudo_label
            and global_round >= args.pseudo_label_start_epoch
            and reference_model is not None
        )

        last_loss_ce = tf.constant(0.0, dtype=tf.float32)
        last_loss_con = tf.constant(0.0, dtype=tf.float32)
        last_loss_con_2 = tf.constant(0.0, dtype=tf.float32)
        last_loss_1 = tf.constant(0.0, dtype=tf.float32)
        last_loss_pi = tf.constant(0.0, dtype=tf.float32)
        last_loss_pa = tf.constant(0.0, dtype=tf.float32)
        use_compiled_train_step = self._supports_compiled_train_step(use_proto_branch)
        compiled_train_step = None
        if use_compiled_train_step:
            start = timer()
            optimizer = workspace.ensure_optimizer(self.args)
            workspace.reset_optimizer(self.args.lr)
            compiled_train_step = self._get_compiled_train_step(workspace, use_pseudo_label)
            stage_times["build_optimizer"] = timer() - start

        epoch_bar = tqdm(
            range(args.local_ep),
            desc=f"Client {self.client_id} local epochs",
            leave=False,
            disable=disable_tqdm,
        )
        for local_epoch in epoch_bar:
            batch_loss = []
            batch_bar = tqdm(
                trainloader,
                desc=f"Client {self.client_id} epoch {local_epoch + 1}",
                leave=False,
                disable=disable_tqdm,
            )
            iter_end = timer()
            data_wait_time = 0.0
            compute_time = 0.0
            compiled_stage_times = {
                "forward_loss": 0.0,
                "contrast": 0.0,
                "pseudo": 0.0,
                "backward": 0.0,
                "apply": 0.0,
            }
            for images, labels in batch_bar:
                batch_start = timer()
                data_wait_time += batch_start - iter_end
                current_lr = lr_schedule(step_idx)
                if use_compiled_train_step:
                    compiled_outputs = compiled_train_step(
                        images,
                        labels,
                        tf.constant(current_lr, dtype=tf.float32),
                        prototypes_tensor,
                        proto_mask_tensor,
                        processed_proto[0],
                        processed_proto[1],
                        missing_class_ids,
                    )
                    if self._profile_runtime:
                        (
                            loss,
                            last_loss_ce,
                            last_loss_con,
                            last_loss_con_2,
                            forward_loss_time,
                            contrast_time,
                            pseudo_time,
                            backward_time,
                            apply_time,
                        ) = compiled_outputs
                        compiled_stage_times["forward_loss"] += float(forward_loss_time)
                        compiled_stage_times["contrast"] += float(contrast_time)
                        compiled_stage_times["pseudo"] += float(pseudo_time)
                        compiled_stage_times["backward"] += float(backward_time)
                        compiled_stage_times["apply"] += float(apply_time)
                    else:
                        loss, last_loss_ce, last_loss_con, last_loss_con_2 = compiled_outputs
                    last_loss_1 = last_loss_ce
                    last_loss_pi = tf.constant(0.0, dtype=tf.float32)
                    last_loss_pa = tf.constant(0.0, dtype=tf.float32)
                else:
                    optimizer.learning_rate.assign(current_lr)
                    with tf.GradientTape() as tape:
                        logits, feat_head, logits_aux = workspace.train_forward(images)
                        labels_ = labels
                        if args.losstype == "bce":
                            labels_ = tf.cast(tf.one_hot(tf.cast(labels_, tf.int32), depth=args.num_classes), tf.float32)

                        loss_pre = self.criteria_pre(labels_, logits)
                        loss_aux = sum(crit(labels_, lgt) for crit, lgt in zip(self.criteria_aux, logits_aux))
                        base_loss = loss_pre + loss_aux
                        loss = base_loss
                        last_loss_ce = base_loss
                        last_loss_con = tf.constant(0.0, dtype=tf.float32)
                        last_loss_con_2 = tf.constant(0.0, dtype=tf.float32)
                        last_loss_pi = tf.constant(0.0, dtype=tf.float32)
                        last_loss_pa = tf.constant(0.0, dtype=tf.float32)
                        last_loss_1 = base_loss

                        if use_proto_branch:
                            feat_h = tf.shape(feat_head)[2]
                            feat_w = tf.shape(feat_head)[3]
                            labels_1 = tf.expand_dims(labels_, axis=-1)
                            labels_1 = tf.image.resize(labels_1, [feat_h, feat_w], method="nearest")
                            labels_1 = self._mask_missing_classes(tf.squeeze(labels_1, axis=-1), missing_class_ids)

                            loss_con = self.criteria_contrast(
                                feat_head,
                                labels_1,
                                prototypes_tensor,
                                proto_mask_tensor,
                                preprocessed_proto=processed_proto,
                            )
                            last_loss_con = loss_con
                            loss += args.con_lamb * loss_con

                            if use_pseudo_label:
                                logits_t, _feat_t_unused, _aux_t_unused = workspace.reference_forward(images)
                                logits_t = tf.transpose(logits_t, [0, 2, 3, 1])
                                labels_2 = tf.image.resize(logits_t, [feat_h, feat_w], method="bilinear")
                                labels_2 = tf.transpose(labels_2, [0, 3, 1, 2])
                                labels_2 = tf.nn.softmax(labels_2, axis=1)
                                props = tf.reduce_max(labels_2, axis=1)
                                labels_2_cls = tf.argmax(labels_2, axis=1, output_type=tf.int32)
                                labels_2_cls = tf.where(props < 0.8, tf.cast(255, tf.int32), labels_2_cls)
                                labels_2_cls = self._mask_missing_classes(labels_2_cls, missing_class_ids)
                                loss_con_2 = self.criteria_contrast(
                                    feat_head,
                                    labels_2_cls,
                                    prototypes_tensor,
                                    proto_mask_tensor,
                                    preprocessed_proto=processed_proto,
                                )
                                last_loss_con_2 = loss_con_2
                                loss += args.con_lamb * loss_con_2

                        if args.fedprox_mu > 0 and reference_model is not None:
                            proximal_term = 0.0
                            for w_var, w_ref in zip(model.trainable_variables, reference_model.trainable_variables):
                                proximal_term += tf.norm(w_var - w_ref, ord=2)
                            loss += (args.fedprox_mu / 2.0) * proximal_term

                        if args.distill and reference_model is not None:
                            _logits_t_unused, feat_head_t, _aux_t_unused = workspace.reference_forward(images)
                            if args.distill_lamb_pi > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                                loss_pi, pixel_seq = self.criteria_distill_pi(feat_head, tf.stop_gradient(feat_head_t), pixel_seq)
                                loss_pi = args.distill_lamb_pi * loss_pi
                                loss += loss_pi
                                last_loss_pi = loss_pi
                            if args.distill_lamb_pa > 0 and args.is_proto and global_round >= args.proto_start_epoch:
                                loss_pa = args.distill_lamb_pa * self.criteria_distill_pa(
                                    feat_head,
                                    tf.stop_gradient(feat_head_t),
                                    prototypes_tensor if prototypes_tensor is not None else prototypes,
                                    proto_mask_tensor if proto_mask_tensor is not None else proto_mask,
                                )
                                loss += loss_pa
                                last_loss_pa = loss_pa

                    grads = tape.gradient(loss, model.trainable_variables)
                    self._apply_optimizer_step(optimizer, grads, model.trainable_variables, current_lr)
                loss_scalar = float(loss)
                batch_loss.append(loss_scalar)
                if not disable_tqdm:
                    batch_bar.set_postfix(loss=f"{loss_scalar:.4f}", lr=f"{current_lr:.2e}")
                step_idx += 1
                iter_end = timer()
                compute_time += iter_end - batch_start

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if not disable_tqdm:
                epoch_bar.set_postfix(loss=f"{epoch_loss[-1]:.4f}")
            if args.verbose:
                logger.debug(
                    "| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}",
                    global_round,
                    local_epoch + 1,
                    self.num_samples,
                    float(batch_loss[-1]),
                )
            if self._profile_runtime:
                logger.info(
                    "Runtime profile | client={} round={} epoch={} data_wait={:.3f}s compute={:.3f}s forward_loss={:.3f}s contrast={:.3f}s pseudo={:.3f}s backward={:.3f}s apply={:.3f}s batches={}",
                    self.client_id,
                    global_round,
                    local_epoch + 1,
                    data_wait_time,
                    compute_time,
                    compiled_stage_times["forward_loss"],
                    compiled_stage_times["contrast"],
                    compiled_stage_times["pseudo"],
                    compiled_stage_times["backward"],
                    compiled_stage_times["apply"],
                    len(batch_loss),
                )

        loss_ce = float(last_loss_ce)
        loss_con_item = float(last_loss_con)
        loss_con_2_item = float(last_loss_con_2)
        loss_1_item = float(last_loss_1)
        loss_pi_item = float(last_loss_pi)
        loss_pa_item = float(last_loss_pa)

        logger.info(
            "| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}",
            global_round,
            args.local_ep,
            self.num_samples,
            float(epoch_loss[-1]),
        )
        if args.distill:
            logger.info("Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}", loss_1_item, loss_pi_item, loss_pa_item)
        if args.is_proto:
            if global_round >= args.proto_start_epoch:
                if args.pseudo_label:
                    logger.info(
                        "Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}",
                        loss_ce,
                        loss_con_item,
                        loss_con_2_item,
                    )
                else:
                    logger.info("Loss_CE:{:.6f} | loss_contrast:{:.6f}", loss_ce, loss_con_item)
            else:
                logger.info("Loss_CE:{:.6f}", loss_ce)

        if self._profile_runtime:
            logger.info(
                "Runtime profile | client={} round={} loader={:.3f}s sync_models={:.3f}s optimizer={:.3f}s",
                self.client_id,
                global_round,
                stage_times.get("build_trainloader", 0.0),
                stage_times.get("sync_local_models", 0.0),
                stage_times.get("build_optimizer", 0.0),
            )

        start = timer()
        avg_epoch_loss = sum(epoch_loss) / len(epoch_loss)
        stage_times["finalize_epoch_loss"] = timer() - start
        start = timer()
        returned_weights = list(model.weights)
        stage_times["collect_weights"] = timer() - start
        start = timer()
        if hasattr(trainloader, "close"):
            trainloader.close()
        stage_times["close_loader"] = timer() - start
        start = timer()
        workspace.last_train_sync_token = None
        stage_times["clear_sync_token"] = timer() - start
        if self._profile_runtime:
            logger.info(
                "Runtime profile | client={} round={} finalize_epoch_loss={:.3f}s collect_weights={:.3f}s close_loader={:.3f}s clear_sync_token={:.3f}s",
                self.client_id,
                global_round,
                stage_times.get("finalize_epoch_loss", 0.0),
                stage_times.get("collect_weights", 0.0),
                stage_times.get("close_loader", 0.0),
                stage_times.get("clear_sync_token", 0.0),
            )
        return returned_weights, avg_epoch_loss

    def inference(self, model):
        testloader = self._build_testloader()
        if bool(getattr(self.args, "eval_fast_mode", False)):
            if bool(getattr(self.args, "eval_tfdata_batch", False)):
                split = DatasetSplit(self.dataset, self._test_split)
                testloader = build_tfdata_shape_batched_eval_loader(
                    split,
                    batch_size=max(1, int(getattr(self.args, "eval_bs", 1))),
                    eval_buckets=parse_eval_buckets(getattr(self.args, "eval_buckets", "")),
                    num_parallel_calls=max(1, int(getattr(self.args, "num_workers", 1))),
                )
            return evaluate_fast_shape_bucket(
                model,
                testloader,
                self.args.num_classes,
                batch_size=max(1, int(getattr(self.args, "eval_bs", 1))),
                dataset_size=self.num_eval_samples,
                eval_buckets=parse_eval_buckets(getattr(self.args, "eval_buckets", "")),
                profile_runtime=bool(getattr(self.args, "profile_runtime", False)),
                tfdata_batch=bool(getattr(self.args, "eval_tfdata_batch", False)),
                desc="Evaluating local fast",
            )
        confmat = evaluate(model, testloader, self.args.num_classes)
        acc_global, _, _, iou_mean = confmat._get_metric_values()
        return acc_global, iou_mean, str(confmat)

    evaluate = inference
    update_weights = train
    get_protos = extract_prototypes


LocalUpdate = Client


def test_inference(args, model, testloader):
    confmat = evaluate(model, testloader, args.num_classes)
    acc_global, _, _, iou_mean = confmat._get_metric_values()
    return acc_global, iou_mean, str(confmat)

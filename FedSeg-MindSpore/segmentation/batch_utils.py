import math
import random
from contextlib import contextmanager

import mindspore as ms
import numpy as np


def resolve_indices(dataset, idxs=None):
    if idxs is None:
        return list(range(len(dataset)))
    return [int(idx) for idx in idxs]


@contextmanager
def temporary_random_seed(seed):
    if seed is None:
        yield
        return

    py_state = random.getstate()
    np_state = np.random.get_state()
    random.seed(int(seed))
    np.random.seed(int(seed) % (2 ** 32))
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def pad_batch(samples, image_pad_value=0.0, label_pad_value=255):
    images = [sample[0] for sample in samples]
    labels = [sample[1] for sample in samples]

    max_h = max(image.shape[1] for image in images)
    max_w = max(image.shape[2] for image in images)

    padded_images = []
    padded_labels = []
    for image, label in zip(images, labels):
        pad_h = max_h - image.shape[1]
        pad_w = max_w - image.shape[2]
        padded_images.append(
            np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                constant_values=image_pad_value,
            )
        )
        padded_labels.append(
            np.pad(
                label,
                ((0, pad_h), (0, pad_w)),
                constant_values=label_pad_value,
            )
        )

    return np.stack(padded_images).astype(np.float32), np.stack(padded_labels).astype(np.int32)


class BatchLoader:
    def __init__(
        self,
        dataset,
        idxs=None,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        seed=None,
        pad_to_max_shape=False,
        synthetic_num_workers=0,
        batch_worker_offset=0,
        worker_seed_base=None,
        worker_states=None,
    ):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.seed = None if seed is None else int(seed)
        self.pad_to_max_shape = bool(pad_to_max_shape)
        self.synthetic_num_workers = max(0, int(synthetic_num_workers))
        self.batch_worker_offset = int(batch_worker_offset)
        self.worker_seed_base = None if worker_seed_base is None else int(worker_seed_base)
        self.indices = resolve_indices(dataset, idxs)
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(self.indices)
        self._num_batches = num_batches(len(self.indices), self.batch_size, self.drop_last)
        if self.synthetic_num_workers > 0:
            if worker_states is None:
                if self.worker_seed_base is None:
                    raise ValueError("worker_seed_base is required when synthetic_num_workers > 0")
                worker_states = [
                    self._make_worker_state(self.worker_seed_base + worker_id)
                    for worker_id in range(self.synthetic_num_workers)
                ]
            self.worker_states = worker_states
        else:
            self.worker_states = None

    def __len__(self):
        return self._num_batches

    def _slice_batch_indices(self, batch_index):
        if batch_index < 0:
            batch_index += len(self)
        if batch_index < 0 or batch_index >= len(self):
            raise IndexError("batch index out of range")

        start = batch_index * self.batch_size
        batch_indices = self.indices[start:start + self.batch_size]
        if self.drop_last and len(batch_indices) < self.batch_size:
            raise IndexError("incomplete batch dropped")
        return batch_indices

    def _materialize_batch(self, batch_indices):
        samples = [self.dataset[idx] for idx in batch_indices]
        if self.pad_to_max_shape:
            image_np, label_np = pad_batch(samples)
        else:
            image_np = np.stack([sample[0] for sample in samples]).astype(np.float32)
            label_np = np.stack([sample[1] for sample in samples]).astype(np.int32)
        return (
            ms.Tensor(image_np, ms.float32),
            ms.Tensor(label_np, ms.int32),
        )

    def _make_worker_state(self, seed):
        py_rng = random.Random(int(seed))
        np_rng = np.random.RandomState(int(seed) % (2 ** 32))
        return {
            "py": py_rng.getstate(),
            "np": np_rng.get_state(),
        }

    def _materialize_batch_with_worker_state(self, batch_indices, worker_id):
        saved_py_state = random.getstate()
        saved_np_state = np.random.get_state()
        worker_state = self.worker_states[int(worker_id)]
        random.setstate(worker_state["py"])
        np.random.set_state(worker_state["np"])
        try:
            batch = self._materialize_batch(batch_indices)
            worker_state["py"] = random.getstate()
            worker_state["np"] = np.random.get_state()
            return batch
        finally:
            random.setstate(saved_py_state)
            np.random.set_state(saved_np_state)

    def __iter__(self):
        with temporary_random_seed(self.seed):
            for batch_index in range(len(self)):
                batch_indices = self._slice_batch_indices(batch_index)
                if self.synthetic_num_workers > 0:
                    worker_id = (self.batch_worker_offset + batch_index) % self.synthetic_num_workers
                    yield self._materialize_batch_with_worker_state(batch_indices, worker_id)
                else:
                    yield self._materialize_batch(batch_indices)

    def __getitem__(self, batch_index):
        batch_indices = self._slice_batch_indices(batch_index)
        seed = None if self.seed is None else self.seed + int(batch_index)
        with temporary_random_seed(seed):
            if self.synthetic_num_workers > 0:
                worker_id = (self.batch_worker_offset + int(batch_index)) % self.synthetic_num_workers
                return self._materialize_batch_with_worker_state(batch_indices, worker_id)
            return self._materialize_batch(batch_indices)


def build_batches(
    dataset,
    idxs=None,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    seed=None,
    pad_to_max_shape=False,
    synthetic_num_workers=0,
    batch_worker_offset=0,
    worker_seed_base=None,
    worker_states=None,
):
    return BatchLoader(
        dataset,
        idxs=idxs,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        seed=seed,
        pad_to_max_shape=pad_to_max_shape,
        synthetic_num_workers=synthetic_num_workers,
        batch_worker_offset=batch_worker_offset,
        worker_seed_base=worker_seed_base,
        worker_states=worker_states,
    )


def num_batches(num_items, batch_size, drop_last=False):
    if drop_last:
        return num_items // batch_size
    return int(math.ceil(num_items / batch_size))

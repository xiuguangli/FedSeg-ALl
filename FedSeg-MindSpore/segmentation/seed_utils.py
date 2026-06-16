import os
import random

import mindspore as ms
import numpy as np


_MERSENNE_STATE_N = 624
_MERSENNE_STATE_M = 397
_MATRIX_A = 0x9908B0DF
_UMASK = 0x80000000
_LMASK = 0x7FFFFFFF
_UINT32_MASK = 0xFFFFFFFF


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    ms.set_seed(seed)


class TorchMT19937:
    """Pure-Python mirror of PyTorch's ATen mt19937 engine."""

    def __init__(self, seed=5489):
        self.state = [0] * _MERSENNE_STATE_N
        self.seeded = True
        self.seed = int(seed) & ((1 << 64) - 1)
        self.left = 1
        self.next = 0
        self.state[0] = self.seed & _UINT32_MASK
        for j in range(1, _MERSENNE_STATE_N):
            prev = self.state[j - 1]
            value = 1812433253 * (prev ^ (prev >> 30)) + j
            self.state[j] = value & _UINT32_MASK

    def _mix_bits(self, u, v):
        return (u & _UMASK) | (v & _LMASK)

    def _twist(self, u, v):
        return ((self._mix_bits(u, v) >> 1) ^ (_MATRIX_A if (v & 1) else 0)) & _UINT32_MASK

    def _next_state(self):
        p = 0
        self.left = _MERSENNE_STATE_N
        self.next = 0

        for _ in range(_MERSENNE_STATE_N - _MERSENNE_STATE_M):
            self.state[p] = (self.state[p + _MERSENNE_STATE_M] ^ self._twist(self.state[p], self.state[p + 1])) & _UINT32_MASK
            p += 1

        for _ in range(_MERSENNE_STATE_M - 1):
            self.state[p] = (self.state[p + _MERSENNE_STATE_M - _MERSENNE_STATE_N] ^ self._twist(self.state[p], self.state[p + 1])) & _UINT32_MASK
            p += 1

        self.state[p] = (self.state[p + _MERSENNE_STATE_M - _MERSENNE_STATE_N] ^ self._twist(self.state[p], self.state[0])) & _UINT32_MASK

    def random(self):
        self.left -= 1
        if self.left == 0:
            self._next_state()
        y = self.state[self.next]
        self.next += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        return y & _UINT32_MASK

    def random64(self):
        random1 = self.random()
        random2 = self.random()
        return ((random1 << 32) | random2) & ((1 << 64) - 1)


def torch_randperm_indices(seed, size):
    size = int(size)
    if size <= 1:
        return list(range(max(0, size)))
    rng = TorchMT19937(seed)
    return torch_randperm_indices_from_rng(rng, size)


def torch_randperm_indices_from_rng(rng, size):
    size = int(size)
    values = list(range(size))
    for index in range(size - 1):
        offset = int(rng.random() % (size - index))
        swap_index = index + offset
        values[index], values[swap_index] = values[swap_index], values[index]
    return values


def torch_multi_epoch_loader_indices(seed, size, local_epoch):
    """Match torch RandomSampler + MultiEpochsDataLoader progression."""
    size = int(size)
    local_epoch = int(local_epoch)
    rng = TorchMT19937(seed)
    # DataLoader iterator creation draws one int64 base_seed before sampler usage.
    rng.random64()
    for _ in range(max(0, local_epoch)):
        torch_randperm_indices_from_rng(rng, size)
        # RandomSampler(replacement=False) performs an extra empty-slice randperm
        # when num_samples % n == 0, which advances the generator by one more pass.
        torch_randperm_indices_from_rng(rng, size)
    return torch_randperm_indices_from_rng(rng, size)


def torch_randperm_subset(seed, size, limit, skip_perm_calls=0, consume_base_seed=False):
    size = int(size)
    limit = int(limit)
    rng = TorchMT19937(seed)
    if consume_base_seed:
        rng.random64()
    for _ in range(max(0, int(skip_perm_calls))):
        torch_randperm_indices_from_rng(rng, size)
    values = torch_randperm_indices_from_rng(rng, size)
    if limit <= 0:
        return []
    return values[: min(size, limit)]

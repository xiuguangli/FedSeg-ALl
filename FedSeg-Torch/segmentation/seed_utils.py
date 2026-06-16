import functools
import os
import random

import numpy as np
import torch


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_torch_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _seed_worker(worker_id, base_seed):
    worker_seed = base_seed + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_worker_init_fn(seed):
    return functools.partial(_seed_worker, base_seed=seed)

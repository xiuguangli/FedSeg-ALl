import os
import random

import numpy as np
import tensorflow as tf


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def make_numpy_rng(seed):
    return np.random.default_rng(seed)

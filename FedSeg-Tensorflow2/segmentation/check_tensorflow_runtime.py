from __future__ import annotations

import argparse

from runtime_utils import configure_tensorflow_runtime

import tensorflow as tf

from tensorflow_runtime import require_tensorflow_device


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TensorFlow runtime for FedSeg.")
    parser.add_argument("--gpu", type=str, default="0", help='GPU id, or "" for CPU')
    args = parser.parse_args()

    configure_tensorflow_runtime(tf)
    device = require_tensorflow_device(tf, args.gpu)
    value = tf.reduce_sum(tf.ones([2, 2], dtype=tf.float32)).numpy().item()
    print("TensorFlow {} runtime OK on {}; test_sum={}".format(tf.__version__, device, value))
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))


if __name__ == "__main__":
    main()

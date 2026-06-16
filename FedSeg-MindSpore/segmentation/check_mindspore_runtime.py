import argparse

import mindspore as ms
import numpy as np

from mindspore_runtime import setup_mindspore_device


def main():
    parser = argparse.ArgumentParser(description="Check MindSpore runtime for FedSeg.")
    parser.add_argument("--gpu", type=str, default="0", help='GPU id, or "" for CPU')
    args = parser.parse_args()

    device = setup_mindspore_device(args.gpu)
    x = ms.Tensor(np.ones((2, 2), dtype=np.float32))
    y = x + x
    print("MindSpore {} runtime OK on {}; test_sum={}".format(ms.__version__, device, float(y.asnumpy().sum())))


if __name__ == "__main__":
    main()

import sys

sys.path.append("/home/pjl/project/FedSeg/paddle_project")
import datetime
import errno
import os

import paddle
from paddle_utils import *


def evaluate(model, data_loader, device, num_classes):
    model.eval()
    loss = 0
    confmat = ConfusionMatrix(num_classes)
    header = "Test:"
    with paddle.no_grad():
        for image, target in data_loader:
            image, target = image.to(device), target.to(device)
            model.aux_mode = "eval"
            output = model(image)[0]
            model.aux_mode = "train"
            confmat.update(target.flatten(), output.argmax(axis=1).flatten())
        confmat.compute()
    return confmat


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0
        self.iou_mean = 0
        self.acc = 0
        self.iu = 0

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = paddle.zeros(shape=(n, n), dtype="int64")
        with paddle.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to("int64") + b[k]
            self.mat += paddle.bincount(x=inds, minlength=n**2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        """compute and update self metrics"""
        h = self.mat.astype(dtype="float32")
        self.acc_global = paddle.diag(x=h).sum() / h.sum()
        self.acc_global = self.acc_global.item() * 100
        self.acc = paddle.diag(x=h) / h.sum(axis=1)
        self.iu = paddle.diag(x=h) / (h.sum(axis=1) + h.sum(axis=0) - paddle.diag(x=h))
        iu = self.iu[~self.iu.isnan()]
        self.iou_mean = iu.mean().item() * 100

    def reduce_from_all_processes(self):
        if not paddle.distributed.is_available():
            return
        if not paddle.distributed.is_initialized():
            return
        paddle.distributed.barrier()
        paddle.distributed.all_reduce(tensor=self.mat)

    def __str__(self):
        self.compute()
        return "global correct: {:.1f}\naverage row correct: {}\nIoU: {}\nmean IoU: {:.1f}".format(
            self.acc_global,
            ["{:.1f}".format(i) for i in (self.acc * 100).tolist()],
            ["{:.1f}".format(i) for i in (self.iu * 100).tolist()],
            self.iou_mean,
        )

# import torch
# import torch.utils.data
# from torch import nn
# import torch.distributed as dist
import errno
import os
import datetime


import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore import Tensor
import numpy as np

def evaluate(model:nn.Cell, data_loader, device, num_classes):
    from tqdm import tqdm
    # model.eval()
    model.set_train(False)
    loss = 0
    confmat = ConfusionMatrix(num_classes)
    header = 'Test:'
    # with torch.no_grad():
    for image, target in tqdm(data_loader,desc="testing",leave=False):
        # image, target = image.to(device), target.to(device)

        # bisenetv2:
        model.aux_mode = 'eval'
        output = model(image)[0] # return logits
        model.aux_mode = 'train'

        # lraspp_mobilenetv3:
        # output = model(image)
        # output = output['out']

        # [optional] return loss
        # batch_loss = criterion(output, target)
        # loss += batch_loss.item()

        confmat.update(target.flatten(), output.argmax(1).flatten())

    #confmat.reduce_from_all_processes()
    confmat.compute()

    return confmat


class ConfusionMatrix0(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0
        self.iou_mean = 0
        self.acc = 0
        self.iu = 0

    # def update(self, a, b):
    #     n = self.num_classes
    #     if self.mat is None:
    #         self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
    #     with torch.no_grad():
    #         k = (a >= 0) & (a < n)
    #         inds = n * a[k].to(torch.int64) + b[k]
    #         self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)
    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            k = (a >= 0) & (a < n)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        self.mat.zero_()

    def compute(self):
        ''' compute and update self metrics '''
        h = self.mat.float()
        self.acc_global = torch.diag(h).sum() / h.sum()
        self.acc_global = self.acc_global.item() * 100
        self.acc = torch.diag(h) / h.sum(1)
        self.iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        # remove nan for calculating mean value
        iu = self.iu[~self.iu.isnan()]
        self.iou_mean = iu.mean().item() * 100
        # return acc_global, acc, iu

    def reduce_from_all_processes(self):
        if not torch.distributed.is_available():
            return
        if not torch.distributed.is_initialized():
            return
        torch.distributed.barrier()
        torch.distributed.all_reduce(self.mat)

    def __str__(self):
        self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            self.acc_global,
            ['{:.1f}'.format(i) for i in (self.acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.iu * 100).tolist()],
            self.iou_mean)
            

class ConfusionMatrix(object):
    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0.0
        self.iou_mean = 0.0
        self.acc = np.array(0)
        self.iu = np.array(0)

    def update(self, a: Tensor, b: Tensor):
        # 将MindSpore Tensor转换为Numpy array
        a_np = a.asnumpy()
        b_np = b.asnumpy()
        
        n = self.num_classes
        if self.mat is None:
            self.mat = np.zeros((n, n), dtype=np.int64)
        
        k = (a_np >= 0) & (a_np < n)
        inds = n * a_np[k].astype(np.int64) + b_np[k]
        update_matrix = np.bincount(inds, minlength=n * n).reshape(n, n)
        self.mat += update_matrix

    def compute(self):
        """ 根据混淆矩阵计算并更新度量指标 (Numpy实现) """
        if self.mat is None:
            print("Warning: Confusion matrix is not updated. Call update() first.")
            return

        h = self.mat.astype(np.float32)
        
        # 全局准确率
        self.acc_global = np.diag(h).sum() / h.sum() * 100

        # 各类别准确率
        self.acc = np.diag(h) / (h.sum(axis=1) + 1e-10)

        # 各类别交并比 (IoU)
        denominator = h.sum(axis=1) + h.sum(axis=0) - np.diag(h)
        self.iu = np.diag(h) / (denominator + 1e-10)

        # 平均交并比 (mIoU)
        iu_not_nan = self.iu[~np.isnan(self.iu)]
        if iu_not_nan.size == 0:
            self.iou_mean = 0.0
        else:
            self.iou_mean = iu_not_nan.mean() * 100

    def __str__(self):
        self.compute()
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            self.acc_global,
            ['{:.1f}'.format(i) for i in (self.acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (self.iu * 100).tolist()],
            self.iou_mean)

# def mkdir(path):
#     try:
#         os.makedirs(path)
#     except OSError as e:
#         if e.errno != errno.EEXIST:
#             raise

def test():
    # num_classes = 150
    num_classes = 4
    # 随机生成10,1,480,640 的整数 tensor
    print("Generating random labels with shape (10, 1, 480, 640) and 150 classes...")
    # 验证 numpy 时间和torch时间相比
    # true_labels = ms.Tensor(np.random.randint(0, num_classes, size=(10, 1, 480, 640)), dtype=ms.int32)
    # pred_labels = ms.Tensor(np.random.randint(0, num_classes, size=(10, 1, 480, 640)), dtype=ms.int32)
    
    true_labels = ms.tensor([0, 1, 2, 3, 0, 1, 2, 2, 4, 1, 3, 0, 2, 4, 4, 1, 0, 3])
    pred_labels = ms.tensor([0, 1, 1, 3, 0, 2, 1, 2, 4, 2, 3, 0, 1, 4, 2, 1, 0, 3])
 
    
    conf_mat = ConfusionMatrix(num_classes)
    
    print("\nTrue labels: ", true_labels)
    print("Pred labels: ", pred_labels)
    
    import time
    # 3. 更新混淆矩阵
    start_time = time.time()
    conf_mat.update(true_labels, pred_labels)
    end_time = time.time()
    print(f"\nUpdate time: {end_time - start_time:.6f} seconds")
    
    print("\nInternal confusion matrix:\n", conf_mat.mat)
    
    # 5. 打印计算出的指标
    # __str__ 方法会自动调用 compute() 并格式化输出
    print("\nComputed metrics:\n" + str(conf_mat))

    print(f"Calculated mIoU: {conf_mat.iou_mean:.1f}")
    
    
    
if __name__ == '__main__':
    test()

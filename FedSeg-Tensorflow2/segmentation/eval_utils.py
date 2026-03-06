import numpy as np
# import torch
# import torch.utils.data
import tensorflow as tf
from tqdm import tqdm
import time
from line_profiler import profile

def evaluate(model, data_loader, num_classes):
    confmat = ConfusionMatrix(num_classes)
    model.aux_mode = 'eval'
    
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, 3, None, None], dtype=tf.float32), 
        tf.TensorSpec(shape=[None, None, None], dtype=tf.int64) 
    ])
    def model_inference(inputs,target):
        output = model(inputs, training=False)[0]
        confmat.update(tf.reshape(target, [-1]), tf.reshape(tf.argmax(output, axis=1), [-1]))
        # return confmat

    for image, target in tqdm(data_loader,desc="Evaluating", leave=False):
    # for image, target in data_loader:
        # output = model(image, training=False)
        model_inference(image, target)
        # break
    
    model.aux_mode = 'train'
    confmat.compute()

    return confmat


class ConfusionMatrix1(object):
    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        self.mat = None
        self.acc_global = 0.0
        self.iou_mean = 0.0
        self.acc = np.array(0)
        self.iu = np.array(0)

    def update(self, a, b):
        # Tensor转换为Numpy array
        a_np = a
        b_np = b
        
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


class ConfusionMatrix(object):
    def __init__(self, num_classes: int):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        # 使用 tf.Variable 存储混淆矩阵，以便在 tf.function 中更新
        self.mat = tf.Variable(
            initial_value=tf.zeros((num_classes, num_classes), dtype=tf.int64),
            trainable=False,
            dtype=tf.int64
        )
        self.acc_global = tf.Variable(0.0, trainable=False)
        self.iou_mean = tf.Variable(0.0, trainable=False)
        self.acc = tf.Variable(tf.zeros(num_classes, dtype=tf.float32), trainable=False)
        self.iu = tf.Variable(tf.zeros(num_classes, dtype=tf.float32), trainable=False)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None], dtype=tf.int64), # 真实标签 a
        tf.TensorSpec(shape=[None], dtype=tf.int64)  # 预测标签 b
    ])
    def update(self, a_flat: tf.Tensor, b_flat: tf.Tensor):
        """ 使用 tf.math.confusion_matrix 累加到 Variable 中 """
        n = self.num_classes
        # 1. 过滤有效范围内的标签
        valid_mask = (a_flat >= 0) & (a_flat < n)
        a_valid = tf.boolean_mask(a_flat, valid_mask)
        b_valid = tf.boolean_mask(b_flat, valid_mask)
        
        # 2. 直接计算当前批次的混淆矩阵，并累加到 Variable 中 (assign_add)
        current_confmat = tf.math.confusion_matrix(
            a_valid, 
            b_valid, 
            num_classes=n, 
            dtype=tf.int64
        )
        self.mat.assign_add(current_confmat)

    @tf.function
    def compute(self):
        """ 根据混淆矩阵计算并更新度量指标 (TensorFlow实现) """
        h = tf.cast(self.mat.read_value(), tf.float32) # 从 Variable 读取值

        # 全局准确率
        global_correct = tf.reduce_sum(tf.linalg.diag_part(h))
        total_sum = tf.reduce_sum(h)
        self.acc_global.assign(global_correct / (total_sum + 1e-10) * 100.0)

        # 各类别准确率
        row_sum = tf.reduce_sum(h, axis=1)
        self.acc.assign(tf.linalg.diag_part(h) / (row_sum + 1e-10))

        # 各类别交并比 (IoU)
        col_sum = tf.reduce_sum(h, axis=0)
        intersection = tf.linalg.diag_part(h)
        union = row_sum + col_sum - intersection
        iu_tensor = intersection / (union + 1e-10)
        self.iu.assign(iu_tensor)

        # 平均交并比 (mIoU)
        # 过滤 NaN 值，只计算有效类别的均值
        is_valid = tf.math.is_finite(iu_tensor) & (union > 0)
        iu_valid = tf.boolean_mask(iu_tensor, is_valid)
        
        # 检查是否所有 IoU 都为 0 或 NaN
        if tf.reduce_sum(tf.cast(is_valid, tf.int32)) == 0:
            self.iou_mean.assign(0.0)
        else:
            self.iou_mean.assign(tf.reduce_mean(iu_valid) * 100.0)

    def reset(self):
        self.mat.assign(tf.zeros_like(self.mat))
        self.acc_global.assign(0.0)
        self.iou_mean.assign(0.0)
        self.acc.assign(tf.zeros_like(self.acc))
        self.iu.assign(tf.zeros_like(self.iu))
    
    def _get_metric_values(self):
        # 显式调用 tf.function 来更新指标
        self.compute() 
        
        # 使用 tf.identity 确保读取的是最新的值，并用 .numpy() 转换为 Python/NumPy
        acc_global = self.acc_global.read_value().numpy()
        acc = self.acc.read_value().numpy()
        iu = self.iu.read_value().numpy()
        iou_mean = self.iou_mean.read_value().numpy()
        
        return acc_global, acc, iu, iou_mean
        
    def __str__(self):        
        acc_global, acc, iu, iou_mean = self._get_metric_values()
        
        return (
            'global correct: {:.1f}\n'
            'average row correct: {}\n'
            'IoU: {}\n'
            'mean IoU: {:.1f}').format(
            acc_global,
            ['{:.1f}'.format(i) for i in (acc * 100).tolist()],
            ['{:.1f}'.format(i) for i in (iu * 100).tolist()],
            iou_mean)
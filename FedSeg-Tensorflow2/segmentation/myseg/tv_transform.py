import numpy as np
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.transforms.functional import InterpolationMode
import tensorflow as tf

# 将图片转换为0-1之间的浮点数，与transform写到一起
class TensorScale_255to1:
    def __call__(self, img):
        # 假设 img 是 tf.Tensor 或 np.ndarray
        img = tf.convert_to_tensor(img, dtype=tf.float32)
        return img / 255.0

class TensorLabeltoLong:
    def __call__(self, label):
        # label shape: (batch, h, w) or (h, w, 1)
        label = tf.convert_to_tensor(label)
        if len(label.shape) == 3:
            label = tf.reshape(label, [label.shape[-2], label.shape[-1]])
        label = tf.cast(label, tf.int64)
        return label

def label_remap(img, old_values, new_values):
    # Replace old values by the new ones
    img = tf.convert_to_tensor(img)
    tmp = tf.zeros_like(img)
    for old, new in zip(old_values, new_values):
        tmp = tf.where(img == old, new, tmp)
    return tmp

def RandomScaleCrop(image, label):
    """
    对图像进行随机缩放和随机裁剪
    scale the images in the range (0.5,1.5) for Cityscapes
    then extract a crop with size 512×1024 for Cityscapes
    """
    # 随机图像缩放scale
    scale = np.random.uniform(0.5, 1.5) # 生成0.5-1.5之间的随机数
    new_h, new_w = int(scale * image.shape[-2]), int(scale * image.shape[-1])

    # tf.image.resize 需要 (h, w, c)
    image = tf.image.resize(image, [new_h, new_w], method='bilinear')
    label = tf.image.resize(label, [new_h, new_w], method='nearest')

    # 随机同时裁剪图片和标签图像crop
    crop_h, crop_w = 512, 1024
    img_shape = tf.shape(image)
    max_y = img_shape[-3] - crop_h
    max_x = img_shape[-2] - crop_w
    offset_y = tf.random.uniform([], 0, max_y + 1, dtype=tf.int32)
    offset_x = tf.random.uniform([], 0, max_x + 1, dtype=tf.int32)
    image = tf.image.crop_to_bounding_box(image, offset_y, offset_x, crop_h, crop_w)
    label = tf.image.crop_to_bounding_box(label, offset_y, offset_x, crop_h, crop_w)

    return image, label

def get_transform():
    def image_transform(img):
        img = tf.image.resize(img, [512, 1024], method='bilinear')
        img = TensorScale_255to1()(img)
        return img

    def label_transform(label):
        label = tf.image.resize(label, [512, 1024], method='nearest')
        label = TensorLabeltoLong()(label)
        return label

    return image_transform, label_transform
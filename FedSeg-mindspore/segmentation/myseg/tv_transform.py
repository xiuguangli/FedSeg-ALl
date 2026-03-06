import numpy as np
import random
# import torch
# import torchvision
# import torchvision.transforms as transforms
# from torchvision.transforms.functional import InterpolationMode

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
from mindspore.dataset.vision import Inter

# 将图片转换为0-1之间的浮点数，与transform写到一起
class TensorScale_255to1:
    def __call__(self, img):
        # return img.float() / 255
        return img.astype(mindspore.float32) / 255.0

class TensorLabeltoLong:
    def __call__(self, label):
        label = label.reshape(label.shape[1], label.shape[2])
        # label = label.long()
        label = label.astype(mindspore.int64)
        return label


def label_remap(img, old_values, new_values):
    # Replace old values by the new ones
    # tmp = torch.zeros_like(img)
    tmp = ops.zeros_like(img)
    for old, new in zip(old_values, new_values):
        tmp[img == old] = new

    return tmp


def RandomScaleCrop(image, label):
    """
    对图像进行随机缩放和随机裁剪
    scale the images in the range (0.5,1.5) for Cityscapes
    then extract a crop with size 512×1024 for Cityscapes
    """
    # 随机图像缩放scale
    scale = np.random.uniform(0.5, 1.5) # 生成0.5-1.5之间的随机数
    #print('scale:', scale)
    new_h, new_w = int(scale * image.shape[-2]), int(scale * image.shape[-1])
    #print(new_h, new_w)
    # image = transforms.functional.resize(image, (new_h, new_w), InterpolationMode.BILINEAR)
    # label = transforms.functional.resize(label, (new_h, new_w), InterpolationMode.NEAREST)
    image = vision.Resize(size=(new_h, new_w), interpolation=Inter.BILINEAR)(image)
    label = vision.Resize(size=(new_h, new_w), interpolation=Inter.NEAREST)(label)
    #print(image.shape, label.shape)

    # 随机同时裁剪图片和标签图像crop
    # rect = transforms.RandomCrop.get_params(image, (512, 1024))
    rect = get_random_crop_params(image, (512, 1024))
    #print(rect)
    # image = transforms.functional.crop(image, *rect)
    # label = transforms.functional.crop(label, *rect)
    
    top, left, height, width = rect
    coordinates = (top, left)
    crop_size = (height, width)
    image = vision.Crop(coordinates, crop_size)(image)
    label = vision.Crop(coordinates, crop_size)(label)
    #print(image.shape, label.shape)

    return image, label

def get_transform():
    # image_transform = transforms.Compose([
    #     transforms.Resize((512, 1024)),
    #     TensorScale_255to1()
    # ])
    
    image_transform = ds.transforms.Compose([
        vision.Resize((512, 1024)),
        TensorScale_255to1()
    ])

    # label_transform = transforms.Compose([
    #     transforms.Resize((512, 1024), InterpolationMode.NEAREST),  # BILINEAR会插入两个标签的中间值，不符合实际
    #     TensorLabeltoLong()
    # ])
    label_transform = ds.transforms.Compose([
        vision.Resize((512, 1024), Inter.NEAREST),  # BILINEAR会插入两个标签的中间值，不符合实际
        TensorLabeltoLong()
    ])
    return image_transform, label_transform

# add
def get_random_crop_params(image, output_size):
    """
    与 torchvision.transforms.RandomCrop.get_params 逻辑等效的函数。
    
    Args:
        image (np.ndarray): 输入图像，用于获取其尺寸。
                            格式通常是 (H, W, C)。
        output_size (tuple or list): 期望的裁剪尺寸 (height, width)。
        
    Returns:
        tuple: (i, j, h, w) 裁剪参数。
               i: 左上角的行坐标 (top)
               j: 左上角的列坐标 (left)
               h: 裁剪高度
               w: 裁剪宽度
    """
    h, w = image.shape[:2]
    th, tw = output_size

    if w < tw or h < th:
        raise ValueError(f"Required crop size {(th, tw)} is larger than "
                         f"input image size {(h, w)}")

    if w == tw and h == th:
        return 0, 0, h, w

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)
    return i, j, th, tw
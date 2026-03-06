import sys

sys.path.append("/home/pjl/project/FedSeg/paddle_project")
import numpy as np
import paddle
from paddle_utils import *
from paddle.vision import transforms


class TensorScale_255to1:
    def __call__(self, img):
        return img.astype(dtype="float32") / 255


class TensorLabeltoLong:
    def __call__(self, label):
        label = label.reshape(tuple(label.shape)[1], tuple(label.shape)[2])
        label = label.astype(dtype="int64")
        return label


def label_remap(img, old_values, new_values):
    tmp = paddle.zeros_like(x=img)
    for old, new in zip(old_values, new_values):
        tmp[img == old] = new
    return tmp


def RandomScaleCrop(image, label):
    """
    对图像进行随机缩放和随机裁剪
    scale the images in the range (0.5,1.5) for Cityscapes
    then extract a crop with size 512×1024 for Cityscapes
    """
    scale = np.random.uniform(0.5, 1.5)
    new_h, new_w = int(scale * tuple(image.shape)[-2]), int(
        scale * tuple(image.shape)[-1]
    )
    image = paddle.vision.transforms.resize(
        img=image,
        size=(new_h, new_w),
    )
    image = transforms.resize(
        image,
        size=(new_h, new_w),
        interpolation='bilinear',
    )
    label = transforms.resize(
        label,
        size=(new_h, new_w),
        interpolation='nearest',
    )

    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(512, 1024))
    image = transforms.crop(image, i, j, h, w)
    label = transforms.crop(label, i, j, h, w)

    return image, label


def get_transform():
    image_transform = paddle.vision.transforms.Compose(
        transforms=[
            paddle.vision.transforms.Resize(size=(512, 1024)),
            TensorScale_255to1(),
        ]
    )
    label_transform = transforms.Compose(
        transforms=[
            transforms.Resize(
                size=(512, 1024),
                interpolation='nearest',   # 改成字符串形式
            ),
            TensorLabeltoLong(),
        ]
    )
    return image_transform, label_transform

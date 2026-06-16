import random

import cv2
import numpy as np


class TensorScale_255to1:
    def __call__(self, image):
        return image.astype(np.float32) / 255.0


class TensorLabeltoLong:
    def __call__(self, label):
        return label.astype(np.int64)


def label_remap(image, old_values, new_values):
    output = np.zeros_like(image)
    for old_value, new_value in zip(old_values, new_values):
        output[image == old_value] = new_value
    return output


def get_transform():
    def image_transform(image):
        image = cv2.resize(image, (1024, 512))
        return TensorScale_255to1()(image)

    def label_transform(label):
        label = cv2.resize(label, (1024, 512), interpolation=cv2.INTER_NEAREST)
        return TensorLabeltoLong()(label)

    return image_transform, label_transform


def get_random_crop_params(image, output_size):
    height, width = image.shape[:2]
    target_h, target_w = output_size
    if width < target_w or height < target_h:
        raise ValueError(
            "Required crop size {} is larger than input image size {}".format(
                (target_h, target_w),
                (height, width),
            )
        )
    if width == target_w and height == target_h:
        return 0, 0, height, width
    top = random.randint(0, height - target_h)
    left = random.randint(0, width - target_w)
    return top, left, target_h, target_w


def RandomScaleCrop(image, label):
    scale = np.random.uniform(0.5, 1.5)
    new_h, new_w = int(scale * image.shape[-2]), int(scale * image.shape[-1])
    image = cv2.resize(image, (new_w, new_h))
    label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    top, left, height, width = get_random_crop_params(image, (512, 1024))
    image = image[top:top + height, left:left + width]
    label = label[top:top + height, left:left + width]
    return image, label


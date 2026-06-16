import math
import random

import cv2
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def __call__(self, image_label):
        result = image_label
        for transform in self.transforms:
            result = transform(result)
        return result


def cv2_get_image(image_path, label_path):
    image = cv2.imread(image_path)[:, :, ::-1].copy()
    label = cv2.imread(label_path, 0)
    return image, label


class TransformationTrain:
    def __init__(self, scales, cropsize):
        self.trans_func = Compose(
            [
                RandomResizedCrop(scales, cropsize),
                RandomHorizontalFlip(),
            ]
        )

    def __call__(self, image_label):
        return self.trans_func(image_label)


class TransformationVal:
    def __init__(self, size=None):
        self.size = size

    def __call__(self, image_label):
        image, label = image_label["im"], image_label["lb"]
        if self.size is not None:
            resize_h, resize_w = self.size
            image = cv2.resize(image, (resize_w, resize_h))
            label = cv2.resize(label, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        return {"im": image, "lb": label}


class RandomResizedCrop:
    def __init__(self, scales=(0.5, 1.0), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, image_label):
        if self.size is None:
            return image_label

        image, label = image_label["im"], image_label["lb"]
        assert image.shape[:2] == label.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        image_h, image_w = [math.ceil(length * scale) for length in image.shape[:2]]
        image = cv2.resize(image, (image_w, image_h))
        label = cv2.resize(label, (image_w, image_h), interpolation=cv2.INTER_NEAREST)

        if (image_h, image_w) == (crop_h, crop_w):
            return {"im": image, "lb": label}

        pad_h = 0
        pad_w = 0
        if image_h < crop_h:
            pad_h = (crop_h - image_h) // 2 + 1
        if image_w < crop_w:
            pad_w = (crop_w - image_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            label = np.pad(label, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=255)

        image_h, image_w, _ = image.shape
        shift_h, shift_w = np.random.random(2)
        shift_h = int(shift_h * (image_h - crop_h))
        shift_w = int(shift_w * (image_w - crop_w))
        return {
            "im": image[shift_h:shift_h + crop_h, shift_w:shift_w + crop_w, :].copy(),
            "lb": label[shift_h:shift_h + crop_h, shift_w:shift_w + crop_w].copy(),
        }


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image_label):
        if np.random.random() < self.p:
            return image_label
        image, label = image_label["im"], image_label["lb"]
        assert image.shape[:2] == label.shape[:2]
        return {
            "im": image[:, ::-1, :],
            "lb": label[:, ::-1],
        }


class ToTensor:
    def __init__(self, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self.std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)

    def __call__(self, image_label):
        image, label = image_label["im"], image_label["lb"]
        image = image.transpose(2, 0, 1).astype(np.float32)
        image = (image / 255.0 - self.mean) / self.std
        if label is not None:
            label = label.astype(np.int64)
        return {"im": image, "lb": label}

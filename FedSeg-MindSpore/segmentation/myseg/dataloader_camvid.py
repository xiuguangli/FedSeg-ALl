import os

import cv2
import numpy as np
from PIL import Image

from logging_utils import logger
import myseg.cv2_transform as cv2_transforms


def read_images_dir(root_dir, folder, is_label=False):
    city_names = sorted(os.listdir(os.path.join(root_dir, folder)))
    image_dirs = []
    for city_name in city_names:
        image_names = os.listdir(os.path.join(root_dir, folder, city_name))
        for image_name in image_names:
            image_dirs.append(os.path.join(root_dir, folder, city_name, image_name))
    return sorted(image_dirs)


class CamVid_Dataset:
    train_folder = "images/train"
    train_lb_folder = "labels/train"
    val_folder = "images/val"
    val_lb_folder = "labels/val"

    def __init__(self, args, root_dir, split="train"):
        self.args = args
        self.root_dir = root_dir
        self.split = split

        if self.split == "train":
            self.image_dirs = read_images_dir(root_dir, self.train_folder)
            self.label_dirs = read_images_dir(root_dir, self.train_lb_folder, is_label=True)
        elif self.split == "val":
            self.image_dirs = read_images_dir(root_dir, self.val_folder)
            self.label_dirs = read_images_dir(root_dir, self.val_lb_folder, is_label=True)
        else:
            raise ValueError("unsupported split: {}".format(split))

        assert len(self.image_dirs) == len(self.label_dirs), "图像和标注数量不匹配"
        logger.info("Found {} {} examples", len(self.image_dirs), self.split)

    def _train_resize_size(self, default_size):
        if not getattr(self.args, "debug_disable_train_aug", False):
            return None
        return (default_size, default_size)

    def __getitem__(self, idx):
        if self.args.dataset == "voc":
            image = np.array(Image.open(self.image_dirs[idx]))
            label = np.array(Image.open(self.label_dirs[idx]))
        else:
            image = cv2.imread(self.image_dirs[idx], cv2.IMREAD_COLOR)[:, :, ::-1]
            label = cv2.imread(self.label_dirs[idx], cv2.IMREAD_GRAYSCALE)

        if self.args.dataset == "camvid":
            if self.split == "val":
                label = np.uint8(label) - 1
        elif self.args.dataset == "ade20k":
            label = np.uint8(label) - 1
        elif self.args.dataset == "voc":
            label[label == 255] = 0
            label = np.uint8(label) - 1

        scale = 480 if self.args.dataset in {"voc", "ade20k"} else 512
        train_resize = self._train_resize_size(scale)

        image_label = {"im": image, "lb": label}
        if self.split == "train":
            if train_resize is None:
                image_label = cv2_transforms.TransformationTrain(
                    scales=(0.5, 1.5),
                    cropsize=(scale, scale),
                )(image_label)
            else:
                image_label = cv2_transforms.TransformationVal(size=train_resize)(image_label)
        elif self.split == "val":
            image_label = cv2_transforms.TransformationVal()(image_label)

        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115),
        )(image_label)
        return image_label["im"], image_label["lb"]

    def __len__(self):
        return len(self.image_dirs)


import os
from collections import OrderedDict

import cv2
import numpy as np

from logging_utils import logger
import myseg.cv2_transform as cv2_transforms


def read_images_dir(root_dir, folder, is_label=False):
    city_names = sorted(os.listdir(os.path.join(root_dir, folder)))
    image_dirs = []
    for city_name in city_names:
        image_names = os.listdir(os.path.join(root_dir, folder, city_name))
        for image_name in image_names:
            if not is_label:
                image_dirs.append(os.path.join(root_dir, folder, city_name, image_name))
            elif image_name.endswith("_labelIds.png"):
                image_dirs.append(os.path.join(root_dir, folder, city_name, image_name))
    return sorted(image_dirs)


class Cityscapes_Dataset:
    train_folder = "leftImg8bit/train"
    train_lb_folder = "gtFine/train"
    val_folder = "leftImg8bit/val"
    val_lb_folder = "gtFine/val"
    test_folder = "leftImg8bit/test"
    test_lb_folder = "gtFine/test"

    full_classes = (
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, -1,
    )
    new_classes = (
        255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6,
        7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18, 255,
    )

    color_encoding = OrderedDict(
        [
            ("unlabeled", (0, 0, 0)),
            ("road", (128, 64, 128)),
            ("sidewalk", (244, 35, 232)),
            ("building", (70, 70, 70)),
            ("wall", (102, 102, 156)),
            ("fence", (190, 153, 153)),
            ("pole", (153, 153, 153)),
            ("traffic_light", (250, 170, 30)),
            ("traffic_sign", (220, 220, 0)),
            ("vegetation", (107, 142, 35)),
            ("terrain", (152, 251, 152)),
            ("sky", (70, 130, 180)),
            ("person", (220, 20, 60)),
            ("rider", (255, 0, 0)),
            ("car", (0, 0, 142)),
            ("truck", (0, 0, 70)),
            ("bus", (0, 60, 100)),
            ("train", (0, 80, 100)),
            ("motorcycle", (0, 0, 230)),
            ("bicycle", (119, 11, 32)),
        ]
    )

    def __init__(self, root_dir, split="train", USE_ERASE_DATA=False):
        self.root_dir = root_dir
        self.split = split
        self.USE_ERASE_DATA = USE_ERASE_DATA

        self.lb_map = np.arange(256).astype(np.uint8)
        for index, class_id in enumerate(self.full_classes):
            self.lb_map[class_id] = self.new_classes[index]

        if self.split == "train":
            self.image_dirs = read_images_dir(root_dir, self.train_folder)
            self.label_dirs = read_images_dir(root_dir, self.train_lb_folder, is_label=True)
        elif self.split == "val":
            self.image_dirs = read_images_dir(root_dir, self.val_folder)
            self.label_dirs = read_images_dir(root_dir, self.val_lb_folder, is_label=True)
        elif self.split == "test":
            self.image_dirs = read_images_dir(root_dir, self.test_folder)
            self.label_dirs = read_images_dir(root_dir, self.test_lb_folder, is_label=True)
        else:
            raise ValueError("unsupported split: {}".format(split))

        assert len(self.image_dirs) == len(self.label_dirs), "图像和标注数量不匹配"
        logger.info("Found {} {} examples", len(self.image_dirs), self.split)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_dirs[idx], cv2.IMREAD_COLOR)[:, :, ::-1]
        label = cv2.imread(self.label_dirs[idx], cv2.IMREAD_GRAYSCALE)

        if not self.USE_ERASE_DATA:
            label = self.lb_map[label]
        elif self.split == "val":
            label = self.lb_map[label]

        image_label = {"im": image, "lb": label}
        if self.split == "train":
            image_label = cv2_transforms.TransformationTrain(
                scales=(0.5, 1.5),
                cropsize=(512, 1024),
            )(image_label)
        elif self.split == "val":
            image_label = cv2_transforms.TransformationVal()(image_label)

        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),
            std=(0.2112, 0.2148, 0.2115),
        )(image_label)
        return image_label["im"], image_label["lb"]

    def __len__(self):
        return len(self.image_dirs)


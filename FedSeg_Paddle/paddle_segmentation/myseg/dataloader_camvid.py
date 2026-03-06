import os
from collections import OrderedDict

import cv2
import myseg.cv2_transform as cv2_transforms
import myseg.tv_transform as my_transforms
import numpy as np
import paddle
import paddle.io as io
from PIL import Image


def read_images_dir(root_dir, folder, is_label=False):
    """
    读取所有图像和标注（的文件路径）

    city_idx ：城市索引 （用于选取某个城市）
    """
    city_names = sorted(os.listdir(os.path.join(root_dir, folder)))
    img_dirs = []
    for city_name in city_names:
        img_names = os.listdir(os.path.join(root_dir, folder, city_name))
        for img_name in img_names:
            if is_label == False:
                img_dirs.append(os.path.join(root_dir, folder, city_name, img_name))
            if is_label == True:
                img_dirs.append(os.path.join(root_dir, folder, city_name, img_name))
    img_dirs = sorted(img_dirs)
    list_ = []
    list_2 = []
    last_c = None
    for ii, ll in enumerate(img_dirs):
        cn = ll.split("/")[-2]
        if cn not in list_:
            list_.append(cn)
        if cn != last_c:
            list_2.append(ii)
            last_c = cn
    print(list_)
    print(list_2)
    return img_dirs


class CamVid_Dataset(paddle.io.Dataset):
    """Cityscapes dataset"""

    train_folder = "images/train"
    train_lb_folder = "labels/train"
    val_folder = "images/val"
    val_lb_folder = "labels/val"

    def __init__(self, args, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split
        self.args = args
        if self.split == "train":
            self.image_dirs, self.label_dirs = read_images_dir(
                root_dir, self.train_folder
            ), read_images_dir(root_dir, self.train_lb_folder, is_label=True)
        elif self.split == "val":
            self.image_dirs, self.label_dirs = read_images_dir(
                root_dir, self.val_folder
            ), read_images_dir(root_dir, self.val_lb_folder, is_label=True)
        assert len(self.image_dirs) == len(self.label_dirs), "图像和标注数量不匹配"
        print("find " + str(len(self.image_dirs)) + " examples")

    def __getitem__(self, idx):
        if self.args.dataset == "voc":
            image = Image.open(self.image_dirs[idx])
            image = np.array(image)
            label = Image.open(self.label_dirs[idx])
            label = np.array(label)
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
        scale_ = 512
        if self.args.dataset == "voc" or self.args.dataset == "ade20k":
            scale_ = 480
        image_label = dict(im=image, lb=label)
        if self.split == "train":
            image_label = cv2_transforms.TransformationTrain(
                scales=(0.5, 1.5), cropsize=(scale_, scale_)
            )(image_label)
        if self.split == "val":
            image_label = cv2_transforms.TransformationVal()(image_label)
        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.369, 0.3223), std=(0.2112, 0.2148, 0.2115)
        )(image_label)
        image, label = image_label["im"], image_label["lb"]
        return image, label

    def __len__(self):
        return len(self.image_dirs)


def load_dataiter(root_dir, batch_size, use_DDP=False):
    """加载dataloader"""
    num_workers = 16
    image_transform, label_transform = my_transforms.get_transform()
    train_set = CamVid_Dataset(root_dir, "train")
    test_set = CamVid_Dataset(root_dir, "val")
    if use_DDP:
        train_sampler = paddle.io.DistributedBatchSampler(
            dataset=train_set, shuffle=True, batch_size=1
        )
        train_iter = io.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=num_workers,
            sampler=train_sampler,
        )
        test_iter = paddle.io.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
        )
    else:
        train_iter = paddle.io.DataLoader(
            dataset=train_set,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )
        test_iter = paddle.io.DataLoader(
            dataset=test_set,
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
        )
    return train_iter, test_iter


if __name__ == "__main__":
    train_iter, test_iter = load_dataiter(root_dir="../data/cityscapes", batch_size=16)
    print("train_iter.len", len(train_iter))
    print("test_iter.len", len(test_iter))
    for i, (images, labels) in enumerate(train_iter):
        if i < 3:
            print(tuple(images.shape), tuple(labels.shape))
            print(labels.dtype)
            print(labels[0, 245:255, 250:260])

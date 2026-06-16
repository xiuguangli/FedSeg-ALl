import os
import time

import numpy as np

from logging_utils import logger
from myseg.dataloader import Cityscapes_Dataset
from myseg.dataloader_camvid import CamVid_Dataset


def get_dataset_cityscapes(args):
    if args.dataset != "cityscapes":
        raise ValueError("Unrecognized dataset")

    if args.data == "train":
        train_dataset = Cityscapes_Dataset(args.root_dir, "train", args.USE_ERASE_DATA)
    elif args.data == "val":
        train_dataset = Cityscapes_Dataset(args.root_dir, "val", args.USE_ERASE_DATA)
    else:
        raise ValueError("Unrecognized split: {}".format(args.data))

    test_dataset = Cityscapes_Dataset(args.root_dir, "val", args.USE_ERASE_DATA)
    if args.iid:
        user_groups = cityscapes_iid(train_dataset, args.num_users, seed=args.seed)
    else:
        user_groups = cityscapes_noniid_extend(
            args.root_dir,
            Cityscapes_Dataset.train_folder,
            args.num_users,
            seed=args.seed,
        )
    return train_dataset, test_dataset, user_groups


def get_dataset_camvid(args):
    if args.dataset != "camvid":
        raise ValueError("Unrecognized dataset")

    if args.data == "train":
        train_dataset = CamVid_Dataset(args, args.root_dir, "train")
    elif args.data == "val":
        train_dataset = CamVid_Dataset(args, args.root_dir, "val")
    else:
        raise ValueError("Unrecognized split: {}".format(args.data))

    test_dataset = CamVid_Dataset(args, args.root_dir, "val")
    user_groups = cityscapes_noniid_extend(
        args.root_dir,
        CamVid_Dataset.train_folder,
        args.num_users,
        seed=args.seed,
    )
    return train_dataset, test_dataset, user_groups


def get_dataset_ade20k(args):
    if args.dataset not in {"ade20k", "voc"}:
        raise ValueError("Unrecognized dataset")

    if args.data == "train":
        train_dataset = CamVid_Dataset(args, args.root_dir, "train")
    elif args.data == "val":
        train_dataset = CamVid_Dataset(args, args.root_dir, "val")
    else:
        raise ValueError("Unrecognized split: {}".format(args.data))

    test_dataset = CamVid_Dataset(args, args.root_dir, "val")
    user_groups = cityscapes_noniid_extend(
        args.root_dir,
        CamVid_Dataset.train_folder,
        args.num_users,
        seed=args.seed,
    )
    return train_dataset, test_dataset, user_groups


def build_dataset_for_split(args, split):
    if args.dataset == "cityscapes":
        return Cityscapes_Dataset(args.root_dir, split, args.USE_ERASE_DATA)
    if args.dataset in {"camvid", "ade20k", "voc"}:
        return CamVid_Dataset(args, args.root_dir, split)
    raise ValueError("Unrecognized dataset")


def get_test_dataset(args):
    return build_dataset_for_split(args, "val")


def cityscapes_iid(dataset, num_users, seed=None):
    rng = np.random.default_rng(seed)
    num_items = int(len(dataset) / num_users)
    dict_users = {}
    all_indices = [idx for idx in range(len(dataset))]
    for user_idx in range(num_users):
        dict_users[user_idx] = set(rng.choice(all_indices, num_items, replace=False))
        all_indices = sorted(set(all_indices) - dict_users[user_idx])
    return dict_users


def cityscapes_noniid(num_users, seed=None):
    rng = np.random.default_rng(seed)
    timer = time.time()

    city_lens = [174, 96, 316, 154, 85, 221, 109, 248, 196, 119, 99, 94, 365, 196, 144, 95, 142, 122]
    num_users_per_city = int(num_users / 18)
    dict_users = {}
    for city_idx in range(18):
        num_items = int(city_lens[city_idx] / num_users_per_city)
        prefix = sum(city_lens[:city_idx])
        all_indices = [offset + prefix for offset in range(city_lens[city_idx])]

        for user_offset in range(num_users_per_city):
            user_id = user_offset + city_idx * num_users_per_city
            dict_users[user_id] = set(rng.choice(all_indices, num_items, replace=False))
            all_indices = sorted(set(all_indices) - dict_users[user_id])

        dict_users[(city_idx + 1) * num_users_per_city - 1] |= set(all_indices)

    logger.info("Time consumed to get non-iid user indices: {:.2f}s", time.time() - timer)
    return dict_users


def cityscapes_noniid_extend(root_dir, train_folder, num_users, seed=None):
    rng = np.random.default_rng(seed)
    timer = time.time()
    logger.info("Getting non-iid user indices")

    city_lens = get_city_num(root_dir, train_folder)
    num_classes = len(city_lens)
    num_users_per_city = int(num_users / num_classes)
    logger.info("num_users_per_city: {} / {} = {}", num_users, num_classes, num_users_per_city)
    assert num_users % num_classes == 0, "num_users % num_classes != 0"

    dict_users = {}
    for city_idx in range(num_classes):
        num_items = int(city_lens[city_idx] / num_users_per_city)
        prefix = sum(city_lens[:city_idx])
        all_indices = [offset + prefix for offset in range(city_lens[city_idx])]

        for user_offset in range(num_users_per_city):
            user_id = user_offset + city_idx * num_users_per_city
            dict_users[user_id] = set(rng.choice(all_indices, num_items, replace=False))
            all_indices = sorted(set(all_indices) - dict_users[user_id])

        dict_users[(city_idx + 1) * num_users_per_city - 1] |= set(all_indices)

    logger.info("Time consumed to get non-iid user indices: {:.2f}s", time.time() - timer)
    return dict_users


def get_city_num(root_dir, train_folder):
    city_names = sorted(os.listdir(os.path.join(root_dir, train_folder)))
    logger.debug("city_names: {}", city_names)
    num_classes = len(city_names)
    logger.debug("num_classes: {}", num_classes)

    city_lens = []
    for city_name in city_names:
        city_lens.append(len(os.listdir(os.path.join(root_dir, train_folder, city_name))))

    for city_name, city_len in zip(city_names, city_lens):
        logger.debug("{} {}", city_name, city_len)

    logger.debug("city_lens: {}", city_lens)
    return city_lens

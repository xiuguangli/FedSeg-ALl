import os
import time

import numpy as np
import paddle
from myseg.dataloader import *
from myseg.dataloader_camvid import CamVid_Dataset
from myseg.tv_transform import get_transform


def get_dataset_cityscapes(args):
    """
    cityscapes dataset:
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == "cityscapes":
        if args.data == "train":
            train_dataset = Cityscapes_Dataset(
                args.root_dir, "train", args.USE_ERASE_DATA
            )
        elif args.data == "val":
            train_dataset = Cityscapes_Dataset(
                args.root_dir, "val", args.USE_ERASE_DATA
            )
        test_dataset = Cityscapes_Dataset(args.root_dir, "val", args.USE_ERASE_DATA)
        if args.iid:
            user_groups = cityscapes_iid(train_dataset, args.num_users)
        else:
            user_groups = cityscapes_noniid_extend(
                args.root_dir, Cityscapes_Dataset.train_folder, args.num_users
            )
    else:
        exit("Unrecognized dataset")
    return train_dataset, test_dataset, user_groups


def get_dataset_camvid(args):
    """
    cityscapes dataset:
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == "camvid":
        if args.data == "train":
            train_dataset = CamVid_Dataset(args, args.root_dir, "train")
        elif args.data == "val":
            train_dataset = CamVid_Dataset(args, args.root_dir, "val")
        test_dataset = CamVid_Dataset(args, args.root_dir, "val")
        user_groups = cityscapes_noniid_extend(
            args.root_dir, CamVid_Dataset.train_folder, args.num_users
        )
    else:
        exit("Unrecognized dataset")
    return train_dataset, test_dataset, user_groups


def get_dataset_ade20k(args):
    """
    cityscapes dataset:
    Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    if args.dataset == "ade20k" or args.dataset == "voc":
        if args.data == "train":
            train_dataset = CamVid_Dataset(args, args.root_dir, "train")
        elif args.data == "val":
            train_dataset = CamVid_Dataset(args, args.root_dir, "val")
        test_dataset = CamVid_Dataset(args, args.root_dir, "val")
        user_groups = cityscapes_noniid_extend(
            args.root_dir, CamVid_Dataset.train_folder, args.num_users
        )
    else:
        exit("Unrecognized dataset")
    return train_dataset, test_dataset, user_groups


def cityscapes_iid(dataset, num_users):
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cityscapes_noniid(num_users):
    """
    Sample non-I.I.D client data from cityscapes

    train_set has 18 cities :
    ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover',
    'jena', 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']

    city_lens = [174, 96, 316, 154, 85, 221, 109, 248, 196, 119, 99, 94, 365, 196, 144, 95, 142, 122]

    num_users = 18 * 8 = 144
    """
    timer = time.time()
    city_lens = [
        174,
        96,
        316,
        154,
        85,
        221,
        109,
        248,
        196,
        119,
        99,
        94,
        365,
        196,
        144,
        95,
        142,
        122,
    ]
    num_users_per_city = int(num_users / 18)
    dict_users = {}
    for city_idx in range(18):
        num_items = int(city_lens[city_idx] / num_users_per_city)
        city_lens_prefix_sum = sum(city_lens[:city_idx])
        all_idxs = [(i + city_lens_prefix_sum) for i in range(city_lens[city_idx])]
        for i in range(num_users_per_city):
            dict_users[i + city_idx * num_users_per_city] = set(
                np.random.choice(all_idxs, num_items, replace=False)
            )
            all_idxs = list(
                set(all_idxs) - dict_users[i + city_idx * num_users_per_city]
            )
        dict_users[(city_idx + 1) * num_users_per_city - 1] |= set(all_idxs)
    print(
        "Time consumed to get non-iid user indices: {:.2f}s".format(time.time() - timer)
    )
    return dict_users


def cityscapes_noniid_extend(root_dir, train_folder, num_users):
    """
    Sample non-I.I.D client data from cityscapes root_dir and train_folder (or other dataset)
    extend : one function to make non-I.I.D split for 18cities or 19classes

    train_set has 18 cities :
    ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 'erfurt', 'hamburg', 'hanover',
    'jena', 'krefeld', 'monchengladbach', 'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    city_lens = [174, 96, 316, 154, 85, 221, 109, 248, 196, 119, 99, 94, 365, 196, 144, 95, 142, 122]
    num_users_per_city = 8
    num_users = 18 * 8 = 144

    manual split train_set has 19 classes :
    city_names:  ['bicycle', 'building', 'bus', 'car', 'fence', 'motorcycle', 'person', 'pole', 'rider',
    'road', 'sidewalk', 'sky', 'terrain', 'traffic_light', 'traffic_sign', 'train', 'truck', 'vegetation', 'wall']
    city_lens:  [156, 156, 156, 181, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 156, 142, 156, 156, 156]
    num_users_per_city = 8
    num_users = 19 * 8 = 152
    """
    timer = time.time()
    print("\nGetting non-iid user indices for cityscapes: ")
    city_lens = get_city_num(root_dir, train_folder)
    num_classes = len(city_lens)
    num_users_per_city = int(num_users / num_classes)
    print(
        "num_users_per_city: {} / {} = {}".format(
            num_users, num_classes, num_users_per_city
        )
    )
    assert num_users % num_classes == 0, "num_users % num_classes != 0"
    dict_users = {}
    for city_idx in range(num_classes):
        num_items = int(city_lens[city_idx] / num_users_per_city)
        city_lens_prefix_sum = sum(city_lens[:city_idx])
        all_idxs = [(i + city_lens_prefix_sum) for i in range(city_lens[city_idx])]
        for i in range(num_users_per_city):
            dict_users[i + city_idx * num_users_per_city] = set(
                np.random.choice(all_idxs, num_items, replace=False)
            )
            all_idxs = list(
                set(all_idxs) - dict_users[i + city_idx * num_users_per_city]
            )
        dict_users[(city_idx + 1) * num_users_per_city - 1] |= set(all_idxs)
    print(
        "Time consumed to get non-iid user indices: {:.2f}s\n".format(
            time.time() - timer
        )
    )
    return dict_users


def get_city_num(root_dir, train_folder):
    city_names = sorted(os.listdir(os.path.join(root_dir, train_folder)))
    print("city_names: ", city_names)
    num_classes = len(city_names)
    print("num_classes: ", num_classes)
    city_lens = []
    for i in range(num_classes):
        city_lens.append(
            len(os.listdir(os.path.join(root_dir, train_folder, city_names[i])))
        )
    for i in range(num_classes):
        print(city_names[i], city_lens[i])
    print("city_lens: ", city_lens)
    return city_lens


if __name__ == "__main__":
    root_dir = "/home/fll/leo_test/data/cityscapes"
    split_root_dir = "/disk1/fll_data/cityscapes_split_erase20"
    user_groups = cityscapes_noniid_extend(
        split_root_dir, Cityscapes_Dataset.train_folder, num_users=152
    )

    def print_user_groups(user_groups):
        data_sum = 0
        for i in range(len(user_groups)):
            print(i, user_groups[i])
            print("len(user_groups[{}]): ".format(i), len(user_groups[i]))
            data_sum += len(user_groups[i])
        print("data_sum: ", data_sum)

    print_user_groups(user_groups)

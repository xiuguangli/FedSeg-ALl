import time
import warnings

import torch
from torch.utils.data import DataLoader

from myseg.dataloader import Cityscapes_Dataset
from myseg.datasplit import get_dataset_cityscapes
from logging_utils import logger, setup_logger
from update import test_inference
from torch_runtime import require_torch_device
from utils import exp_details

from federated_main import make_model

import os
import argparse

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=str, default='0',
                        help='index of GPU to use')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='test colab gpu num_workers=1 is faster')

    # model arguments
    parser.add_argument('--model', type=str, default='bisenetv2',
                        choices=['lraspp_mobilenetv3', 'bisenetv2'],
                        help='model name')
    parser.add_argument('--num_classes', type=int, default=21, help="number of classes max is 81, pretrained is 21")
    parser.add_argument('--checkpoint', type=str, default='', help='full file name of the checkpoint')

    # datasets and training
    parser.add_argument('--dataset', type=str, default='cityscapes', help="name of dataset")
    parser.add_argument('--root_dir', type=str, default='/home/data/cityscapes/', help="root of dataset")
    parser.add_argument('--root', type=str, default='./', help='home directory')
    parser.add_argument('--data', type=str, default='train', choices=['train', 'val', 'test'],
                        help='cityscapes train or val or test')

    parser.add_argument('--USE_ERASE_DATA', type=str2bool,  help="name of dataset")
    parser.add_argument('--proj_dim', type=int, default=256, help="name of dataset")
    parser.add_argument('--rand_init', type=str2bool, default=False, help="name of dataset")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = args_parser()
    setup_logger(verbose=False, logs_dir="logs/eval", log_name="eval")

    start_time = time.time()
    #exp_details(args)

    #torch.manual_seed(args.seed)
    device = require_torch_device(args.gpu)
    logger.info('device: {}', device)

    # load dataset and user groups
    if args.dataset == 'cityscapes':
        #train_dataset, test_dataset, user_groups = get_dataset_cityscapes(args)
        test_dataset = Cityscapes_Dataset(args.root_dir, args.data, args.USE_ERASE_DATA)
        logger.info('args.data: {}', args.data)
    else:
        exit('Error: unrecognized dataset')

    test_loader = DataLoader(test_dataset, batch_size=4, num_workers=args.num_workers, shuffle=False,
                             pin_memory=True)  # for global model test

    # BUILD MODEL
    global_model = make_model(args)

    # print global_model
    # from torchinfo import summary
    # print(global_model) # 根据__init__的参数顺序，输出网络结构
    # summary(global_model, input_size=(1, 3, 512, 1024), device='cpu', depth=5)
    # exit()

    # Set the model to train and send it to device.
    global_model.to(device)

    # resume from checkpoint
    # args.checkpoint = "fed_train_bisenetv2_c19_e1500_frac[0.035]_iid[1]_E[2]_B[8]_lr[0.05]_acti[relu]_users[144]_opti[sgd]_sche[lambda].pth"
    if args.checkpoint != "":
        checkpoint = torch.load(
            os.path.join(args.root, 'save/checkpoints', args.checkpoint),
            map_location=device)
        global_model.load_state_dict(checkpoint['model'])
        start_ep = checkpoint['epoch'] + 1
        logger.info("resume from: {}", args.checkpoint)
    else:
        exit('Error: args.checkpoint is empty')


    # ----------------------------下面的全是evaluate部分----------------------------
    global_model.eval()

    # Evaluate GLOBAL model on test dataset every 'global_test_frequency' rounds
    logger.info('Evaluate global model on global Test dataset')
    test_acc, test_iou, confmat = test_inference(args, global_model, test_loader)
    logger.debug("Confusion matrix:\n{}", confmat)
    logger.info('Results after {} global rounds of training', start_ep)
    logger.info("Global Test Accuracy: {:.2f}%", test_acc)
    logger.info("Global Test IoU: {:.2f}%", test_iou)
    logger.info('Total Run Time: {:.2f}s', time.time() - start_time)

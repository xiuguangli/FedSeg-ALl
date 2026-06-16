import sys

sys.path.append("/home/pjl/project/FedSeg/paddle_project")
import copy
import os
import pickle
import time
import warnings

import numpy as np
import paddle
from eval_utils import evaluate
from myseg.bisenet_utils import set_model_bisenetv2
from myseg.datasplit import (get_dataset_ade20k, get_dataset_camvid,
                             get_dataset_cityscapes)
from options import args_parser
from paddle_utils import *
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from tqdm import tqdm
from update import LocalUpdate, test_inference
from utils import EMA, average_weights, exp_details, weighted_average_weights

warnings.filterwarnings("ignore")
print("os.getcwd(): ", os.getcwd())


def make_model(args):
    if args.model == "bisenetv2":
        global_model = set_model_bisenetv2(args=args, num_classes=args.num_classes)
    else:
        exit("Error: unrecognized model")
    return global_model


def get_exp_name(args):
    exp_name = "fed_{}_{}_{}_c{}_e{}_frac[{}]_iid[{}]_E[{}]_B[{}]_lr[{}]_users[{}]_opti[{}]_sche[{}]".format(
        args.date_now,
        args.data,
        args.model,
        args.num_classes,
        args.epochs,
        args.frac_num,
        args.iid,
        args.local_ep,
        args.local_bs,
        args.lr,
        args.num_users,
        args.optimizer,
        args.lr_scheduler,
    )
    return exp_name


def init_wandb(args, wandb_id, project_name="myseg"):
    if wandb_id is None:
        print("wandb new run")
        wandb.init(project=project_name, name=args.date_now)
    else:
        print("wandb resume")
        wandb.init(project=project_name, resume="must", id=wandb_id)
    try:
        print("wandb_id now: ", wandb.run.id)
    except:
        print("wandb not init")


if __name__ == "__main__":
    args = args_parser()
    start_time = time.time()
    exp_details(args)
    paddle.device.set_device(device=device2str(int(args.gpu)))
    paddle.seed(seed=args.seed)
    device = "cuda" if paddle.device.cuda.device_count() >= 1 else "cpu"
    print("device: " + device)
    if args.dataset == "cityscapes":
        train_dataset, test_dataset, user_groups = get_dataset_cityscapes(args)
    elif args.dataset == "camvid":
        train_dataset, test_dataset, user_groups = get_dataset_camvid(args)
    elif args.dataset == "ade20k":
        train_dataset, test_dataset, user_groups = get_dataset_ade20k(args)
    elif args.dataset == "voc":
        train_dataset, test_dataset, user_groups = get_dataset_ade20k(args)
    else:
        exit("Error: unrecognized dataset")
    test_loader = paddle.io.DataLoader(
        dataset=test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False
    )
    global_model = make_model(args)
    global_model.to(device)
    global_model.train()
    global_weights = global_model.state_dict()
    if args.checkpoint != "":
        checkpoint = paddle.load(
            path=str(os.path.join(args.root, "save/checkpoints", args.checkpoint))
        )
        global_model.set_state_dict(state_dict=checkpoint["model"])
        start_ep = checkpoint["epoch"] + 1
        wandb_id = checkpoint["wandb_id"]
        print("resume from: ", args.checkpoint)
    else:
        start_ep = 0
        wandb_id = None
    if args.USE_WANDB:
        init_wandb(args, wandb_id, project_name="Fedavg_seg")
        try:
            wandb_id = wandb.run.id
        except:
            wandb_id = None
    exp_name = get_exp_name(args)
    print("exp_name :" + exp_name)
    print(
        "\nTraining global model on {} of {} users locally for {} epochs".format(
            args.frac_num, args.num_users, args.epochs
        )
    )
    train_loss, local_test_accuracy, local_test_iou = [], [], []
    if args.globalema:
        ema = EMA(global_model, args.momentum)
        ema.register()
    IoU_record = []
    Acc_record = []
    for epoch in range(start_ep, args.epochs):
        local_weights, local_losses = [], []
        client_dataset_len = []
        print("\n\n| Global Training Round : {} |".format(epoch))
        if args.globalema:
            ema.apply_shadow()
            global_model = ema.model
        global_model.train()
        idxs_users = np.random.choice(
            range(args.num_users), int(args.frac_num), replace=False
        )
        print("local update")
        for idx in idxs_users:
            print("\nUser idx : " + str(idx))
            local_model = LocalUpdate(
                args=args, dataset=train_dataset, idxs=user_groups[idx]
            )
            if not args.is_proto:
                local_mem = None
                local_mask = None
            elif args.localmem and epoch >= args.proto_start_epoch:
                print("Extracting prototypes...")
                proto_tmp, label_list, label_mask_ = local_model.get_protos(
                    model=copy.deepcopy(global_model), global_round=epoch
                )
                if args.kmean_num > 0:
                    proto_tmp = paddle.nn.functional.normalize(x=proto_tmp, axis=2)
                else:
                    proto_tmp = proto_tmp.mean(axis=0)
                    proto_tmp = paddle.nn.functional.normalize(x=proto_tmp, axis=1)
                    label_mask_ = label_mask_.sum(axis=0) > 0
                local_mem = proto_tmp
                local_mask = label_mask_
            else:
                local_mem = None
                local_mask = None
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model),
                global_round=epoch,
                prototypes=local_mem,
                proto_mask=local_mask,
            )
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            client_dataset_len.append(len(user_groups[idx]))
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        print("\n| Global Training Round {} Summary |".format(epoch))
        print("Local Train One global epoch loss_avg: {:.6f}".format(loss_avg))
        try:
            wandb.log({"train_loss": loss_avg}, commit=False, step=epoch + 1)
            wandb.log(
                {"epoch_time (s)": time.time() - local_train_start_time},
                commit=False,
                step=epoch + 1,
            )
        except:
            pass
        print("\nWeight averaging")
        if args.iid:
            print("using average_weights")
            global_weights = average_weights(local_weights)
        else:
            print("using weighted_average_weights")
            global_weights = weighted_average_weights(local_weights, client_dataset_len)
        if args.globalema:
            ema.model.load_state_dict(global_weights)
            ema.update()
        else:
            global_model.set_state_dict(state_dict=global_weights)
        if (epoch + 1) % args.save_frequency == 0 or epoch == args.epochs - 1:
            paddle.save(
                obj={
                    "model": global_model.state_dict(),
                    "epoch": epoch,
                    "exp_name": exp_name,
                    "wandb_id": wandb_id,
                },
                path=os.path.join(args.root, "save/checkpoints", exp_name + ".pth"),
            )
            print("\nGlobal model weights save to checkpoint")
        global_model.eval()
        if (epoch + 1) % args.local_test_frequency == 0:
            local_test_start_time = time.time()
            print(
                """
Testing global model on 50% of train dataset on {} Local users after {} epochs""".format(
                    len(idxs_users), epoch + 1
                )
            )
            list_acc, list_iou = [], []
            for idx in idxs_users:
                local_model = LocalUpdate(
                    args=args, dataset=train_dataset, idxs=user_groups[idx]
                )
                print("\nLocal Test user idx: {}".format(idx))
                print("user_groups[idx]: {}".format(user_groups[idx]))
                acc, iou, confmat = local_model.inference(model=global_model)
                print(confmat)
                list_acc.append(acc)
                list_iou.append(iou)
            local_test_accuracy.append(sum(list_acc) / len(list_acc))
            local_test_iou.append(sum(list_iou) / len(list_iou))
            print("\nLocal test Stats after {} global rounds:".format(epoch + 1))
            print("Training Avg Loss : {:.6f}".format(np.mean(np.array(train_loss))))
            print("Local Test Accuracy: {:.2f}% ".format(local_test_accuracy[-1]))
            print("Local Test IoU: {:.2f}%".format(local_test_iou[-1]))
            print(
                "Local Test Run Time: {:.2f}s\n".format(
                    time.time() - local_test_start_time
                )
            )
            try:
                wandb.log(
                    {"train_acc": local_test_accuracy[-1]}, commit=False, step=epoch + 1
                )
                wandb.log(
                    {"train_MIOU": local_test_iou[-1]}, commit=False, step=epoch + 1
                )
            except:
                pass
        if not args.train_only and (epoch + 1) % args.global_test_frequency == 0:
            print("\n*******************************************")
            print("Evaluate global model on global Test dataset")
            test_acc, test_iou, confmat = test_inference(
                args, global_model, test_loader
            )
            print(confmat)
            print("\nResults after {} global rounds of training:".format(epoch + 1))
            print("|---- Global Test Accuracy: {:.2f}%".format(test_acc))
            print("|---- Global Test IoU: {:.2f}%".format(test_iou))
            print("\nTotal Run Time: {:.2f}min".format((time.time() - start_time) / 60))
            print("*******************************************")
            IoU_record.append(test_iou)
            Acc_record.append(test_acc)
            try:
                wandb.log({"test_acc": test_acc}, commit=False, step=epoch + 1)
                wandb.log({"test_MIOU": test_iou}, commit=False, step=epoch + 1)
            except:
                pass
        try:
            wandb.log({}, commit=True)
            print("\nwandb commit at epoch {}".format(epoch + 1))
        except:
            print("\nwandb not init")
    print("@" * 100)
    print("Average Results of final 5 epochs")
    print("|---- Global Test Accuracy: {:.2f}%".format(sum(Acc_record[-5:]) / 5.0))
    print("|---- Global Test IoU: {:.2f}%".format(sum(IoU_record[-5:]) / 5.0))
    print("@" * 100)

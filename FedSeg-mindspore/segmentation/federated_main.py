import os
from pyexpat import model
import numpy as np

os.environ['GLOG_v'] = '3'
import copy
import json
import time
import pickle
#import wandb
# import torch.nn.functional as F
# import mindspore.mint.nn.functional as F
import mindspore.ops as F

# from torch import nn
from tqdm import tqdm

# import torch
import mindspore
import mindspore.nn as nn
# from torch.utils.data import DataLoader

from mindspore.dataset import GeneratorDataset

import pynvml

from options import args_parser
from update import LocalUpdate, test_inference
from utils import average_weights, weighted_average_weights, exp_details,EMA
from myseg.datasplit import get_dataset_cityscapes,get_dataset_camvid,get_dataset_ade20k
from myseg.bisenet_utils import set_model_bisenetv2


# from sklearn.cluster import KMeans
# from scipy.optimize  import linear_sum_assignment


import warnings
warnings.filterwarnings("ignore") # 忽略warning

print('os.getcwd(): ', os.getcwd())



def make_model(args):
    if args.model == 'bisenetv2':
        #global_model, criteria_pre, criteria_aux = set_model_bisenetv2(num_classes=args.num_classes)
        global_model = set_model_bisenetv2(args=args,num_classes=args.num_classes)

    else:
        exit('Error: unrecognized model')

    # if args.freeze_backbone: # test for DP-SGD
    #     for p in global_model.backbone.parameters():
    #         p.requires_grad = False


    return global_model


def get_exp_name(args):
    # my exp_name
    # exp_name = 'fed_{}_{}_c{}_e{}_frac[{}]_iid[{}]_E[{}]_B[{}]_lr[{}]_acti[{}]_users[{}]_opti[{}]_sche[{}]'. \
    #     format(args.data, args.model, args.num_classes, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs, args.lr, args.activation, args.num_users,
    #            args.optimizer, args.lr_scheduler,
    #            )
    exp_name = 'fed_{}_{}_{}_c{}_e{}_frac[{}]_iid[{}]_E[{}]_B[{}]_lr[{}]_users[{}]_opti[{}]_sche[{}]'. \
        format(args.date_now, args.data, args.model, args.num_classes, args.epochs, args.frac_num, args.iid,
               args.local_ep, args.local_bs, args.lr, args.num_users, args.optimizer, args.lr_scheduler,
               )
    return exp_name

def init_wandb(args, wandb_id, project_name='myseg'):
    # wandb 可视化
    # wandb+pdb 会卡住
    if wandb_id is None: # new run
        print("wandb new run")
        wandb.init(project=project_name,
                   name=args.date_now)
    else:                # resume
        print("wandb resume")
        wandb.init(project=project_name,
                   resume='must',
                   id=wandb_id)
    try:
        print("wandb_id now: ", wandb.run.id)
    except:
        print("wandb not init")




if __name__ == '__main__':
    args = args_parser()

    start_time = time.time()
    exp_details(args)

    # torch.cuda.set_device(int(args.gpu))
    mindspore.set_device(device_target="GPU", device_id=int(args.gpu))

    # torch.manual_seed(args.seed)
    mindspore.set_seed(args.seed)
    idxs_users_list = [np.random.choice(range(args.num_users), int(args.frac_num), replace=False) for _ in range(args.epochs)] # 预先生成每个epoch的用户选择列表，保证可复现
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # print('device: ' + device)
    # ===============================
    
    # load dataset and user groups
    if args.dataset == 'cityscapes':
        train_dataset, test_dataset, user_groups = get_dataset_cityscapes(args)
    elif args.dataset =='camvid':
        train_dataset, test_dataset, user_groups = get_dataset_camvid(args)
    elif args.dataset =='ade20k':
        train_dataset, test_dataset, user_groups = get_dataset_ade20k(args)
    elif args.dataset =='voc':
        train_dataset, test_dataset, user_groups = get_dataset_ade20k(args)
    else:
        exit('Error: unrecognized dataset')
    
    
    
    # test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=True) # for global model test
    test_loader = GeneratorDataset(source=test_dataset, column_names=["image", "label"], num_parallel_workers=args.num_workers, shuffle=False).batch(batch_size=1)
    train_loader = GeneratorDataset(source=train_dataset, column_names=["image", "label"], num_parallel_workers=args.num_workers, shuffle=True).batch(batch_size=args.local_bs)
    print("len(train_dataset): {}, len(test_dataset): {}".format(len(train_dataset), len(test_dataset)))
    

    # BUILD MODEL
    global_model = make_model(args)

    # print global_model
    # from torchinfo import summary
    # print(global_model) # 根据__init__的参数顺序，输出网络结构
    # summary(global_model, input_size=(1, 3, 512, 1024), device='cpu', depth=5)
    # exit()

    # Set the model to train and send it to device.
    # global_model.to(device)
    # global_model.train()
    global_model.set_train()

    # copy weights
    global_weights = global_model.state_dict()

    # resume from checkpoint
    #args.checkpoint = "fed_train_bisenetv2_c19_e1500_frac[0.035]_iid[1]_E[2]_B[8]_lr[0.05]_acti[relu]_users[144]_opti[sgd]_sche[lambda].pth"
    # if args.checkpoint != "":
    #     checkpoint = torch.load(
    #         os.path.join(args.root, 'save/checkpoints', args.checkpoint),
    #         map_location=device)
    #     global_model.load_state_dict(checkpoint['model'])
    #     start_ep = checkpoint['epoch'] + 1
    #     wandb_id = checkpoint['wandb_id']
    #     print("resume from: ", args.checkpoint)
    # else:
    #     # a new run
    #     start_ep = 0
    #     wandb_id = None
    start_ep = 0
    wandb_id = None


    # wandb可视化 init
    if args.USE_WANDB:
        init_wandb(args, wandb_id, project_name='Fedavg_seg')

        try:
            wandb_id = wandb.run.id  # get wandb id
        except:
            wandb_id = None

    
    # set exp name for logging
    exp_name = get_exp_name(args)
    print("exp_name :" + exp_name)

    ## Global rounds / Training
    print('\nTraining global model on {} of {} users locally for {} epochs'.format(args.frac_num, args.num_users, args.epochs))
    train_loss, local_test_accuracy, local_test_iou = [], [], []
    # weights = [] # comment off for checking weights update

    if args.globalema:
        ema = EMA(global_model, args.momentum)
        ema.register()
#        prototypes = torch.randn((args.num_classes,args.proto_dim)).to('cuda:'+str(args.gpu))

    IoU_record =[]
    Acc_record = []
    local_models_list = []
    print('Creating LocalUpdate instances for each user...')
    for idx_user in tqdm(range(args.num_users), desc='Creating LocalUpdate',leave=False):
        local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx_user],model=global_model)
        local_models_list.append(local_model)
        # break
    print('Created {} LocalUpdate instances'.format(len(local_models_list)))
    print(len(user_groups[0]))
    
    all_trained_clients = set() # 记录所有被训练过的client
    # args.is_proto = False
    
    for epoch in range(start_ep, args.epochs):
        local_weights, local_losses = [], []
        client_dataset_len = [] # for non-IID weighted_average_weights
        print('\n\n| Global Training Round : {} |'.format(epoch))

        if args.globalema:
            ema.apply_shadow()
            global_model = ema.model
        # global_model.train()
        global_model.set_train()
        # print(11111111111111111111111111111111)
        # local_models_list[0].train(model = copy.deepcopy(global_model),test_loader=test_loader, train_loader=train_loader)
        # exit()
        
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # idxs_users = np.random.choice(range(args.num_users), int(args.frac_num), replace=False) 
        idxs_users = idxs_users_list[epoch]
        # idxs_users = [0]

        # idxs_users = [0]
        # idxs_users = [i for i in range(args.num_users)] # for debugging, all users participate
        print('local update')
        for idx in idxs_users:
        # for idx in range(args.num_users):
            all_trained_clients.add(idx)
            if len(all_trained_clients)==args.num_users:
                print('All clients have been trained!!!')
            print('\nUser idx : ' + str(idx))
            # local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx])
            local_model = local_models_list[idx]
            # local_model = local_models_list[0]
            local_model.set_model_parameters(global_model)
            local_model.set_global_model(global_model)
            
            if not args.is_proto:
                local_mem = None
                local_mask = None

            else:
                if args.localmem and epoch >= args.proto_start_epoch:


                    print('Extracting prototypes...')
                    proto_tmp,label_list,label_mask_ = local_model.get_protos(model=None,global_round=epoch)

                    if args.kmean_num>0:
                        # proto_tmp = F.normalize(proto_tmp,dim=2)
                        proto_tmp = F.L2Normalize(axis=2)(proto_tmp)
                    
                    else:
                        proto_tmp = proto_tmp.mean(0)
                        # proto_tmp = F.normalize(proto_tmp,dim=1)
                        proto_tmp = F.L2Normalize(axis=1)(proto_tmp)
                        
                        label_mask_ = label_mask_.sum(0)>0

                    local_mem=proto_tmp
                    local_mask = label_mask_
                else:
                    local_mem = None
                    local_mask = None         

            # local_model.train(test_loader=test_loader, train_loader=train_loader)
            # exit()
            
            # w, loss = local_model.update_weights(model=None,global_round=epoch,prototypes = local_mem,proto_mask = local_mask)
            w, loss = local_model.update_weights(model=None,global_round=epoch,prototypes = local_mem,proto_mask = local_mask,test_loader=test_loader)
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            client_dataset_len.append(len(user_groups[idx])) # for non-IID weighted_average_weights
            # client_dataset_len.append(len(train_dataset)) # for non-IID weighted_average_weights

            #print('create LocalUpdate time: {:.2f}s'.format(LocalUpdate_time))
            #print('update_weights time: {:.2f}s'.format(update_weights_time))
            #print("Time per user: {:.2f}s".format(time.time() - time_per_user))
        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)
        print('\n| Global Training Round {} Summary |'.format(epoch))
        print('Local Train One global epoch loss_avg: {:.6f}'.format(loss_avg))
        #print('Local Train One global epoch Time: {:.2f}s'.format((time.time() - local_train_start_time)))
        try:
            wandb.log({'train_loss': loss_avg}, commit=False, step=epoch + 1)
            wandb.log({'epoch_time (s)': (time.time() - local_train_start_time)}, commit=False, step=epoch + 1)
        except:
            pass


        ## UPDATE global weights （fedavg: average_weights)
        print('\nWeight averaging')
        if args.iid:  # IID
            print('using average_weights')
            global_weights = average_weights(local_weights)
        else:  # non-IID
            print('using weighted_average_weights')
            global_weights = weighted_average_weights(local_weights, client_dataset_len)


        if args.globalema:
            ema.model.load_state_dict(global_weights)
            ema.update()
        else:
            global_model.load_state_dict(global_weights)


        # weights.append(global_weights)# comment off for checking weights update

        # save global model to checkpoint                 
        if (epoch+1) % args.save_frequency == 0 or epoch == args.epochs-1:
            # torch.save(
            #     {
            #         'model': global_model.state_dict(),
            #         'epoch': epoch,
            #         'exp_name': exp_name,
            #         'wandb_id': wandb_id
            #     },
            #     os.path.join(args.root, 'save/checkpoints', exp_name+'.pth')
            # )
            save_dir = os.path.join(args.root, 'save/checkpoints')
            os.makedirs(save_dir, exist_ok=True)
            checkpoint_path = os.path.join(save_dir, exp_name + '.ckpt') # MindSpore 推荐使用 .ckpt 后缀
            metadata_path = os.path.join(save_dir, exp_name + '_meta.json')
            # --- 1. 保存模型参数 ---
            mindspore.save_checkpoint(global_model, checkpoint_path)
            metadata = {
                'epoch': epoch,
                'exp_name': exp_name,
                'wandb_id': wandb_id
            }
            # --- 2. 保存元数据 ---
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print('\nGlobal model weights save to checkpoint')
        
        # continue
        # torch.save(weights, 'weights.pt')# comment off for checking weights update

        # ----------------------------下面的全是evaluate部分----------------------------
        # global_model.eval()
        global_model.set_train(False)
        
        # origin : Calculate avg test accuracy over train data of a fraction of users at every epoch
        # my code : Calculate avg accuracy over LOCAL train data of users in [idxs_users] trained already at every 'local_test_frequency' epoch
        #           print global training loss on train set after every 'local_test_frequency' rounds
        if (epoch+1) % args.local_test_frequency == 0:
            local_test_start_time = time.time()
            # test_users = int(args.local_test_frac * args.num_users)
            # print('Testing global model on {} users'.format(test_users))
            print('\nTesting global model on 50% of train dataset on {} Local users after {} epochs'.format(len(idxs_users), epoch+1))
            list_acc, list_iou = [], []

            # for c in tqdm(range(test_users)):
            for idx in idxs_users:
                # local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx])
                local_model = local_models_list[idx]
                local_model.set_model_parameters(global_model)
                print("\nLocal Test user idx: {}".format(idx))
                print("user_groups[idx]: {}".format(user_groups[idx]))
                acc, iou, confmat = local_model.inference(model=global_model)
                print(confmat) # 输出太多，不在最终log中打印
                list_acc.append(acc)
                list_iou.append(iou)
            local_test_accuracy.append(sum(list_acc) / len(list_acc))
            local_test_iou.append(sum(list_iou) / len(list_iou))

            print('\nLocal test Stats after {} global rounds:'.format(epoch+1))
            print('Training Avg Loss : {:.6f}'.format(np.mean(np.array(train_loss)))) # 历史平均值
            print('Local Test Accuracy: {:.2f}% '.format(local_test_accuracy[-1]))
            print('Local Test IoU: {:.2f}%'.format(local_test_iou[-1]))
            print('Local Test Run Time: {:.2f}s\n'.format((time.time()-local_test_start_time)))

            try:
                wandb.log({'train_acc': local_test_accuracy[-1]}, commit=False, step=epoch+1)
                wandb.log({'train_MIOU': local_test_iou[-1]}, commit=False, step=epoch+1)
            except:
                pass


        # Evaluate GLOBAL model on test dataset every 'global_test_frequency' rounds
        if not args.train_only and (epoch+1) % args.global_test_frequency == 0:
            print('\n*******************************************') # use * to mark the Evaluation of GLOBAL model on TEST dataset
            print('Evaluate global model on global Test dataset')
            test_acc, test_iou, confmat = test_inference(args, global_model, test_loader)
            print(confmat)
            print('\nResults after {} global rounds of training:'.format(epoch+1))
            print("|---- Global Test Accuracy: {:.2f}%".format(test_acc))
            print("|---- Global Test IoU: {:.2f}%".format(test_iou))
            print('\nTotal Run Time: {:.2f}min'.format((time.time()-start_time)/60))
            print('*******************************************')
            IoU_record.append(test_iou)
            Acc_record.append(test_acc)

            try:
                wandb.log({'test_acc': test_acc}, commit=False, step=epoch+1)
                wandb.log({'test_MIOU': test_iou}, commit=False, step=epoch+1)
            except:
                pass

        # one epoch ending
        try:
            wandb.log({}, commit=True)  # 每个epoch的最后统一commit
            print('\nwandb commit at epoch {}'.format(epoch+1))
        except:
            print('\nwandb not init')

    print('@'*100)
    print('Average Results of final 5 epochs')
    print("|---- Global Test Accuracy: {:.2f}%".format(sum(Acc_record[-5:])/5.))
    print("|---- Global Test IoU: {:.2f}%".format(sum(IoU_record[-5:])/5.))
    print('@'*100)
    
    # Plot Loss curve
    # if args.epochs > 1:
    #     # Plot Training Loss vs Communication rounds (train_loss)
    #     plt.figure()
    #     plt.title('Training Loss vs Communication rounds')
    #     plt.plot(range(len(train_loss)), train_loss, color='r')
    #     plt.ylabel('Training loss')
    #     plt.xlabel('Communication Rounds')
    #     plt.savefig(os.path.join(args.root, 'save/training_curves', exp_name+'_loss.png'))
    #
    #     # Plot Average Accuracy vs Communication rounds (local_test_accuracy, local_test_iou)
    #     plt.figure()
    #     plt.title('Average Accuracy vs Communication rounds')
    #     plt.plot(range(len(local_test_accuracy)), local_test_accuracy, color='k', label='local test accuracy')
    #     plt.plot(range(len(local_test_iou)), local_test_iou, color='b', label='local test IoU')
    #     plt.ylabel('Average Accuracy')
    #     plt.xlabel('Communication Rounds')
    #     plt.legend()
    #     plt.savefig(os.path.join(args.root, 'save/training_curves', exp_name+'_metrics.png'))

    # Logging
    # filename = os.path.join(args.root, 'save/logs', exp_name+'_log.txt')
    # with open(filename, 'w') as w:
    #     for line in log:
    #         w.write(line + '\n')
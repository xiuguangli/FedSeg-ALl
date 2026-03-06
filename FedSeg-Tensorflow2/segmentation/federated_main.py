from ast import arg

from torch import le
from options import args_parser
args = args_parser()

import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import copy
import time
import pickle
#import wandb
# import torch.nn.functional as F
import numpy as np
# from torch import nn
import tensorflow as tf
from tqdm import tqdm
import concurrent.futures
from box import Box

# import torch
# from torch.utils.data import DataLoader

from update import LocalUpdate, test_inference
from utils import average_weights, weighted_average_weights, exp_details,EMA
from eval_utils import evaluate
from utils import get_weights_dict
from myseg.bisenetv2 import BiSeNetV2

# from sklearn.cluster import KMeans
# from scipy.optimize  import linear_sum_assignment

from myseg.datasplit import get_dataset_cityscapes,get_dataset_camvid,get_dataset_ade20k
from myseg.bisenet_utils import set_model_bisenetv2
from myseg.magic import create_tf_dataloader_from_custom_dataset_test

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



    
def setup_local_updates_parallel(args, train_dataset, user_groups):
    def init_local_update(user_id, args, train_dataset, user_groups):
        """
        一个用于在单独线程中执行 LocalUpdate 初始化的函数。
        """
        # local_updater = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[user_id])
        # return user_id, local_updater
        # 确保 LocalUpdate 可以在多线程环境中安全运行
        try:
            local_updater = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[user_id])
            return user_id, local_updater
        except Exception as e:
            print(f"Error initializing LocalUpdate for user {user_id}: {e}")
            return user_id, None # 返回 None 表示失败
    
    
    # 存储所有 LocalUpdate 实例的字典
    local_updaters = {}
    
    # 确定最大工作线程数。通常是 CPU 核心数或一个经验值。
    max_workers = min(args.num_users, os.cpu_count() * 2) / 2
    # max_workers = 2
    print(f"Using up to {max_workers} threads for LocalUpdate initialization.")
    
    # 使用 ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        # 提交所有用户的初始化任务
        # map() 会按提交顺序返回结果
        futures = {executor.submit(init_local_update, k, args, train_dataset, user_groups): k 
                   for k in range(args.num_users)}
        
        # 使用 tqdm 监控进度
        results = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), 
            total=args.num_users, 
            desc="Initializing Local Updates",
            leave=False
        ):
            user_id, local_updater = future.result()
            
            if local_updater is not None:
                local_updaters[user_id] = local_updater
            
    # 排序并返回 LocalUpdate 列表 (如果需要)
    sorted_updaters = [local_updaters[k] for k in sorted(local_updaters.keys())]
    return sorted_updaters



if __name__ == '__main__':
    # -------------------------------------------------------------
    # 关键设置：启用内存增长
    # -------------------------------------------------------------
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.optimizer.set_jit(False) 
    if gpus:
        try:
            # 限制 GPU 内存增长，让 TensorFlow 只在需要时分配内存
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("TensorFlow 已启用内存增长模式 (set_memory_growth=True)。")
        except RuntimeError as e:
            # 必须在程序启动时设置
            print(e)
    args = args_parser()

    start_time = time.time()
    exp_details(args)

    # torch.cuda.set_device(int(args.gpu))

    # torch.manual_seed(args.seed)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda' if tf.config.list_physical_devices('GPU') else 'cpu'
    print('device: ' + device)

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
    test_loader = create_tf_dataloader_from_custom_dataset_test(test_dataset)

    # BUILD MODEL
    global_model = make_model(args)
    _ = global_model(train_dataset[0][0][None, ...], training=False)  # 模型build，指定输入shape
    

    start_ep = 0
    wandb_id = None

    # test_acc, test_iou, confmat = test_inference(args, global_model, test_loader)
    # print(confmat)
    # exit(0)

    
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

    if args.globalema:  # False
        ema = EMA(global_model, args.momentum)
        ema.register()
        # TensorFlow下原型初始化请用np或tf实现
        # prototypes = np.random.randn(args.num_classes, args.proto_dim)

    # 创建客户端
    print(f'Creating clients {args.num_users}...')
    local_updaters = setup_local_updates_parallel(args, train_dataset, user_groups)
    print('Clients created.')
        
    IoU_record =[]
    Acc_record = []
    train_client_set = set()
    for epoch in range(start_ep, args.epochs):
        local_weights, local_losses = [], []
        client_dataset_len = [] # for non-IID weighted_average_weights
        print('\n\n| Global Training Round : {} |'.format(epoch))

        if len(train_client_set) == args.num_users:
            print("All clients have been selected for training in this epoch.!!!!!!")
        
        if args.globalema:
            ema.apply_shadow()
            global_model = ema.model
        # global_model.train()  # Keras模型训练模式自动管理
        # m = max(int(args.frac * args.num_users), 1)
        # idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        idxs_users = np.random.choice(range(args.num_users), int(args.frac_num), replace=False) # 直接指定frac_num个local user
        # idxs_users = np.random.choice(range(args.num_users), int(args.num_users), replace=False) # 直接指定frac_num个local user
        train_client_set.update(idxs_users.tolist())
       # #local_train_start_time = time.time()
        # Local training
        print('local update')
        # idxs_users = [0]
        for idx in sorted(idxs_users):
            print('\nUser idx : ' + str(idx))

            # local_model = LocalUpdate(args=args, dataset=train_dataset,idxs=user_groups[idx])
            local_model: LocalUpdate = local_updaters[idx]
            # print('Extracting prototypes...')
            # proto_tmp,label_list,label_mask_ = local_model.get_protos(model=copy.deepcopy(global_model),global_round=epoch)
            # exit(0)
            # continue
            
            if not args.is_proto:
                local_mem = None
                local_mask = None
            else:
                if args.localmem and epoch >= args.proto_start_epoch:
                    print('Extracting prototypes...')
                    proto_tmp,label_list,label_mask_ = local_model.get_protos(model=copy.deepcopy(global_model),global_round=epoch)

                    if args.kmean_num>0:
                        # proto_tmp = F.normalize(proto_tmp,dim=2)
                        proto_tmp = proto_tmp / (np.linalg.norm(proto_tmp, axis=2, keepdims=True) + 1e-8)
                    
                    else:
                        proto_tmp = proto_tmp.mean(0)
                        # proto_tmp = F.normalize(proto_tmp,dim=1)
                        proto_tmp = proto_tmp / (np.linalg.norm(proto_tmp, axis=1, keepdims=True) + 1e-8)
                        label_mask_ = label_mask_.sum(0)>0

                    local_mem=proto_tmp
                    local_mask = label_mask_
                else:
                    local_mem = None
                    local_mask = None
            # t0 = time.time()
            model1 = copy.deepcopy(global_model)
            global_model.trainable = False
            w, loss = local_model.update_weights(model=model1,global_round=epoch,prototypes = local_mem,proto_mask = local_mask,global_model=global_model)
            global_model.trainable = True
            # print("Time consumed for local update: {:.2f}s".format(time.time() - t0))
            # exit()
            
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            client_dataset_len.append(len(user_groups[idx])) # for non-IID weighted_average_weights

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

        global_model.set_weights(global_weights)
        
        # save global model to checkpoint                 
        if (epoch+1) % args.save_frequency == 0 or epoch == args.epochs-1:
            os.makedirs(os.path.join(args.root, 'save/checkpoints'), exist_ok=True)
            # torch.save(
            #     {
            #         'model': global_model.state_dict(),
            #         'epoch': epoch,
            #         'exp_name': exp_name,
            #         'wandb_id': wandb_id
            #     },
            #     os.path.join(args.root, 'save/checkpoints', exp_name+'.pth')
            # )
            # print('\nGlobal model weights save to checkpoint')
            global_model.save_weights(os.path.join(args.root, 'save/checkpoints', exp_name+'.weights.h5'))
            print('\nGlobal model weights save to checkpoint')
        # torch.save(weights, 'weights.pt')# comment off for checking weights update


        # ----------------------------下面的evaluate部分----------------------------
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


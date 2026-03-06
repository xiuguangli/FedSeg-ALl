from itertools import count
from os import cpu_count
import time
import copy
from tracemalloc import start
import numpy as np
from tqdm import tqdm

# import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, Dataset

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds

from line_profiler import profile

from eval_utils import evaluate
import myseg.bisenet_utils
from myseg.bisenet_utils import OhemCELoss,BackCELoss,CriterionPixelPair,CriterionPixelRegionPair,ContrastLoss,ContrastLossLocal,CriterionPixelPairG,CriterionPixelPairSeq
from myseg.magic import MultiEpochsDataLoader
#from segmentation_models_pytorch.losses import JaccardLoss,DiceLoss,FocalLoss,LovaszLoss,SoftBCEWithLogitsLoss


# class DatasetSplit(Dataset):
class DatasetSplit():
    """
    An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        image: mindspore.Tensor = mindspore.Tensor(image)
        label: mindspore.Tensor = mindspore.Tensor(label)
        # return torch.tensor(image), torch.tensor(label)
        # pytorch warning and suggest below 
        # return image.clone().detach().float(), label.clone().detach()
        return image.astype(mindspore.float32), label


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, model):
        self.args = args
        # idxs = [i for i in range(len(dataset))]  # use all data for training
        self.trainloader, self.testloader,self.trainloader_eval = self.train_val_test(dataset, list(idxs))
        self.trainloader_iterator = self.trainloader.create_tuple_iterator()
        self.trainloader_eval_iterator = self.trainloader_eval.create_tuple_iterator()
        self.testloader_iterator = self.testloader.create_tuple_iterator()
        #self.device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cuda' if mindspore.device_context.gpu.is_available() else 'cpu'
        self.initial(args,model)

    
    def initial(self,args,model):
        self.model:nn.Cell = copy.deepcopy(model)
        
        self.global_round = 0
        self.initial_lr = args.lr
        scheduler_dict = {
            'step': nn.piecewise_constant_lr(
                milestone=[1000,600000],
                learning_rates=[self.initial_lr, self.initial_lr * 0.1]
            ),
            'poly': nn.polynomial_decay_lr(
                learning_rate=self.initial_lr,
                end_learning_rate=0.0,
                total_step=self.trainloader.get_dataset_size() * max(1, args.local_ep),
                step_per_epoch=self.trainloader.get_dataset_size(),
                decay_epoch=args.local_ep,
                power=0.9
            )
        }
        args.lr_scheduler_ = scheduler_dict[args.lr_scheduler]
        optimizer = myseg.bisenet_utils.set_optimizer(self.model, args)
        if args.model == 'bisenetv2':
            if args.losstype=='ohem':
                criteria_pre = OhemCELoss(0.7)
                criteria_aux = [OhemCELoss(0.7) for _ in range(4)]  # num_aux_heads=4

            elif args.losstype=='ce':
                criteria_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
                criteria_aux = [nn.CrossEntropyLoss(ignore_index=255, reduction='mean') for _ in range(4)]  # num_aux_heads=4
            elif args.losstype =='back':
                criteria_pre = BackCELoss(args)
                criteria_aux = [BackCELoss(args) for _ in range(4)]  # num_aux_heads=4
            elif args.losstype == 'lovasz':
                criteria_pre = LovaszLoss('multiclass',ignore_index=255)
                criteria_aux = [LovaszLoss('multiclass',ignore_index=255) for _ in range(4)]  # num_aux_heads=4
 
            elif args.losstype == 'dice':
                criteria_pre = DiceLoss('multiclass',args.num_classes,ignore_index=255)
                criteria_aux = [DiceLoss('multiclass',args.num_classes,ignore_index=255) for _ in range(4)]  # num_aux_heads=4
            elif args.losstype == 'focal':

                criteria_pre = FocalLoss('multiclass',alpha=0.25,ignore_index=255)
                criteria_aux = [FocalLoss('multiclass',alpha=0.25,ignore_index=255) for _ in range(4)]  # num_aux_heads=4
             
            elif args.losstype == 'bce':
                criteria_pre = SoftBCEWithLogitsLoss(ignore_index=255)
                criteria_aux = [SoftBCEWithLogitsLoss(ignore_index=255) for _ in range(4)]  # num_aux_heads=4
             
            else:
                raise ValueError('loss type is not defined')

        else:
            exit('Error: unrecognized model')
        self.criteria_pre = criteria_pre
        self.criteria_aux = criteria_aux
        # self.criteria_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        # self.criteria_pre = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
        # self.criteria_aux = [nn.CrossEntropyLoss(ignore_index=255, reduction='mean') for _ in range(4)]  # num_aux_heads=4

        self.optimizer = optimizer
          
        # 1.ce + trainable_params() + learning_rate=0.0002 + momentum=0.9 正常
        # 2.backce + trainable_params() + learning_rate=0.0002 + momentum=0.9  正常，略低
        # 3.backce + params_list + learning_rate=0.0002 + momentum=0.9 不正常
        # 4.backce + params_list + learning_rate=0.002 + momentum=0.9 不正常
        
          
        self.grad_fn = mindspore.value_and_grad(self.forward_fn, None, self.optimizer.parameters, has_aux=True)
        self.global_model:nn.Cell = copy.deepcopy(model)
        self.global_model.set_train(False)
        
        if args.distill or args.fedprox_mu >0:
            # self.global_model = copy.deepcopy(model)
            # self.global_model.set_train(False)
            # global_model.eval()
            # for param in global_model.parameters():
            #     param.requires_grad = False
            self.global_model.set_grad(False)
            self.criteria_distill_pi = CriterionPixelPairSeq(args,temperature=args.temp_dist)
            self.criteria_distill_pa =CriterionPixelRegionPair(args)
            self.pixel_seq = []
            

        if args.is_proto:
            self.criteria_contrast = ContrastLoss(args)
            self.global_model.set_grad(False)
            
            # self.global_model = copy.deepcopy(model)
            # global_model.eval()
            # for param in global_model.parameters():
            #     param.requires_grad = False
            # self.global_model.set_train(False)
    
    def set_global_model(self,model):
        self.global_model.load_state_dict(model.state_dict())
        self.global_model.set_train(False)
        
    def set_model_parameters(self,model):
        self.model.load_state_dict(model.state_dict())
    
    def forward_fn(self,global_round,images, labels,prototypes=None,proto_mask=None):
        model = self.model
        args = self.args
        if args.model == 'bisenetv2':
            # start_time = time.perf_counter()
            logits, feat_head, *logits_aux = model(images)
            # print('Local Train Run Time: {0:0.2f}s'.format(time.perf_counter()-start_time))
            # exit()
            labels_ = labels
            if args.losstype == 'bce':
                # cl_ = torch.arange(args.num_classes)
                cl_ = ops.arange(end=args.num_classes)
                cl_ = cl_.unsqueeze(0).unsqueeze(2).unsqueeze(2)
                # cl_ = cl_.to(labels_.device)
                labels_ = labels_.unsqueeze(1) ==cl_
                labels_ = labels_.float()

#                    print(logits.size())
#                    print(labels.size())
#                    exit()
            loss_pre = self.criteria_pre(logits, labels_)
            # loss = loss_pre
            # return loss,0,0,0,0,0,0
# ===============================        
            loss_aux = [crit(lgt, labels_) for crit, lgt in zip(self.criteria_aux, logits_aux)]
            loss = loss_pre + sum(loss_aux)
        else:
            exit('Error: unrecognized model')

        ##########
        if args.is_proto and global_round>= args.proto_start_epoch:

            # _,_,h,w = feat_head.size()
            _,_,h,w = feat_head.shape

            labels_1 = labels_.unsqueeze(1)
            # labels_1 = F.interpolate(labels_1.float(),size=(h,w),mode='nearest')
            labels_1 = ops.interpolate(labels_1.float(),size=(h,w),mode='nearest')
            labels_1 = labels_1.squeeze(1)
            #print(feat_head.size())
            #print(labels_1.size())
            #print(prototypes.size())
            #print(proto_mask.size())
            #exit()
            if args.kmean_num>0:

                proto_mask_tmp = proto_mask.sum(1)<1
            else:
                proto_mask_tmp = proto_mask<1
            for ii, bo in enumerate(proto_mask_tmp):
                if bo:
                    labels_1[labels_1==ii]=255

            loss_con = self.criteria_contrast(feat_head,labels_1,prototypes,proto_mask)
            loss_con_item = loss_con.item()
            # loss_con_item = float(loss_con.asnumpy())
            loss_ce = loss.item()
            # loss_ce = float(loss.asnumpy())
            loss +=args.con_lamb*loss_con 
            
            if args.pseudo_label and global_round>=args.pseudo_label_start_epoch:
                # device = prototypes.device
                # with torch.no_grad():
                #     logits_t, feat_head_t, *logits_aux_t = global_model(images)
                self.global_model.set_train(False)
                logits_t, feat_head_t, *logits_aux_t = self.global_model(images)
                self.global_model.set_train(True)
                # labels_2 = F.interpolate(logits_t.float(),size=(h,w),mode='bilinear')
                labels_2 = ops.interpolate(logits_t.float(),size=(h,w),mode='bilinear')
                # labels_2 = torch.softmax(labels_2,dim=1) 
                labels_2 = ops.softmax(labels_2,axis=1) 
                # props, labels_2 = torch.max(labels_2,dim=1)
                props, labels_2 = ops.max(labels_2,axis=1)
#                        print(props.max())
#                        print(props.min())


                mask_ = props<0.8
                labels_2[mask_]=255
        

                for ii, bo in enumerate(proto_mask_tmp):
                    if bo:
                        labels_2[labels_2==ii]=255
                    
                loss_con_2 = self.criteria_contrast(feat_head,labels_2,prototypes,proto_mask)
                loss_con_2_item = loss_con_2.item()
                # loss_con_2_item = float(loss_con_2.asnumpy())
                loss +=args.con_lamb*loss_con_2
                
        else:
            loss_ce = loss.item()
            # loss_ce = float(loss.asnumpy())
            loss_con_item=0
            loss_con_2_item = 0

        ########
        if args.fedprox_mu >0:
            proximal_term = 0.0
            # for w, w_t in zip(model.parameters(), global_model.parameters()):
            #     proximal_term += (w - w_t).norm(2)
            for w, w_t in zip(model.get_parameters(), self.global_model.get_parameters()):
                proximal_term += float(ops.norm(w - w_t, ord=2))
            loss += (args.fedprox_mu / 2) * proximal_term
        
        if args.distill:
            loss_1_item = loss.item()
            
            # with torch.no_grad():
            #     logits_t, feat_head_t, *logits_aux_t = global_model(images)    
            self.global_model.set_train(False)
            logits_t, feat_head_t, *logits_aux_t = self.global_model(images)                 
                    
            if args.distill_lamb_pi>0 and args.is_proto and global_round>= args.proto_start_epoch:
                # loss_pi,pixel_seq=criteria_distill_pi(feat_head,feat_head_t.detach(),pixel_seq)
                loss_pi, pixel_seq = self.criteria_distill_pi(feat_head,ops.stop_gradient(feat_head_t),pixel_seq)  # ops.stop_gradient(feat_head_t)等价于 .detach()
                loss_pi = args.distill_lamb_pi *loss_pi
                        
                loss+=loss_pi
                loss_pi_item = loss_pi.item()
            else:
                loss_pi_item=0
            if args.distill_lamb_pa>0 and args.is_proto and global_round>= args.proto_start_epoch:
                # loss_pa=args.distill_lamb_pa*criteria_distill_pa(feat_head,feat_head_t.detach(),prototypes,proto_mask)
                loss_pa=args.distill_lamb_pa*self.criteria_distill_pa(feat_head,ops.stop_gradient(feat_head_t),prototypes,proto_mask)
                loss+=loss_pa
                loss_pa_item = loss_pa.item()
            else:
                loss_pa_item=0
        else:
            loss_1_item=0
            loss_pi_item=0
            loss_pa_item=0
        return loss,loss_ce,loss_1_item,loss_pi_item,loss_pa_item,loss_con_item,loss_con_2_item
# ===============================        
           


    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, and test (80%, 20%)
        # idxs_train = idxs[:int(0.8*len(idxs))]
        # idxs_test = idxs[int(0.8*len(idxs)):]

        # split indexes for train, and test (100%, 50%)
        idxs_train = idxs[:]
        idxs_test = idxs[:int(0.5*len(idxs))]

        # try to change num_workers, to see if can speed up training. (num_workers=4 is better for training speed)

        # trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
        #                          batch_size=self.args.local_bs, num_workers=self.args.num_workers,
        #                          shuffle=True, drop_last=True, pin_memory=True)

        # use MultiEpochsDataLoader to speed up training
        # trainloader = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
        #                                     batch_size=self.args.local_bs, num_workers=self.args.num_workers,
        #                                     shuffle=True, drop_last=True, pin_memory=True)

        # trainloader_eval = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
        #                                     batch_size=1, num_workers=self.args.num_workers,
        #                                     shuffle=False, drop_last=False, pin_memory=True)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=1, num_workers=self.args.num_workers,
        #                         shuffle=False)
        # trainloader = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
        #                                     num_parallel_workers=self.args.num_workers,
        #                                     # shuffle=True).batch(self.args.local_bs, drop_remainder=True).repeat(self.args.local_ep)
        #                                     shuffle=True).batch(1, drop_remainder=True).repeat(self.args.local_ep)
        # trainloader_eval = MultiEpochsDataLoader(DatasetSplit(dataset, idxs_train),
        #                                     num_parallel_workers=self.args.num_workers,
        #                                     shuffle=False).batch(1, drop_remainder=False).repeat(1)
        trainloader = ds.GeneratorDataset(DatasetSplit(dataset, idxs_train),
                                            # num_parallel_workers=self.args.num_workers,column_names=["image", "label"],
                                            num_parallel_workers=self.args.num_workers,column_names=["image", "label"],
                                            shuffle=True).batch(self.args.local_bs, drop_remainder=True)
        trainloader_eval = ds.GeneratorDataset(DatasetSplit(dataset, idxs_train),
                                            num_parallel_workers=self.args.num_workers,column_names=["image", "label"],
                                            shuffle=False).batch(1, drop_remainder=False)
        testloader = ds.GeneratorDataset(DatasetSplit(dataset, idxs_test),
                                num_parallel_workers=self.args.num_workers,column_names=["image", "label"],
                                shuffle=False).batch(1, drop_remainder=False) 

        self.trainloader_idx = idxs_train
        self.testloader_idx =idxs_test
        
        return trainloader, testloader,trainloader_eval

    # @torch.no_grad()
    def get_protos(self,model:nn.Cell,global_round:int):
        args = self.args
        # model.eval()
        model = self.model
        model.set_train(False)
        tmp_ = []
        label_list =  []
        label_mask_list = []
        
        for batch_idx, (images, labels) in enumerate(self.trainloader_eval):
        # for batch_idx, (images, labels) in enumerate(self.trainloader_eval_iterator):
            # images, labels = images.to(self.device), labels.to(self.device)

            if args.model == 'bisenetv2':
                logits, feat_head, *logits_aux = model(images)

            # _,_,h,w = feat_head.size()
            _,_,h,w = feat_head.shape
            # labels_2 = F.interpolate(logits.float(),size=(h,w),mode='bilinear')
            labels_2 = ops.interpolate(logits.float(),size=(h,w),mode='bilinear')
            # labels_2 = torch.softmax(labels_2,dim=1)
            labels_2 = ops.softmax(labels_2,axis=1)
            # props, labels_2 = torch.max(labels_2,dim=1)
            props, labels_2 = ops.max(labels_2,axis=1)
#                        print(props.max())
#                        print(props.min())
            mask_ = props<0.8
            labels_2[mask_]=255

            feat_head = feat_head.unsqueeze(1)

            labels = labels.unsqueeze(1)
            labels = ops.interpolate(labels.float(),size=(h,w),mode='nearest')
            labels = labels.unsqueeze(1)

#            print(labels.size())
#            print(labels_2.size())
#            exit()


            labels_2 = labels_2.unsqueeze(1).unsqueeze(1)

            #labels_2[labels!=255]=labels


            # labels = torch.where(labels.float()!=255,labels.float(),labels_2.float())
            labels = ops.where(labels.float()!=255,labels.float(),labels_2.float())
            # unique_l = torch.unique(labels.cpu()).numpy().tolist()
            unique_l = ops.unique(labels)[0].numpy().tolist()
            label_list.extend(unique_l)
            # one_hot_ = torch.zeros(args.num_classes).to(self.device)
            one_hot_ = ops.zeros(args.num_classes)
            for ll in unique_l:
                ll = int(ll)
                if ll !=255:
                    one_hot_[ll]=1
            label_mask_list.append(one_hot_)
            

            # class_ = torch.arange(args.num_classes).to(self.device).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            class_ = ops.arange(args.num_classes).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            weight_ = class_ == labels
            weight_ = weight_/(weight_.sum(3,keepdim=True).sum(4,keepdim=True)+1e-5)
            out = weight_*feat_head
            out = out.sum(-1).sum(-1)
            tmp_.append(out)
        # tmp_ = torch.cat(tmp_,0)
        tmp_ = ops.cat(tmp_,0)
        tmp_ =  tmp_.permute(1,0,2)
            
#        print(tmp_.size())
#        tmp_ = sum(tmp_)/len(tmp_)
        # label_mask_ = torch.stack(label_mask_list,1)
        label_mask_ = ops.stack(label_mask_list,1)
        model.set_train(True)
        return tmp_,label_list,label_mask_

    
    def print_param(self,model,prefix='start'):
        for name, param in model.parameters_and_names():
            print(f"{prefix} {name}: {param.flatten()[:5]}")
            break
    
    
    def update_weights(self, model, global_round,prototypes=None,proto_mask=None,test_loader=None):
        # Set mode to train model
        # model.train()
        # 在 调用 update 之前，先调用 self.set_model_parameters(model)
        model = self.model
        model.set_train()
        epoch_loss = []
        args = self.args
        
        # self.print_param(model,prefix='Start')

        # Set optimizer and lr_scheduler for the local updates

        from tqdm import tqdm
        # training
        start_time = time.time()
        # for iter in range(args.local_ep):
        for iter in range(1):
        # for iter in range(1000):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(tqdm(self.trainloader,desc=f"Train Epoch {iter}",leave=False)):
                labels = labels.astype(mindspore.int32)
                # images, labels = images.to(self.device), labels.to(self.device)
                #print(labels.shape) # torch.Size([8, 512, 1024])
                (loss,loss_ce,loss_1_item,loss_pi_item,loss_pa_item,loss_con_item,loss_con_2_item),grads = self.grad_fn(global_round,images, labels,prototypes,proto_mask)
                self.optimizer(grads)
                
                # optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()  # update params
                batch_loss.append(loss.item())
                # batch_loss.append(float(loss.asnumpy()))

                # 打印学习率
                # print("Local Epoch: {}, batch_idx: {}, lr: {:.3e}".format(iter, batch_idx, lr_scheduler.get_lr()[0]))
                # lr_scheduler.step() # lr_scheduler:poly,根据iter(每个batch)更新lr, (不是根据local_epoch更新)
                # break
            
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            # 打印学习率
            #print("Local Epoch: {}, lr: {:.3e}".format(iter, lr_scheduler.get_lr()[0]))
            #print("Local Epoch: {}, lr: {:.3e}".format(iter, optimizer.param_groups[0]['lr'])) #两个打印学习率的方式都可以
            #lr_scheduler.step()
            
            # test_acc, test_iou, confmat = test_inference(args, model, test_loader)
            # print(f"iter {iter} Test Accuracy: {test_acc:.6f} | Test IoU: {test_iou:.6f} loss: {epoch_loss[iter]:.6f}")
            

            if args.verbose:
                string = '| Global Round : {} | Local Epoch : {} | {} images\tLoss: {:.6f}'.format(
                    # global_round, iter+1, len(self.trainloader.dataset), loss.item())
                    global_round, iter+1, len(self.trainloader.dataset), float(loss.asnumpy()))
                print(string)
        
        # after training, print logs
        # strings = [
        #     '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
        #     global_round, args.local_ep, len(self.trainloader.dataset), loss.item()),
        #     '\nLocal Train Run Time: {0:0.2f}s'.format(time.time()-start_time),
        #     ]

        # 不输出Local Train Run Time了
        # strings = [
        #     '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
        #         global_round, args.local_ep, len(self.trainloader.dataset), loss.item())
        # ]
        strings = [
            '| Global Round : {} | Local Epochs : {} | {} images\tLoss: {:.6f}'.format(
                # global_round, args.local_ep, len(self.trainloader.dataset),float(loss.asnumpy()))
                global_round, args.local_ep, len(self.trainloader_idx),float(loss.asnumpy()))
        ]
        print(''.join(strings))
        if args.distill:
            print('Loss_CE:{:.6f} | loss_pi:{:.6f} | loss_pa:{:.6f}'.format(loss_1_item,loss_pi_item,loss_pa_item))
        
        if args.is_proto:
            if global_round>= args.proto_start_epoch:

                if args.pseudo_label:
                    print('Loss_CE:{:.6f} | loss_contrast:{:.6f} loss_pseudo: {:.6f}'.format(loss_ce,loss_con_item,loss_con_2_item))
                else:
                    print('Loss_CE:{:.6f} | loss_contrast:{:.6f}'.format(loss_ce,loss_con_item))
            else:
                print('Loss_CE:{:.6f}'.format(loss_ce))
            
        # self.print_param(model,prefix='After')
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """
        confmat = evaluate(model, self.testloader, self.device, self.args.num_classes)
        # print(str(confmat)) # local test也输出信息
        return confmat.acc_global, confmat.iou_mean, str(confmat)

   
    

def test_inference(args, model, testloader):
    """ Returns the test accuracy and loss.
    """
    #device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cuda' if mindspore.device_context.gpu.is_available() else 'cpu'
    confmat = evaluate(model, testloader, device, args.num_classes)
    return confmat.acc_global, confmat.iou_mean, str(confmat)
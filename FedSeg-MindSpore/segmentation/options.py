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

def args_parser(argv=None):
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    # parser.add_argument('--frac', type=float, default=0.1,
    #                     help='the fraction of clients used for training: C')
    parser.add_argument('--frac_num', type=int, default=5,
                        help="the fraction num of clients used for training")
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E, default is 10")
    parser.add_argument('--local_bs', type=int, default=1,
                        help="local batch size: B, default=8, local gpu can only set 1")
    parser.add_argument('--num_workers', type=int, default=1,
                        help='test colab gpu num_workers=1 is faster')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                        help='weight decay (default: 0.0005)')
    parser.add_argument('--gpu', type=str, default='0',
                        help='index of GPU to use')
    parser.add_argument('--ms_mode', type=str, default='pynative', choices=['pynative', 'graph'],
                        help='MindSpore execution mode for training and evaluation')
    parser.add_argument('--ms_deterministic', type=str2bool, default=False,
                        help='enable MindSpore deterministic GPU execution where supported')
    parser.add_argument('--ms_disable_tf32', type=str2bool, default=False,
                        help='disable TensorFloat-32 for MindSpore GPU conv/matmul kernels')
    parser.add_argument('--ms_conv_fprop_algo', type=str, default='',
                        help='override MindSpore GPU convolution forward algorithm')
    parser.add_argument('--ms_conv_dgrad_algo', type=str, default='',
                        help='override MindSpore GPU convolution input-gradient algorithm')
    parser.add_argument('--ms_conv_wgrad_algo', type=str, default='',
                        help='override MindSpore GPU convolution weight-gradient algorithm')
    parser.add_argument('--USE_WANDB', type=str2bool, default=0,
                        help='if use wandb')
    parser.add_argument('--USE_ERASE_DATA', type=str2bool, default=0,
                        help='if USE_ERASE_DATA')
    parser.add_argument('--is_proto', type=str2bool, default=False,
                        help='if proto')
#    parser.add_argument('--proto_dim', type=int, default=128,
#                        help='proto_dim')
    parser.add_argument('--label_online_gen', type=str2bool, default=True,
                        help='online Pseudo label generation')
    parser.add_argument('--losstype', type=str, default='ce',choices = ['ce','ohem','back','lovasz','dice','focal','bce'],
                        help='background loss')
    parser.add_argument('--warmstep', type=int, default=500,
                        help='')
    parser.add_argument('--globalema', type=str2bool, default=True,
                        help='')
    parser.add_argument('--mixlabel', type=str2bool, default=False,
                        help='')

    parser.add_argument('--fedprox_mu', type=float, default=0.,
                        help='')
    parser.add_argument('--distill', type=str2bool, default=False,
                        help='')
    parser.add_argument('--distill_lamb_pi', type=float, default=1,
                        help='')
    parser.add_argument('--distill_lamb_pa', type=float, default=1,
                        help='')
    parser.add_argument('--rand_init', type=str2bool, default=False,
                        help='')
#########################PROTO_NEW
    parser.add_argument('--proj_dim', type=int, default=256,
                        help='')
    parser.add_argument('--proto_start_epoch', type=int, default=1,
                        help='')
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='')
    parser.add_argument('--con_lamb', type=float, default=0.1,
                        help='')
    parser.add_argument('--con_lamb_local', type=float, default=0.1,
                        help='')
    parser.add_argument('--max_anchor', type=int, default=1024,
                        help='')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='')

    parser.add_argument('--kmean_num', type=int, default=0,
                        help='')
    parser.add_argument('--pseudo_label', type=str2bool, 
                        help='')
    parser.add_argument('--pseudo_label_start_epoch', type=int,default=1, 
                        help='')
    parser.add_argument('--localmem', type=str2bool,default=False, 
                        help='')
    parser.add_argument('--mom_update', type=str2bool,default=False, 
                        help='')
    parser.add_argument('--temp_dist', type=float, default=0.07,
                        help='')
##############################
    # model arguments
    parser.add_argument('--model', type=str, default='bisenetv2',
                        choices=['lraspp_mobilenetv3', 'bisenetv2'],
                        help='model name')    
    parser.add_argument('--num_classes', type=int, default=21, help="number of classes max is 81, pretrained is 21")
    #parser.add_argument('--cpu_only', action='store_true', help="indicate to use cpu only")
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'],
                        help="type of optimizer")
    # parser.add_argument('-aux', '--aux_lr', type=int, default=2,
    #                     help='times of normal learning rate used for auxiliary classifier ')
    parser.add_argument('--lr_scheduler', default='poly', choices=['poly', 'step'], help='learning rate scheduler')
    parser.add_argument('--checkpoint', type=str, default='', help='full file name of the checkpoint')
    parser.add_argument('--init_checkpoint', type=str, default='',
                        help='load model weights from a MindSpore checkpoint as round-0 initialization without resuming training metadata')
    parser.add_argument('--auto_align_torch_init', type=str2bool, default=True,
                        help='when training VOC BiSeNetV2 from scratch and no checkpoint/init_checkpoint is given, auto-load save/checkpoints/fedseg_torch_init_full.ckpt if available to match the torch baseline initialization')
    parser.add_argument('--save_frequency', type=int, default=5, help='number of epochs to save checkpoint')
    parser.add_argument('--keep_round_checkpoints', type=str2bool, default=False,
                        help='keep an additional immutable checkpoint snapshot tagged with the saved global round')
    #parser.add_argument('--test_frequency', type=int, default=5, help='number of epochs to eval global test data')
    parser.add_argument('--local_test_frequency', type=int, default=1, help='number of epochs to eval global model on train data')
    parser.add_argument('--global_test_frequency', type=int, default=5, help='number of epochs to eval global model on test data')
    parser.add_argument('--train_only', action='store_true')
    parser.add_argument('--pretrained', action='store_true',
                        help='only available for deeplab_mobilenetv3 and lraspp_mobilenetv3')        
    # parser.add_argument('--activation', default='relu', choices=['relu', 'tanh'],
    #                     help='set activatition function in models as argument')

    # datasets and training
    parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes','camvid','ade20k','voc'],help="name of dataset")
    parser.add_argument('--root_dir', type=str, default='/home/data/cityscapes/', help="root of dataset")
    parser.add_argument('--iid', type=str2bool, default=1,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--profile_runtime', type=str2bool, default=False,
                        help='log runtime breakdown for hot-path debugging')
    parser.add_argument('--profile_runtime_detail', type=str2bool, default=False,
                        help='log batch-level runtime detail for MindSpore hot-path debugging')
    parser.add_argument('--profile_runtime_sync', type=str2bool, default=False,
                        help='synchronize device inside the training loop for runtime debugging; may perturb PYNATIVE numerics')
    parser.add_argument('--check_finite_per_batch', type=str2bool, default=True,
                        help='validate loss/grad/metric finiteness after every local batch; useful for debugging but adds overhead')
    parser.add_argument('--prototype_forward_batch_size', type=int, default=4,
                        help='number of prototype-extraction samples to batch into one forward pass; preserves sample order while reducing per-image launch overhead')
    parser.add_argument('--debug_log_memory', type=str2bool, default=False,
                        help='log MindSpore allocator memory stats after client updates')
    parser.add_argument('--debug_empty_cache_per_client', type=str2bool, default=False,
                        help='call mindspore.hal.empty_cache() after each local client update')
    parser.add_argument('--debug_disable_train_aug', type=str2bool, default=False,
                        help='replace train augmentation with a deterministic resize for debugging')
    parser.add_argument('--debug_disable_dropout', type=str2bool, default=False,
                        help='disable dropout during training for debugging parity')
    parser.add_argument('--debug_deterministic_dropout', type=str2bool, default=False,
                        help='replace dropout with a deterministic mask during training for debugging parity')
    parser.add_argument('--debug_deterministic_contrast', type=str2bool, default=False,
                        help='replace random contrast anchor sampling with a deterministic prefix selection for debugging parity')
    parser.add_argument('--debug_freeze_bn_stats', type=str2bool, default=False,
                        help='freeze BatchNorm running statistics during local training')
    parser.add_argument('--debug_train_order_file', type=str, default='',
                        help='JSON file with an explicit per-epoch local train sample order for debugging parity')
    parser.add_argument('--debug_emulate_torch_worker_rng', type=str2bool, default=True,
                        help='emulate torch DataLoader worker-local RNG progression for parity with the torch baseline')
    parser.add_argument('--reuse_local_trainer', type=str2bool, default=False,
                        help='reuse a single local MindSpore trainer across clients; may change proto-training numerics')
    parser.add_argument('--aggregate_bn_stats', type=str2bool, default=True,
                        help='aggregate BatchNorm moving statistics across clients during global averaging')
    parser.add_argument('--eval_only', type=str2bool, default=False,
                        help='skip training and only evaluate the loaded checkpoint')
    parser.add_argument('--eval_batch_size', type=int, default=24,
                        help='batch size for the fast grouped global evaluation path')
    parser.add_argument('--final_eval_batch_size', type=int, default=1,
                        help='batch size for the final precise evaluation after all rounds')
    parser.add_argument('--final_eval_precise', type=str2bool, default=True,
                        help='run one final global evaluation with final_eval_batch_size after training')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--root', type=str, default='./', help='home directory')
    parser.add_argument('--data', type=str, default='train', choices=['train', 'val'],
                        help='cityscapes train or val')
    # parser.add_argument('--local_test_frac', default=0.1, type=float, help='frac of num_users for local testing')
    # parser.add_argument('--freeze_backbone', action='store_true', help='choose to not train backbone')
    # parser.add_argument('--weight', default=1.0, type=float, help='the weight assigned to computing loss of background class')


    # inference
    #parser.add_argument('--filename', default='', type=str, help='image filename for inference.')
    parser.add_argument('--date_now', default='unknown', type=str, help='for name of my wandb run')
    
    args = parser.parse_args(argv)
    return args

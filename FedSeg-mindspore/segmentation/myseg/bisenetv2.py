
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.model_zoo as modelzoo
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import XavierNormal, initializer,HeNormal
from mindspore.common.parameter import Parameter
import math


backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'


# class ConvBNReLU(nn.Module):
class ConvBNReLU(nn.Cell):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, bias=False):
        super(ConvBNReLU, self).__init__()
        # self.conv = nn.Conv2d(
        #         in_chan, out_chan, kernel_size=ks, stride=stride,
        #         padding=padding, dilation=dilation,
        #         groups=groups, bias=bias)
        self.conv = nn.Conv2d(
                in_chan, out_chan, kernel_size=ks, stride=stride,
                pad_mode='pad',padding=padding, dilation=dilation,
                group=groups, has_bias=bias)
        self.bn = nn.BatchNorm2d(out_chan)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    # def forward(self, x):
    def construct(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


# class UpSample(nn.Module):
class UpSample(nn.Cell):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        # self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.proj = nn.Conv2d(in_channels=n_chan, 
                              out_channe=out_chan, 
                              kernel_size=1, 
                              stride=1, 
                              pad_mode='pad',
                              padding=0,
                              has_bias=True)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    # def forward(self, x):
    def construct(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        # nn.init.xavier_normal_(self.proj.weight, gain=1.)
        """
        手动对 self.proj.weight 进行 Xavier Normal 初始化
        """
        # 1. 获取要初始化的参数
        weight_param = self.proj.weight

        # 2. 创建 XavierNormal 初始化器实例
        xavier_initializer = XavierNormal(gain=1.0)

        # 3. 使用 initializer 生成新的数据，并用 set_data 更新参数
        new_data = initializer(xavier_initializer, weight_param.shape, weight_param.dtype)
        weight_param.set_data(new_data)



# class DetailBranch(nn.Module):
class DetailBranch(nn.Cell):

    def __init__(self):
        super(DetailBranch, self).__init__()
        # self.S1 = nn.Sequential(
        self.S1 = nn.SequentialCell(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        # self.S2 = nn.Sequential(
        self.S2 = nn.SequentialCell(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        # self.S3 = nn.Sequential(
        self.S3 = nn.SequentialCell(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    # def forward(self, x):
    def construct(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


# class StemBlock(nn.Module):
class StemBlock(nn.Cell):

    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        # self.left = nn.Sequential(
        self.left = nn.SequentialCell(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(
            kernel_size=3, stride=2, pad_mode='pad',padding=1, ceil_mode=False)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    # def forward(self, x):
    def construct(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        # feat = torch.cat([feat_left, feat_right], dim=1)
        feat = ops.cat((feat_left, feat_right), axis=1)
        feat = self.fuse(feat)
        return feat


# class CEBlock(nn.Module):
class CEBlock(nn.Cell):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    # def forward(self, x):
    def construct(self, x):
        # feat = torch.mean(x, dim=(2, 3), keepdim=True)
        feat = ops.mean(x, axis=(2, 3), keep_dims=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


# class GELayerS1(nn.Module):
class GELayerS1(nn.Cell):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        # self.dwconv = nn.Sequential(
        self.dwconv = nn.SequentialCell(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=1,
                # padding=1, groups=in_chan, bias=False),
                pad_mode='pad',padding=1, group=in_chan, has_bias=False),
            nn.BatchNorm2d(mid_chan),
            # nn.ReLU(inplace=True), # not shown in paper
            nn.ReLU(), # not shown in paper
        )
        # self.conv2 = nn.Sequential(
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                pad_mode='pad',padding=0, has_bias=False),
            # nn.BatchNorm2d(out_chan),
            nn.BatchNorm2d(out_chan,gamma_init='zeros'),
        )
        # self.conv2[1].last_bn = True   # nn.BatchNorm2d(out_chan),
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    # def forward(self, x):
    def construct(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


# class GELayerS2(nn.Module):
class GELayerS2(nn.Cell):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        # self.dwconv1 = nn.Sequential(
        self.dwconv1 = nn.SequentialCell(
            nn.Conv2d(
                in_chan, mid_chan, kernel_size=3, stride=2,
                pad_mode='pad',padding=1, group=in_chan, has_bias=False),
                # padding=1, groups=in_chan, bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        # self.dwconv2 = nn.Sequential(
        self.dwconv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan, mid_chan, kernel_size=3, stride=1,
                # padding=1, groups=mid_chan, bias=False),
                pad_mode='pad',padding=1, group=mid_chan, has_bias=False),
            nn.BatchNorm2d(mid_chan),
            # nn.ReLU(inplace=True), # not shown in paper
            nn.ReLU(), # not shown in paper
        )
        # self.conv2 = nn.Sequential(
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan, out_chan, kernel_size=1, stride=1,
                # padding=0, bias=False),
                pad_mode='pad',padding=0, has_bias=False),
            nn.BatchNorm2d(out_chan,gamma_init='zeros'),
        )
        # self.conv2[1].last_bn = True
        # self.shortcut = nn.Sequential(
        self.shortcut = nn.SequentialCell(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    # padding=1, groups=in_chan, bias=False),
                    pad_mode='pad',padding=1, group=in_chan, has_bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    # padding=0, bias=False),
                    pad_mode='pad',padding=0, has_bias=False),
                nn.BatchNorm2d(out_chan),
        )
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU()

    # def forward(self, x):
    def construct(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


# class SegmentBranch(nn.Module):
class SegmentBranch(nn.Cell):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        # self.S3 = nn.Sequential(
        self.S3 = nn.SequentialCell(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        # self.S4 = nn.Sequential(
        self.S4 = nn.SequentialCell(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        # self.S5_4 = nn.Sequential(
        self.S5_4 = nn.SequentialCell(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    # def forward(self, x):
    def construct(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


# class BGALayer(nn.Module):
class BGALayer(nn.Cell):

    def __init__(self):
        super(BGALayer, self).__init__()
        # self.left1 = nn.Sequential(
        self.left1 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                # padding=1, groups=128, bias=False),
                pad_mode='pad',padding=1, group=128, has_bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                # padding=0, bias=False),
                pad_mode='pad',padding=0, has_bias=False),
        )
        # self.left2 = nn.Sequential(
        self.left2 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                # padding=1, bias=False),
                pad_mode='pad',padding=1, has_bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='pad',padding=1, ceil_mode=False)
        )
        # self.right1 = nn.Sequential(
        self.right1 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                # padding=1, bias=False),
                pad_mode='pad',padding=1, has_bias=False),
            nn.BatchNorm2d(128),
        )
        # self.right2 = nn.Sequential(
        self.right2 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                # padding=1, groups=128, bias=False),
                pad_mode='pad',padding=1, group=128, has_bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                # padding=0, bias=False),
                pad_mode='pad',padding=0, has_bias=False),
        )
        self.scale_factor = 4
        # self.up1 = nn.Upsample(scale_factor=4.0)
        # self.up2 = nn.Upsample(scale_factor=4.0)
        self.up1 = nn.Upsample(size=(60,60)) # voc 到这里的原始尺寸是 n c 15 15，扩大4倍，就是60。 mindspore 中不支持在4D数据情况下，在nearest模式下以scale方式插值
        self.up2 = nn.Upsample(size=(60,60))
        ##TODO: does this really has no relu?
        # self.conv = nn.Sequential(
        self.conv = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                # padding=1, bias=False),
                pad_mode='pad',padding=1, has_bias=False),
            nn.BatchNorm2d(128),
            # nn.ReLU(inplace=True), # not shown in paper
            nn.ReLU(), # not shown in paper
        )

    # def forward(self, x_d, x_s):
    def construct(self, x_d:ms.Tensor, x_s):
        # dsize = x_d.size()[2:]
        dsize = x_d.shape[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        # right1 = self.up1(right1)
        right1 = ops.interpolate(input=right1,size=(self.scale_factor*right1.shape[2],self.scale_factor*right1.shape[3]))
        # left = left1 * torch.sigmoid(right1)
        left = left1 * ops.sigmoid(right1)
        # right = left2 * torch.sigmoid(right2)
        right = left2 * ops.sigmoid(right2)
        # right = self.up2(right)
        right = ops.interpolate(input=right,size=(self.scale_factor*right.shape[2],self.scale_factor*right.shape[3]))
        out = self.conv(left + right)
        return out



# class SegmentHead(nn.Module):
class SegmentHead(nn.Cell):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        # self.drop = nn.Dropout(0.1)
        self.drop = nn.Dropout(p=0.1)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.scale_factor = 2
        self.up_factor = up_factor
        self.aux = aux
        # self.conv_out = nn.Sequential(
        self.conv_out = nn.SequentialCell(
            # nn.Sequential(
            nn.SequentialCell(
                # nn.Upsample(scale_factor=2),
                # nn.Upsample(scale_factor=self.scale_factor),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1)
                ) if aux else nn.Identity(),
            # nn.Conv2d(mid_chan2, out_chan, 1, 1, 0, bias=True),
            nn.Conv2d(
                in_channels=mid_chan2,
                out_channels=out_chan,
                kernel_size=1,
                stride=1,
                pad_mode='pad',padding=0,
                has_bias=True  # 使用 has_bias
            ),
            # nn.Upsample(scale_factor=up_factor, mode='bilinear', align_corners=False)
        )

    # def forward(self, x):
    def construct(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        if self.aux:
            feat = ops.interpolate(input=feat,size=(self.scale_factor*feat.shape[2],self.scale_factor*feat.shape[3]))
        feat = self.conv_out(feat)
        feat = ops.interpolate(input=feat,size=(self.up_factor*feat.shape[2],self.up_factor*feat.shape[3]),mode='bilinear',align_corners=False)
        return feat

from line_profiler import profile

# class BiSeNetV2(nn.Module):
class BiSeNetV2(nn.Cell):

    def __init__(self,args, n_classes, aux_mode='train'):
        super(BiSeNetV2, self).__init__()
        self.args = args
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        ## TODO: what is the number of mid chan ?
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=self.args.proj_dim)

        self.init_weights()
    
    # def forward(self, x:ms.Tensor):
    def construct(self, x:ms.Tensor):
        # size = x.size()[2:]
        size = x.shape[2:]
        
        ######
        if self.aux_mode=='eval':
            h_,w_ = size
            if h_%32!=0:
                new_h = math.ceil(h_/32)*32
                pad_h  = new_h-h_
            else: 
                pad_h=0
            if w_%32!=0:
                new_w = math.ceil(w_/32)*32
                pad_w  = new_w-w_
            else:
                pad_w=0
            
            # x = torch.nn.functional.pad(x,(0,pad_w,0,pad_h),mode='reflect')    
            x = ops.pad(x, [0,pad_w,0,pad_h], mode='reflect')
        #####
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        emb = self.proj_head(feat_head)
        logits = self.head(feat_head)
        
        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits,emb,logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            logits = logits[:,:,:h_,:w_]
            return logits,
        elif self.aux_mode == 'pred':
            # pred = logits.argmax(dim=1)
            pred = logits.argmax(axis=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights0(self):
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
                if not module.bias is None: nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                if hasattr(module, 'last_bn') and module.last_bn:
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if not self.args.rand_init:
            self.load_pretrain()
    
    
    
    def init_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(initializer(HeNormal(mode='fan_out', nonlinearity='relu'),cell.weight.shape, cell.weight.dtype))
                # cell.weight.set_data(initializer("ones",cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(initializer('zeros', cell.bias.shape, cell.bias.dtype))
            
            elif isinstance(cell, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                if hasattr(cell, 'last_bn') and cell.last_bn:
                    cell.gamma.set_data(initializer('zeros', cell.gamma.shape, cell.gamma.dtype))
                else:
                    cell.gamma.set_data(initializer('ones', cell.gamma.shape, cell.gamma.dtype))
                cell.beta.set_data(initializer('zeros', cell.beta.shape, cell.beta.dtype))

        if not self.args.rand_init:
            self.load_pretrain()

    def load_pretrain(self):
        # state = modelzoo.load_url(backbone_url)
        # state = torch.load('segmentation/myseg/backbone_v2.pth')  # baidu server
        # for name, child in self.named_children():
        #     if name in state.keys():
        #         child.load_state_dict(state[name], strict=True)
        ckpt_file_path = 'segmentation/myseg/backbone_v2.ckpt'
        # 1. 加载整个检查点文件到一个参数字典
        #    这个字典是扁平的，key 是参数的全名，如 'backbone.layer1.conv1.weight'
        param_dict = ms.load_checkpoint(ckpt_file_path)
        
        # 2. 一次性将参数加载到当前网络 (self) 中
        #    strict_load=False 允许只加载 ckpt 中存在的参数，
        #    如果网络中有些参数在 ckpt 里没有，也不会报错。
        #    这完美地模拟了 'if name in state.keys()' 的效果。
        unloaded_params = ms.load_param_into_net(self, param_dict, strict_load=False)
        print_tag = False
        if unloaded_params and print_tag:
            print(f"[WARNING] Some parameters were not loaded: {unloaded_params}")

    def get_params0(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.parameters():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    def get_params1(self):
        """
        将模型的参数分组，用于优化器设置。
        - 1D 参数 (如 bias, beta, gamma) 不设置权重衰减。
        - 4D 参数 (如 conv.weight) 设置权重衰减。
        - head 和 aux 模块的参数被单独分组，以便应用不同的学习率。
        """
        def add_param_to_list(cell:nn.Cell, wd_params, nowd_params, wd_params_name, nowd_params_name):
            # 使用 cell.trainable_params() 获取可训练参数
            for param in cell.trainable_params():
                # 使用 param.ndim 获取参数维度
                if param.ndim == 1:
                    # 1D 参数通常是 bias, LayerNorm/BatchNorm 的 beta/gamma
                    nowd_params.append(param)
                    nowd_params_name.append(f"{param.name}")
                elif param.ndim == 4:
                    # 4D 参数通常是 Conv2d 的 weight
                    wd_params.append(param)
                    wd_params_name.append(f"{param.name}")

            
            # for name, param in cell.parameters_and_names():
            #     if param.ndim == 1:
            #         # 1D 参数通常是 bias, LayerNorm/BatchNorm 的 beta/gamma
            #         nowd_params.append(param)
            #         print(f"{name=},{param.shape=}")
            #     elif param.ndim == 4:
            #         # 4D 参数通常是 Conv2d 的 weight
            #         wd_params.append(param)
                
        def print_param_list(param_list, param_list_name):
            print("=====print_param_list=====")
            for idx, (name, param ) in enumerate(zip(param_list_name, param_list)):
                print(f" {idx=} {name=}, {param.shape=}")
        

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        wd_params_name, nowd_params_name, lr_mul_wd_params_name, lr_mul_nowd_params_name = [], [], [], []
        
        # 使用 self.name_cells() 遍历直接子 Cell
        # print(type(self.name_cells()))
        # exit()
        for name, child in self.name_cells().items():
            # print(1)
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params, lr_mul_wd_params_name, lr_mul_nowd_params_name)
            else:
                add_param_to_list(child, wd_params, nowd_params, wd_params_name, nowd_params_name)
        print_param_list(lr_mul_nowd_params, lr_mul_nowd_params_name)
        print_param_list(lr_mul_wd_params, lr_mul_wd_params_name)
        print_param_list(nowd_params, nowd_params_name)
        print_param_list(wd_params, wd_params_name)
        exit()
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
    
    def get_params(self):
        """
        将模型的参数分组，用于优化器设置。
        - 1D 参数 (如 bias, beta, gamma) 不设置权重衰减。
        - 4D 参数 (如 conv.weight) 设置权重衰减。
        - head 和 aux 模块的参数被单独分组，以便应用不同的学习率。
        """
        def add_param_to_list(cell:nn.Cell, wd_params, nowd_params):
            # 使用 cell.trainable_params() 获取可训练参数
            for param in cell.trainable_params():
                # 使用 param.ndim 获取参数维度
                if param.ndim == 1:
                    # 1D 参数通常是 bias, LayerNorm/BatchNorm 的 beta/gamma
                    nowd_params.append(param)
                elif param.ndim == 4:
                    # 4D 参数通常是 Conv2d 的 weight
                    wd_params.append(param)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        
        # 使用 self.name_cells() 遍历直接子 Cell
        # print(type(self.name_cells()))
        # exit()
        for name, child in self.name_cells().items():
            # print(1)
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
                
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
 
# class ProjectionHead(nn.Module):
class ProjectionHead(nn.Cell):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', ):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            # self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1, has_bias=True)
        elif proj == 'convmlp':
            # self.proj = nn.Sequential(
            self.proj = nn.SequentialCell(
                # nn.Conv2d(dim_in, dim_in, kernel_size=1),
                nn.Conv2d(dim_in, dim_in, kernel_size=1, has_bias=True),
                nn.BatchNorm2d(dim_in),
                # nn.ReLU(inplace=True),
                nn.ReLU(),
                # nn.Conv2d(dim_in, proj_dim, kernel_size=1)
                nn.Conv2d(dim_in, proj_dim, kernel_size=1, has_bias=True)
            )
    # def forward(self, x):
    def construct(self, x):
        # return F.normalize(self.proj(x), p=2, dim=1)
        return ops.L2Normalize(axis=1)(self.proj(x))



if __name__ == "__main__":
    #  x = torch.randn(16, 3, 1024, 2048)
    #  detail = DetailBranch()
    #  feat = detail(x)
    #  print('detail', feat.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  stem = StemBlock()
    #  feat = stem(x)
    #  print('stem', feat.size())
    #
    #  x = torch.randn(16, 128, 16, 32)
    #  ceb = CEBlock()
    #  feat = ceb(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 32, 16, 32)
    #  ge1 = GELayerS1(32, 32)
    #  feat = ge1(x)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 16, 16, 32)
    #  ge2 = GELayerS2(16, 32)
    #  feat = ge2(x)
    #  print(feat.size())
    #
    #  left = torch.randn(16, 128, 64, 128)
    #  right = torch.randn(16, 128, 16, 32)
    #  bga = BGALayer()
    #  feat = bga(left, right)
    #  print(feat.size())
    #
    #  x = torch.randn(16, 128, 64, 128)
    #  head = SegmentHead(128, 128, 19)
    #  logits = head(x)
    #  print(logits.size())
    #
    #  x = torch.randn(16, 3, 1024, 2048)
    #  segment = SegmentBranch()
    #  feat = segment(x)[0]
    #  print(feat.size())
    #
    # x = torch.randn(16, 3, 1024, 2048)
    x = ops.randn(16, 3, 1024, 2048)
    model = BiSeNetV2(n_classes=19)
    outs = model(x)
    for out in outs:
        print(out.size())
    #  print(logits.size())

    #  for name, param in model.named_parameters():
    #      if len(param.size()) == 1:
    #          print(name)

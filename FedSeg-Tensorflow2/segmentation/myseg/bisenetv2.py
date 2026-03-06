import tensorflow as tf
from keras.saving import register_keras_serializable 

backbone_url = 'https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth'

@register_keras_serializable()
class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, use_bias=False):
        super().__init__()
        
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        
        self.conv = tf.keras.layers.Conv2D(
            out_chan, ks, strides=stride, padding='same' if padding > 0 else 'valid',
            dilation_rate=dilation, groups=groups, use_bias=use_bias)
        self.bn = tf.keras.layers.BatchNormalization(axis=1)
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'ks': self.ks,  
            'stride': self.stride,
            'padding': self.padding,
            'dilation': self.dilation,
            'groups': self.groups,
            'use_bias': self.use_bias,
        })
        return config

@register_keras_serializable()
class UpSample(tf.keras.layers.Layer):
    def __init__(self, n_chan, factor=2):
        super().__init__()
        out_chan = n_chan * factor * factor
        
        self.n_chan = n_chan
        self.factor = factor
        self.out_chan = out_chan
        
        self.proj = tf.keras.layers.Conv2D(out_chan, 1, strides=1, padding='valid')
        self.up = tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, factor))

    def call(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_chan': self.n_chan,
            'factor': self.factor,
            'out_chan': self.out_chan,
        })
        return config

@register_keras_serializable()
class DetailBranch(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.S1 = tf.keras.Sequential([
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        ])
        self.S2 = tf.keras.Sequential([
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        ])
        self.S3 = tf.keras.Sequential([
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        ])

    def call(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat
    

    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class StemBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = tf.keras.Sequential([
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        ])
        self.right = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same')
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def call(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = tf.concat([feat_left, feat_right], axis=1)
        feat = self.fuse(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class CEBlock(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.bn = tf.keras.layers.BatchNormalization(axis=1)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def call(self, x):
        feat = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class GELayerS1(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.exp_ratio = exp_ratio
        
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.ReLU(),
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_chan, 1, strides=1, padding='valid'),
            tf.keras.layers.BatchNormalization(axis=1),
        ])
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'exp_ratio': self.exp_ratio,
        })
        return config

@register_keras_serializable()
class GELayerS2(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.exp_ratio = exp_ratio
        
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
        ])
        self.dwconv2 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.ReLU(),
        ])
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(out_chan, 1, strides=1, padding='valid'),
            tf.keras.layers.BatchNormalization(axis=1),
        ])
        self.shortcut = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Conv2D(out_chan, 1, strides=1, padding='valid'),
            tf.keras.layers.BatchNormalization(axis=1),
        ])
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_chan': self.in_chan,
            'out_chan': self.out_chan,
            'exp_ratio': self.exp_ratio,
        })
        return config

@register_keras_serializable()
class SegmentBranch(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = tf.keras.Sequential([
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        ])
        self.S4 = tf.keras.Sequential([
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        ])
        self.S5_4 = tf.keras.Sequential([
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        ])
        self.S5_5 = CEBlock()

    def call(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5
    

    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class BGALayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.left1 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Conv2D(128, 1, strides=1, padding='valid'),
        ])
        self.left2 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='same'),
        ])
        self.right1 = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
        ])
        self.right2 = tf.keras.Sequential([
            tf.keras.layers.DepthwiseConv2D(3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.Conv2D(128, 1, strides=1, padding='valid'),
        ])
        self.up1 = tf.keras.layers.UpSampling2D(size=4)
        self.up2 = tf.keras.layers.UpSampling2D(size=4)
        self.conv = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, 3, strides=1, padding='same'),
            tf.keras.layers.BatchNormalization(axis=1),
            tf.keras.layers.ReLU(),
        ])

    def call(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * tf.sigmoid(right1)
        right = left2 * tf.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out
    
    def get_config(self):
        config = super().get_config()
        return config

@register_keras_serializable()
class SegmentHead(tf.keras.layers.Layer):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super().__init__()
        self.in_chan = in_chan
        self.mid_chan = mid_chan
        self.n_classes = n_classes
        self.aux = aux
        self.up_factor = up_factor
        self.dropout_rate = 0.1 # 保留比率
        
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        # self.drop = tf.keras.layers.Dropout(0.1)
        self.drop = tf.keras.layers.SpatialDropout2D(
            0.1, 
            data_format='channels_first' # 强制 NCHW 格式
        )
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=2) if aux else tf.keras.layers.Lambda(lambda x: x),
            ConvBNReLU(mid_chan, mid_chan2, 3, stride=1) if aux else tf.keras.layers.Lambda(lambda x: x),
            tf.keras.layers.Conv2D(out_chan, 1, strides=1, padding='valid'),
            tf.keras.layers.UpSampling2D(size=up_factor, interpolation='bilinear'),
        ])
     

    def call0(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat
    
    def call(self, x, training=False):
        feat = self.conv(x)
        if training and self.dropout_rate > 0:
            feat_nhwc = tf.transpose(feat, perm=[0, 2, 3, 1])
            feat_nhwc_dropped = tf.nn.dropout(feat_nhwc, rate=self.dropout_rate)
            feat = tf.transpose(feat_nhwc_dropped, perm=[0, 3, 1, 2])
            # print("Using dropout in SegmentHead")
        
        # feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'in_chan': self.in_chan,
            'mid_chan': self.mid_chan,
            'n_classes': self.n_classes,
            'aux': self.aux,
            'up_factor': self.up_factor,
            "dropout_rate": self.dropout_rate,
        })
        return config

from line_profiler import profile

@register_keras_serializable()
class BiSeNetV2(tf.keras.Model):
    def __init__(self, n_classes, proj_dim=256, aux_mode='train',**kwargs):
        super().__init__(**kwargs) # ⬅️ Keras 元数据（name, trainable, dtype）在这里被处理
        # self.args = args
        self.proj_dim = proj_dim
        self.n_classes = n_classes
        self.aux_mode = aux_mode
        
        
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        # self.proj_head = ProjectionHead(dim_in=128, proj_dim=self.args.proj_dim)
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=self.proj_dim)

    def call(self, x, training=False):
        size = tf.shape(x)[-2:]  # H, W
        if self.aux_mode == 'eval':
            h_ = size[0]
            w_ = size[1]
            rem_h = h_ % 32
            rem_w = w_ % 32
            pad_h = tf.where(rem_h == 0, 0, 32 - rem_h)
            pad_w = tf.where(rem_w == 0, 0, 32 - rem_w)
            paddings = [[0, 0],[0, 0], [0, pad_h], [0, pad_w]]
            x = tf.pad(x, paddings, mode='REFLECT')
                    
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        emb = self.proj_head(feat_head)
        logits = self.head(feat_head, training=training)
        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, emb, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            logits = logits[:, :, :h_, :w_]
            return logits,
        elif self.aux_mode == 'pred':
            pred = tf.argmax(logits, axis=1)
            return pred
        else:
            raise NotImplementedError
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'proj_dim': self.proj_dim, 
            'n_classes': self.n_classes,
            'aux_mode': self.aux_mode,
        })
        return config

    @classmethod
    def from_config(cls, config):
        # 1. 弹出所有自定义参数
        proj_dim = config.pop('proj_dim')
        n_classes = config.pop('n_classes') 
        aux_mode = config.pop('aux_mode')

        
        # 3. 使用重建的自定义参数和剩余的 Keras 元数据 (即剩下的 **config) 重建实例
        return cls(proj_dim=proj_dim, 
                   n_classes=n_classes, 
                   aux_mode=aux_mode, 
                   **config) # ⬅️ 剩余的 config 包含 name, trainable, dtype 等
    

@register_keras_serializable()
class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp'):
        super().__init__()
        self.dim_in = dim_in   
        self.proj_dim = proj_dim
        self.proj = proj
        
        if proj == 'linear':
            self.proj = tf.keras.layers.Conv2D(proj_dim, 1)
        elif proj == 'convmlp':
            self.proj = tf.keras.Sequential([
                tf.keras.layers.Conv2D(dim_in, 1),
                tf.keras.layers.BatchNormalization(axis=1),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Conv2D(proj_dim, 1)
            ])
    def call(self, x):
        x = self.proj(x)
        x = tf.nn.l2_normalize(x, axis=1)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'dim_in': self.dim_in,
            'proj_dim': self.proj_dim,
            'proj': self.proj,
        })
        return config

# 测试代码部分已注释
if __name__ == "__main__":
    from argparse import Namespace
    from tqdm import tqdm
    args = Namespace()
    args.proj_dim = 256
    x = tf.random.normal([16, 3, 366, 500])
    # model = BiSeNetV2(args, n_classes=20, aux_mode='train')
    model = BiSeNetV2(args, n_classes=20, aux_mode='eval')
    for i in tqdm(range(10000)):
        out = model(x)
    print("Output shapes:")
    for o in out:
        print(o.shape)
    
    
    
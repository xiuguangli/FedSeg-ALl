import math

import paddle

backbone_url = (
    "https://github.com/CoinCheung/BiSeNet/releases/download/0.0.0/backbone_v2.pth"
)


class ConvBNReLU(paddle.nn.Layer):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super(ConvBNReLU, self).__init__()
        self.conv = paddle.nn.Conv2D(
            in_channels=in_chan,
            out_channels=out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
        )
        self.bn = paddle.nn.BatchNorm2D(num_features=out_chan)
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class UpSample(paddle.nn.Layer):
    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = paddle.nn.Conv2D(
            in_channels=n_chan,
            out_channels=out_chan,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.up = paddle.nn.PixelShuffle(upscale_factor=factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        init_XavierNormal = paddle.nn.initializer.XavierNormal(gain=1.0)
        init_XavierNormal(self.proj.weight)


class DetailBranch(paddle.nn.Layer):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = paddle.nn.Sequential(
            ConvBNReLU(3, 64, 3, stride=2), ConvBNReLU(64, 64, 3, stride=1)
        )
        self.S2 = paddle.nn.Sequential(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = paddle.nn.Sequential(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def forward(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(paddle.nn.Layer):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = paddle.nn.Sequential(
            ConvBNReLU(16, 8, 1, stride=1, padding=0), ConvBNReLU(8, 16, 3, stride=2)
        )
        self.right = paddle.nn.MaxPool2D(
            kernel_size=3, stride=2, padding=1, ceil_mode=False
        )
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def forward(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = paddle.concat(x=[feat_left, feat_right], axis=1)
        feat = self.fuse(feat)
        return feat


class CEBlock(paddle.nn.Layer):
    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = paddle.nn.BatchNorm2D(num_features=128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def forward(self, x):
        feat = paddle.mean(x=x, axis=(2, 3), keepdim=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat


class GELayerS1(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_chan,
                out_channels=mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=in_chan,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=mid_chan),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=mid_chan,
                out_channels=out_chan,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        feat = self.relu(feat)
        return feat


class GELayerS2(paddle.nn.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_chan,
                out_channels=mid_chan,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_chan,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=mid_chan),
        )
        self.dwconv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=mid_chan,
                out_channels=mid_chan,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=mid_chan,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=mid_chan),
            paddle.nn.ReLU(),
        )
        self.conv2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=mid_chan,
                out_channels=out_chan,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=in_chan,
                out_channels=in_chan,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=in_chan,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=in_chan),
            paddle.nn.Conv2D(
                in_channels=in_chan,
                out_channels=out_chan,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=out_chan),
        )
        self.relu = paddle.nn.ReLU()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(paddle.nn.Layer):
    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = paddle.nn.Sequential(GELayerS2(16, 32), GELayerS1(32, 32))
        self.S4 = paddle.nn.Sequential(GELayerS2(32, 64), GELayerS1(64, 64))
        self.S5_4 = paddle.nn.Sequential(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def forward(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(paddle.nn.Layer):
    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            ),
        )
        self.left2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.AvgPool2D(
                kernel_size=3, stride=2, padding=1, ceil_mode=False, exclusive=False
            ),
        )
        self.right1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=128),
        )
        self.right2 = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=128,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=False,
            ),
        )
        self.up1 = paddle.nn.Upsample(scale_factor=4)
        self.up2 = paddle.nn.Upsample(scale_factor=4)
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
                bias_attr=False,
            ),
            paddle.nn.BatchNorm2D(num_features=128),
            paddle.nn.ReLU(),
        )

    def forward(self, x_d, x_s):
        dsize = tuple(x_d.shape)[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * paddle.nn.functional.sigmoid(x=right1)
        right = left2 * paddle.nn.functional.sigmoid(x=right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out


class SegmentHead(paddle.nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = paddle.nn.Dropout(p=0.1)
        self.up_factor = up_factor
        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        self.conv_out = paddle.nn.Sequential(
            paddle.nn.Sequential(
                paddle.nn.Upsample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1),
            )
            if aux
            else paddle.nn.Identity(),
            paddle.nn.Conv2D(
                in_channels=mid_chan2,
                out_channels=out_chan,
                kernel_size=1,
                stride=1,
                padding=0,
                bias_attr=True,
            ),
            paddle.nn.Upsample(
                scale_factor=up_factor, mode="bilinear", align_corners=False
            ),
        )

    def forward(self, x):
        feat = self.conv(x)
        feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(paddle.nn.Layer):
    def __init__(self, args, n_classes, aux_mode="train"):
        super(BiSeNetV2, self).__init__()
        self.args = args
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == "train":
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=self.args.proj_dim)
        self.init_weights()

    def forward(self, x):
        size = tuple(x.shape)[2:]
        if self.aux_mode == "eval":
            h_, w_ = size
            if h_ % 32 != 0:
                new_h = math.ceil(h_ / 32) * 32
                pad_h = new_h - h_
            else:
                pad_h = 0
            if w_ % 32 != 0:
                new_w = math.ceil(w_ / 32) * 32
                pad_w = new_w - w_
            else:
                pad_w = 0
            x = paddle.nn.functional.pad(
                x=x, pad=(0, pad_w, 0, pad_h), mode="reflect", pad_from_left_axis=False
            )
        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        emb = self.proj_head(feat_head)
        logits = self.head(feat_head)
        if self.aux_mode == "train":
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return (logits, emb, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4)
        elif self.aux_mode == "eval":
            logits = logits[:, :, :h_, :w_]
            return (logits,)
        elif self.aux_mode == "pred":
            pred = logits.argmax(axis=1)
            return pred
        else:
            raise NotImplementedError

    def init_weights(self):
        for name, module in self.named_sublayers(include_self=True):
            if isinstance(module, (paddle.nn.Conv2D, paddle.nn.Linear)):
                init_KaimingNormal = paddle.nn.initializer.KaimingNormal(
                    mode="fan_out", nonlinearity="leaky_relu"
                )
                init_KaimingNormal(module.weight)
                if not module.bias is None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(module.bias)
            elif isinstance(module, paddle.nn.layer.norm._BatchNormBase):
                if hasattr(module, "last_bn") and module.last_bn:
                    init_Constant = paddle.nn.initializer.Constant(value=0.0)
                    init_Constant(module.weight)
                else:
                    init_Constant = paddle.nn.initializer.Constant(value=1.0)
                    init_Constant(module.weight)
                init_Constant = paddle.nn.initializer.Constant(value=0.0)
                init_Constant(module.bias)
        if not self.args.rand_init:
            self.load_pretrain()

    def load_pretrain(self):
        state = paddle.load(path=str("segmentation/myseg/backbone_v2.pth"))
        for name, child in self.named_children():
            if name in state.keys():
                child.set_state_dict(state_dict=state[name])

    def get_params(self):
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
            if "head" in name or "aux" in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params


class ProjectionHead(paddle.nn.Layer):
    def __init__(self, dim_in, proj_dim=256, proj="convmlp"):
        super(ProjectionHead, self).__init__()
        if proj == "linear":
            self.proj = paddle.nn.Conv2D(
                in_channels=dim_in, out_channels=proj_dim, kernel_size=1
            )
        elif proj == "convmlp":
            self.proj = paddle.nn.Sequential(
                paddle.nn.Conv2D(
                    in_channels=dim_in, out_channels=dim_in, kernel_size=1
                ),
                paddle.nn.BatchNorm2D(num_features=dim_in),
                paddle.nn.ReLU(),
                paddle.nn.Conv2D(
                    in_channels=dim_in, out_channels=proj_dim, kernel_size=1
                ),
            )

    def forward(self, x):
        return paddle.nn.functional.normalize(x=self.proj(x), p=2, axis=1)


if __name__ == "__main__":
    x = paddle.randn(shape=[16, 3, 1024, 2048])
    model = BiSeNetV2(n_classes=19)
    outs = model(x)
    for out in outs:
        print(tuple(out.shape))

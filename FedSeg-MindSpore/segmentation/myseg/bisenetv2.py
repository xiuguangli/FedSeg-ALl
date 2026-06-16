import math
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import HeNormal, XavierNormal, initializer

from logging_utils import logger
from mindspore import Parameter


BACKBONE_CKPT_PATH = Path(__file__).with_name("backbone_v2.ckpt")
L2_NORM_EPS = 1e-12

def fedseg_bn2d(num_features):
    return nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.9)


def fedseg_l2_normalize(axis):
    return ops.L2Normalize(axis=axis, epsilon=L2_NORM_EPS)


class ConvBNReLU(nn.Cell):
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
        super().__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            pad_mode="pad",
            padding=padding,
            dilation=dilation,
            group=groups,
            has_bias=bias,
        )
        self.bn = fedseg_bn2d(out_chan)
        self.relu = nn.ReLU()

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UpSample(nn.Cell):
    def __init__(self, n_chan, factor=2):
        super().__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(
            n_chan,
            out_chan,
            kernel_size=1,
            stride=1,
            pad_mode="pad",
            padding=0,
            has_bias=True,
        )
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def construct(self, x):
        return self.up(self.proj(x))

    def init_weight(self):
        weight = initializer(XavierNormal(gain=1.0), self.proj.weight.shape, self.proj.weight.dtype)
        self.proj.weight.set_data(weight)


class ResizeNearest(nn.Cell):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = int(scale_factor)

    def construct(self, x):
        return ops.interpolate(
            x,
            size=(x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor),
            mode="nearest",
        )


class ResizeBilinear(nn.Cell):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = int(scale_factor)

    def construct(self, x):
        return ops.interpolate(
            x,
            size=(x.shape[2] * self.scale_factor, x.shape[3] * self.scale_factor),
            mode="bilinear",
            align_corners=False,
        )


class DetailBranch(nn.Cell):
    def __init__(self):
        super().__init__()
        self.S1 = nn.SequentialCell(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.SequentialCell(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.SequentialCell(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def construct(self, x):
        x = self.S1(x)
        x = self.S2(x)
        x = self.S3(x)
        return x


class StemBlock(nn.Cell):
    def __init__(self):
        super().__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.SequentialCell(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="pad", padding=1)
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def construct(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = ops.cat((feat_left, feat_right), axis=1)
        return self.fuse(feat)


class CEBlock(nn.Cell):
    def __init__(self):
        super().__init__()
        self.bn = fedseg_bn2d(128)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def construct(self, x):
        feat = ops.mean(x, axis=(2, 3), keep_dims=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        return self.conv_last(feat)


class GELayerS1(nn.Cell):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.SequentialCell(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                group=in_chan,
                has_bias=False,
            ),
            fedseg_bn2d(mid_chan),
            nn.ReLU(),
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan,
                out_chan,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=False,
            ),
            fedseg_bn2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU()

    def construct(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = feat + x
        return self.relu(feat)


class GELayerS2(nn.Cell):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super().__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.SequentialCell(
            nn.Conv2d(
                in_chan,
                mid_chan,
                kernel_size=3,
                stride=2,
                pad_mode="pad",
                padding=1,
                group=in_chan,
                has_bias=False,
            ),
            fedseg_bn2d(mid_chan),
        )
        self.dwconv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan,
                mid_chan,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                group=mid_chan,
                has_bias=False,
            ),
            fedseg_bn2d(mid_chan),
            nn.ReLU(),
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan,
                out_chan,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=False,
            ),
            fedseg_bn2d(out_chan),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.SequentialCell(
            nn.Conv2d(
                in_chan,
                in_chan,
                kernel_size=3,
                stride=2,
                pad_mode="pad",
                padding=1,
                group=in_chan,
                has_bias=False,
            ),
            fedseg_bn2d(in_chan),
            nn.Conv2d(
                in_chan,
                out_chan,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=False,
            ),
            fedseg_bn2d(out_chan),
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        return self.relu(feat)


class SegmentBranch(nn.Cell):
    def __init__(self):
        super().__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.SequentialCell(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.SequentialCell(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.SequentialCell(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def construct(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Cell):
    def __init__(self):
        super().__init__()
        self.left1 = nn.SequentialCell(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                group=128,
                has_bias=False,
            ),
            fedseg_bn2d(128),
            nn.Conv2d(
                128,
                128,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=False,
            ),
        )
        self.left2 = nn.SequentialCell(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=2,
                pad_mode="pad",
                padding=1,
                has_bias=False,
            ),
            fedseg_bn2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, pad_mode="pad", padding=1),
        )
        self.right1 = nn.SequentialCell(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                has_bias=False,
            ),
            fedseg_bn2d(128),
        )
        self.right2 = nn.SequentialCell(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                group=128,
                has_bias=False,
            ),
            fedseg_bn2d(128),
            nn.Conv2d(
                128,
                128,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=False,
            ),
        )
        self.up1 = ResizeNearest(4)
        self.up2 = ResizeNearest(4)
        self.conv = nn.SequentialCell(
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                stride=1,
                pad_mode="pad",
                padding=1,
                has_bias=False,
            ),
            fedseg_bn2d(128),
            nn.ReLU(),
        )

    def construct(self, x_d, x_s):
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)
        left = left1 * ops.sigmoid(right1)
        right = left2 * ops.sigmoid(right2)
        right = self.up2(right)
        return self.conv(left + right)


class SegmentHead(nn.Cell):
    def __init__(self, args, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super().__init__()
        self.args = args
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = nn.Identity() if getattr(args, "debug_disable_dropout", False) else nn.Dropout(p=0.1)
        self.dropout_p = 0.1
        self.aux = aux

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        out_scale = up_factor // 2 if aux else up_factor
        self.conv_out = nn.SequentialCell(
            nn.SequentialCell(
                ResizeNearest(2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1),
            ) if aux else nn.Identity(),
            nn.Conv2d(
                mid_chan2,
                out_chan,
                kernel_size=1,
                stride=1,
                pad_mode="pad",
                padding=0,
                has_bias=True,
            ),
            ResizeBilinear(out_scale),
        )

    def _deterministic_dropout(self, feat):
        if not self.training:
            return feat
        keep_prob = 1.0 - self.dropout_p
        n, c, h, w = feat.shape
        n_idx = ops.reshape(ops.arange(0, n, 1), (n, 1, 1, 1))
        c_idx = ops.reshape(ops.arange(0, c, 1), (1, c, 1, 1))
        h_idx = ops.reshape(ops.arange(0, h, 1), (1, 1, h, 1))
        w_idx = ops.reshape(ops.arange(0, w, 1), (1, 1, 1, w))
        flat_idx = (((n_idx * c + c_idx) * h + h_idx) * w + w_idx).astype(ms.int64)
        seed = int(getattr(self.args, "seed", 1))
        hashed = ops.floor_mod(flat_idx * 1103515245 + 12345 + seed, 2147483647)
        mask = (hashed.astype(ms.float32) / 2147483647.0) < keep_prob
        mask = mask.astype(feat.dtype)
        return feat * mask / keep_prob

    def construct(self, x):
        feat = self.conv(x)
        if getattr(self.args, "debug_deterministic_dropout", False) and not getattr(self.args, "debug_disable_dropout", False):
            feat = self._deterministic_dropout(feat)
        else:
            feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class ProjectionHead(nn.Cell):
    def __init__(self, dim_in, proj_dim=256, proj="convmlp"):
        super().__init__()
        if proj == "linear":
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1, has_bias=True)
        else:
            self.proj = nn.SequentialCell(
                nn.Conv2d(dim_in, dim_in, kernel_size=1, has_bias=True),
                fedseg_bn2d(dim_in),
                nn.ReLU(),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1, has_bias=True),
            )
        self.normalize = fedseg_l2_normalize(axis=1)

    def construct(self, x):
        return self.normalize(self.proj(x))


class BiSeNetV2(nn.Cell):
    def __init__(self, args, n_classes, aux_mode="train"):
        super().__init__()
        self.args = args
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        self.head = SegmentHead(args, 128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == "train":
            self.aux2 = SegmentHead(args, 16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(args, 32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(args, 64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(args, 128, 128, n_classes, up_factor=32)
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=args.proj_dim)

        self.init_weights()

    def construct(self, x):
        size = x.shape[2:]
        if self.aux_mode == "eval":
            h, w = size
            pad_h = math.ceil(h / 32) * 32 - h if h % 32 != 0 else 0
            pad_w = math.ceil(w / 32) * 32 - w if w % 32 != 0 else 0
            if pad_h or pad_w:
                x = ops.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            h, w = size

        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        logits = self.head(feat_head)

        if self.aux_mode == "train":
            emb = self.proj_head(feat_head)
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, emb, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        if self.aux_mode == "eval":
            logits = logits[:, :, :h, :w]
            return (logits,)
        if self.aux_mode == "pred":
            return logits.argmax(axis=1)
        raise NotImplementedError

    def init_weights(self):
        for _, module in self.cells_and_names():
            if isinstance(module, (nn.Conv2d, nn.Dense)):
                if hasattr(module, "weight") and module.weight is not None:
                    weight = initializer(HeNormal(mode="fan_out", nonlinearity="relu"), module.weight.shape, module.weight.dtype)
                    module.weight.set_data(weight)
                if hasattr(module, "bias") and module.bias is not None:
                    module.bias.set_data(ops.zeros_like(module.bias))
            elif isinstance(module, nn.BatchNorm2d):
                if getattr(module, "last_bn", False):
                    module.gamma.set_data(ops.zeros_like(module.gamma))
                else:
                    module.gamma.set_data(ops.ones_like(module.gamma))
                module.beta.set_data(ops.zeros_like(module.beta))
        if not getattr(self.args, "rand_init", False):
            self.load_pretrain()

    def load_pretrain(self, checkpoint_path=None):
        checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else BACKBONE_CKPT_PATH
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Backbone checkpoint not found: {}. Please place a MindSpore backbone_v2.ckpt at this path first.".format(
                    checkpoint_path
                )
            )

        raw_state = ms.load_checkpoint(str(checkpoint_path))
        load_summary = {}
        total_loaded = 0
        for top_name in ["detail", "segment", "bga"]:
            child = getattr(self, top_name)
            child_state = {}
            prefix = top_name + "."
            for name, value in raw_state.items():
                if not name.startswith(prefix):
                    continue
                tensor = value.data if hasattr(value, "data") else value
                child_state[name] = Parameter(tensor, name=name)
            missing, unexpected = ms.load_param_into_net(child, child_state, strict_load=False)
            load_summary[top_name] = {
                "loaded": len(child_state),
                "missing": list(missing),
                "unexpected": list(unexpected),
            }
            total_loaded += len(child_state)
            if missing or unexpected:
                raise RuntimeError(
                    "Backbone preload mismatch for {}: missing={}, unexpected={}".format(
                        top_name, list(missing), list(unexpected)
                    )
                )
        logger.info(
            "Loaded MindSpore backbone pretrain from {}: loaded={}, detail={}, segment={}, bga={}",
            checkpoint_path,
            total_loaded,
            load_summary["detail"]["loaded"],
            load_summary["segment"]["loaded"],
            load_summary["bga"]["loaded"],
        )
        return {
            "loaded": total_loaded,
            "summary": load_summary,
        }

    def get_params(self):
        def add_param_to_list(cell, wd_params, nowd_params):
            for param in cell.get_parameters():
                if not getattr(param, "requires_grad", True):
                    continue
                if param.ndim == 1:
                    nowd_params.append(param)
                elif param.ndim == 4:
                    wd_params.append(param)

        wd_params, nowd_params = [], []
        lr_mul_wd_params, lr_mul_nowd_params = [], []
        for name, child in self.name_cells().items():
            # Match the original torch optimizer grouping: any module name that
            # contains "head" (including proj_head) or "aux" uses the 10x lr group.
            if ("head" in name) or ("aux" in name):
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            elif name:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

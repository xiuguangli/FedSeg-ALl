import os
import math

import tensorflow as tf
from keras.saving import register_keras_serializable


def _use_native_nchw() -> bool:
    override = os.environ.get("FEDSEG_FORCE_NHWC_CONV")
    if override is not None and override.strip().lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        return bool(tf.config.get_visible_devices("GPU"))
    except Exception:
        return False


def _make_bn():
    return TorchBatchNorm2D()


def _normalize_pair(value):
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise TypeError(f"unsupported pair value: {value!r}")


def _pad_nchw(x, padding, constant_values=0.0):
    pad_h, pad_w = _normalize_pair(padding)
    if pad_h == 0 and pad_w == 0:
        return x
    return tf.pad(x, [[0, 0], [0, 0], [pad_h, pad_h], [pad_w, pad_w]], constant_values=constant_values)


@register_keras_serializable()
class TorchBatchNorm2D(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, momentum=0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.momentum = momentum

    def build(self, input_shape):
        channels = int(input_shape[1])
        self.gamma = self.add_weight(
            name="gamma",
            shape=(channels,),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            name="beta",
            shape=(channels,),
            initializer="zeros",
            trainable=True,
        )
        self.moving_mean = self.add_weight(
            name="moving_mean",
            shape=(channels,),
            initializer="zeros",
            trainable=False,
        )
        self.moving_variance = self.add_weight(
            name="moving_variance",
            shape=(channels,),
            initializer="ones",
            trainable=False,
        )
        super().build(input_shape)

    def _normalize(self, x, mean, var):
        mean = tf.reshape(mean, [1, -1, 1, 1])
        var = tf.reshape(var, [1, -1, 1, 1])
        gamma = tf.reshape(self.gamma, [1, -1, 1, 1])
        beta = tf.reshape(self.beta, [1, -1, 1, 1])
        x = (x - mean) / tf.sqrt(var + self.epsilon)
        return x * gamma + beta

    def _train_call(self, x):
        mean = tf.reduce_mean(x, axis=[0, 2, 3])
        var = tf.math.reduce_variance(x, axis=[0, 2, 3])

        sample_count = tf.cast(tf.shape(x)[0] * tf.shape(x)[2] * tf.shape(x)[3], x.dtype)
        unbiased_var = tf.where(
            sample_count > 1,
            var * sample_count / (sample_count - 1.0),
            var,
        )

        self.moving_mean.assign((1.0 - self.momentum) * self.moving_mean + self.momentum * mean)
        self.moving_variance.assign((1.0 - self.momentum) * self.moving_variance + self.momentum * unbiased_var)
        return self._normalize(x, mean, var)

    def _infer_call(self, x):
        if _use_native_nchw():
            y, _, _ = tf.compat.v1.nn.fused_batch_norm(
                x,
                self.gamma,
                self.beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                data_format="NCHW",
                is_training=False,
            )
            return y
        return self._normalize(x, self.moving_mean, self.moving_variance)

    def call(self, x, training=False):
        if training is None:
            training = False
        if isinstance(training, bool):
            return self._train_call(x) if training else self._infer_call(x)
        return tf.cond(tf.cast(training, tf.bool), lambda: self._train_call(x), lambda: self._infer_call(x))

    def get_config(self):
        config = super().get_config()
        config.update({"epsilon": self.epsilon, "momentum": self.momentum})
        return config


@register_keras_serializable()
class TorchConv2D(tf.keras.layers.Layer):
    def __init__(
        self,
        filters,
        kernel_size,
        strides=1,
        padding=0,
        dilation_rate=1,
        groups=1,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.dilation_rate = dilation_rate
        self.groups = groups
        self.use_bias = use_bias
        self.native_nchw = _use_native_nchw()
        self._kernel_size = _normalize_pair(kernel_size)
        self._strides = _normalize_pair(strides)
        self._dilation_rate = _normalize_pair(dilation_rate)
        self._in_channels = None
        self._depthwise_multiplier = None
        self.kernel_var = None
        self.bias_var = None

    @property
    def kernel(self):
        return self.kernel_var

    @property
    def bias(self):
        return self.bias_var

    def build(self, input_shape):
        in_channels = int(input_shape[1])
        if self.groups not in {1, in_channels}:
            raise ValueError(f"unsupported groups={self.groups} for input channels={in_channels}")

        self._in_channels = in_channels
        if self.groups == 1:
            kernel_shape = (*self._kernel_size, in_channels, self.filters)
        else:
            if self.filters % in_channels != 0:
                raise ValueError(
                    f"depthwise filters must be divisible by input channels: filters={self.filters}, in_channels={in_channels}"
                )
            self._depthwise_multiplier = self.filters // in_channels
            # Keep the same weight layout Keras grouped Conv2D expects so checkpoint conversion stays simple.
            kernel_shape = (*self._kernel_size, 1, self.filters)

        self.kernel_var = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer="glorot_uniform",
            trainable=True,
        )
        if self.use_bias:
            self.bias_var = self.add_weight(
                name="bias",
                shape=(self.filters,),
                initializer="zeros",
                trainable=True,
            )
        super().build(input_shape)

    def _conv2d(self, x):
        if self.native_nchw:
            return tf.nn.conv2d(
                x,
                self.kernel_var,
                strides=[1, 1, self._strides[0], self._strides[1]],
                padding="VALID",
                data_format="NCHW",
                dilations=[1, 1, self._dilation_rate[0], self._dilation_rate[1]],
            )
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.conv2d(
            x,
            self.kernel_var,
            strides=[1, self._strides[0], self._strides[1], 1],
            padding="VALID",
            data_format="NHWC",
            dilations=[1, self._dilation_rate[0], self._dilation_rate[1], 1],
        )
        return tf.transpose(x, [0, 3, 1, 2])

    def _depthwise_conv2d(self, x):
        kernel = tf.reshape(
            self.kernel_var,
            [self._kernel_size[0], self._kernel_size[1], self._in_channels, self._depthwise_multiplier],
        )
        if self.native_nchw:
            return tf.nn.depthwise_conv2d(
                x,
                kernel,
                strides=[1, 1, self._strides[0], self._strides[1]],
                padding="VALID",
                data_format="NCHW",
                dilations=[self._dilation_rate[0], self._dilation_rate[1]],
            )
        x = tf.transpose(x, [0, 2, 3, 1])
        x = tf.nn.depthwise_conv2d(
            x,
            kernel,
            strides=[1, self._strides[0], self._strides[1], 1],
            padding="VALID",
            data_format="NHWC",
            dilations=[self._dilation_rate[0], self._dilation_rate[1]],
        )
        return tf.transpose(x, [0, 3, 1, 2])

    def call(self, x):
        x = _pad_nchw(x, self.padding)
        x = self._conv2d(x) if self.groups == 1 else self._depthwise_conv2d(x)
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias_var, data_format="NCHW")
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "padding": self.padding,
                "dilation_rate": self.dilation_rate,
                "groups": self.groups,
                "use_bias": self.use_bias,
            }
        )
        return config


def _resize_bilinear_nchw(x, scale=None, size=None):
    x_nhwc = tf.transpose(x, [0, 2, 3, 1])
    if size is None:
        shape = tf.shape(x_nhwc)
        size = [shape[1] * scale, shape[2] * scale]
    y = tf.raw_ops.ResizeBilinear(
        images=x_nhwc,
        size=tf.cast(size, tf.int32),
        align_corners=False,
        half_pixel_centers=True,
    )
    return tf.transpose(y, [0, 3, 1, 2])


def _resize_nearest_nchw(x, size):
    x_nhwc = tf.transpose(x, [0, 2, 3, 1])
    y = tf.raw_ops.ResizeNearestNeighbor(
        images=x_nhwc,
        size=tf.cast(size, tf.int32),
        align_corners=False,
        half_pixel_centers=False,
    )
    return tf.transpose(y, [0, 3, 1, 2])


def _avg_pool2d_nchw(x, ksize, strides, padding):
    if _use_native_nchw():
        if isinstance(padding, str):
            return tf.nn.avg_pool2d(x, ksize=ksize, strides=strides, padding=padding, data_format="NCHW")
        return tf.nn.avg_pool2d(_pad_nchw(x, padding), ksize=ksize, strides=strides, padding="VALID", data_format="NCHW")

    if isinstance(padding, str):
        x_nhwc = tf.transpose(x, [0, 2, 3, 1])
        padding_mode = padding
    else:
        x_nhwc = tf.transpose(_pad_nchw(x, padding), [0, 2, 3, 1])
        padding_mode = "VALID"
    y = tf.nn.avg_pool2d(x_nhwc, ksize=ksize, strides=strides, padding=padding_mode, data_format="NHWC")
    return tf.transpose(y, [0, 3, 1, 2])


def _max_pool2d_nchw(x, ksize, strides, padding):
    if _use_native_nchw():
        if isinstance(padding, str):
            return tf.nn.max_pool2d(x, ksize=ksize, strides=strides, padding=padding, data_format="NCHW")
        padded = _pad_nchw(x, padding, constant_values=tf.cast(tf.float32.min, x.dtype))
        return tf.nn.max_pool2d(padded, ksize=ksize, strides=strides, padding="VALID", data_format="NCHW")

    if isinstance(padding, str):
        x_nhwc = tf.transpose(x, [0, 2, 3, 1])
        padding_mode = padding
    else:
        x_nhwc = tf.transpose(_pad_nchw(x, padding, constant_values=tf.cast(tf.float32.min, x.dtype)), [0, 2, 3, 1])
        padding_mode = "VALID"
    y = tf.nn.max_pool2d(x_nhwc, ksize=ksize, strides=strides, padding=padding_mode, data_format="NHWC")
    return tf.transpose(y, [0, 3, 1, 2])


@register_keras_serializable()
class ConvBNReLU(tf.keras.layers.Layer):
    def __init__(
        self,
        in_chan,
        out_chan,
        ks=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        use_bias=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.use_bias = use_bias
        self.conv = TorchConv2D(
            filters=out_chan,
            kernel_size=ks,
            strides=stride,
            padding=padding,
            dilation_rate=dilation,
            groups=groups,
            use_bias=use_bias,
        )
        self.bn = _make_bn()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.bn(x, training=training)
        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chan": self.in_chan,
                "out_chan": self.out_chan,
                "ks": self.ks,
                "stride": self.stride,
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups,
                "use_bias": self.use_bias,
            }
        )
        return config


@register_keras_serializable()
class UpSample(tf.keras.layers.Layer):
    def __init__(self, n_chan, factor=2, **kwargs):
        super().__init__(**kwargs)
        self.n_chan = n_chan
        self.factor = factor
        out_chan = n_chan * factor * factor
        self.proj = TorchConv2D(
            out_chan,
            kernel_size=1,
            strides=1,
            padding=0,
            use_bias=True,
        )

    def call(self, x):
        x_nhwc = tf.transpose(self.proj(x), [0, 2, 3, 1])
        x_nhwc = tf.nn.depth_to_space(x_nhwc, self.factor)
        return tf.transpose(x_nhwc, [0, 3, 1, 2])

    def get_config(self):
        config = super().get_config()
        config.update({"n_chan": self.n_chan, "factor": self.factor})
        return config


@register_keras_serializable()
class DetailBranch(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.S1 = tf.keras.Sequential(
            [ConvBNReLU(3, 64, 3, stride=2), ConvBNReLU(64, 64, 3, stride=1)],
            name="S1",
        )
        self.S2 = tf.keras.Sequential(
            [
                ConvBNReLU(64, 64, 3, stride=2),
                ConvBNReLU(64, 64, 3, stride=1),
                ConvBNReLU(64, 64, 3, stride=1),
            ],
            name="S2",
        )
        self.S3 = tf.keras.Sequential(
            [
                ConvBNReLU(64, 128, 3, stride=2),
                ConvBNReLU(128, 128, 3, stride=1),
                ConvBNReLU(128, 128, 3, stride=1),
            ],
            name="S3",
        )

    def call(self, x, training=False):
        x = self.S1(x, training=training)
        x = self.S2(x, training=training)
        return self.S3(x, training=training)


@register_keras_serializable()
class StemBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = tf.keras.Sequential(
            [
                ConvBNReLU(16, 8, 1, stride=1, padding=0),
                ConvBNReLU(8, 16, 3, stride=2),
            ],
            name="left",
        )
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def call(self, x, training=False):
        feat = self.conv(x, training=training)
        feat_left = self.left(feat, training=training)
        feat_right = _max_pool2d_nchw(feat, ksize=3, strides=2, padding=1)
        feat = tf.concat([feat_left, feat_right], axis=1)
        return self.fuse(feat, training=training)


@register_keras_serializable()
class CEBlock(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bn = _make_bn()
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def call(self, x, training=False):
        feat = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        feat = self.bn(feat, training=training)
        feat = self.conv_gap(feat, training=training)
        feat = feat + x
        return self.conv_last(feat, training=training)


@register_keras_serializable()
class GELayerS1(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6, **kwargs):
        super().__init__(**kwargs)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.exp_ratio = exp_ratio
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = tf.keras.Sequential(
            [
                TorchConv2D(
                    mid_chan,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    groups=in_chan,
                    use_bias=False,
                ),
                _make_bn(),
                tf.keras.layers.ReLU(),
            ],
            name="dwconv",
        )
        self.conv2_conv = TorchConv2D(
            out_chan,
            kernel_size=1,
            strides=1,
            padding=0,
            use_bias=False,
        )
        self.conv2_bn = _make_bn()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        feat = self.conv1(x, training=training)
        feat = self.dwconv(feat, training=training)
        feat = self.conv2_conv(feat)
        feat = self.conv2_bn(feat, training=training)
        feat = feat + x
        return self.relu(feat)


@register_keras_serializable()
class GELayerS2(tf.keras.layers.Layer):
    def __init__(self, in_chan, out_chan, exp_ratio=6, **kwargs):
        super().__init__(**kwargs)
        self.in_chan = in_chan
        self.out_chan = out_chan
        self.exp_ratio = exp_ratio
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = tf.keras.Sequential(
            [
                TorchConv2D(
                    mid_chan,
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    groups=in_chan,
                    use_bias=False,
                ),
                _make_bn(),
            ],
            name="dwconv1",
        )
        self.dwconv2 = tf.keras.Sequential(
            [
                TorchConv2D(
                    mid_chan,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    groups=mid_chan,
                    use_bias=False,
                ),
                _make_bn(),
                tf.keras.layers.ReLU(),
            ],
            name="dwconv2",
        )
        self.conv2_conv = TorchConv2D(
            out_chan,
            kernel_size=1,
            strides=1,
            padding=0,
            use_bias=False,
        )
        self.conv2_bn = _make_bn()
        self.shortcut = tf.keras.Sequential(
            [
                TorchConv2D(
                    in_chan,
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    groups=in_chan,
                    use_bias=False,
                ),
                _make_bn(),
                TorchConv2D(
                    out_chan,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    use_bias=False,
                ),
                _make_bn(),
            ],
            name="shortcut",
        )
        self.relu = tf.keras.layers.ReLU()

    def call(self, x, training=False):
        feat = self.conv1(x, training=training)
        feat = self.dwconv1(feat, training=training)
        feat = self.dwconv2(feat, training=training)
        feat = self.conv2_conv(feat)
        feat = self.conv2_bn(feat, training=training)
        shortcut = self.shortcut(x, training=training)
        feat = feat + shortcut
        return self.relu(feat)


@register_keras_serializable()
class SegmentBranch(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.S1S2 = StemBlock()
        self.S3 = tf.keras.Sequential([GELayerS2(16, 32), GELayerS1(32, 32)], name="S3")
        self.S4 = tf.keras.Sequential([GELayerS2(32, 64), GELayerS1(64, 64)], name="S4")
        self.S5_4 = tf.keras.Sequential(
            [GELayerS2(64, 128), GELayerS1(128, 128), GELayerS1(128, 128), GELayerS1(128, 128)],
            name="S5_4",
        )
        self.S5_5 = CEBlock()

    def call(self, x, training=False):
        feat2 = self.S1S2(x, training=training)
        feat3 = self.S3(feat2, training=training)
        feat4 = self.S4(feat3, training=training)
        feat5_4 = self.S5_4(feat4, training=training)
        feat5_5 = self.S5_5(feat5_4, training=training)
        return feat2, feat3, feat4, feat5_4, feat5_5


@register_keras_serializable()
class BGALayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.left1 = tf.keras.Sequential(
            [
                TorchConv2D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    groups=128,
                    use_bias=False,
                ),
                _make_bn(),
                TorchConv2D(
                    128,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    use_bias=False,
                ),
            ],
            name="left1",
        )
        self.left2 = tf.keras.Sequential(
            [
                TorchConv2D(
                    128,
                    kernel_size=3,
                    strides=2,
                    padding=1,
                    use_bias=False,
                ),
                _make_bn(),
            ],
            name="left2",
        )
        self.right1 = tf.keras.Sequential(
            [
                TorchConv2D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    use_bias=False,
                ),
                _make_bn(),
            ],
            name="right1",
        )
        self.right2 = tf.keras.Sequential(
            [
                TorchConv2D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    groups=128,
                    use_bias=False,
                ),
                _make_bn(),
                TorchConv2D(
                    128,
                    kernel_size=1,
                    strides=1,
                    padding=0,
                    use_bias=False,
                ),
            ],
            name="right2",
        )
        self.conv = tf.keras.Sequential(
            [
                TorchConv2D(
                    128,
                    kernel_size=3,
                    strides=1,
                    padding=1,
                    use_bias=False,
                ),
                _make_bn(),
                tf.keras.layers.ReLU(),
            ],
            name="conv",
        )

    def call(self, x_d, x_s, training=False):
        left1 = self.left1(x_d, training=training)
        left2 = self.left2(x_d, training=training)
        left2 = _avg_pool2d_nchw(left2, ksize=3, strides=2, padding=1)
        right1 = self.right1(x_s, training=training)
        right2 = self.right2(x_s, training=training)
        right1 = _resize_nearest_nchw(right1, size=tf.shape(left1)[2:])
        left = left1 * tf.sigmoid(right1)
        right = left2 * tf.sigmoid(right2)
        right = _resize_nearest_nchw(right, size=tf.shape(left)[2:])
        return self.conv(left + right, training=training)


@register_keras_serializable()
class SegmentHead(tf.keras.layers.Layer):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True, **kwargs):
        super().__init__(**kwargs)
        self.in_chan = in_chan
        self.mid_chan = mid_chan
        self.n_classes = n_classes
        self.up_factor = up_factor
        self.aux = aux
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        self.drop = tf.keras.layers.Dropout(0.1)
        out_chan = n_classes
        self.mid_chan2 = up_factor * up_factor if aux else mid_chan
        self.final_up_factor = up_factor // 2 if aux else up_factor
        if aux:
            self.aux_pre = ConvBNReLU(mid_chan, self.mid_chan2, 3, stride=1)
        else:
            self.aux_pre = None
        self.conv_out = TorchConv2D(
            out_chan,
            kernel_size=1,
            strides=1,
            padding=0,
            use_bias=True,
        )

    def call(self, x, training=False):
        feat = self.conv(x, training=training)
        feat = self.drop(feat, training=training)
        if self.aux_pre is not None:
            feat = _resize_nearest_nchw(feat, size=[tf.shape(feat)[2] * 2, tf.shape(feat)[3] * 2])
            feat = self.aux_pre(feat, training=training)
        feat = self.conv_out(feat)
        return _resize_bilinear_nchw(feat, scale=self.final_up_factor)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_chan": self.in_chan,
                "mid_chan": self.mid_chan,
                "n_classes": self.n_classes,
                "up_factor": self.up_factor,
                "aux": self.aux,
            }
        )
        return config


@register_keras_serializable()
class ProjectionHead(tf.keras.layers.Layer):
    def __init__(self, dim_in, proj_dim=256, proj="convmlp", **kwargs):
        super().__init__(**kwargs)
        self.dim_in = dim_in
        self.proj_dim = proj_dim
        self.proj_kind = proj
        if proj == "linear":
            self.proj = TorchConv2D(proj_dim, kernel_size=1, strides=1, padding=0, use_bias=True)
        else:
            self.proj = tf.keras.Sequential(
                [
                    TorchConv2D(dim_in, kernel_size=1, strides=1, padding=0, use_bias=True),
                    _make_bn(),
                    tf.keras.layers.ReLU(),
                    TorchConv2D(proj_dim, kernel_size=1, strides=1, padding=0, use_bias=True),
                ],
                name="proj",
            )

    def call(self, x, training=False):
        x = self.proj(x, training=training) if isinstance(self.proj, tf.keras.Sequential) else self.proj(x)
        return tf.nn.l2_normalize(x, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"dim_in": self.dim_in, "proj_dim": self.proj_dim, "proj": self.proj_kind})
        return config


@register_keras_serializable()
class BiSeNetV2(tf.keras.Model):
    def __init__(self, n_classes, proj_dim=256, aux_mode="train", **kwargs):
        super().__init__(**kwargs)
        self.n_classes = n_classes
        self.proj_dim = proj_dim
        self.aux_mode = aux_mode
        self.detail = DetailBranch(name="detail")
        self.segment = SegmentBranch(name="segment")
        self.bga = BGALayer(name="bga")
        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False, name="head")
        if self.aux_mode == "train":
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4, name="aux2")
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8, name="aux3")
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16, name="aux4")
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32, name="aux5_4")
        self.proj_head = ProjectionHead(dim_in=128, proj_dim=proj_dim, name="proj_head")

    def call(self, x, training=False):
        spatial_size = tf.shape(x)[2:]
        if self.aux_mode == "eval":
            h_ = spatial_size[0]
            w_ = spatial_size[1]
            h_f = tf.cast(h_, tf.float32)
            w_f = tf.cast(w_, tf.float32)
            pad_h = tf.where(tf.equal(h_ % 32, 0), 0, tf.cast(tf.math.ceil(h_f / 32.0), tf.int32) * 32 - h_)
            pad_w = tf.where(tf.equal(w_ % 32, 0), 0, tf.cast(tf.math.ceil(w_f / 32.0), tf.int32) * 32 - w_)
            x = tf.pad(x, [[0, 0], [0, 0], [0, pad_h], [0, pad_w]], mode="REFLECT")
        feat_d = self.detail(x, training=training)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x, training=training)
        feat_head = self.bga(feat_d, feat_s, training=training)
        emb = self.proj_head(feat_head, training=training)
        logits = self.head(feat_head, training=training)
        if self.aux_mode == "train":
            logits_aux2 = self.aux2(feat2, training=training)
            logits_aux3 = self.aux3(feat3, training=training)
            logits_aux4 = self.aux4(feat4, training=training)
            logits_aux5_4 = self.aux5_4(feat5_4, training=training)
            return logits, emb, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        if self.aux_mode == "eval":
            return (logits[:, :, : spatial_size[0], : spatial_size[1]],)
        if self.aux_mode == "pred":
            return tf.argmax(logits, axis=1)
        raise NotImplementedError(self.aux_mode)

    def get_config(self):
        config = super().get_config()
        config.update({"n_classes": self.n_classes, "proj_dim": self.proj_dim, "aux_mode": self.aux_mode})
        return config

    def get_params(self):
        def add_var_to_list(var, wd_params, nowd_params):
            if len(var.shape) == 1:
                nowd_params.append(var)
            elif len(var.shape) in {2, 4}:
                wd_params.append(var)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name in ["detail", "segment", "bga", "head", "aux2", "aux3", "aux4", "aux5_4", "proj_head"]:
            if not hasattr(self, name):
                continue
            child = getattr(self, name)
            target_wd = lr_mul_wd_params if ("head" in name or "aux" in name) else wd_params
            target_nowd = lr_mul_nowd_params if ("head" in name or "aux" in name) else nowd_params
            for var in child.trainable_variables:
                add_var_to_list(var, target_wd, target_nowd)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params

    @classmethod
    def from_config(cls, config):
        return cls(**config)

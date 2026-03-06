import random
import math
import numpy as np
import cv2
# import torch
# from torchvision.transforms import Compose
import tensorflow as tf

def cv2_get_image(impth, lbpth):
    image = cv2.imread(impth)[:, :, ::-1].copy()
    label = cv2.imread(lbpth, 0)
    return image, label

class ComposeTF(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

class TransformationTrain(object):
    """
    Feddrive: We apply a standard data augmentation pipeline

    1、对图像进行随机缩放和随机裁剪
    scale the images in the range (0.5,1.5) for Cityscapes
    then extract a crop with size 512×1024 for Cityscapes

    2、随机翻转图像 (not implemented)
    We point out that when the clients do not have enough samples to
    use 16 as batch size, we virtually double their dataset by flipping the images horizontally.
    """

    def __init__(self, scales, cropsize):
        self.trans_func = ComposeTF([
            RandomResizedCrop(scales, cropsize),
            RandomHorizontalFlip()  # p=0.5
        ])

    def __call__(self, im_lb):
        im_lb = self.trans_func(im_lb)
        return im_lb

class TransformationVal(object):

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        return dict(im=im, lb=lb)

class RandomResizedCrop(object):
    '''
    size should be a tuple of (H, W)
    '''

    def __init__(self, scales=(0.5, 1.), size=(384, 384)):
        self.scales = scales
        self.size = size

    def __call__(self, im_lb):
        if self.size is None:
            return im_lb

        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]

        crop_h, crop_w = self.size
        scale = np.random.uniform(min(self.scales), max(self.scales))
        im_h, im_w = [math.ceil(el * scale) for el in im.shape[:2]]
        im = cv2.resize(im, (im_w, im_h))
        lb = cv2.resize(lb, (im_w, im_h), interpolation=cv2.INTER_NEAREST)

        if (im_h, im_w) == (crop_h, crop_w): return dict(im=im, lb=lb)
        pad_h, pad_w = 0, 0
        if im_h < crop_h:
            pad_h = (crop_h - im_h) // 2 + 1
        if im_w < crop_w:
            pad_w = (crop_w - im_w) // 2 + 1
        if pad_h > 0 or pad_w > 0:
            im = np.pad(im, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
            lb = np.pad(lb, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=255)

        im_h, im_w, _ = im.shape
        sh, sw = np.random.random(2)
        sh, sw = int(sh * (im_h - crop_h)), int(sw * (im_w - crop_w))
        return dict(
            im=im[sh:sh + crop_h, sw:sw + crop_w, :].copy(),
            lb=lb[sh:sh + crop_h, sw:sw + crop_w].copy()
        )

class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im_lb):
        if np.random.random() < self.p:
            return im_lb
        im, lb = im_lb['im'], im_lb['lb']
        assert im.shape[:2] == lb.shape[:2]
        return dict(
            im=im[:, ::-1, :],
            lb=lb[:, ::-1],
        )

class ToTensor(object):
    '''
    mean and std should be of the channel order 'bgr'
    '''

    def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, im_lb):
        im, lb = im_lb['im'], im_lb['lb']
        im = im.transpose(2, 0, 1).astype(np.float32)
        # im = torch.from_numpy(im).div_(255)
        im = tf.convert_to_tensor(im, dtype=tf.float32) / 255.0
        mean = tf.constant(self.mean, dtype=tf.float32)[:, None, None]
        std = tf.constant(self.std, dtype=tf.float32)[:, None, None]
        im = (im - mean) / std
        im = tf.identity(im)
        if lb is not None:
            lb = tf.convert_to_tensor(lb.astype(np.int64).copy())
            lb = tf.identity(lb)
        return dict(im=im, lb=lb)



# import tensorflow as tf
# import math
# import random
# # 确保在 TF 模式下操作，通常在 tf.data.Dataset.map 中已经是
# # Eager Execution 环境下，tf.random.* 可以直接使用

# # **注意**：
# # 1. TF 图像默认是 HWC (高, 宽, 通道) 顺序。
# # 2. TF 的随机操作是可图化的（Graph-compatible）。

# class ComposeTF(object):
#     """
#     TF 版本的 ComposeTF
#     """
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, x):
#         for t in self.transforms:
#             # 这里的 x 始终是 {im: Tensor, lb: Tensor} 字典
#             x = t(x)
#         return x

# class TransformationTrain(object):
#     """
#     TF 版本的 TransformationTrain
#     """
#     def __init__(self, scales, cropsize):
#         self.trans_func = ComposeTF([
#             TFRandomResizedCrop(scales, cropsize),
#             TFRandomHorizontalFlip(p=0.5)
#         ])

#     def __call__(self, im_lb):
#         return self.trans_func(im_lb)

# class TransformationVal(object):
#     """
#     TF 版本的 TransformationVal (无操作)
#     """
#     def __call__(self, im_lb):
#         # 验证集不执行数据增强，只进行 ToTensor 之前的直通
#         return im_lb

# class TFRandomResizedCrop(object):
#     '''
#     TensorFlow 实现的随机缩放和裁剪
#     '''

#     def __init__(self, scales=(0.5, 1.), size=(384, 384)):
#         self.scales = scales
#         self.size = size # (H, W)

#     def __call__(self, im_lb):
#         if self.size is None:
#             return im_lb

#         im, lb = im_lb['im'], im_lb['lb']
        
#         crop_h, crop_w = self.size
        
#         # 1. 随机缩放 (Random Scaling)
#         # tf.random.uniform 生成 (min, max) 范围内的随机浮点数
#         scale = tf.random.uniform(shape=[], minval=min(self.scales), maxval=max(self.scales), dtype=tf.float32)
        
#         # 获取当前 H, W
#         im_h = tf.cast(tf.shape(im)[0], tf.float32)
#         im_w = tf.cast(tf.shape(im)[1], tf.float32)
        
#         # 计算缩放后的尺寸，并向上取整 (类似 math.ceil)
#         new_h = tf.cast(tf.math.ceil(im_h * scale), tf.int32)
#         new_w = tf.cast(tf.math.ceil(im_w * scale), tf.int32)

#         # 图像缩放 (双线性插值)
#         # tf.image.resize 接收 [H, W] 或 [N, H, W, C]，输出 float32
#         im_scaled = tf.image.resize(im, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
#         im_scaled = tf.cast(im_scaled, im.dtype) # 保持原始数据类型，通常是 uint8

#         # 标签缩放 (最近邻插值)
#         lb_scaled = tf.image.resize(lb[..., None], [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
#         lb_scaled = tf.cast(tf.squeeze(lb_scaled, axis=-1), lb.dtype) # 恢复 (H, W) 形状

#         im, lb = im_scaled, lb_scaled
#         im_h, im_w = new_h, new_w # 更新尺寸
        
#         # 2. 填充 (Padding)
#         # 如果缩放后的尺寸小于裁剪尺寸，则进行填充
#         pad_h = tf.maximum(0, (crop_h - im_h) // 2 + 1)
#         pad_w = tf.maximum(0, (crop_w - im_w) // 2 + 1)
        
#         if pad_h > 0 or pad_w > 0:
#             # 图像填充：((pad_h, pad_h), (pad_w, pad_w), (0, 0))
#             im = tf.pad(im, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode='REFLECT') # REFLECT 适合图像，或用 CONSTANT
#             # 标签填充：constant_values=255
#             lb = tf.pad(lb, [[pad_h, pad_h], [pad_w, pad_w]], mode='CONSTANT', constant_values=255)
            
#             # 更新尺寸
#             im_h = tf.shape(im)[0]
#             im_w = tf.shape(im)[1]

#         # 3. 随机裁剪 (Random Cropping)
#         # tf.image.random_crop 对 im 和 lb 进行随机同步裁剪
        
#         # 组合 im 和 lb (通常需要堆叠)
#         # 标签需要扩展一个通道维度 (H, W) -> (H, W, 1)
#         stacked_img = tf.concat([im, tf.expand_dims(lb, axis=-1)], axis=-1) 
        
#         # 定义裁剪尺寸 (H, W, C)
#         crop_size = [crop_h, crop_w, tf.shape(stacked_img)[-1]]
        
#         # 随机裁剪
#         cropped_img = tf.image.random_crop(stacked_img, crop_size)
        
#         # 分离 im 和 lb
#         im_cropped = cropped_img[..., :3] # 假设 im 是 3 通道
#         lb_cropped = tf.squeeze(cropped_img[..., 3:], axis=-1) # 标签是最后一个通道，并移除通道维度
        
#         # 确保标签类型不变 (裁剪后的标签是 int32/uint8)
#         lb_cropped = tf.cast(lb_cropped, lb.dtype)

#         return dict(im=im_cropped, lb=lb_cropped)

# class TFRandomHorizontalFlip(object):
#     '''
#     TensorFlow 实现的随机水平翻转
#     '''
#     def __init__(self, p=0.5):
#         self.p = p

#     def __call__(self, im_lb):
#         im, lb = im_lb['im'], im_lb['lb']
        
#         # tf.random.uniform() 生成一个 0 到 1 之间的随机数
#         should_flip = tf.less(tf.random.uniform(shape=[]), self.p)
        
#         # 使用 tf.cond 进行条件执行
#         def flip_fn():
#             im_flipped = tf.image.flip_left_right(im)
#             lb_flipped = tf.image.flip_left_right(lb[..., None]) # 标签需要 (H, W, 1) 
#             lb_flipped = tf.squeeze(lb_flipped, axis=-1)
#             return im_flipped, lb_flipped

#         def no_flip_fn():
#             return im, lb

#         im_res, lb_res = tf.cond(should_flip, flip_fn, no_flip_fn)
        
#         return dict(im=im_res, lb=lb_res)

# class TFTensorNormalizer(object):
#     '''
#     TensorFlow 实现的 ToTensor 和归一化
#     '''
#     def __init__(self, mean=(0, 0, 0), std=(1., 1., 1.)):
#         # mean 和 std (RGB 顺序, 3 元素)
#         self.mean = tf.constant(mean, dtype=tf.float32)
#         self.std = tf.constant(std, dtype=tf.float32)

#     def __call__(self, im_lb):
#         im, lb = im_lb['im'], im_lb['lb']
        
#         # 1. 图像 (HWC) -> 浮点数并归一化到 [0, 1]
#         im = tf.cast(im, tf.float32) / 255.0
        
#         # 2. 图像 (HWC) -> CHW 
#         # tf.transpose(im, [2, 0, 1])
#         # 注意：通常在 TF 中会保持 HWC，只有 PyTorch 需要 CHW。
#         # 如果你的模型输入要求 CHW (如 PyTorch 模型转的 TF 部署)，则执行：
#         im = tf.transpose(im, [2, 0, 1])
        
#         # 3. 归一化 (使用广播)
#         # 扩展 mean 和 std 形状为 [C, 1, 1] 以匹配 im [C, H, W]
#         mean_c11 = self.mean[:, None, None]
#         std_c11 = self.std[:, None, None]
#         im = (im - mean_c11) / std_c11
        
#         # 4. 标签
#         if lb is not None:
#             # 标签保持 H x W，并转为 int64
#             lb = tf.cast(lb, tf.int64) 
        
#         return dict(im=im, lb=lb)

# # -------------------------------------------------------------
# # 5. 最终的 _tf_map_func 移除 tf.numpy_function
# # -------------------------------------------------------------
# # 现在所有的转换都在 TF 环境中执行，我们只需要一个 TF map 函数

# def _tf_map_func_final(img_path, lbl_path, args, split):
#     # 假设你已经定义了 _tf_read_decode 函数来处理 I/O 和解码
#     # 例如：
#     # def _tf_read_decode(img_path, lbl_path, args):
#     #     ... (TF I/O 和解码逻辑)
#     #     return image_tensor_HWC_uint8, label_tensor_HW_uint8
    
#     # 1. TF I/O 和解码 (HWC uint8, HW uint8)
#     # 假设 _tf_read_decode 已在前面定义
#     image, label = _tf_read_decode(img_path, lbl_path, args)
    
#     # 2. 转换成字典
#     im_lb = dict(im=image, lb=label)
    
#     # 3. 复杂变换
#     scale_ = 512
#     if args.dataset=='voc' or args.dataset=='ade20k':
#         scale_ = 480
        
#     if split == 'train':
#         transformer = TransformationTrain(scales=(0.5, 1.5), cropsize=(scale_, scale_))
#         im_lb = transformer(im_lb)
#     elif split == 'val':
#         transformer = TransformationVal()
#         im_lb = transformer(im_lb)

#     # 4. Remap
#     # 这部分逻辑也需要是 TF 操作
    
#     # camvid:
#     if args.dataset=='camvid' and split == 'val':
#         im_lb['lb'] = tf.cast(im_lb['lb'], tf.uint8) - 1
#     # ade20k:
#     elif args.dataset=='ade20k':
#         im_lb['lb'] = tf.cast(im_lb['lb'], tf.uint8) - 1
#     # voc:
#     elif args.dataset=='voc':
#         # 假设 255 是背景或忽略标签
#         lb = im_lb['lb']
#         lb = tf.where(tf.equal(lb, 255), tf.zeros_like(lb), lb) # 255 -> 0
#         lb = tf.cast(lb, tf.uint8) - 1 # 所有标签减 1
#         im_lb['lb'] = lb

#     # 5. ToTensor/Normalizer
#     normalizer = TFTensorNormalizer(
#         mean=(0.3257, 0.3690, 0.3223),
#         std=(0.2112, 0.2148, 0.2115),
#     )
#     im_lb = normalizer(im_lb)

#     image, label = im_lb['im'], im_lb['lb']
    
#     # 6. 补全形状 (HWC -> CHW，形状 (3, H, W) 和 (H, W))
#     image.set_shape((3, None, None)) 
#     label.set_shape((None, None))
#     return image, label









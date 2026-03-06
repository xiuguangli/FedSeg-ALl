import cv2
import tensorflow as tf
import numpy as np

class MultiEpochsDataLoader:
    """
    一个基于 tf.data.Dataset 的数据加载器类，模仿 PyTorch DataLoader 的接口。
    
    注意：在 tf.data 中，repeat() 行为通常用于训练，它使迭代器无限循环。
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=False):
        """
        初始化 TFDataloader。
        
        Args:
            dataset: 包含特征和标签的元组 (features, labels)，或一个 tf.data.Dataset 实例。
            batch_size (int): 批次大小。
            shuffle (bool): 是否在每个 epoch 开始时打乱数据。
            num_workers (int): 用于并行处理的 CPU 核心数。在 tf.data 中影响 map 和 prefetch。
            drop_last (bool): 如果数据集大小不能被 batch_size 整除，是否丢弃最后一个不完整的批次。
            pin_memory (bool): 在 tf 中不适用，仅为保持 PyTorch 接口兼容性。
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_workers = tf.data.AUTOTUNE if num_workers > 1 else None
        
        # 1. 创建基础 Dataset 并计算长度
        if isinstance(dataset, tuple) or isinstance(dataset, list):
            # 假设 dataset 是 (features, labels)
            self._tf_dataset = tf.data.Dataset.from_tensor_slices(dataset)
            self._data_size = tf.nest.flatten(dataset)[0].shape[0] # 获取样本总数
        elif isinstance(dataset, tf.data.Dataset):
            self._tf_dataset = dataset
            # 尝试获取长度，如果失败则设为 None
            try:
                self._data_size = len(list(dataset.as_numpy_iterator()))
            except Exception:
                self._data_size = None
        else:
            raise TypeError("dataset 必须是 (features, labels) 元组或 tf.data.Dataset 实例。")

        # 2. 构建数据流 pipeline
        self._tf_dataset = self._build_pipeline(self._tf_dataset)
        
    def _build_pipeline(self, ds):
        """构建 tf.data 的完整数据流 pipeline"""
        
        # 混洗 (Shuffle) - 放在 batch 之前
        if self.shuffle and self._data_size is not None:
            # 使用整个数据集大小作为 buffer_size 实现充分混洗
            ds = ds.shuffle(buffer_size=self._data_size, reshuffle_each_iteration=True)
            
        # 批量化 (Batching)
        ds = ds.batch(self.batch_size, drop_remainder=self.drop_last)
        
        # 预取 (Prefetching) - 放在 pipeline 的最后以实现并行
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        return ds

    def __len__(self):
        """返回批次数 (Steps per Epoch)"""
        if self._data_size is None:
            raise NotImplementedError("Dataset size is unknown. Cannot return length.")
            
        if self.drop_last:
            return self._data_size // self.batch_size
        else:
            return (self._data_size + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        """返回一个可迭代对象，用于循环"""
        # 每次调用 iter() 都会创建一个新的迭代器
        # 注意：由于没有使用 .repeat()，迭代器会在一个 epoch 后结束
        return iter(self._tf_dataset)


def dataset_generator(pytorch_style_dataset):
    """
    一个 Python 生成器函数，利用 PyTorch 风格 Dataset 的 __len__ 和 __getitem__。
    
    Args:
        pytorch_style_dataset: 您的 CamVid_Dataset 实例。
    """
    for i in range(len(pytorch_style_dataset)):
        # 调用 __getitem__ 获取一个样本 (这里是 numpy/PIL 对象)
        image, label = pytorch_style_dataset[i] 
        
        # ⚠️ 注意: __getitem__ 已经返回了经过 ToTensor 处理的 NumPy 数组（因为您的代码是用 cv2_transforms.ToTensor 处理的）
        # 如果 ToTensor 返回的是 NumPy 数组，则直接 yield
        # 如果返回的是 PyTorch Tensor，需要先转换为 NumPy 数组 (例如：image.numpy(), label.numpy())
        
        # 假设您的 ToTensor 返回的是 NumPy 数组
        yield image, label
        

def create_tf_dataloader_from_custom_dataset_train(
    custom_dataset_instance, batch_size, shuffle=True, repeat=True, drop_last=True,
    output_img_shape=(480, 480, 3), output_lbl_shape=(480, 480)
):
    """
    从 CamVid_Dataset 实例创建 tf.data.Dataset。
    
    Args:
        custom_dataset_instance: CamVid_Dataset 实例。
        batch_size (int): 批次大小。
        shuffle (bool): 是否混洗。
        repeat (bool): 是否无限重复。
        output_img_shape (tuple): 图像输出形状 (H, W, C)。
        output_lbl_shape (tuple): 标签输出形状 (H, W)。
    """
    
    # 获取生成器
    generator = lambda: dataset_generator(custom_dataset_instance)

    # 1. 使用 from_generator 创建基础 Dataset
    # 必须显式指定生成器返回的元素的类型和形状
    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int64), # 根据您 ToTensor 后的类型确定
        output_shapes=(output_img_shape, output_lbl_shape)
    )

    # 2. 混洗 (Shuffle)
    if shuffle:
        # 使用数据集大小作为 buffer_size
        tf_dataset = tf_dataset.shuffle(buffer_size=len(custom_dataset_instance))

    # 3. 批量化 (Batching)
    tf_dataset = tf_dataset.batch(batch_size, drop_remainder=drop_last,num_parallel_calls=tf.data.AUTOTUNE,deterministic=False)
    
    # 4. 重复 (Repeat)
    if repeat:
        tf_dataset = tf_dataset.repeat() 

    # 5. 预取 (Prefetch)
    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset

import myseg.tv_transform as my_transforms
import myseg.cv2_transform as cv2_transforms
from PIL import Image
def create_tf_dataloader_from_custom_dataset_test111(
    custom_dataset_instance, 
    batch_size=1, 
    shuffle=False, 
    repeat=False, 
    num_channels=3
): 
    def _parse_function(img_path, lbl_path):
        # 读入图片
        if args.dataset=='voc':
            image = Image.open(img_path)
            image = np.array(image)
            label = Image.open(lbl_path)
            label = np.array(label)
        else:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1]
            label = cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
        # print(image.shape, label.shape)  # (1024, 2048, 3) (1024, 2048)

        # 将label进行remap
        #label = lb_map[label]
        if args.dataset=='camvid':
            if split == 'val':  
                label = np.uint8(label)-1
        elif args.dataset=='ade20k':
            label = np.uint8(label)-1
        elif args.dataset=='voc':
            label[label==255]=0
            label =  np.uint8(label)-1
        scale_ = 512
        if  args.dataset=='voc' or args.dataset=='ade20k':
            scale_ = 480

        # transform : 同时处理image和label
        image_label = dict(im=image, lb=label)

        if split == 'train':
            image_label = cv2_transforms.TransformationTrain(scales=(0.5, 1.5), cropsize=(scale_, scale_))(image_label)

        if split == 'val':
            image_label = cv2_transforms.TransformationVal()(image_label)

        # ToTensor
        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )(image_label)

        image, label = image_label['im'], image_label['lb']
        # print(image.shape, label.shape) # torch.Size([3, 512, 1024]) torch.Size([512, 1024])

        return (image, label)
        
    def _tf_map_func(img_path, lbl_path):
        # 注意：inp 参数里不要传 args，因为 args 不是 Tensor
        # _cv2_process_func 会自动捕获 args
        image, label = tf.numpy_function(
            func=_parse_function,
            inp=[img_path, lbl_path], 
            Tout=[tf.float32, tf.int64]
        )
        # 补全形状
        image.set_shape((3, None, None)) # 这里也可以用 args
        label.set_shape((None, None))
        return image, label
        
    
    images_path, labels_path = custom_dataset_instance.image_dirs, custom_dataset_instance.label_dirs
    args = custom_dataset_instance.args
    split = custom_dataset_instance.split
    
    ds = tf.data.Dataset.from_tensor_slices((images_path, labels_path))
    ds = ds.shuffle(len(images_path))
    ds = ds.map(_tf_map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds 
    

def create_tf_dataloader_from_custom_dataset_test(
    custom_dataset_instance, 
    batch_size=1, 
    shuffle=False, 
    repeat=False, 
    num_channels=3
): 
    def _np_transform_remap(image_tensor_np, label_tensor_np):
        # image_tensor_np 和 label_tensor_np 已经是 NumPy 数组
        image = image_tensor_np
        label = label_tensor_np

        # 将label进行remap (保持原逻辑)
        if args.dataset=='camvid':
            if split == 'val':  
                label = np.uint8(label)-1
        elif args.dataset=='ade20k':
            label = np.uint8(label)-1
        elif args.dataset=='voc':
            # 如果是 VOC，且之前在 TF 阶段未能解码为 NumPy 数组，则这部分可能需要调整
            # 但我们假设 TF 解码已经正确处理了。
            # 如果 TF 解码将 255 保留，则这里的 remap 保持不变
            label[label==255]=0
            label =  np.uint8(label)-1
        
        # 确保 label 是 uint8 类型，以匹配 cv2_transforms 的预期
        label = np.uint8(label) 
        
        scale_ = 512
        if  args.dataset=='voc' or args.dataset=='ade20k':
            scale_ = 480

        # transform : 同时处理image和label
        image_label = dict(im=image, lb=label)

        # *** 这部分将使用你稍后提供的 cv2_transforms 代码 ***
        if split == 'train':
            image_label = cv2_transforms.TransformationTrain(scales=(0.5, 1.5), cropsize=(scale_, scale_))(image_label)
        elif split == 'val': # 更改为 elif 以避免同时执行
            image_label = cv2_transforms.TransformationVal()(image_label)

        # ToTensor
        image_label = cv2_transforms.ToTensor(
            mean=(0.3257, 0.3690, 0.3223),  # city, rgb
            std=(0.2112, 0.2148, 0.2115),
        )(image_label)

        image, label = image_label['im'], image_label['lb']
        
        # 返回 torch.Tensor 格式的 NumPy 数组 (通常是 C x H x W)
        return (image, label)


    # -------------------------------------------------------------
    # 2. TensorFlow I/O 和解码函数
    #    - 接收：文件路径 (String Tensor)
    #    - 输出：解码后的图像/标签 (Numeric Tensor)
    # -------------------------------------------------------------
    def _tf_read_decode(img_path, lbl_path):
        # 1. I/O: 读取文件内容
        img_bytes = tf.io.read_file(img_path)
        lbl_bytes = tf.io.read_file(lbl_path)
        
        # 2. Decoding: 高效的 C++ 图像解码
        if args.dataset=='voc':
            # 假设 VOC 图像是 JPEG 格式，标签是 PNG 格式
            image = tf.image.decode_jpeg(img_bytes, channels=3)
            # tf.image.decode_png 适用于单通道标签图 (即使是 8-bit)
            label = tf.image.decode_png(lbl_bytes, channels=1)
            label = tf.squeeze(label, axis=-1) # (H, W, 1) -> (H, W)
            
        else: # 对应于原来的 cv2.imread 逻辑
            # 图像：cv2.imread(img_path, cv2.IMREAD_COLOR)[:, :, ::-1] 是 BGR -> RGB
            # TF 默认解码为 RGB
            image = tf.image.decode_image(img_bytes, channels=3, dtype=tf.uint8)
            # 标签：cv2.imread(lbl_path, cv2.IMREAD_GRAYSCALE)
            label = tf.image.decode_image(lbl_bytes, channels=1, dtype=tf.uint8)
            label = tf.squeeze(label, axis=-1) # (H, W, 1) -> (H, W)
        
        # 3. 确保数据类型正确，以便 tf.numpy_function 转换为 NumPy
        image = tf.cast(image, tf.uint8) # HWC uint8
        label = tf.cast(label, tf.uint8) # HW uint8
        
        return image, label
    
    def _tf_map_func(img_path, lbl_path):
        # 1. TF I/O 和解码
        image, label = _tf_read_decode(img_path, lbl_path)
        
        # 2. 调用 tf.numpy_function 执行 NumPy/CV2 变换
        image, label = tf.numpy_function(
            func=_np_transform_remap,
            inp=[image, label], # 传入 Tensor (TF 会转为 NumPy)
            Tout=[tf.float32, tf.int64] # 返回 Tensor 类型
        )
        
        # 3. 补全形状
        # 假设 ToTensor 后的形状是 C x H x W
        image.set_shape((3, None, None)) 
        label.set_shape((None, None))
        return image, label
        
    
    images_path, labels_path = custom_dataset_instance.image_dirs, custom_dataset_instance.label_dirs
    args = custom_dataset_instance.args
    split = custom_dataset_instance.split
    
    ds = tf.data.Dataset.from_tensor_slices((images_path, labels_path))
    ds = ds.shuffle(len(images_path))
    ds = ds.map(_tf_map_func, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds 
    
    
def create_pure_tf_dataset(image_paths, label_paths, batch_size):
    
    def _parse_function(img_path, lbl_path):
        # 读取文件
        img_str = tf.io.read_file(img_path)
        lbl_str = tf.io.read_file(lbl_path)
        
        # 解码
        img = tf.image.decode_jpeg(img_str, channels=3)
        lbl = tf.image.decode_png(lbl_str, channels=1) # 假设 label 是 png
        
        # Resize
        img = tf.image.resize(img, [480, 480])
        lbl = tf.image.resize(lbl, [480, 480], method='nearest') # 标签必须用最近邻插值
        
        # 归一化 (模拟 ToTensor)
        img = tf.cast(img, tf.float32) / 255.0
        # 减均值除方差等需用 tf.math 操作...
        
        # 转置为 NCHW (如果需要)
        img = tf.transpose(img, [2, 0, 1]) 
        
        return img, lbl

    ds = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))
    ds = ds.shuffle(len(image_paths))
    ds = ds.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds    


def create_tf_dataloader_from_custom_dataset_test000(
    custom_dataset_instance, 
    batch_size=1, 
    shuffle=False, 
    repeat=False, 
    num_channels=3
):  
    # 数据是 NCHW 格式
    # 假设 dataset_generator 已经定义好并返回 (image_nchw, label_hw)
    generator = lambda: dataset_generator(custom_dataset_instance)

    # NCHW 动态形状: (Channels, None, None)
    img_shape = tf.TensorShape([num_channels, None, None])
    # Label 动态形状: (None, None)
    lbl_shape = tf.TensorShape([None, None])

    tf_dataset = tf.data.Dataset.from_generator(
        generator,
        output_types=(tf.float32, tf.int64),
        output_shapes=(img_shape, lbl_shape)
    )

    if shuffle:
        tf_dataset = tf_dataset.shuffle(buffer_size=len(custom_dataset_instance))

    # 对于动态形状，batch_size 建议为 1
    tf_dataset = tf_dataset.batch(batch_size)
    
    if repeat:
        tf_dataset = tf_dataset.repeat() 

    tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
    
    return tf_dataset


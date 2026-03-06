
# import torch

# # MultiEpochsDataLoader：解决每个循环开始时多次重建self.trainloader的问题
# class MultiEpochsDataLoader(torch.utils.data.DataLoader):

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self._DataLoader__initialized = False
#         self.batch_sampler = _RepeatSampler(self.batch_sampler)
#         self._DataLoader__initialized = True
#         self.iterator = super().__iter__()

#     def __len__(self):
#         return len(self.batch_sampler.sampler)

#     def __iter__(self):
#         for i in range(len(self)):
#             yield next(self.iterator)


# class _RepeatSampler(object):
#     """ Sampler that repeats forever.
#     Args:
#         sampler (Sampler)
#     """

#     def __init__(self, sampler):
#         self.sampler = sampler

#     def __iter__(self):
#         while True:
#             yield from iter(self.sampler)
            

import mindspore.dataset as ds

class _RepeatSampler:
    """无限重复采样器"""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class MultiEpochsDataLoader(ds.GeneratorDataset):
    """
    兼容你原来的链式写法：
    loader = MultiEpochsDataLoader(..., shuffle=True).batch(bs, drop_remainder=True)
    """
    def __init__(self, dataset, num_parallel_workers=1, shuffle=False):
        # 1. 构造 sampler
        # sampler = (ds.RandomSampler(num_samples=len(dataset)) if shuffle
        #           else ds.SequentialSampler(num_samples=len(dataset)))
        
        sampler = ds.RandomSampler() if shuffle else ds.SequentialSampler()
        # repeater = _RepeatSampler(sampler)

        # 3. 先初始化父类
        super().__init__(
            source=dataset,
            column_names=["image", "label"],
            sampler=sampler,
            num_parallel_workers=num_parallel_workers,
            python_multiprocessing=(num_parallel_workers > 1)
        )

        # 4. 预生成一次迭代器，保证每个 epoch 不复建
        # self._iterator = super().__iter__()

    # def __iter__(self):
    #     for _ in range(len(self)):
    #         yield next(self._iterator)

    # def __len__(self):
    #     return self.get_dataset_size()

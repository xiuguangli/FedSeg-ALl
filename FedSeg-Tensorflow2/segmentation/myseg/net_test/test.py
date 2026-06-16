import numpy as np
import torch
from logging_utils import logger
from myseg.tv_transform import RandomScaleCrop

# image, label = torch.randn(1, 3, 1024, 2048), torch.randn(1, 3, 1024, 2048)
# image, label = RandomScaleCrop(image, label)

strings = ['\n'
           'Local Train Run Time: {0:0.2f}s'
           '5555'
           ]

logger.info("{}", strings)
logger.info("{}", len(strings))

'''
TODO:
check the code of acc and miou (look in Enet)
'''

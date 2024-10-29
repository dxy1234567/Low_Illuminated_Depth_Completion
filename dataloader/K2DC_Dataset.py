
"""
This script is modified from the work of Abdelrahman Eldesokey.
Find more details from https://github.com/abdo-eldesokey/nconv
"""

########################################
__author__ = "Abdelrahman Eldesokey"
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Abdelrahman Eldesokey"
__email__ = "abdo.eldesokey@gmail.com"
########################################

from PIL import Image
import torch
import numpy as np
import glob
import random
import cv2
from torch.utils.data import  Dataset


class KittiDepthDataset(Dataset):
    def __init__(self, data_path, setname='train', transform=None, flip = False):
        self.data_path = data_path  # KITTI_to_DC/dataset/
        self.setname = setname
        self.transform = transform
        self.flip = flip
        
        self.types = ['depth', 'depth_gt', 'gray']

        # Lists of depth, gt, and gray.
        # KITTI_to_DC/dataset/train/depth/xx_xxxxxx.png
        self.depth = sorted(glob.iglob(self.data_path + setname + '/depth' + "/*.png", recursive=False)) 
        self.gt = sorted(glob.iglob(self.data_path + setname + '/depth_gt' + "/*.png", recursive=False))
        self.gray = sorted(glob.iglob(self.data_path + setname + '/gray' + "/*.png", recursive=False))

        assert (len(self.gt) == len(self.depth))

    def __len__(self):
        return len(self.depth)

    def __getitem__(self, item):
        if item < 0 or item >= self.__len__():
            return None

        # Read images and convert them to 4D floats
        depth = Image.open(str(self.depth[item])).convert('L')
        gt = Image.open(str(self.gt[item])).convert('L')
        gray = Image.open(str(self.gray[item])).convert('L')

        # Apply transformations if given
        if self.transform is not None:
            depth = self.transform(depth)
            gt = self.transform(gt)
            gray = self.transform(gray)

        # 数据增强，数据翻转
        if self.flip and random.randint(0, 1) and self.setname == 'train':
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
            gray = gray.transpose(Image.FLIP_LEFT_RIGHT)

        # Convert to numpy
        depth = np.array(depth, dtype=np.float16)
        gt = np.array(gt, dtype=np.float16)
        gray = np.array(gray, dtype=np.float16)
        C = (depth > 0).astype(float)

        # Normalize the depth
        depth = depth / 255  # [0,1]
        gt = gt / 16
        gray = gray /255

        # Expand dims into Pytorch format
        depth = np.expand_dims(depth, 0)
        gt = np.expand_dims(gt, 0)
        gray = np.expand_dims(gray, 0)
        C = np.expand_dims(C, 0)
        # Convert to Pytorch Tensors
        depth = torch.tensor(depth, dtype=torch.float)
        gt = torch.tensor(gt, dtype=torch.float)
        gray = torch.tensor(gray, dtype=torch.float)
        C = torch.tensor(C, dtype=torch.float)
    

        return depth, gt, item, gray, C
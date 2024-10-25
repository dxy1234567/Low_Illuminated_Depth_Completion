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

import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from dataloader.K2DC_Dataset import KittiDepthDataset
import random
import glob
num_worker = 8

def KittiDataLoader(params):
    # Input images are 16-bit, but only 15-bits are utilized, so we normalized the data to [0:1] using a normalization factor
    num_worker = 8

    image_datasets = {}
    dataloaders = {}
    datasizes = {}

    data_path = params['dir_ds']

    ###### Training Set ######
    image_datasets['train'] = KittiDepthDataset(data_path, setname='train', transform=None, flip=False)

    # Select the desired number of images from the training set
    if params['train_on'] != 'full':
        image_datasets['train'].data = image_datasets['train'].data[0:params['train_on']]  # file directions
        image_datasets['train'].gt = image_datasets['train'].gt[0:params['train_on']]

    dataloaders['train'] = DataLoader(image_datasets['train'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)
    datasizes['train'] = {len(image_datasets['train'])}
    ###### Test Set ######
    image_datasets['test'] = KittiDepthDataset(data_path, setname='test', transform=None, flip=False)

    dataloaders['test'] = DataLoader(image_datasets['test'], shuffle=True, batch_size=params['train_batch_sz'],
                                      num_workers=num_worker)
    datasizes['test'] = {len(image_datasets['test'])}

    ###### Validation Set ######
    image_datasets['val'] = KittiDepthDataset(data_path, setname='val', transform=None, flip=False)
    
    dataloaders['val'] = DataLoader(image_datasets['val'], shuffle=False, batch_size=params['val_batch_sz'],
                                    num_workers=num_worker)
    datasizes['val'] = {len(image_datasets['val'])}

    return dataloaders








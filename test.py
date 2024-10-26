import cv2
import torch
import numpy as np


img = cv2.imread('/root/ChenJiasheng/_20_1_optimal_lr_shce/workspace/exp_msg_chn/test_output_epoch_1/0000000000.png')
img_KDC = cv2.imread('/data/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png')

max = np.max(img)
max_KDC = np.max(img_KDC)
print()

print("End")

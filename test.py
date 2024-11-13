import cv2

path_img = '/data/data_depth_annotated/train/2011_09_26_drive_0001_sync/proj_depth/groundtruth/image_02/0000000005.png'

img = cv2.imread(path_img, cv2.IMREAD_UNCHANGED)

print('End')
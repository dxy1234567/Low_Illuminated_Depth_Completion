import glob


data_path = "/home/cjs/data/KITTI/data_odometry_gray/dataset/sequences/01/image_0"

data1 = glob.iglob(data_path + "/*.png", recursive=True)
data2 = sorted(data1)
data = list(data2)

print("End")

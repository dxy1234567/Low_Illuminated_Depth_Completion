import os
import random
import shutil
from torch.utils.data import random_split

def extract_random_frames_with_types(all_dataset, sequences_dir, output_dir, test_ratio=0.1, val_ratio=0.05):
    """
        从每个序列中随机抽取num_frames个帧，并确保depth, depth_gt, gray三种数据对应。
        
        Params: 
            sequences_dir: 存放序列的根目录，每个序列有depth, depth_gt, gray三个子文件夹
            output_dir: 测试集输出目录
            # num_frames: 抽取的随机帧数
            test_ratio: 测试集比例
            val_ratio: 验证集比例
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    sequences = {"00", "05", "07"}
    types = ['depth', 'depth_gt', 'gray']
    types_ds = ['train', 'test', 'val']

    for sequence in sequences:
        # 获取三个类型的文件夹路径，在每个序列中
        type_dirs = {t: os.path.join(sequences_dir, sequence, t) for t in types}
        
        # 确保三个子目录都存在
        for t, t_dir in type_dirs.items():
            if not os.path.isdir(t_dir):
                print(f"目录 {t_dir} 不存在！")
                return
        
        # 获取depth文件夹中的所有帧（以gt为基准【gt少】）
        frame_files = sorted([f for f in os.listdir(type_dirs['depth_gt']) if f.endswith(('.jpg', '.png'))])
        
        dataset_size = len(frame_files)
        
        # 计算每个子集的大小
        test_size = int(test_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)  # 验证集相对于剩余数据
        train_size = dataset_size - test_size - val_size

        # selected_frames = random.sample(frame_files, num_frames)

        train_dataset, val_dataset, test_dataset = random_split(all_dataset, [train_size, val_size, test_size])
        
        # 为每种类型创建对应的输出目录
        for t in types:
            t_output_dir = os.path.join(output_dir, t)
            os.makedirs(t_output_dir, exist_ok=True)
            
            # 复制选中的帧到输出目录
            for frame in selected_frames:
                src = os.path.join(type_dirs[t], frame)
                dst = os.path.join(t_output_dir, sequence + "_" + frame)
                shutil.copy(src, dst)
                
        print(f"从序列中随机抽取了 {len(selected_frames)} 个帧，并保存到 {output_dir}")

# 使用示例
sequence_dir = '/data/KITTI_to_DC/'  # 原始序列的根目录
output_dir = '/data/KITTI_to_DC/dataset/'      # 测试集输出目录
extract_random_frames_with_types(sequence_dir, output_dir)

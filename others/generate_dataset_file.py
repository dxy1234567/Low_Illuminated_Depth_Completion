import os
import random
import shutil

def select_random_frames(source_dir, dst_dir, num_frames=100):
    """
    从每个序列文件夹中随机抽取 num_frames 帧图像，作为测试集保存到目标目录。
    :param source_dir: 源数据集的根目录（每个子文件夹对应一个序列）
    :param dst_dir: 保存测试集的目标目录
    :param num_frames: 从每个序列中随机抽取的帧数
    """
    # 确保目标目录存在
    os.makedirs(dst_dir, exist_ok=True)

    # 遍历每个序列（子文件夹）
    for sequence_folder in os.listdir(source_dir):
        if sequence_folder
        sequence_path = os.path.join(source_dir, sequence_folder)

        # 检查是否是文件夹
        if os.path.isdir(sequence_path):
            # 获取该序列中的所有图像文件
            frame_files = [f for f in os.listdir(sequence_path) if f.endswith(('.jpg', '.png'))]

            # 如果图像文件数量不足num_frames，发出警告
            if len(frame_files) < num_frames:
                print(f"Warning: {sequence_folder} has less than {num_frames} frames.")
                continue

            # 随机抽取 num_frames 帧
            selected_frames = random.sample(frame_files, num_frames)

            # 创建对应的测试集目录
            test_sequence_path = os.path.join(dst_dir, sequence_folder)
            os.makedirs(test_sequence_path, exist_ok=True)

            # 将选中的帧复制到目标目录
            for frame in selected_frames:
                src = os.path.join(sequence_path, frame)
                dst = os.path.join(test_sequence_path, frame)
                shutil.copy(src, dst)

            print(f"Selected {num_frames} frames from {sequence_folder}.")

# 示例用法
source_dir = 'path/to/source/dataset'  # 源数据集目录
dst_dir = 'path/to/test/dataset'      # 测试集保存目录
select_random_frames(source_dir, dst_dir, num_frames=100)

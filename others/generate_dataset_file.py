import os
import random
import shutil

def extract_random_frames_with_types(base_dir, output_dir, num_frames=100):
    """
    从每个序列中随机抽取num_frames个帧，并确保depth, depth_gt, gray三种数据对应。
    
    :param base_dir: 存放序列的根目录，每个序列有depth, depth_gt, gray三个子文件夹
    :param output_dir: 测试集输出目录
    :param num_frames: 抽取的随机帧数
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义三个类型
    types = ['depth', 'depth_gt', 'gray']
    
    # 获取三个类型的文件夹路径
    type_dirs = {t: os.path.join(base_dir, t) for t in types}
    
    # 确保三个子目录都存在
    for t, t_dir in type_dirs.items():
        if not os.path.isdir(t_dir):
            print(f"目录 {t_dir} 不存在！")
            return
    
    # 获取depth文件夹中的所有帧（假设所有类型帧数量和命名一致）
    frame_files = sorted([f for f in os.listdir(type_dirs['depth']) if f.endswith(('.jpg', '.png'))])
    
    # 如果帧数少于num_frames，则取全部帧
    if len(frame_files) < num_frames:
        selected_frames = frame_files
    else:
        # 随机选择num_frames个帧
        selected_frames = random.sample(frame_files, num_frames)
    
    # 为每种类型创建对应的输出目录
    for t in types:
        t_output_dir = os.path.join(output_dir, t)
        os.makedirs(t_output_dir, exist_ok=True)
        
        # 复制选中的帧到输出目录
        for frame in selected_frames:
            src = os.path.join(type_dirs[t], frame)
            dst = os.path.join(t_output_dir, frame)
            shutil.copy(src, dst)
            
    print(f"从序列中随机抽取了 {len(selected_frames)} 个帧，并保存到 {output_dir}")

# 使用示例
base_dir = 'path/to/your/sequence'  # 原始序列的根目录
output_dir = 'path/to/test_set'      # 测试集输出目录
extract_random_frames_with_types(base_dir, output_dir, num_frames=100)

import torch
from workspace.exp_msg_chn.network_exp_msg_chn import network
import sys
sys.path.append('./workspace/exp_msg_chn')

model = network()
model = torch.nn.DataParallel(model)

params = torch.load('workspace/exp_msg_chn/checkpoints/DataParallel_ep0020.pth.tar', map_location=torch.device('cpu'))
for key, value in params.items():
    print(f"{key}: {value.shape}")



# model.load_state_dict(torch.load('/root/ChenJiasheng/Low_Illuminated_Depth_Completion/workspace/exp_msg_chn/final_model.pth', map_location=torch.device('cpu')))

# # 查看参数
# for name, param in model.named_parameters():
#     print(f"{name}: {param.shape}")
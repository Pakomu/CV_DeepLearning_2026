import torch

# 1. 指定你的 Top 3 FFT 權重
ckpt1_path = "train_ckpt/promptir-epoch=004-train_loss=0.0156.ckpt" # 請替換為實際檔名
ckpt2_path = "train_ckpt/promptir-epoch=017-train_loss=0.0110.ckpt"
ckpt3_path = "train_ckpt/promptir-epoch=047-train_loss=0.0169.ckpt"

# 2. 載入權重字典 (放 CPU 處理避免 OOM)
ckpt1 = torch.load(ckpt1_path, map_location='cpu')
ckpt2 = torch.load(ckpt2_path, map_location='cpu')
ckpt3 = torch.load(ckpt3_path, map_location='cpu')

avg_state_dict = {}

# 3. 三者平均
for key in ckpt1['state_dict'].keys():
    avg_state_dict[key] = (ckpt1['state_dict'][key] + 
                           ckpt2['state_dict'][key] + 
                           ckpt3['state_dict'][key]) / 3.0

ckpt1['state_dict'] = avg_state_dict
torch.save(ckpt1, "train_ckpt/swa_fft_model.ckpt")
print("成功產生 FFT 平均權重模型：swa_fft_model.ckpt")
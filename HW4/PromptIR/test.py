import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import lightning.pytorch as pl
import torch.nn as nn

# 匯入原本的 PromptIR 模型架構
from net.model import PromptIR

# 必須定義這個 LightningModule，才能正確載入 .ckpt 權重檔
class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    def forward(self, x):
        return self.net(x)

def main():
    # 自動偵測 GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ==========================================
    # 1. 設定你的路徑 (請根據你的實際狀況修改)
    # ==========================================
    # 填入你剛剛訓練出來的 ckpt 檔案路徑 (通常在 train_ckpt 資料夾下)
    ckpt_path = "train_ckpt/last.ckpt" 
    
    # 填入作業測試集資料夾 (裡面應該要是 0.png 到 99.png)
    test_folder = './data/test/degraded' 
    output_npz = 'pred.npz'

    # ==========================================
    # 2. 載入模型
    # ==========================================
    print(f"Loading model from {ckpt_path}...")
    model = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval() # 切換到測試模式，關閉 Dropout 等機制

    images_dict = {}

    # ==========================================
    # 3. 推論與轉換 NPZ 迴圈
    # ==========================================
    print("Starting inference...")
    with torch.no_grad(): # 測試時不需要計算梯度，節省大量 VRAM
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_folder, filename)
                
                # 讀取圖片並轉為 RGB
                img = Image.open(img_path).convert('RGB')
                
                # 將 PIL Image 轉為 Tensor (此時形狀為 C, H, W，數值 0~1)
                img_tensor = TF.to_tensor(img).to(device)
                
                # 增加 Batch 維度 (變成 1, C, H, W) 以符合模型輸入格式
                img_tensor = img_tensor.unsqueeze(0)

                # 丟入模型進行還原
                # 丟入模型進行還原 (加入 TTA 機制)
                # 1. 原始視角
                restored_orig = model(img_tensor)
                
                # 2. 水平翻轉視角 (輸入前翻轉，預測完再翻轉回來)
                img_hflip = TF.hflip(img_tensor)
                restored_hflip = TF.hflip(model(img_hflip))
                
                # 3. 垂直翻轉視角
                img_vflip = TF.vflip(img_tensor)
                restored_vflip = TF.vflip(model(img_vflip))

                # 將三個結果平均，消除單一視角的雜訊與瑕疵
                restored_tensor = (restored_orig + restored_hflip + restored_vflip) / 3.0

                # ==========================================
                # 格式轉換: Tensor -> Numpy (3, H, W) -> uint8
                # ==========================================
                # 移除 Batch 維度，移回 CPU，轉成 numpy
                restored_np = restored_tensor.squeeze(0).cpu().numpy() 
                
                # 限制數值在 0~1 之間，乘上 255，轉為整數 (0-255 uint8)
                restored_np = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)

                # 存入字典 (Key 為檔名，例如 '0.png')
                images_dict[filename] = restored_np
                print(f"Processed: {filename}")

    # ==========================================
    # 4. 輸出最終檔案
    # ==========================================
    np.savez(output_npz, **images_dict)
    print(f"\nSuccess! Saved {len(images_dict)} images to {output_npz}")
    print("Shape check for the last image:", restored_np.shape) # 確保是 (3, H, W)

if __name__ == '__main__':
    main()
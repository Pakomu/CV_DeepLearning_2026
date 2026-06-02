import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import lightning.pytorch as pl
import torch.nn.functional as F

from net.model import PromptIR

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
    def forward(self, x):
        return self.net(x)

def pad_input(input_, img_multiple_of=8):
    """將全圖長寬補齊為 8 的倍數，確保 Transformer 能整除"""
    height, width = input_.shape[2], input_.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_, height, width

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ⚠️ 這裡填入你退回的「最強權重 (Epoch 42 或是 SWA 都可以)」
    ckpt_path = "train_ckpt/promptir-epoch=042-train_loss=0.0656.ckpt" 
    test_folder = './data/test/degraded' 
    output_npz = 'pred.npz'

    model = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    images_dict = {}

    print("Starting Full-Image TTA Inference (No Tiling)...")
    with torch.no_grad():
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_folder, filename)
                img = Image.open(img_path).convert('RGB')
                
                img_tensor = TF.to_tensor(img).to(device).unsqueeze(0)
                
                # 1. 全圖 Padding (不切塊！)
                img_tensor, h, w = pad_input(img_tensor)

                # ==========================================
                # 全圖 TTA (保持全局特徵完整)
                # ==========================================
                
                # 視角 A：原始全圖
                res_orig = model(img_tensor)
                
                # 視角 B：水平翻轉全圖
                img_hflip = TF.hflip(img_tensor)
                res_hflip = TF.hflip(model(img_hflip))
                
                # 視角 C：垂直翻轉全圖
                img_vflip = TF.vflip(img_tensor)
                res_vflip = TF.vflip(model(img_vflip))

                # 將三個結果平均
                restored_tensor = (res_orig + res_hflip + res_vflip) / 3.0

                # ==========================================
                
                # 2. 裁切回原本的圖片大小
                restored_tensor = restored_tensor[:, :, :h, :w]

                # 轉換為 Numpy uint8
                restored_np = restored_tensor.squeeze(0).cpu().numpy() 
                restored_np = np.clip(restored_np * 255.0, 0, 255).astype(np.uint8)

                images_dict[filename] = restored_np
                print(f"Processed: {filename}")

    np.savez(output_npz, **images_dict)
    print(f"\nSuccess! Saved {len(images_dict)} images to {output_npz}")

if __name__ == '__main__':
    main()
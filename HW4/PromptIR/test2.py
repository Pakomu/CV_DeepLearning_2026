import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import lightning.pytorch as pl
import torch.nn as nn
import torch.nn.functional as F

from net.model import PromptIR

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        
    def forward(self, x):
        return self.net(x)

# ==========================================
# 官方的 Padding 函數 (確保切割時長寬是 8 的倍數)
# ==========================================
def pad_input(input_, img_multiple_of=8):
    height, width = input_.shape[2], input_.shape[3]
    H, W = ((height + img_multiple_of) // img_multiple_of) * img_multiple_of, ((width + img_multiple_of) // img_multiple_of) * img_multiple_of
    padh = H - height if height % img_multiple_of != 0 else 0
    padw = W - width if width % img_multiple_of != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')
    return input_, height, width

# ==========================================
# 官方的 Tile 推論函數 (重疊切割)
# ==========================================
def tile_eval(model, input_, tile=192, tile_overlap=32):
    b, c, h, w = input_.shape
    tile = min(tile, h, w)
    assert tile % 8 == 0, "tile size should be multiple of 8"

    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    E = torch.zeros(b, c, h, w).type_as(input_)
    W = torch.zeros_like(E)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch)
            W[..., h_idx:(h_idx+tile), w_idx:(w_idx+tile)].add_(out_patch_mask)
            
    restored = E.div_(W)
    restored = torch.clamp(restored, 0, 1) # 確保數值在 0~1 之間
    return restored


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ⚠️ 這裡填入你平均後的權重檔，或是你最強的那顆權重
    ckpt_path = "train_ckpt/last.ckpt" 
    test_folder = './data/test/degraded' 
    output_npz = 'pred.npz'

    # ⚠️ 這裡的 tile 請設定為你「最後一次訓練的 patch_size」
    # 如果你最後是用 192x192 訓練的，這裡就設 192；如果是 128 就設 128
    TILE_SIZE = 128
    TILE_OVERLAP = 64

    model = PromptIRModel.load_from_checkpoint(ckpt_path).to(device)
    model.eval()

    images_dict = {}

    print("Starting Ultimate Inference (Tile + TTA)...")
    with torch.no_grad():
        for filename in os.listdir(test_folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_folder, filename)
                img = Image.open(img_path).convert('RGB')
                
                img_tensor = TF.to_tensor(img).to(device).unsqueeze(0)
                
                # 1. 先做 Padding (避免圖片大小不能整除)
                img_tensor, h, w = pad_input(img_tensor)

                # ==========================================
                # 終極組合技：重疊切割 (Tile) + 測試期擴增 (TTA)
                # ==========================================
                
                # 視角 A：原始圖片切割推論
                res_orig = tile_eval(model, img_tensor, tile=TILE_SIZE, tile_overlap=TILE_OVERLAP)
                
                # 視角 B：水平翻轉切割推論
                img_hflip = TF.hflip(img_tensor)
                res_hflip = TF.hflip(tile_eval(model, img_hflip, tile=TILE_SIZE, tile_overlap=TILE_OVERLAP))
                
                # 視角 C：垂直翻轉切割推論
                img_vflip = TF.vflip(img_tensor)
                res_vflip = TF.vflip(tile_eval(model, img_vflip, tile=TILE_SIZE, tile_overlap=TILE_OVERLAP))

                # 將三個結果平均
                restored_tensor = (res_orig + res_hflip + res_vflip) / 3.0

                # ==========================================
                
                # 2. 裁切回原本的圖片大小 (去除 Padding)
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
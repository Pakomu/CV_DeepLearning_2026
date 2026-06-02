import numpy as np

# 載入兩個最強的預測結果
npz_fft = np.load('pred_fft.npz')
npz_old = np.load('pred_24.05.npz')

ensemble_dict = {}

print("開始融合模型預測...")
for filename in npz_old.files:
    # 轉為 float32 以便精確計算
    img_fft = npz_fft[filename].astype(np.float32)
    img_old = npz_old[filename].astype(np.float32)
    
    # 50% / 50% 完美平均融合
    img_ensemble = (img_fft * 0.5) + (img_old * 0.5)
    
    # 限制範圍並轉回 uint8
    ensemble_dict[filename] = np.clip(img_ensemble, 0, 255).astype(np.uint8)

np.savez('pred.npz', **ensemble_dict)
print("✅ 融合完成！你的終極檔案是：pred_final.npz")
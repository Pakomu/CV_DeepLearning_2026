import numpy as np

# 載入兩個不同模型產生的預測結果
npz_A = np.load('pred_1.npz')
npz_B = np.load('pred_2.npz')

ensemble_dict = {}

# 將圖片陣列相加平均
for filename in npz_A.files:
    img_A = npz_A[filename].astype(np.float32)
    img_B = npz_B[filename].astype(np.float32)
    
    # 權重可以自己調，例如 0.6 給強模型，0.4 給弱模型
    img_ensemble = (img_A * 0.6 + img_B * 0.4) 
    
    ensemble_dict[filename] = np.clip(img_ensemble, 0, 255).astype(np.uint8)

np.savez('pred.npz', **ensemble_dict)
print("Ensemble 融合完成！")
import pandas as pd
import matplotlib.pyplot as plt

# 1. 讀取剛才建立的 CSV 檔案
# 請確保 'trainlog.csv' 與此 Python 腳本放在同一個目錄下
file_path = 'F:/CV_DeepLearning_2026/HW2/checkpoints/training_log.csv'
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"找不到檔案：{file_path}，請確認檔案路徑是否正確。")
    exit()
# 💡 新增這行：印出所有欄位名稱，讓您肉眼確認實際的名稱長怎樣
print("目前的欄位名稱有：", df.columns.tolist())

# 💡 新增這行：自動清除所有欄位名稱前後可能隱藏的空白字元
df.columns = df.columns.str.strip()
# 2. 設定畫布大小與風格
plt.figure(figsize=(10, 6))

# 3. 繪製 Training Loss 和 Validation Loss 曲線
# 使用 marker 標出每個 Epoch 的點，讓資料更清晰
plt.plot(df['Epoch'], df['Train_Loss'], label='Training Loss', color='blue', marker='o', linestyle='-', markersize=4)
plt.plot(df['Epoch'], df['Val_Loss'], label='Validation Loss', color='red', marker='s', linestyle='-', markersize=4)

# 4. 加入標題與軸標籤
plt.title('Model Loss Curve', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)

# 5. 加入圖例與背景格線
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# 6. 調整版面並輸出
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=300)  # 將圖表存成高畫質圖片
print("曲線圖已儲存為 'loss_curve.png'")

# 如果您是在 Jupyter Notebook 或支援 GUI 的環境中，可以取消下面這行的註解來直接顯示圖表
# plt.show()
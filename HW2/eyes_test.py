import os
import json
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# --- 參數設定 ---
TEST_IMG_DIR = "./nycu-hw2-data/test"  # 測試集圖片資料夾
JSON_PATH = "pred.json"          # 你的預測結果
OUTPUT_DIR = "./visual_check"          # 畫完框框的圖片要存去哪裡

# 確保輸出資料夾存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 讀取預測結果
with open(JSON_PATH, "r", encoding="utf-8") as f:
    predictions = json.load(f)

# 按照 image_id 把預測結果分組 (因為一張圖可能有多個框)
preds_by_image = {}
for p in predictions:
    img_id = str(p["image_id"])
    if img_id not in preds_by_image:
        preds_by_image[img_id] = []
    preds_by_image[img_id].append(p)

# 2. 隨機挑選 5 張圖片來驗證 (或者你可以改成想看的特定檔名)
all_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg'))]
sample_images = random.sample(all_images, min(5, len(all_images)))

print("--> 開始生成肉眼檢查圖片...")

for filename in sample_images:
    # 抓取檔名中的數字作為 ID (需對應你的 get_image_id 邏輯)
    img_id = str(int(''.join(filter(str.isdigit, os.path.splitext(filename)[0]))))
    
    img_path = os.path.join(TEST_IMG_DIR, filename)
    img = Image.open(img_path).convert("RGB")
    
    # 設定畫布
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img)
    
    # 如果這張圖有預測結果，就畫上去
    if img_id in preds_by_image:
        for pred in preds_by_image[img_id]:
            bbox = pred["bbox"]
            cat_id = pred["category_id"]
            score = pred["score"]
            
            # COCO bbox 格式: [x_min, y_min, width, height]
            x_min, y_min, width, height = bbox
            
            # 建立矩形框 (紅色框，線條粗細為2)
            rect = patches.Rectangle(
                (x_min, y_min), width, height, 
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(rect)
            
            # 在框框左上角加上標籤文字 (類別與分數)
            label_text = f"Class: {cat_id} ({score:.2f})"
            ax.text(
                x_min, y_min - 5, label_text, 
                color='white', fontsize=12, fontweight='bold',
                bbox=dict(facecolor='red', alpha=0.5, edgecolor='none')
            )
    else:
        # 如果 JSON 裡找不到這張圖的預測
        ax.set_title(f"Image {filename}: NO PREDICTIONS", color='red')

    # 隱藏座標軸並存檔
    plt.axis('off')
    out_path = os.path.join(OUTPUT_DIR, f"check_{filename}")
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"已儲存檢查圖: {out_path}")

print(f"--> 檢查完畢！請去 {OUTPUT_DIR} 資料夾查看結果。")
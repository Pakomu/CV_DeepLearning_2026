import json
from pycocotools import mask as maskUtils

def convert_to_coco_format(input_json_path, output_json_path):
    """
    讀取缺少 bbox 的預測結果，計算 bbox 後，儲存為符合 COCO 格式的 JSON。
    """
    print(f"Reading predictions from {input_json_path}...")
    with open(input_json_path, 'r') as f:
        predictions = json.load(f)

    formatted_predictions = []
    
    print("Converting formats and calculating bounding boxes...")
    for pred in predictions:
        # 1. 確保 RLE 的 'counts' 是字串格式 (在 Python 3 中通常是)
        rle = pred['segmentation']
        
        # 為了讓 pycocotools 能正確處理，我們需要確保 counts 是位元組 (bytes)
        # 如果你的 JSON 裡已經是字串，我們把它編碼成 bytes
        if isinstance(rle['counts'], str):
            encoded_counts = rle['counts'].encode('utf-8')
            # 建立一個暫時的 RLE 字典給 pycocotools 用
            temp_rle = {'size': rle['size'], 'counts': encoded_counts}
        else:
             temp_rle = rle

        # 2. 關鍵步驟：從 RLE 計算 Bounding Box
        # toBbox 會回傳 [x, y, width, height]
        bbox = maskUtils.toBbox(temp_rle).tolist() 
        
        # 3. 組裝成正確的字典結構
        formatted_pred = {
            "image_id": int(pred['image_id']),
            "bbox": bbox,
            "score": float(pred['score']),
            "category_id": int(pred['category_id']),
            "segmentation": pred['segmentation'] # 保持原本的 segmentation 不變
        }
        
        formatted_predictions.append(formatted_pred)
        
    # 4. 存檔
    print(f"Saving formatted predictions to {output_json_path}...")
    with open(output_json_path, 'w') as f:
        json.dump(formatted_predictions, f)
    
    print("Conversion complete!")

# ==========================================
# 使用方法：
# 假設你原本輸出的檔案叫做 'my_partial_submission.json'
# 你希望產生正確格式的檔案叫做 'final_submission.json'
# ==========================================
if __name__ == "__main__":
    INPUT_FILE = 'sub2.json' # 替換成你現在的 JSON 檔名
    OUTPUT_FILE = 'test-results.json'     # 轉換後的檔名
    
    convert_to_coco_format(INPUT_FILE, OUTPUT_FILE)
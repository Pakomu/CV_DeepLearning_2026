import json

json_path = "./nycu-hw2-data/train.json"

print("正在讀取 JSON，請稍候...")
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("\n這份 JSON 最外層包含的 Keys:", data.keys())

# 漂亮地印出第一張圖片和第一個標註，indent=4 會自動排版
print("\n--- 偷看第一張圖片 (images) ---")
print(json.dumps(data["images"][0], indent=4))

print("\n--- 偷看第一個標註 (annotations) ---")
print(json.dumps(data["annotations"][0], indent=4))

print("\n--- 偷看類別定義 (categories) ---")
print(json.dumps(data["categories"], indent=4))
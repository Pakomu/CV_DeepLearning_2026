import os
import json
import torch
import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import functional as F
from pycocotools import mask as maskUtils
from tqdm import tqdm

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
# Tuning these is key to beating the strong-baseline (0.35 AP50)
TRAIN_DIR = './train'
TEST_DIR = './test_release'
TEST_MAPPING_JSON = './test_image_name_to_ids.json'
OUTPUT_SUBMISSION = './test-results.json'

NUM_CLASSES = 5  # 4 cell classes + 1 background
BATCH_SIZE = 1   # Keep at 2 to avoid Out of Memory
NUM_EPOCHS = 50  # Increased for better convergence with Cosine Annealing
LEARNING_RATE = 0.0002 # Tuned specifically for AdamW
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MedicalCellDataset(Dataset):
    """
    Custom PyTorch Dataset for loading the Medical Image Instance Segmentation data.
    Handles the specific directory structure: train/[image_name]/image.tif and classX.tif
    """
    def __init__(self, root_dir, is_train=True):
        self.root_dir = root_dir
        self.is_train = is_train
        
        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Directory {self.root_dir} not found. Please ensure data is extracted.")
            
        if self.is_train:
            # Train directory contains subdirectories
            self.image_items = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        else:
            # Test directory contains .tif files directly
            self.image_items = [f for f in os.listdir(root_dir) if f.endswith('.tif')]
        
    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        item_name = self.image_items[idx]
        
        if not self.is_train:
            # For testing: direct path to the .tif file
            img_path = os.path.join(self.root_dir, item_name)
            img = Image.open(img_path).convert("RGB")
            img_tensor = F.to_tensor(img)
            # Return image and filename (e.g., "abc.tif") to map to image_id later
            return img_tensor, item_name

        # For training: path requires going into the subdirectory
        img_dir = os.path.join(self.root_dir, item_name)
        
        # 1. Load the main image
        img_path = os.path.join(img_dir, "image.tif")
        img = Image.open(img_path).convert("RGB")
        img_tensor = F.to_tensor(img) # Converts to [0, 1] range float tensor

        # 2. Load masks for Training
        masks = []
        labels = []
        
        # Iterate through the 4 possible classes
        for class_id in range(1, 5): 
            mask_path = os.path.join(img_dir, f"class{class_id}.tif")
            
            # FIX 1: Check if file exists AND has content (> 0 bytes)
            if os.path.exists(mask_path) and os.path.getsize(mask_path) > 0:
                try:
                    # 改用 OpenCV 來讀取醫療影像的 .tif 檔，並保留原始格式
                    mask_np = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                    
                    # 如果 OpenCV 還是讀不到影像資料，就跳過
                    if mask_np is None:
                        continue
                    
                    # Each unique pixel value > 0 represents a distinct instance
                    obj_ids = np.unique(mask_np)
                    obj_ids = obj_ids[obj_ids > 0] # Remove background (0)
                    
                    for obj_id in obj_ids:
                        # Create a binary mask for this specific instance
                        binary_mask = (mask_np == obj_id)
                        masks.append(binary_mask)
                        labels.append(class_id) # The class ID matches the filename
                except Exception as e:
                    # 把下面這行 print 註解掉，讓它安靜地跳過，不要洗版
                    # print(f"\n[Warning] Skipping corrupted mask: {mask_path} | Error: {e}")
                    continue

        # FIX 2: Handle edge cases safely if an image has 0 valid masks
        if len(masks) > 0:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        else:
            # Create an empty tensor with correct dimensions [0, H, W]
            _, H, W = img_tensor.shape
            masks = torch.empty((0, H, W), dtype=torch.uint8)
            
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 3. Generate bounding boxes dynamically from the masks
        num_objs = len(masks)
        boxes = []
        for i in range(num_objs):
            pos = torch.where(masks[i])
            # Bounding box coordinates: [xmin, ymin, xmax, ymax]
            xmin = torch.min(pos[1])
            xmax = torch.max(pos[1])
            ymin = torch.min(pos[0])
            ymax = torch.max(pos[0])
            
            # Handle edge case where a mask might be just 1 pixel thick
            if xmax == xmin: xmax += 1
            if ymax == ymin: ymax += 1
                
            boxes.append([xmin, ymin, xmax, ymax])
            
        if num_objs > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        else:
            # Mask R-CNN expects empty boxes to have a specific shape (0, 4)
            boxes = torch.empty((0, 4), dtype=torch.float32)

        # Apply robust Data Augmentation (Random Flips) for Training
        
        
        if self.is_train:
            # Random Horizontal Flip
            if torch.rand(1).item() > 0.5:
                img_tensor = F.hflip(img_tensor)
                masks = F.hflip(masks)
                width = img_tensor.shape[2]
                # Flip bounding boxes: xmin becomes width - xmax, xmax becomes width - xmin
                boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                
            # Random Vertical Flip
            if torch.rand(1).item() > 0.5:
                img_tensor = F.vflip(img_tensor)
                masks = F.vflip(masks)
                height = img_tensor.shape[1]
                # Flip bounding boxes: ymin becomes height - ymax, ymax becomes height - ymin
                boxes[:, [1, 3]] = height - boxes[:, [3, 1]]

        # 4. Construct target dictionary required by Mask R-CNN
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = torch.tensor([idx])
        
        # Calculate area for COCO evaluation compatibility (optional but good practice)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target["area"] = area
        target["iscrowd"] = torch.zeros((num_objs,), dtype=torch.int64)

        return img_tensor, target


def get_instance_segmentation_model(num_classes):
    """
    Loads a state-of-the-art Mask R-CNN v2 (better FPN and training recipes).
    """
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    
    model = maskrcnn_resnet50_fpn_v2(
        weights=weights,
        # [關鍵優化 1] 解凍骨幹：允許更新 3 層網路，讓模型適應醫療影像
        trainable_backbone_layers=3,  
        # [關鍵優化 2] 內部縮放：將圖片縮小以節省記憶體，取代暴力刪減細胞
        min_size=256,
        max_size=512,
        rpn_pre_nms_top_n_train=1000,
        rpn_post_nms_top_n_train=300,
        # [關鍵優化 3] 正規顯存控制：讓 PyTorch 每張圖最多只挑 256 個候選框來算 Loss
        box_batch_size_per_image=256  
    )

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Object detection datasets have varying numbers of instances per image, 
    so standard tensor stacking won't work. We return tuples instead.
    """
    return tuple(zip(*batch))

def encode_mask_to_rle(binary_mask):
    """
    Converts a 2D numpy array binary mask to COCO RLE format.
    Ensures the array is Fortran-contiguous as required by pycocotools.
    """
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    # Decode bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def train_model(model, dataloader, optimizer, lr_scheduler, num_epochs):
    print("Starting training...")
    model.train()
    scaler = torch.amp.GradScaler('cuda')
    
    # [關鍵優化 4] 梯度累積步數 (模擬 Batch Size = 4)
    accumulation_steps = 4 
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        optimizer.zero_grad() # 移到迴圈外初始化
        
        for i, (images, targets) in enumerate(progress_bar):
            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                # 將 loss 除以累積步數，保持梯度大小正確
                loss = losses / accumulation_steps 
            
            # 反向傳播並累積梯度
            scaler.scale(loss).backward()
            
            # 當達到累積步數，才真正更新模型權重
            if ((i + 1) % accumulation_steps == 0) or ((i + 1) == len(dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() # 更新完後清空梯度
            
            epoch_loss += losses.item()
            progress_bar.set_postfix(loss=losses.item())
            
        lr_scheduler.step()
        print(f"Epoch {epoch+1} Complete | Average Loss: {epoch_loss/len(dataloader):.4f}")
        torch.cuda.empty_cache()
        
    return model


def generate_submission(model, test_dataloader, mapping_json_path, output_path):
    print("Generating submission file...")
    model.eval()
    
    # Load the mapping dictionary: {"filename.tif": {"id": 1, ...}, ...}
    with open(mapping_json_path, 'r') as f:
        # Convert list of dicts to a lookup dictionary based on filename
        mapping_data = json.load(f)
        filename_to_id = {item['file_name']: item['id'] for item in mapping_data}

    submission = []
    
    with torch.no_grad(): # No gradients needed for inference
        for images, img_names in tqdm(test_dataloader, desc="Evaluating"):
            images = list(img.to(DEVICE) for img in images)
            
            # Forward pass: Model returns list of prediction dictionaries
            predictions = model(images)
            
            for i, prediction in enumerate(predictions):
                img_name = img_names[i]
                # Fallback to image name if .tif is missing, just in case
                img_name_with_ext = img_name + ".tif" if not img_name.endswith('.tif') else img_name
                image_id = filename_to_id.get(img_name_with_ext, -1)
                
                scores = prediction['scores'].cpu().numpy()
                labels = prediction['labels'].cpu().numpy()
                masks = prediction['masks'].cpu().numpy()
                boxes = prediction['boxes'].cpu().numpy()

                # Iterate through all detected objects in this image
                for j in range(len(scores)):
                    # HIGH SCORE TIP: Lowered threshold to 0.35 to increase recall, 
                    # which often heavily boosts the AP50 score in dense medical images.
                    if scores[j] > 0.35: 
                        # Extract the mask and threshold it (Mask R-CNN outputs probabilities)
                        mask_prob = masks[j, 0]
                        binary_mask = (mask_prob > 0.5).astype(np.uint8)
                        
                        # Encode to RLE
                        rle = encode_mask_to_rle(binary_mask)
                        
                        # Format precisely as requested by the assignment
                        pred_dict = {
                            "image_id": int(image_id),
                            "category_id": int(labels[j]),
                            "segmentation": rle,
                            "score": float(scores[j])
                        }
                        submission.append(pred_dict)

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(submission, f)
    print(f"Submission saved to {output_path}")


def main():
    print(f"Using device: {DEVICE}")
    
    # 1. Prepare Datasets and DataLoaders
    train_dataset = MedicalCellDataset(root_dir=TRAIN_DIR, is_train=True)
    test_dataset = MedicalCellDataset(root_dir=TEST_DIR, is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, # 請確保這裡目前是設定 BATCH_SIZE = 1
        shuffle=True, 
        num_workers=0, # [極限省記憶體修改] 在 Windows 上必須設為 0 以避免記憶體洩漏
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=1, # Keep at 1 for testing to simplify mapping logic
        shuffle=False, 
        num_workers=0, # [極限省記憶體修改]
        collate_fn=collate_fn
    )

    # 2. Initialize Model, Optimizer, and Scheduler
    model = get_instance_segmentation_model(NUM_CLASSES)
    model.to(DEVICE)
    
    # Filter parameters to only train those that require gradients
    params = [p for p in model.parameters() if p.requires_grad]
    
    #    # [終極省記憶體修改] 
    # 放棄 AdamW，因為 AdamW 每個參數都要儲存 momentum 跟 variance，會吃掉 2 倍的優化器記憶體。
    # 改回標準的 SGD 配合 Momentum，能省下近 500MB 的 VRAM。
    optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    
    # Cosine Annealing learning rate scheduler: smoothly reduces LR over epochs, avoids plateauing early
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 3. Train the Model
    # (Comment this out and add logic to load a torch.load() state_dict if you want to skip training later)
    model = train_model(model, train_loader, optimizer, lr_scheduler, NUM_EPOCHS)
    
    # Save the trained weights
    torch.save(model.state_dict(), "medical_maskrcnn_weights.pth")
    print("Model weights saved to medical_maskrcnn_weights.pth")

    # 4. Run Inference and Generate Submission
    generate_submission(model, test_loader, TEST_MAPPING_JSON, OUTPUT_SUBMISSION)

if __name__ == "__main__":
    # Ensure you have run: pip install torch torchvision pycocotools Pillow numpy tqdm
    # And that your directory structure matches:
    # ./train/[image_name]/image.tif
    # ./test_release/[image_name].tif
    # ./test_image_name_to_ids.json
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] Missing Files: {e}")
        print("Please ensure your 'train', 'test', and 'json' files are in the same directory as this script.")
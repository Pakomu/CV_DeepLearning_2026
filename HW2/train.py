import csv
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import torch
import numpy as np
import albumentations as A
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import (
    DetrImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
    get_cosine_schedule_with_warmup
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.amp import autocast, GradScaler


CONFIG = {
    "experiment_name": "DeformableDETR_ResNet50",
    "batch_size": 2,
    "accumulation_steps": 8,
    "epochs": 50,
    "resume_epoch": 0,
    "learning_rate": 1e-4,
    "backbone_lr": 1e-5,          
    "weight_decay": 1e-4,
    "checkpoint_dir": "checkpoints",
    "best_model_name": "best_model",
    "resume_checkpoint": "",
    "score_threshold": 0.3,
    "num_workers": 4,
}
train_transforms = A.Compose([
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.GaussNoise(p=0.3),
    A.CLAHE(clip_limit=2.0, p=0.3),
])

# class num. and the dataset path.
NUM_CLASSES = 11          
TRAIN_IMG_DIR = "./nycu-hw2-data/train"
TRAIN_JSON = "./nycu-hw2-data/train.json"
VAL_IMG_DIR = "./nycu-hw2-data/valid"
VAL_JSON = "./nycu-hw2-data/valid.json"

# Image Processor
processor = DetrImageProcessor.from_pretrained(
    "SenseTime/deformable-detr",
    size={"shortest_edge": 480, "longest_edge": 800},
)



class DigitDataset(CocoDetection):

    def __init__(self, img_folder, ann_file, processor, transforms=None):
        super().__init__(img_folder, ann_file)
        self.processor = processor
        self.albu_transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]

        if self.albu_transforms is not None:
            img_np = np.array(img)
            img_h, img_w = img_np.shape[:2]
            
            bboxes = []
            indices = []

            for i, ann in enumerate(target):
                x, y, w, h = ann['bbox']
                
                x = max(0, min(x, img_w))
                y = max(0, min(y, img_h))
                w = max(0, min(w, img_w - x))
                h = max(0, min(h, img_h - y))

                if w > 1 and h > 1:
                    bboxes.append([x, y, w, h])
                    indices.append(i)

            augmented = self.albu_transforms(image=img_np, bboxes=bboxes, indices=indices)
            img = Image.fromarray(augmented["image"])

            new_target = []
            for bbox, orig_idx in zip(augmented['bboxes'], augmented['indices']):
                new_ann = target[int(orig_idx)].copy()
                new_ann['bbox'] = list(bbox)
                new_target.append(new_ann)
            target = new_target
        target = {"image_id": image_id, "annotations": target}
        return img, target


def collate_fn(batch):
    """Batch collator that delegates padding/normalisation to the processor."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    encoding = processor(images=images, annotations=targets, return_tensors="pt")
    return {
        "pixel_values": encoding["pixel_values"],
        "pixel_mask": encoding["pixel_mask"],
        "labels": encoding["labels"],
    }


# ─────────────────────────────────────────────
# Model builder
# ─────────────────────────────────────────────
def build_model(device):
    """Build Deformable-DETR and optionally load a checkpoint."""
    # FIX 4: Initialise FROM a pretrained Deformable-DETR checkpoint so the
    # backbone AND transformer weights are already good.  Only the final
    # classification head (num_labels) is re-initialised.
    config = DeformableDetrConfig(
        backbone="resnet50",
        use_pretrained_backbone=True,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,   # head size changes → ok to ignore
        encoder_layers=2,       # from 4
        decoder_layers=4,       # from 4
        num_queries=100,        # from 300
        auxiliary_loss=True,
    )
    model = DeformableDetrForObjectDetection(config=config)
    model.to(device)
    return model


# ─────────────────────────────────────────────
# Differential learning-rate optimizer
# ─────────────────────────────────────────────
def build_optimizer(model):
    """
    FIX 5: Use differential LR.
    Backbone (pretrained ResNet-50) → very small LR.
    Transformer encoder/decoder → normal LR.
    """
    backbone_params = [
        p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad
    ]
    other_params = [
        p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad
    ]
    param_groups = [
        {"params": backbone_params, "lr": CONFIG["backbone_lr"]},
        {"params": other_params,    "lr": CONFIG["learning_rate"]},
    ]
    return AdamW(param_groups, weight_decay=CONFIG["weight_decay"])


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler,scheduler, device, epoch):
    model.train()
    total_loss = 0.0
    accumulation_steps = CONFIG.get("accumulation_steps", 1)
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} Train")):

        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                labels=[
                    {k: v.to(device) for k, v in t.items()}
                    for t in batch["labels"]
                ],
            )
            loss = outputs.loss / accumulation_steps

        scaler.scale(loss).backward()
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
            # Unscale 梯度，準備做 gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 更新參數
            scaler.step(optimizer)
            scaler.update()
            
            # 更新 Learning Rate
            scheduler.step()
            
            # 清空梯度，準備下一輪的累積
            optimizer.zero_grad()

        total_loss += (loss.item() * accumulation_steps)

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device, epoch):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader, desc=f"Epoch {epoch} Valid"):
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                pixel_values=batch["pixel_values"].to(device),
                pixel_mask=batch["pixel_mask"].to(device),
                labels=[
                    {k: v.to(device) for k, v in t.items()}
                    for t in batch["labels"]
                ],
            )
            total_loss += outputs.loss.item()

    return total_loss / len(loader)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)

    model = build_model(device)
    optimizer = build_optimizer(model)

    # FIX 7: Add LR scheduler – cosine annealing works well for DETR

    csv_file_path = os.path.join(CONFIG["checkpoint_dir"], "training_log.csv")
    
    # 2. 判斷是否為接續訓練，如果不是，就建立新檔案並寫入標題列 (Header)
    is_resuming = bool(CONFIG["resume_checkpoint"]) and os.path.exists(CONFIG["resume_checkpoint"])
    file_mode = 'a' if is_resuming else 'w'
    with open(csv_file_path, mode=file_mode, newline='') as f:
        writer = csv.writer(f)
        if file_mode == 'w':
            writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "Learning_Rate"])
    
    if os.path.exists(CONFIG["resume_checkpoint"]):
        print(f" acclerate...")
        for _ in range(CONFIG["resume_epoch"]):
            scheduler.step()
    train_dataset = DigitDataset(TRAIN_IMG_DIR, TRAIN_JSON, processor, transforms=train_transforms)
    val_dataset = DigitDataset(VAL_IMG_DIR, VAL_JSON, processor, transforms=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        collate_fn=collate_fn,
        num_workers=CONFIG["num_workers"],
        pin_memory=True,
    )

    num_training_steps = len(train_loader) * CONFIG["epochs"]
    num_warmup_steps = 500  # 前 500 個 Batch 慢慢增加學習率

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # --- 處理接續訓練的 Scheduler 快進 ---
    if is_resuming:
        print(" accelerating scheduler...")
        # 注意：因為現在是算 Step (Batch) 而不是 Epoch，所以要乘以 len(train_loader)
        for _ in range(CONFIG["resume_epoch"] * len(train_loader)):
            scheduler.step()
    
    scaler = GradScaler("cuda")


    best_val_loss = float("inf")
    start_epoch = 1
    resume_path = CONFIG["resume_checkpoint"]
    if resume_path and os.path.exists(resume_path):
        print(f"[INFO] Resuming everything from {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        
        # 載入所有狀態
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float("inf"))
        
        # 不需要再跑迴圈 fast-forward scheduler 了，狀態已經完美還原！
        print(f"[INFO] Resumed successfully from epoch {checkpoint['epoch']}.")
    else:
        start_epoch = CONFIG["resume_epoch"] + 1 if CONFIG["resume_epoch"] > 0 else 1





    for epoch in range(start_epoch, CONFIG["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler,scheduler, device, epoch)
        val_loss = validate(model, val_loader, device, epoch)
        # scheduler.step()

        current_lr = scheduler.get_last_lr()
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | LR: {current_lr}"
        )
        with open(csv_file_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, current_lr])

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(CONFIG["checkpoint_dir"], f"{CONFIG['best_model_name']}_{epoch}.pth")
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss
            }
            torch.save(checkpoint, save_path)
            print(f"  --> New best model saved to {save_path} (val_loss={val_loss:.4f})")

        # Also save periodic checkpoint every 10 epochs
        if epoch % 5 == 0:
            ckpt = os.path.join(CONFIG["checkpoint_dir"], f"epoch_{epoch:02d}.pth")
            torch.save(model.state_dict(), ckpt)

    print("Training done!")


if __name__ == "__main__":
    main()
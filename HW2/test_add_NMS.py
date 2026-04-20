"""
inference.py - Generate pred.json for CodaBench submission.
Follows PEP8 style guidelines.

Usage:
    python inference.py \
        --img_dir ./nycu-hw2-data/test \
        --checkpoint checkpoints/best_model.pth \
        --output pred.json \
        --threshold 0.3
"""

import os
import json
import argparse
import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from transformers import (
    DetrImageProcessor,
    DeformableDetrConfig,
    DeformableDetrForObjectDetection,
)
TEST_DIR = "./test_checkpoint/best_model.pth"
OUTPUT = "pred.json"
NUM_CLASSES = 11   # digits 0-9

# Same processor settings as training
processor = DetrImageProcessor.from_pretrained(
    "SenseTime/deformable-detr",
    size={"shortest_edge": 480, "longest_edge": 800},
)


def build_model(checkpoint_path: str, device: torch.device):
    """Load trained Deformable-DETR model from checkpoint."""
    config = DeformableDetrConfig(
        backbone="resnet50",
        use_pretrained_backbone=False, # 推論不需下載權重
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        encoder_layers=6,
        decoder_layers=6,
        num_queries=300,
    )
    model = DeformableDetrForObjectDetection(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint, strict=True)
    model.to(device)
    model.eval()
    print(f"[INFO] Loaded checkpoint from {checkpoint_path}")
    return model


def get_image_ids_from_folder(img_dir: str):
    """
    Returns sorted list of (image_id, file_path) tuples.
    image_id is the integer stem of the filename, e.g. '000123.jpg' -> 123.
    Falls back to 1-based index if filename is not numeric.
    """
    valid_ext = {".jpg", ".jpeg", ".png"}
    files = sorted(
        f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_ext
    )
    results = []
    for i, fname in enumerate(files):
        stem = os.path.splitext(fname)[0]
        try:
            image_id = int(stem)
        except ValueError:
            image_id = i + 1
        results.append((image_id, os.path.join(img_dir, fname)))
    return results



@torch.no_grad()
def run_inference(model, img_dir: str, threshold: float, device: torch.device):
    """
    Run inference over all images in img_dir.

    Returns a list of dicts ready to be written as pred.json:
        [{"image_id": int, "category_id": int, "bbox": [x,y,w,h], "score": float}, ...]
    """
    image_files = get_image_ids_from_folder(img_dir)
    predictions = []

    for image_id, img_path in tqdm(image_files, desc="Inference"):
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        # Pre-process
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward
        outputs = model(**inputs)

        # Post-process: converts logits → scores + COCO [x,y,w,h] boxes
        target_sizes = torch.tensor([[orig_h, orig_w]], device=device)
        results = processor.post_process_object_detection(
            outputs,
            threshold=threshold,
            target_sizes=target_sizes,
        )[0]   # single image

        scores = results["scores"].cpu()
        labels = results["labels"].cpu()
        boxes  = results["boxes"].cpu()   # [x1, y1, x2, y2]

        if len(boxes) > 0:
            # 改用 batched_nms，多傳入 labels 作為參數
            keep_idx = torchvision.ops.batched_nms(boxes, scores, labels, iou_threshold=0.5)
            scores = scores[keep_idx]
            labels = labels[keep_idx]
            boxes = boxes[keep_idx]

        scores = scores.tolist()
        labels = labels.tolist()
        boxes  = boxes.tolist()

        # 【過濾完畢後，再轉成 list 交給後面的迴圈】
        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1
            # FIX: category_id in the dataset starts from 1;
            category_id = int(label)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "score": round(score, 6),
                }
            )

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Deformable-DETR Inference")
    parser.add_argument(
        "--img_dir", default="./nycu-hw2-data/test", help="Test image directory"
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/best_model_25.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output", default="pred.json", help="Output JSON file name"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Confidence score threshold",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model(args.checkpoint, device)
    predictions = run_inference(model, args.img_dir, args.threshold, device)

    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"[INFO] Saved {len(predictions)} predictions to {args.output}")
    print("[INFO] Remember to zip pred.json and submit to CodaBench!")


if __name__ == "__main__":
    main()
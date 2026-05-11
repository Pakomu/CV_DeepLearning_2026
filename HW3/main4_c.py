import os
import csv
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
cv2.utils.logging.setLogLevel(0)
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.roi_heads import (
    RoIHeads,
    maskrcnn_inference,
    project_masks_on_boxes,
)
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as TF
from pycocotools import mask as maskUtils
from tqdm import tqdm
import albumentations as A


# =============================================================================
# CONFIGURATION
# =============================================================================
TRAIN_DIR         = './train'
TEST_DIR          = './test_release'
TEST_MAPPING_JSON = './test_image_name_to_ids.json'
OUTPUT_SUBMISSION = './test-results.json'
CHECKPOINT_PATH   = './best_model.pth'
LOG_CSV           = './training_log.csv'    

NUM_CLASSES        = 5       # background + 4 cell types
BATCH_SIZE         = 1
ACCUMULATION_STEPS = 4       
NUM_EPOCHS         = 80

LEARNING_RATE = 1e-4         
MAX_LR        = 1e-3         
WEIGHT_DECAY  = 1e-4
USE_ONECYCLE  = True        

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture knobs
BACKBONE         = 'resnet101'
TRAINABLE_LAYERS = 3
USE_CASCADE      = True
DICE_WEIGHT      = 0.5
FOCAL_GAMMA      = 2.0


SCORE_THRESHOLD = 0.05       # [FIX 1] was 0.30
NMS_THRESHOLD   = 0.60       # [FIX 2] was 0.50
USE_TTA         = True



_HED_FROM_RGB = np.array([[0.6500286, 0.7044268, 0.2860126],
                            [0.0704478, 0.9897211, 0.1250150],
                            [0.2682671, 0.5704815, 0.7759092]], dtype=np.float32)
_RGB_FROM_HED = np.linalg.inv(_HED_FROM_RGB)


def _augment_hed(image: np.ndarray, alpha: float = 0.05, beta: float = 0.05) -> np.ndarray:
    img    = np.clip(image.astype(np.float32) / 255.0, 1e-6, 1.0)
    OD     = -np.log(img)
    HED    = OD @ _RGB_FROM_HED
    HED    = HED * np.random.uniform(1-alpha, 1+alpha, 3).astype(np.float32) \
                 + np.random.uniform(-beta, beta, 3).astype(np.float32)
    return (np.clip(np.exp(-(HED @ _HED_FROM_RGB)), 0.0, 1.0) * 255).astype(np.uint8)


class HEDStainAugmentation(A.ImageOnlyTransform):
    def __init__(self, alpha=0.05, beta=0.05, always_apply=False, p=0.5):
        super().__init__(p=p)
        self.alpha = alpha
        self.beta  = beta

    def apply(self, img, **params):
        return _augment_hed(img, self.alpha, self.beta)

    def get_transform_init_args_names(self):
        return ("alpha", "beta")


def get_train_transforms() -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ElasticTransform(alpha=80, sigma=80*0.05,
                           interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_REFLECT_101, p=0.4),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        HEDStainAugmentation(alpha=0.05, beta=0.05, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def apply_albumentations(image_np, masks_np, transform):
    if not masks_np:
        return transform(image=image_np, masks=[])['image'], []
    res = transform(image=image_np, masks=masks_np)
    return res['image'], res['masks']



class MedicalCellDataset(Dataset):
    def __init__(self, root_dir: str, is_train: bool = True):
        self.root_dir  = root_dir
        self.is_train  = is_train
        self.transform = get_train_transforms() if is_train else None

        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Directory {root_dir} not found.")

        if is_train:
            self.image_items = sorted(
                d for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )
        else:
            self.image_items = sorted(
                f for f in os.listdir(root_dir) if f.endswith('.tif')
            )

    def __len__(self):
        return len(self.image_items)

    def __getitem__(self, idx):
        item_name = self.image_items[idx]

        if not self.is_train:
            img = Image.open(os.path.join(self.root_dir, item_name)).convert("RGB")
            return TF.to_tensor(img), item_name

        img_dir  = os.path.join(self.root_dir, item_name)
        img_np   = cv2.imread(os.path.join(img_dir, "image.tif"))
        if img_np is None:
            img_np = np.array(Image.open(os.path.join(img_dir, "image.tif")).convert("RGB"))
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        masks_np, labels = [], []
        for cls_id in range(1, 5):
            mp = os.path.join(img_dir, f"class{cls_id}.tif")
            if not (os.path.exists(mp) and os.path.getsize(mp) > 0):
                continue
            try:
                raw = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
                if raw is None:
                    continue
                for obj_id in np.unique(raw):
                    if obj_id == 0:
                        continue
                    masks_np.append((raw == obj_id).astype(np.uint8))
                    labels.append(cls_id)
            except Exception:
                continue

        if self.transform is not None:
            img_np, masks_np = apply_albumentations(img_np, masks_np, self.transform)

        img_tensor = TF.to_tensor(Image.fromarray(img_np))
        H, W = img_tensor.shape[1], img_tensor.shape[2]

        if masks_np:
            masks_t = torch.as_tensor(np.stack(masks_np, 0), dtype=torch.uint8)
        else:
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)
        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        boxes, valid = [], []
        for i, m in enumerate(masks_np):
            ys, xs = np.where(m)
            if not len(xs):
                continue
            xmn, xmx = int(xs.min()), int(xs.max())
            ymn, ymx = int(ys.min()), int(ys.max())
            if xmx == xmn: xmx += 1
            if ymx == ymn: ymx += 1
            boxes.append([xmn, ymn, xmx, ymx])
            valid.append(i)

        if boxes:
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            masks_t  = masks_t[valid]
            labels_t = labels_t[valid]
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            masks_t  = torch.zeros((0, H, W), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        area    = (boxes_t[:, 3]-boxes_t[:, 1]) * (boxes_t[:, 2]-boxes_t[:, 0])
        target  = {
            "boxes": boxes_t, "labels": labels_t, "masks": masks_t,
            "image_id": torch.tensor([idx]),
            "area": area, "iscrowd": torch.zeros(len(labels_t), dtype=torch.int64),
        }
        return img_tensor, target



def combined_maskrcnn_loss(
    mask_logits, proposals, gt_masks, gt_labels,
    mask_matched_idxs, dice_weight=0.5, smooth=1.0,
):
    M            = mask_logits.shape[-1]
    labels_list  = [g[i] for g, i in zip(gt_labels, mask_matched_idxs)]
    targets_list = [project_masks_on_boxes(m.float(), p, i, M)
                    for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)]

    lbl_cat = torch.cat(labels_list, 0)
    tgt_cat = torch.cat(targets_list, 0)

    pos_inds = torch.where(lbl_cat > 0)[0]
    pos_lbls = lbl_cat[pos_inds]
    if tgt_cat.numel() == 0:
        return mask_logits.sum() * 0.0

    pred = mask_logits[pos_inds, pos_lbls - 1]
    gt   = tgt_cat[pos_inds].float()

    bce  = F.binary_cross_entropy_with_logits(pred, gt)

    prob = torch.sigmoid(pred)
    pf   = prob.view(prob.size(0), -1)
    gf   = gt.view(gt.size(0), -1)
    intr = (pf * gf).sum(1)
    dice = 1.0 - (2*intr + smooth) / (pf.sum(1) + gf.sum(1) + smooth)

    return bce + dice_weight * dice.mean()



class CustomRoIHeads(RoIHeads):
    def __init__(self, dice_weight=0.5, focal_gamma=2.0, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            matched_idxs = labels = regression_targets = None

        box_feats          = self.box_roi_pool(features, proposals, image_shapes)
        box_feats          = self.box_head(box_feats)
        cls_logits, box_reg = self.box_predictor(box_feats)

        result: List[Dict[str, torch.Tensor]] = []
        losses: Dict[str, torch.Tensor]       = {}

        if self.training:
            flat_lbl = torch.cat(labels, 0)
            flat_reg = torch.cat(regression_targets, 0)

            ce       = F.cross_entropy(cls_logits, flat_lbl, reduction='none')
            cls_loss = ((1.0 - torch.exp(-ce)) ** self.focal_gamma * ce).mean()

            pos      = torch.where(flat_lbl > 0)[0]
            pos_lbls = flat_lbl[pos]
            N, C     = cls_logits.shape
            box_loss = F.smooth_l1_loss(
                box_reg.reshape(N, C, 4)[pos, pos_lbls],
                flat_reg[pos], beta=1/9, reduction="sum",
            ) / flat_lbl.numel()
            losses   = {"loss_classifier": cls_loss, "loss_box_reg": box_loss}

        else:
            boxes, scores, out_lbls = self.postprocess_detections(
                cls_logits, box_reg, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": out_lbls[i], "scores": scores[i]})

        if self.has_mask():
            if self.training:
                mask_props, pos_midxs = [], []
                for img_id in range(len(proposals)):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_props.append(proposals[img_id][pos])
                    pos_midxs.append(matched_idxs[img_id][pos])
            else:
                mask_props = [r["boxes"] for r in result]

            mf          = self.mask_roi_pool(features, mask_props, image_shapes)
            mf          = self.mask_head(mf)
            mask_logits = self.mask_predictor(mf)

            if self.training:
                losses["loss_mask"] = combined_maskrcnn_loss(
                    mask_logits, mask_props,
                    [t["masks"] for t in targets], [t["labels"] for t in targets],
                    pos_midxs, dice_weight=self.dice_weight,
                )
            else:
                out_lbl_list = [r["labels"] for r in result]
                for mp, r in zip(maskrcnn_inference(mask_logits, out_lbl_list), result):
                    r["masks"] = mp

        return result, losses


class CascadeRoIHeads(CustomRoIHeads):
    def __init__(self, box_head_s2, box_predictor_s2, cascade_fg_iou=0.6, **kwargs):
        super().__init__(**kwargs)
        self.box_head_s2      = box_head_s2
        self.box_predictor_s2 = box_predictor_s2
        self._casc_fg_iou     = cascade_fg_iou

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            matched_idxs = labels = regression_targets = None

        box_feats = self.box_roi_pool(features, proposals, image_shapes)
        box_feats = self.box_head(box_feats)
        cls_logits, box_reg = self.box_predictor(box_feats)

        result = []
        losses = {}

        if self.training:
            flat_lbl, flat_reg = torch.cat(labels, dim=0), torch.cat(regression_targets, dim=0)
            ce = F.cross_entropy(cls_logits, flat_lbl, reduction='none')
            cls_loss = ((1.0 - torch.exp(-ce)) ** self.focal_gamma * ce).mean()

            pos = torch.where(flat_lbl > 0)[0]
            pos_lbls = flat_lbl[pos]
            N, C = cls_logits.shape
            box_loss = F.smooth_l1_loss(
                box_reg.reshape(N, C, 4)[pos, pos_lbls], flat_reg[pos], beta=1.0 / 9, reduction="sum"
            ) / flat_lbl.numel()
            losses.update({"loss_classifier": cls_loss, "loss_box_reg": box_loss})
        else:
            boxes, scores, out_lbls = self.postprocess_detections(cls_logits, box_reg, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": out_lbls[i], "scores": scores[i]})

        if self.training:
            orig_hi, orig_lo = self.proposal_matcher.high_threshold, self.proposal_matcher.low_threshold
            self.proposal_matcher.high_threshold, self.proposal_matcher.low_threshold = self._casc_fg_iou, self._casc_fg_iou - 0.1
            try:
                p2, m2, lbl2, reg2 = self.select_training_samples(proposals, targets)
            finally:
                self.proposal_matcher.high_threshold, self.proposal_matcher.low_threshold = orig_hi, orig_lo

            f2 = self.box_roi_pool(features, p2, image_shapes)
            f2 = self.box_head_s2(f2)
            cl2, br2 = self.box_predictor_s2(f2)

            flat_lbl2, flat_reg2 = torch.cat(lbl2), torch.cat(reg2)
            ce2 = F.cross_entropy(cl2, flat_lbl2, reduction='none')
            cls_loss2 = ((1.0 - torch.exp(-ce2)) ** self.focal_gamma * ce2).mean()

            pos2 = torch.where(flat_lbl2 > 0)[0]
            if len(pos2) > 0:
                N2, C2 = cl2.shape
                box_loss2 = F.smooth_l1_loss(
                    br2.reshape(N2, C2, 4)[pos2, flat_lbl2[pos2]], flat_reg2[pos2], beta=1.0 / 9, reduction="sum"
                ) / flat_lbl2.numel()
            else:
                box_loss2 = cl2.sum() * 0.0
            losses.update({"loss_classifier_s2": cls_loss2 * 0.5, "loss_box_reg_s2": box_loss2 * 0.5})
        else:
            s2_in = [r["boxes"] for r in result]
            if any(b.numel() > 0 for b in s2_in):
                f2 = self.box_roi_pool(features, s2_in, image_shapes)
                if f2.numel() > 0:
                    f2 = self.box_head_s2(f2)
                    cl2, br2 = self.box_predictor_s2(f2)
                    bx2, sc2, lb2 = self.postprocess_detections(cl2, br2, s2_in, image_shapes)
                    for i, r in enumerate(result):
                        r.update({"boxes": bx2[i], "labels": lb2[i], "scores": sc2[i]})

        if self.has_mask():
            if self.training:
                mask_props, pos_midxs = [], []
                for img_id in range(len(proposals)):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_props.append(proposals[img_id][pos])
                    pos_midxs.append(matched_idxs[img_id][pos])

                mask_feats  = self.mask_roi_pool(features, mask_props, image_shapes)
                mask_feats  = self.mask_head(mask_feats)
                mask_logits = self.mask_predictor(mask_feats)
                gt_masks, gt_lbl_list = [t["masks"] for t in targets], [t["labels"] for t in targets]
                
                losses["loss_mask"] = combined_maskrcnn_loss(
                    mask_logits, mask_props, gt_masks, gt_lbl_list, pos_midxs, dice_weight=self.dice_weight
                )
            else:
                mask_props = [r["boxes"] for r in result]
                if any(b.numel() > 0 for b in mask_props):
                    mask_feats  = self.mask_roi_pool(features, mask_props, image_shapes)
                    mask_feats  = self.mask_head(mask_feats)
                    mask_logits = self.mask_predictor(mask_feats)
                    out_lbl_list = [r["labels"] for r in result]
                    for m_prob, r in zip(maskrcnn_inference(mask_logits, out_lbl_list), result):
                        r["masks"] = m_prob

        return result, losses




def _build_roi_heads(
    num_classes:  int,
    backbone_out: int   = 256,
    dice_weight:  float = DICE_WEIGHT,
    focal_gamma:  float = FOCAL_GAMMA,
    use_cascade:  bool  = USE_CASCADE,
) -> CustomRoIHeads:
    from torchvision.models.detection.mask_rcnn import MaskRCNNHeads

    box_roi_pool  = MultiScaleRoIAlign(['0','1','2','3'], output_size=7,  sampling_ratio=2)
    mask_roi_pool = MultiScaleRoIAlign(['0','1','2','3'], output_size=14, sampling_ratio=2)

    res  = box_roi_pool.output_size[0]
    rep  = 1024
    bh   = TwoMLPHead(backbone_out * res**2, rep)
    bp   = FastRCNNPredictor(rep, num_classes)
    mh   = MaskRCNNHeads(backbone_out, (256,256,256,256), dilation=1)
    mp   = MaskRCNNPredictor(256, 256, num_classes)

    kw = dict(
        box_roi_pool=box_roi_pool, box_head=bh, box_predictor=bp,
        fg_iou_thresh=0.5, bg_iou_thresh=0.5,
        batch_size_per_image=512, positive_fraction=0.25,
        bbox_reg_weights=None,
        score_thresh=SCORE_THRESHOLD,      # [FIX 1] was 0.30, now 0.05
        nms_thresh=NMS_THRESHOLD,          # [FIX 2] was 0.50, now 0.60
        detections_per_img=500,            # [FIX 5] was 300
        mask_roi_pool=mask_roi_pool, mask_head=mh, mask_predictor=mp,
        dice_weight=dice_weight, focal_gamma=focal_gamma,
    )

    if use_cascade:
        return CascadeRoIHeads(
            box_head_s2=TwoMLPHead(backbone_out * res**2, rep),
            box_predictor_s2=FastRCNNPredictor(rep, num_classes),
            cascade_fg_iou=0.6,
            **kw,
        )
    return CustomRoIHeads(**kw)


def get_model(num_classes: int = NUM_CLASSES) -> MaskRCNN:
    backbone = resnet_fpn_backbone(
        backbone_name='resnet101',
        weights='IMAGENET1K_V1',
        trainable_layers=TRAINABLE_LAYERS,
    )

    anchor_gen = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )

    roi_heads = _build_roi_heads(
        num_classes=num_classes, backbone_out=backbone.out_channels,
    )

    model = MaskRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_gen,
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1500,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_nms_thresh=0.7,
        rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,

        min_size=[256, 320, 384, 448, 512],   
        max_size=640,                         
    )
    model.roi_heads = roi_heads

    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] Trainable parameters: {n/1e6:.1f}M")
    return model


# =============================================================================
# UTILITIES
# =============================================================================

def collate_fn(batch):
    return tuple(zip(*batch))


def encode_mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = maskUtils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle



def _write_csv_log(
    filepath:         str,
    epoch:            int,
    avg_loss:         float,
    lr:               float,
    is_best:          bool,
    component_losses: Dict[str, float],
    write_header:     bool = False,
) -> None:
    """
    Append one epoch row to `filepath`.

    Columns
    -------
    epoch, avg_loss, lr, is_best, <one column per loss component, alphabetical>

    The header is written only when `write_header=True` (first epoch or new
    file).  All subsequent epochs append without re-writing the header.

    Usage in report
    ---------------
    Load with:  pd.read_csv('training_log.csv')
    Plot with:  df[['epoch','avg_loss','loss_mask','loss_classifier']].plot(x='epoch')
    """
    standard_cols  = ['epoch', 'avg_loss', 'lr', 'is_best']
    component_cols = sorted(component_losses.keys())     # alphabetical = reproducible
    all_cols       = standard_cols + component_cols

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        if write_header:
            writer.writeheader()
        row = {
            'epoch':    epoch,
            'avg_loss': f"{avg_loss:.6f}",
            'lr':       f"{lr:.6e}",
            'is_best':  is_best,
        }
        for col in component_cols:
            row[col] = f"{component_losses.get(col, 0.0):.6f}"
        writer.writerow(row)



def train_model(
    model:        nn.Module,
    dataloader:   DataLoader,
    optimizer:    torch.optim.Optimizer,
    lr_scheduler,
    num_epochs:   int,
    is_onecycle:  bool = True,   # True → step scheduler per optimizer update
) -> nn.Module:
    """
    Training loop with:
      • Per-component loss accumulation across the epoch
      • [FIX 3] OneCycleLR: scheduler.step() called after each optimizer step
      • [FIX 6] CSV logging of all metrics at end of every epoch
    """
    print("Starting training …")
    model.train()
    scaler    = torch.amp.GradScaler('cuda')
    best_loss = float('inf')

    write_header = not os.path.exists(LOG_CSV)

    for epoch in range(num_epochs):
        epoch_total_loss = 0.0
        epoch_comp: Dict[str, float] = {}   
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{num_epochs}")
        optimizer.zero_grad(set_to_none=True)

        for i, (images, targets) in enumerate(pbar):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())
                loss      = losses / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(dataloader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if is_onecycle:
                    lr_scheduler.step()

            # [FIX 6] accumulate per-component losses
            epoch_total_loss += losses.item()
            for k, v in loss_dict.items():
                epoch_comp[k] = epoch_comp.get(k, 0.0) + v.item()
            n_batches += 1

            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items()})
            del images, targets, loss_dict, losses, loss

        if not is_onecycle:
            lr_scheduler.step()

        avg_loss     = epoch_total_loss / n_batches
        avg_comp     = {k: v / n_batches for k, v in epoch_comp.items()}
        current_lr   = optimizer.param_groups[-1]['lr']   # report head LR
        is_best      = avg_loss < best_loss

        # Pretty-print epoch summary with all sub-losses
        comp_str = "  |  ".join(
            f"{k}={v:.4f}" for k, v in sorted(avg_comp.items())
        )
        print(
            f"\nEpoch {epoch+1:03d}/{num_epochs} | "
            f"avg={avg_loss:.4f} | LR={current_lr:.3e} | best={is_best}\n"
            f"  {comp_str}"
        )

        if is_best:
            best_loss = avg_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ Checkpoint saved  (avg_loss={avg_loss:.4f})")

        _write_csv_log(
            filepath=LOG_CSV, epoch=epoch+1,
            avg_loss=avg_loss, lr=current_lr, is_best=is_best,
            component_losses=avg_comp, write_header=write_header,
        )
        write_header = False   # only write header on very first epoch

        torch.cuda.empty_cache()

    return model


@torch.no_grad()
def _predict_single(model, image, device):
    model.eval()
    pred = model([image.to(device)])[0]
    return {k: v.cpu() for k, v in pred.items()}


def _hflip_pred(pred, W):
    p = {k: v.clone() for k, v in pred.items()}
    if p["boxes"].numel():
        p["boxes"][:, [0, 2]] = W - p["boxes"][:, [2, 0]]
    if "masks" in p and p["masks"].numel():
        p["masks"] = p["masks"].flip(-1)
    return p


def _vflip_pred(pred, H):
    p = {k: v.clone() for k, v in pred.items()}
    if p["boxes"].numel():
        p["boxes"][:, [1, 3]] = H - p["boxes"][:, [3, 1]]
    if "masks" in p and p["masks"].numel():
        p["masks"] = p["masks"].flip(-2)
    return p


def _merge_predictions(preds, iou_thresh=NMS_THRESHOLD):
    from torchvision.ops import batched_nms
    if not preds:
        return {"boxes": torch.zeros((0,4)), "labels": torch.zeros(0,dtype=torch.int64),
                "scores": torch.zeros(0), "masks": torch.zeros((0,1,1,1))}
    all_boxes  = torch.cat([p["boxes"]  for p in preds])
    all_labels = torch.cat([p["labels"] for p in preds])
    all_scores = torch.cat([p["scores"] for p in preds])
    all_masks  = torch.cat([p["masks"]  for p in preds]) if "masks" in preds[0] else None
    keep   = batched_nms(all_boxes, all_scores, all_labels, iou_thresh)
    merged = {"boxes": all_boxes[keep], "labels": all_labels[keep], "scores": all_scores[keep]}
    if all_masks is not None:
        merged["masks"] = all_masks[keep]
    return merged


def _filter_low_scores(pred: Dict, thresh: float = 0.05) -> Dict:
    """在進入 TTA 合併前，提早過濾掉低信心框，拯救記憶體與計算量"""
    if "scores" not in pred or pred["scores"].numel() == 0:
        return pred
    keep = pred["scores"] > thresh
    return {k: v[keep] for k, v in pred.items()}

@torch.no_grad()
def predict_with_tta(model: nn.Module, image: torch.Tensor, device) -> Dict:
    """
    執行 TTA (原圖 + 水平翻轉 + 垂直翻轉) 並合併結果。
    """
    _, H, W = image.shape

    p_orig  = _filter_low_scores(_predict_single(model, image, device))
    p_hflip = _filter_low_scores(_hflip_pred(_predict_single(model, TF.hflip(image), device), W))
    p_vflip = _filter_low_scores(_vflip_pred(_predict_single(model, TF.vflip(image), device), H))

    return _merge_predictions([p_orig, p_hflip, p_vflip])



def generate_submission(model, test_dataloader, mapping_json_path, output_path,
                        use_tta=USE_TTA):
    print("Generating submission …")
    model.eval()
    with open(mapping_json_path) as f:
        mapping_data = json.load(f)
    fid = {item['file_name']: item['id'] for item in mapping_data}

    submission = []
    for images, img_names in tqdm(test_dataloader, desc="Inference"):
        for img_tensor, img_name in zip(images, img_names):
            img_name_ext = img_name if img_name.endswith('.tif') else img_name + '.tif'
            image_id     = fid.get(img_name_ext, -1)

            # [加入這行] 使用半精度 (FP16) 進行推論，大幅降低 VRAM 消耗
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                if use_tta:
                    pred = predict_with_tta(model, img_tensor, DEVICE)
                else:
                    with torch.no_grad():
                        pred = model([img_tensor.to(DEVICE)])[0]
                
            pred = {k: v.cpu() for k, v in pred.items()}

            scores = pred['scores'].numpy()
            labels = pred['labels'].numpy()
            masks  = pred['masks'].numpy()
            boxes  = pred['boxes'].numpy()

            for j in range(len(scores)):
                if scores[j] <= SCORE_THRESHOLD:
                    continue
                bm = (masks[j, 0] > 0.5).astype(np.uint8)
                if not np.any(bm):
                    continue
                xmn, ymn, xmx, ymx = boxes[j]
                submission.append({
                    "image_id":     int(image_id),
                    "bbox":         [float(xmn), float(ymn),
                                     float(xmx-xmn), float(ymx-ymn)],
                    "score":        float(scores[j]),
                    "category_id":  int(labels[j]),
                    "segmentation": encode_mask_to_rle(bm),
                })
            

            torch.cuda.empty_cache()

    with open(output_path, 'w') as f:
        json.dump(submission, f)
    print(f"Saved → {output_path}  ({len(submission)} predictions)")




def main():
    print(f"Device: {DEVICE}")

    train_dataset = MedicalCellDataset(TRAIN_DIR, is_train=True)
    test_dataset  = MedicalCellDataset(TEST_DIR,  is_train=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, collate_fn=collate_fn, pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=0, collate_fn=collate_fn,
    )

    model = get_model(NUM_CLASSES)
    model.to(DEVICE)


    backbone_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'backbone' in n]
    head_params     = [p for n, p in model.named_parameters()
                       if p.requires_grad and 'backbone' not in n]

    optimizer = torch.optim.AdamW(
        [
            {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},  # 1e-5
            {'params': head_params,     'lr': LEARNING_RATE},         # 1e-4
        ],
        weight_decay=WEIGHT_DECAY,
        eps=1e-8,
    )


    steps_per_epoch   = math.ceil(len(train_loader) / ACCUMULATION_STEPS)
    total_optim_steps = NUM_EPOCHS * steps_per_epoch

    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = [MAX_LR * 0.1, MAX_LR],   # [backbone_peak, head_peak]
        total_steps     = total_optim_steps,
        pct_start       = 0.15,                      # 15% warmup
        div_factor      = 10.0,                      # start_lr = max_lr / 10
        final_div_factor= 1000.0,                    # final_lr = max_lr / 1000
        anneal_strategy = 'cos',
    )

    print(
        f"[optimizer] AdamW  backbone_lr={LEARNING_RATE*0.1:.1e}  "
        f"head_lr={LEARNING_RATE:.1e}"
    )
    print(
        f"[scheduler] OneCycleLR  max_lr=[{MAX_LR*0.1:.1e}, {MAX_LR:.1e}]  "
        f"total_steps={total_optim_steps}  warmup={int(0.15*total_optim_steps)} steps"
    )

    model = train_model(
        model, train_loader, optimizer, lr_scheduler,
        NUM_EPOCHS, is_onecycle=USE_ONECYCLE,
    )

    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Loaded best checkpoint for inference.")

    generate_submission(model, test_loader, TEST_MAPPING_JSON, OUTPUT_SUBMISSION)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
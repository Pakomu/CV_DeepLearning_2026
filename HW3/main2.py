#!/usr/bin/env python3
"""
main.py  –  Advanced Medical Cell Instance Segmentation
========================================================
Baseline: maskrcnn_resnet50_fpn_v2  |  Target: beat AP50 > 0.35

Modifications over your baseline:
  [MOD 1] Albumentations pipeline: ElasticTransform + GridDistortion +
          HED stain augmentation tailored for H&E microscopy images.
  [MOD 2] ResNet-101-FPN backbone (63M params, well under 200M).
          Optional: swap-in Swin-S backbone (~64M) for further gain.
  [MOD 3] Custom RoI Heads:
            • Focal Loss for box-classification branch
            • BCE + Dice Loss for mask-prediction branch
  [MOD 4] Test-Time Augmentation (TTA) – H-flip + V-flip ensemble
          with per-class NMS merging at inference time.
  [MOD 5] (optional) 2-stage Cascade Box Head (Cai & Vasconcelos, 2018)
          that trains two box heads with increasing IoU thresholds.

Dependencies (pip install):
    torch torchvision albumentations pycocotools tifffile tqdm numpy
"""

import os
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

# ==========================================
# CONFIGURATION
# ==========================================
TRAIN_DIR        = './train'
TEST_DIR         = './test_release'
TEST_MAPPING_JSON = './test_image_name_to_ids.json'
OUTPUT_SUBMISSION = './test-results.json'
CHECKPOINT_PATH  = './best_model.pth'

NUM_CLASSES          = 5       # background + 4 cell types
BATCH_SIZE           = 1
ACCUMULATION_STEPS   = 4       # effective batch = 4
NUM_EPOCHS           = 60
LEARNING_RATE        = 0.005   # SGD + momentum
WEIGHT_DECAY         = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Architecture knobs
BACKBONE               = 'resnet101'   # 'resnet50' | 'resnet101' | 'resnext101_32x8d'
TRAINABLE_LAYERS       = 3
USE_CASCADE            = True          # enable 2-stage cascade box head
DICE_WEIGHT            = 0.5           # weight for Dice term in mask loss
FOCAL_GAMMA            = 2.0           # γ for Focal loss

# Inference knobs
SCORE_THRESHOLD  = 0.30
NMS_THRESHOLD    = 0.50
USE_TTA          = True


# =============================================================================
# [MOD 1]  ADVANCED MEDICAL AUGMENTATIONS
# =============================================================================
# Theoretical justification:
#   Standard flips only double effective dataset size.  Medical microscopy
#   images benefit from:
#   (a) Elastic / grid distortions – simulate tissue-preparation deformations.
#       Ref: Simard et al. (2003); Ronneberger et al. UNet (2015).
#   (b) HED stain augmentation – randomly perturbs the Haematoxylin &
#       Eosin colour channels in the HED optical-density space, mimicking
#       inter-lab staining variation without altering tissue morphology.
#       Ref: Tellez et al. "Quantifying the effects of data augmentation
#       and stain color normalization in convolutional neural networks for
#       computational pathology." Medical Image Analysis (2019).
# =============================================================================

# --- 1a. HED stain perturbation (pure numpy, no extra library needed) --------

# Colour-deconvolution matrix (Ruifrok & Johnston, 2001)
_HED_FROM_RGB = np.array([[0.6500286, 0.7044268, 0.2860126],
                           [0.0704478, 0.9897211, 0.1250150],
                           [0.2682671, 0.5704815, 0.7759092]], dtype=np.float32)
_RGB_FROM_HED = np.linalg.inv(_HED_FROM_RGB)


def _augment_hed(image: np.ndarray, alpha: float = 0.05, beta: float = 0.05) -> np.ndarray:
    """
    Randomly perturb H & E stain channels.
    `image` must be uint8 RGB (H, W, 3).
    """
    img = image.astype(np.float32) / 255.0
    img = np.clip(img, 1e-6, 1.0)
    OD  = -np.log(img)                          # optical density

    HED = OD @ _RGB_FROM_HED                    # colour-deconvolution

    # random multiplicative (alpha) + additive (beta) perturbation
    alpha_v = np.random.uniform(1 - alpha, 1 + alpha, 3).astype(np.float32)
    beta_v  = np.random.uniform(-beta,     beta,      3).astype(np.float32)
    HED     = HED * alpha_v + beta_v

    # reconstruct
    OD_rec = HED @ _HED_FROM_RGB
    img_rec = np.clip(np.exp(-OD_rec), 0.0, 1.0)
    return (img_rec * 255).astype(np.uint8)


class HEDStainAugmentation(A.ImageOnlyTransform):
    """Albumentations-compatible HED stain augmentation."""

    def __init__(self, alpha: float = 0.05, beta: float = 0.05,
                 always_apply: bool = False, p: float = 0.5):
        super().__init__(p=p)
        self.alpha = alpha
        self.beta  = beta

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return _augment_hed(img, self.alpha, self.beta)

    def get_transform_init_args_names(self) -> Tuple[str, ...]:
        return ("alpha", "beta")


# --- 1b. Full training augmentation pipeline ---------------------------------

def get_train_transforms() -> A.Compose:
    """
    Returns an Albumentations Compose that operates on the image and on a
    LIST of binary instance masks simultaneously.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Shape-preserving tissue deformations
        A.ElasticTransform(
            alpha=80, sigma=80 * 0.05,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101, p=0.4
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        # Stain & intensity augmentations
        HEDStainAugmentation(alpha=0.05, beta=0.05, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def apply_albumentations(image_np: np.ndarray,
                          masks_np: List[np.ndarray],
                          transform: A.Compose) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Apply an Albumentations transform to an image and its instance masks.
    `masks_np`: list of (H, W) uint8 binary arrays, one per instance.
    Returns transformed (image_np, masks_np).
    """
    if len(masks_np) == 0:
        res = transform(image=image_np, masks=[])
        return res['image'], []

    result = transform(image=image_np, masks=masks_np)
    return result['image'], result['masks']


# =============================================================================
# [MOD 1 cont.]  DATASET WITH ALBUMENTATIONS
# =============================================================================

class MedicalCellDataset(Dataset):
    """
    Loads the Medical Image Instance Segmentation data.
    Applies Albumentations-based augmentation during training.
    """

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

    def __len__(self) -> int:
        return len(self.image_items)

    def __getitem__(self, idx: int):
        item_name = self.image_items[idx]

        # ── TEST ──────────────────────────────────────────────────────────
        if not self.is_train:
            img_path = os.path.join(self.root_dir, item_name)
            img = Image.open(img_path).convert("RGB")
            return TF.to_tensor(img), item_name

        # ── TRAIN ─────────────────────────────────────────────────────────
        img_dir  = os.path.join(self.root_dir, item_name)
        img_path = os.path.join(img_dir, "image.tif")
        img_np   = cv2.imread(img_path)
        if img_np is None:
            img_np = np.array(Image.open(img_path).convert("RGB"))
        else:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

        masks_np: List[np.ndarray] = []
        labels:   List[int]        = []

        for class_id in range(1, 5):
            mask_path = os.path.join(img_dir, f"class{class_id}.tif")
            if not (os.path.exists(mask_path) and os.path.getsize(mask_path) > 0):
                continue
            try:
                mask_raw = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                if mask_raw is None:
                    continue
                for obj_id in np.unique(mask_raw):
                    if obj_id == 0:
                        continue
                    binary = (mask_raw == obj_id).astype(np.uint8)
                    masks_np.append(binary)
                    labels.append(class_id)
            except Exception:
                continue

        # ── [MOD 1] Apply Albumentations ──────────────────────────────────
        if self.transform is not None:
            img_np, masks_np = apply_albumentations(img_np, masks_np, self.transform)

        img_tensor = TF.to_tensor(Image.fromarray(img_np))
        H, W = img_tensor.shape[1], img_tensor.shape[2]

        # ── Build target dict ─────────────────────────────────────────────
        if len(masks_np) > 0:
            masks_t = torch.as_tensor(np.stack(masks_np, axis=0), dtype=torch.uint8)
        else:
            masks_t = torch.zeros((0, H, W), dtype=torch.uint8)

        labels_t = torch.as_tensor(labels, dtype=torch.int64)

        boxes = []
        valid_mask_idx = []
        for i, m in enumerate(masks_np):
            ys, xs = np.where(m)
            if len(xs) == 0:
                continue
            xmin, xmax = int(xs.min()), int(xs.max())
            ymin, ymax = int(ys.min()), int(ys.max())
            if xmax == xmin: xmax += 1
            if ymax == ymin: ymax += 1
            boxes.append([xmin, ymin, xmax, ymax])
            valid_mask_idx.append(i)

        if len(boxes) > 0:
            boxes_t  = torch.as_tensor(boxes, dtype=torch.float32)
            masks_t  = masks_t[valid_mask_idx]
            labels_t = labels_t[valid_mask_idx]
        else:
            boxes_t  = torch.zeros((0, 4), dtype=torch.float32)
            masks_t  = torch.zeros((0, H, W), dtype=torch.uint8)
            labels_t = torch.zeros((0,), dtype=torch.int64)

        area     = (boxes_t[:, 3] - boxes_t[:, 1]) * (boxes_t[:, 2] - boxes_t[:, 0])
        iscrowd  = torch.zeros(len(labels_t), dtype=torch.int64)

        target = {
            "boxes":    boxes_t,
            "labels":   labels_t,
            "masks":    masks_t,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  iscrowd,
        }
        return img_tensor, target


# =============================================================================
# [MOD 3]  CUSTOM LOSS FUNCTIONS
# =============================================================================
# Theoretical justifications:
#
# Focal Loss (Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017):
#   In a scene with many negative RoI proposals (background), cross-entropy is
#   dominated by easy negatives.  The modulating factor (1−p_t)^γ down-weights
#   well-classified examples so the gradient focuses on the hard, misclassified
#   ones.  This is particularly relevant in dense cell images where most RoIs
#   are background.
#
# Dice Loss (Milletari et al., "V-Net", 3DV 2016):
#   BCE treats each pixel independently.  For small cells the foreground pixels
#   can be heavily outnumbered by background pixels, causing the BCE to be
#   dominated by background and produce over-smooth masks with poor boundaries.
#   Dice loss measures the *overlap* between prediction and ground truth,
#   making it inherently robust to foreground/background imbalance and directly
#   optimizing the F1-like IoU that AP50 depends on.
#
#   Combined BCE + Dice loss retains per-pixel accuracy (BCE) while adding
#   region-level overlap optimization (Dice).
# =============================================================================

def combined_maskrcnn_loss(
    mask_logits:       torch.Tensor,          # (N_all, C, M, M)
    proposals:         List[torch.Tensor],    # pos proposals per image
    gt_masks:          List[torch.Tensor],    # all GT masks per image
    gt_labels:         List[torch.Tensor],    # all GT labels per image
    mask_matched_idxs: List[torch.Tensor],    # GT idx per pos proposal
    dice_weight:       float = 0.5,
    smooth:            float = 1.0,
) -> torch.Tensor:
    """
    Combined BCE + Dice mask loss.
    Replicates torchvision's maskrcnn_loss and adds a Dice term.
    """
    M = mask_logits.shape[-1]

    # ── Gather resized GT masks aligned with predictions ──────────────────
    labels_list = [
        gt_lbl[idxs] for gt_lbl, idxs in zip(gt_labels, mask_matched_idxs)
    ]
    mask_targets_list = [
        project_masks_on_boxes(m.float(), p, i, M)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]

    labels_cat  = torch.cat(labels_list,       dim=0)   # (N_all,)
    targets_cat = torch.cat(mask_targets_list, dim=0)   # (N_all, M, M)

    pos_inds = torch.where(labels_cat > 0)[0]
    pos_lbls = labels_cat[pos_inds]

    if targets_cat.numel() == 0:
        return mask_logits.sum() * 0.0

    pred_logits = mask_logits[pos_inds, pos_lbls - 1]   # (N_pos, M, M)
    gt          = targets_cat[pos_inds].float()          # (N_pos, M, M)

    # ── BCE ───────────────────────────────────────────────────────────────
    bce_loss = F.binary_cross_entropy_with_logits(pred_logits, gt)

    # ── Dice ──────────────────────────────────────────────────────────────
    probs = torch.sigmoid(pred_logits)
    p_f   = probs.view(probs.size(0), -1)
    g_f   = gt.view(gt.size(0), -1)
    inter = (p_f * g_f).sum(dim=1)
    dice  = 1.0 - (2.0 * inter + smooth) / (p_f.sum(dim=1) + g_f.sum(dim=1) + smooth)

    return bce_loss + dice_weight * dice.mean()


# =============================================================================
# [MOD 3 cont.] + [MOD 5]  CUSTOM ROI HEADS  (Focal + Dice + Cascade)
# =============================================================================

class CustomRoIHeads(RoIHeads):
    """
    Drop-in replacement for torchvision RoIHeads that injects:
      • Focal Loss  for the box-classification branch
      • BCE + Dice  for the mask-prediction branch
    """

    def __init__(self, dice_weight: float = 0.5, focal_gamma: float = 2.0, **kwargs):
        super().__init__(**kwargs)
        self.dice_weight = dice_weight
        self.focal_gamma = focal_gamma

    # ── unified forward ──────────────────────────────────────────────────

    def forward(
        self,
        features:     Dict[str, torch.Tensor],
        proposals:    List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets:      Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:

        if self.training:
            proposals, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            matched_idxs = labels = regression_targets = None

        # ── Box head ─────────────────────────────────────────────────────
        box_feats  = self.box_roi_pool(features, proposals, image_shapes)
        box_feats  = self.box_head(box_feats)
        cls_logits, box_reg = self.box_predictor(box_feats)

        result: List[Dict[str, torch.Tensor]] = []
        losses: Dict[str, torch.Tensor]       = {}

        if self.training:
            flat_lbl = torch.cat(labels, dim=0)
            flat_reg = torch.cat(regression_targets, dim=0)

            # [MOD 3a] Focal Loss – replaces cross-entropy
            ce  = F.cross_entropy(cls_logits, flat_lbl, reduction='none')
            pt  = torch.exp(-ce)
            cls_loss = ((1.0 - pt) ** self.focal_gamma * ce).mean()

            # Smooth-L1 box regression (standard, unchanged)
            pos       = torch.where(flat_lbl > 0)[0]
            pos_lbls  = flat_lbl[pos]
            N, C      = cls_logits.shape
            box_loss  = F.smooth_l1_loss(
                box_reg.reshape(N, C, 4)[pos, pos_lbls],
                flat_reg[pos], beta=1.0 / 9, reduction="sum",
            ) / flat_lbl.numel()

            losses = {"loss_classifier": cls_loss, "loss_box_reg": box_loss}

        else:
            boxes, scores, out_lbls = self.postprocess_detections(
                cls_logits, box_reg, proposals, image_shapes
            )
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": out_lbls[i], "scores": scores[i]})

        # ── Mask head ────────────────────────────────────────────────────
        if self.has_mask():
            if self.training:
                mask_props, pos_midxs = [], []
                for img_id in range(len(proposals)):
                    pos = torch.where(labels[img_id] > 0)[0]
                    mask_props.append(proposals[img_id][pos])
                    pos_midxs.append(matched_idxs[img_id][pos])
            else:
                mask_props = [r["boxes"] for r in result]

            mask_feats  = self.mask_roi_pool(features, mask_props, image_shapes)
            mask_feats  = self.mask_head(mask_feats)
            mask_logits = self.mask_predictor(mask_feats)

            if self.training:
                gt_masks     = [t["masks"]  for t in targets]
                gt_lbl_list  = [t["labels"] for t in targets]
                # [MOD 3b] Combined BCE + Dice mask loss
                losses["loss_mask"] = combined_maskrcnn_loss(
                    mask_logits, mask_props, gt_masks, gt_lbl_list,
                    pos_midxs, dice_weight=self.dice_weight,
                )
            else:
                out_lbl_list = [r["labels"] for r in result]
                for m_prob, r in zip(maskrcnn_inference(mask_logits, out_lbl_list), result):
                    r["masks"] = m_prob

        return result, losses


class CascadeRoIHeads(CustomRoIHeads):
    """
    [MOD 5]  Two-stage Cascade Mask R-CNN.

    Theoretical justification:
    A single box head trained at IoU ≥ 0.5 sees proposals whose IoU
    distribution [0.5, 1.0] is mis-matched with inference, where most
    proposals have IoU < 0.6 and only a handful are truly tight.
    Cascade R-CNN (Cai & Vasconcelos, CVPR 2018) addresses this by training
    sequential heads with increasing IoU thresholds (0.5 → 0.6), where each
    stage is trained on the *output* of the previous stage – matching the
    distribution seen at inference.  The result is consistently ~2–4 AP
    improvement on COCO-style metrics with only ~10% parameter increase.

    Here stage 2 trains on the same RPN proposals but with the matcher
    threshold raised to `cascade_fg_iou` (0.6).  During inference, stage 2
    refines stage-1 boxes before the mask head runs.
    """

    def __init__(
        self,
        box_head_s2:     nn.Module,
        box_predictor_s2: nn.Module,
        cascade_fg_iou:  float = 0.6,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.box_head_s2      = box_head_s2
        self.box_predictor_s2 = box_predictor_s2
        self._casc_fg_iou     = cascade_fg_iou

    def forward(
        self,
        features:     Dict[str, torch.Tensor],
        proposals:    List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        targets:      Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]:

        # ── Stage 1 (IoU ≥ 0.5) via parent ──────────────────────────────
        # NOTE: proposals in *this* scope is unchanged by select_training_samples
        # (it reassigns a local copy inside); safe to reuse below.
        result, losses = super().forward(features, proposals, image_shapes, targets)

        if self.training:
            # ── Stage 2 training (IoU ≥ 0.6, same RPN proposals) ────────
            orig_hi = self.proposal_matcher.high_threshold
            orig_lo = self.proposal_matcher.low_threshold
            self.proposal_matcher.high_threshold = self._casc_fg_iou
            self.proposal_matcher.low_threshold  = self._casc_fg_iou - 0.1
            try:
                p2, _, lbl2, reg2 = self.select_training_samples(proposals, targets)
            finally:
                self.proposal_matcher.high_threshold = orig_hi
                self.proposal_matcher.low_threshold  = orig_lo

            f2 = self.box_roi_pool(features, p2, image_shapes)
            f2 = self.box_head_s2(f2)
            cl2, br2 = self.box_predictor_s2(f2)

            flat_lbl2 = torch.cat(lbl2)
            flat_reg2 = torch.cat(reg2)
            ce2 = F.cross_entropy(cl2, flat_lbl2, reduction='none')
            cls_loss2 = ((1.0 - torch.exp(-ce2)) ** self.focal_gamma * ce2).mean()

            pos2 = torch.where(flat_lbl2 > 0)[0]
            if len(pos2) > 0:
                N2, C2 = cl2.shape
                box_loss2 = F.smooth_l1_loss(
                    br2.reshape(N2, C2, 4)[pos2, flat_lbl2[pos2]],
                    flat_reg2[pos2], beta=1.0 / 9, reduction="sum",
                ) / flat_lbl2.numel()
            else:
                box_loss2 = cl2.sum() * 0.0

            losses["loss_classifier_s2"] = cls_loss2 * 0.5
            losses["loss_box_reg_s2"]    = box_loss2 * 0.5

        else:
            # ── Stage 2 inference: refine stage-1 boxes ──────────────────
            s2_in = [r["boxes"] for r in result]
            if any(b.numel() > 0 for b in s2_in):
                f2 = self.box_roi_pool(features, s2_in, image_shapes)
                if f2.numel() > 0:
                    f2 = self.box_head_s2(f2)
                    cl2, br2 = self.box_predictor_s2(f2)
                    bx2, sc2, lb2 = self.postprocess_detections(cl2, br2, s2_in, image_shapes)
                    for i, r in enumerate(result):
                        r["boxes"]  = bx2[i]
                        r["labels"] = lb2[i]
                        r["scores"] = sc2[i]

        return result, losses


# =============================================================================
# [MOD 2]  ARCHITECTURAL UPGRADE — ResNet-101-FPN Backbone
# =============================================================================
# Theoretical justification:
#   ResNet-101 has 23 more layers than ResNet-50 (adding a further residual
#   stage in layer3 with 23 bottleneck blocks vs. 6).  The deeper network
#   learns richer hierarchical features, especially important for texture-heavy
#   H&E images where cell-type discrimination relies on fine chromatin patterns.
#   The FPN (Lin et al., 2017) multi-scale pyramid ensures small cells (which
#   map to early FPN levels) still benefit from deep semantic features via
#   top-down pathways.  Total param count ≈ 63M < 200M limit.
# =============================================================================

def _build_roi_heads(
    num_classes:  int,
    backbone_out: int = 256,
    dice_weight:  float = DICE_WEIGHT,
    focal_gamma:  float = FOCAL_GAMMA,
    use_cascade:  bool  = USE_CASCADE,
) -> CustomRoIHeads:
    """Construct (optionally cascade) RoI heads."""

    # Shared building blocks
    box_roi_pool  = MultiScaleRoIAlign(['0','1','2','3'], output_size=7,  sampling_ratio=2)
    mask_roi_pool = MultiScaleRoIAlign(['0','1','2','3'], output_size=14, sampling_ratio=2)

    resolution     = box_roi_pool.output_size[0]
    representation = 1024
    box_head       = TwoMLPHead(backbone_out * resolution ** 2, representation)
    box_predictor  = FastRCNNPredictor(representation, num_classes)

    from torchvision.models.detection.mask_rcnn import MaskRCNNHeads
    mask_head      = MaskRCNNHeads(backbone_out, (256, 256, 256, 256), dilation=1)
    mask_predictor = MaskRCNNPredictor(256, 256, num_classes)

    common_kwargs = dict(
        # Box
        box_roi_pool            = box_roi_pool,
        box_head                = box_head,
        box_predictor           = box_predictor,
        # Box scoring
        fg_iou_thresh       = 0.5,
        bg_iou_thresh       = 0.5,
        batch_size_per_image= 512,
        positive_fraction   = 0.25,
        bbox_reg_weights        = None,
        score_thresh        = SCORE_THRESHOLD,
        nms_thresh          = NMS_THRESHOLD,
        detections_per_img  = 300,
        # Mask
        mask_roi_pool           = mask_roi_pool,
        mask_head               = mask_head,
        mask_predictor          = mask_predictor,
        # Custom loss weights
        dice_weight             = dice_weight,
        focal_gamma             = focal_gamma,
    )

    if use_cascade:
        box_head_s2      = TwoMLPHead(backbone_out * resolution ** 2, representation)
        box_predictor_s2 = FastRCNNPredictor(representation, num_classes)
        return CascadeRoIHeads(
            box_head_s2=box_head_s2,
            box_predictor_s2=box_predictor_s2,
            cascade_fg_iou=0.6,
            **common_kwargs,
        )
    else:
        return CustomRoIHeads(**common_kwargs)


def get_model(num_classes: int = NUM_CLASSES) -> MaskRCNN:
    """
    Build the full Mask R-CNN with ResNet-101-FPN backbone,
    custom anchor sizes tuned for dense small cells, and
    custom RoI heads (Focal + Dice + optional Cascade).
    """
    # ── Backbone ─────────────────────────────────────────────────────────
    backbone = resnet_fpn_backbone(
        backbone_name     = BACKBONE,
        weights           = 'IMAGENET1K_V1',
        trainable_layers  = TRAINABLE_LAYERS,
    )

    # ── Anchors – tuned for typical cell diameters (8–128 px) ────────────
    anchor_gen = AnchorGenerator(
        sizes        = ((8,), (16,), (32,), (64,), (128,)),
        aspect_ratios= ((0.5, 1.0, 2.0),) * 5,
    )

    # ── RoI heads ────────────────────────────────────────────────────────
    roi_heads = _build_roi_heads(
        num_classes = num_classes,
        backbone_out= backbone.out_channels,   # 256 for FPN
        use_cascade = USE_CASCADE,
    )

    # ── Assemble MaskRCNN ────────────────────────────────────────────────
    model = MaskRCNN(
        backbone             = backbone,
        num_classes          = num_classes,           # we replace roi_heads below
        rpn_anchor_generator = anchor_gen,
        # RPN knobs
        rpn_pre_nms_top_n_train  = 2000,
        rpn_pre_nms_top_n_test   = 1000,
        rpn_post_nms_top_n_train = 2000,
        rpn_post_nms_top_n_test  = 1000,
        rpn_nms_thresh           = 0.7,
        rpn_fg_iou_thresh        = 0.7,
        rpn_bg_iou_thresh        = 0.3,
        rpn_batch_size_per_image = 256,
        rpn_positive_fraction    = 0.5,
        # Image size
        min_size = 256,
        max_size = 512,
    )
    # Inject custom heads (replaces the default heads built by MaskRCNN.__init__)
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


# =============================================================================
# TRAINING LOOP
# =============================================================================

def train_model(model, dataloader, optimizer, lr_scheduler, num_epochs: int):
    print("Starting training …")
    model.train()
    scaler = torch.amp.GradScaler('cuda')

    best_loss = float('inf')

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        optimizer.zero_grad()

        for i, (images, targets) in enumerate(pbar):
            images  = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss_dict = model(images, targets)
                losses    = sum(loss_dict.values())
                loss      = losses / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if ((i + 1) % ACCUMULATION_STEPS == 0) or ((i + 1) == len(dataloader)):
                # Gradient clipping prevents exploding gradients (important with Focal)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += losses.item()
            pbar.set_postfix({k: f"{v.item():.3f}" for k, v in loss_dict.items()})
            del images, targets, loss_dict, losses, loss
        avg = epoch_loss / len(dataloader)
        lr_scheduler.step()
        print(f"Epoch {epoch+1} | avg_loss={avg:.4f} | LR={optimizer.param_groups[0]['lr']:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  ✓ New best model saved ({avg:.4f})")

        torch.cuda.empty_cache()

    return model


# =============================================================================
# [MOD 4]  TEST-TIME AUGMENTATION (TTA)
# =============================================================================
# Theoretical justification:
#   A single-pass prediction is subject to the model's sensitivity to the
#   specific image orientation.  TTA generates predictions for K augmented
#   versions of each test image (original, H-flip, V-flip), maps all
#   predictions back to the original coordinate frame, concatenates them, and
#   applies per-class NMS.  This reduces variance and consistently yields
#   +0.5–2 AP points at zero extra training cost.
#   Ref: standard practice; surveyed in Lim et al. (2020) "Fast AutoAugment".
# =============================================================================

@torch.no_grad()
def _predict_single(model: nn.Module, image: torch.Tensor, device) -> Dict:
    """Run model on a single image tensor (C, H, W) and return raw prediction."""
    model.eval()
    pred = model([image.to(device)])[0]
    return {k: v.cpu() for k, v in pred.items()}


def _hflip_pred(pred: Dict, W: int) -> Dict:
    """Map H-flip prediction back to original coordinates."""
    p = {k: v.clone() for k, v in pred.items()}
    if p["boxes"].numel():
        p["boxes"][:, [0, 2]] = W - p["boxes"][:, [2, 0]]
    if "masks" in p and p["masks"].numel():
        p["masks"] = p["masks"].flip(-1)
    return p


def _vflip_pred(pred: Dict, H: int) -> Dict:
    """Map V-flip prediction back to original coordinates."""
    p = {k: v.clone() for k, v in pred.items()}
    if p["boxes"].numel():
        p["boxes"][:, [1, 3]] = H - p["boxes"][:, [3, 1]]
    if "masks" in p and p["masks"].numel():
        p["masks"] = p["masks"].flip(-2)
    return p


def _merge_predictions(preds: List[Dict], iou_thresh: float = NMS_THRESHOLD) -> Dict:
    """
    Merge N predictions via per-class NMS.
    For each kept box, the mask is taken from whichever prediction had the
    highest score for that detection.
    """
    from torchvision.ops import batched_nms

    if not preds:
        return {"boxes": torch.zeros((0, 4)), "labels": torch.zeros(0, dtype=torch.int64),
                "scores": torch.zeros(0), "masks": torch.zeros((0, 1, 1, 1))}

    all_boxes  = torch.cat([p["boxes"]  for p in preds], dim=0)
    all_labels = torch.cat([p["labels"] for p in preds], dim=0)
    all_scores = torch.cat([p["scores"] for p in preds], dim=0)
    all_masks  = torch.cat([p["masks"]  for p in preds], dim=0) if "masks" in preds[0] else None

    keep = batched_nms(all_boxes, all_scores, all_labels, iou_thresh)

    merged = {
        "boxes":  all_boxes[keep],
        "labels": all_labels[keep],
        "scores": all_scores[keep],
    }
    if all_masks is not None:
        merged["masks"] = all_masks[keep]
    return merged


@torch.no_grad()
def predict_with_tta(model: nn.Module, image: torch.Tensor, device) -> Dict:
    """
    Run TTA (original + H-flip + V-flip) and merge results.
    `image`: (C, H, W) float tensor in [0, 1].
    """
    _, H, W = image.shape

    p_orig  = _predict_single(model, image, device)
    p_hflip = _hflip_pred(_predict_single(model, TF.hflip(image), device), W)
    p_vflip = _vflip_pred(_predict_single(model, TF.vflip(image), device), H)

    return _merge_predictions([p_orig, p_hflip, p_vflip])


# =============================================================================
# INFERENCE & SUBMISSION GENERATION
# =============================================================================

def generate_submission(
    model:            nn.Module,
    test_dataloader:  DataLoader,
    mapping_json_path: str,
    output_path:      str,
    use_tta:          bool = USE_TTA,
):
    print("Generating submission file …")
    model.eval()

    with open(mapping_json_path) as f:
        mapping_data = json.load(f)
    filename_to_id = {item['file_name']: item['id'] for item in mapping_data}

    submission = []

    for images, img_names in tqdm(test_dataloader, desc="Inference"):
        for img_tensor, img_name in zip(images, img_names):
            img_name_ext = img_name if img_name.endswith('.tif') else img_name + '.tif'
            image_id     = filename_to_id.get(img_name_ext, -1)

            if use_tta:
                prediction = predict_with_tta(model, img_tensor, DEVICE)
            else:
                with torch.no_grad():
                    prediction = model([img_tensor.to(DEVICE)])[0]
                prediction = {k: v.cpu() for k, v in prediction.items()}

            scores = prediction['scores'].numpy()
            labels = prediction['labels'].numpy()
            masks  = prediction['masks'].numpy()
            boxes  = prediction['boxes'].numpy()

            for j in range(len(scores)):
                if scores[j] <= SCORE_THRESHOLD:
                    continue
                mask_prob   = masks[j, 0]
                binary_mask = (mask_prob > 0.5).astype(np.uint8)
                
                # 確保 mask 不是空的
                if not np.any(binary_mask):
                    continue

                # 直接取用模型預測的 boxes，並轉換成 COCO 格式: [x, y, width, height]
                xmin, ymin, xmax, ymax = boxes[j]
                bbox = [
                    float(xmin), 
                    float(ymin), 
                    float(xmax - xmin), 
                    float(ymax - ymin)
                ]

                submission.append({
                    "image_id":     int(image_id),
                    "bbox":         bbox,
                    "score":        float(scores[j]),
                    "category_id":  int(labels[j]),
                    "segmentation": encode_mask_to_rle(binary_mask)
                })

    with open(output_path, 'w') as f:
        json.dump(submission, f)
    print(f"Submission saved → {output_path}  ({len(submission)} predictions)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print(f"Device: {DEVICE}")

    # ── Datasets ─────────────────────────────────────────────────────────
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

    # ── Model ─────────────────────────────────────────────────────────────
    model = get_model(NUM_CLASSES)
    model.to(DEVICE)

    # ── Optimiser  (SGD saves ≈500MB vs AdamW on VRAM-constrained setups) ─
    params        = [p for p in model.parameters() if p.requires_grad]
    optimizer     = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9,
                                     weight_decay=WEIGHT_DECAY)
    lr_scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    # ── Train ─────────────────────────────────────────────────────────────
    model = train_model(model, train_loader, optimizer, lr_scheduler, NUM_EPOCHS)

    # ── Load best checkpoint ──────────────────────────────────────────────
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Loaded best checkpoint for inference.")

    # ── Inference ─────────────────────────────────────────────────────────
    generate_submission(model, test_loader, TEST_MAPPING_JSON, OUTPUT_SUBMISSION)


if __name__ == "__main__":
    # pip install torch torchvision albumentations pycocotools tifffile tqdm numpy opencv-python
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
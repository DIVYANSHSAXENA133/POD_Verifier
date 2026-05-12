"""
POD Quality Classifier — Augmentation Pipelines.

Provides train/val transforms using albumentations.
Includes Mixup and CutMix as batch-level augmentations.
"""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch


def get_train_transforms(input_size: int = 320) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(size=(input_size, input_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.25),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.2),
        A.GaussNoise(p=0.2),
        A.ImageCompression(quality_range=(50, 95), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(input_size: int = 320) -> A.Compose:
    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def mixup_batch(
    images: torch.Tensor,
    targets: dict[str, torch.Tensor],
    alpha: float = 0.2,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply Mixup augmentation to a batch."""
    if alpha <= 0:
        return images, targets

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    mixed_targets = {
        key: lam * val + (1 - lam) * val[index]
        for key, val in targets.items()
    }
    return mixed_images, mixed_targets


def cutmix_batch(
    images: torch.Tensor,
    targets: dict[str, torch.Tensor],
    alpha: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Apply CutMix augmentation to a batch."""
    if alpha <= 0:
        return images, targets

    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    _, _, h, w = images.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)

    cy = np.random.randint(h)
    cx = np.random.randint(w)

    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)

    mixed_images = images.clone()
    mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]

    lam_adjusted = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)
    mixed_targets = {
        key: lam_adjusted * val + (1 - lam_adjusted) * val[index]
        for key, val in targets.items()
    }
    return mixed_images, mixed_targets

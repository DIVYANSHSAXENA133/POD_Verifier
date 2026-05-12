"""
POD Quality Classifier — Model Definitions.

Contains:
- PODNet: Custom CNN from scratch (production)
- MultiHeadEfficientNet: Fine-tuned EfficientNet-B0 (prototype)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ATTRIBUTE_NAMES = ["context_valid", "package_visible", "label_readable", "image_clarity"]
ATTRIBUTE_WEIGHTS = torch.tensor([0.35, 0.30, 0.20, 0.15])


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y


class ConvBlock(nn.Module):
    """Standard conv block: Conv -> BN -> ReLU -> optional MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        pool: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualConvBlock(nn.Module):
    """Conv block with residual skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Identity()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return self.pool(out)


class ClassificationHead(nn.Module):
    """Per-attribute classification head: FC 512->128->1."""

    def __init__(self, in_features: int = 512, hidden: int = 128):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class PODNet(nn.Module):
    """
    Custom CNN for POD quality classification.
    5 conv blocks (2 standard + 2 residual + 1 SE-attention) + 4 classification heads.
    ~8M parameters, 320x320 input.
    """

    def __init__(self, num_attributes: int = 4, dropout: float = 0.4):
        super().__init__()
        self.num_attributes = num_attributes

        self.block1 = ConvBlock(3, 64, kernel_size=3, stride=2, pool=True)
        self.block2 = ConvBlock(64, 128, kernel_size=3, stride=1, pool=True)
        self.block3 = ResidualConvBlock(128, 256)
        self.block4 = ResidualConvBlock(256, 512)
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            SEBlock(512, reduction=16),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)

        self.heads = nn.ModuleList([ClassificationHead(512, 128) for _ in range(num_attributes)])

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> dict:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x).flatten(1)
        x = self.dropout(x)

        logits = {
            ATTRIBUTE_NAMES[i]: self.heads[i](x).squeeze(-1)
            for i in range(self.num_attributes)
        }
        return logits

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns composite POD quality score (0-1)."""
        logits = self.forward(x)
        probs = {k: torch.sigmoid(v) for k, v in logits.items()}
        weights = ATTRIBUTE_WEIGHTS.to(x.device)
        score = sum(probs[ATTRIBUTE_NAMES[i]] * weights[i] for i in range(self.num_attributes))
        return score


class MultiHeadEfficientNet(nn.Module):
    """
    Fine-tuned EfficientNet-B0 with 4 classification heads.
    Used as prototype/baseline on M4 (MPS).
    """

    def __init__(self, num_attributes: int = 4, pretrained: bool = True):
        super().__init__()
        self.num_attributes = num_attributes

        import timm

        self.backbone = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features

        self.heads = nn.ModuleList([ClassificationHead(feature_dim, 128) for _ in range(num_attributes)])

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        logits = {
            ATTRIBUTE_NAMES[i]: self.heads[i](features).squeeze(-1)
            for i in range(self.num_attributes)
        }
        return logits

    def predict_score(self, x: torch.Tensor) -> torch.Tensor:
        """Returns composite POD quality score (0-1)."""
        logits = self.forward(x)
        probs = {k: torch.sigmoid(v) for k, v in logits.items()}
        weights = ATTRIBUTE_WEIGHTS.to(x.device)
        score = sum(probs[ATTRIBUTE_NAMES[i]] * weights[i] for i in range(self.num_attributes))
        return score

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

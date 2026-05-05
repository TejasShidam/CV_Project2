from __future__ import annotations

import torch
from torch import nn


class CNNBackbone(nn.Module):
    """ResNet-18 (or EfficientNet-B0) feature extractor.

    Uses ImageNet-pretrained weights for fast convergence on small datasets.
    The final FC + AvgPool are stripped; output is a spatial feature map
    projected to `out_dim` channels via a 1×1 convolution.

    Args:
        out_dim:       Channel dimension fed into the Transformer decoder.
        backbone:      ``"resnet18"`` (default) or ``"efficientnet_b0"``.
        pretrained:    Load ImageNet weights (strongly recommended).
        freeze_layers: Number of ResNet layer groups to freeze (0–4).
                       Set >0 to stabilise early training on tiny datasets.
    """

    def __init__(
        self,
        out_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_layers: int = 2,
    ) -> None:
        super().__init__()

        if backbone == "efficientnet_b0":
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            base = efficientnet_b0(weights=weights)
            # strip classifier; keep features block → (B, 1280, H, W)
            self.features = base.features
            self.proj = nn.Conv2d(1280, out_dim, kernel_size=1)
        else:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base = resnet18(weights=weights)
            # strip AvgPool + FC → (B, 512, H, W)
            self.features = nn.Sequential(*list(base.children())[:-2])
            self.proj = nn.Conv2d(512, out_dim, kernel_size=1)

            # Optionally freeze early layer groups to avoid destroying pretrained
            # representations while the decoder is warming up.
            layers = list(self.features.children())
            for layer in layers[:freeze_layers]:
                for param in layer.parameters():
                    param.requires_grad = False

        self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        feat = self.bn(self.proj(feat))
        return feat

"""Residual GroupNorm U-Net used for London daily PM2.5 mapping."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _groups(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.dropout = nn.Dropout2d(dropout) if dropout else nn.Identity()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.se = SqueezeExcitation(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        x = F.silu(self.norm1(self.conv1(x)), inplace=True)
        x = self.dropout(x)
        x = self.norm2(self.conv2(x))
        x = self.se(x)
        return F.silu(x + identity, inplace=True)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2), ResidualBlock(in_channels, out_channels, dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ResidualBlock(out_channels + skip_channels, out_channels, dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.block(torch.cat([skip, x], dim=1))


class LondonResidualUNet(nn.Module):
    """Predicts a correction to a leakage-free covariate baseline map."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int = 32,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        b = base_channels
        self.encoder1 = ResidualBlock(in_channels, b, dropout * 0.5)
        self.encoder2 = Down(b, b * 2, dropout * 0.75)
        self.encoder3 = Down(b * 2, b * 4, dropout)
        self.bottleneck = Down(b * 4, b * 8, dropout * 1.25)
        self.decoder3 = Up(b * 8, b * 4, b * 4, dropout)
        self.decoder2 = Up(b * 4, b * 2, b * 2, dropout * 0.75)
        self.decoder1 = Up(b * 2, b, b, dropout * 0.5)
        self.head = nn.Conv2d(b, 1, 1)
        # Start exactly at the strong tabular baseline; learn only useful correction.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        xb = self.bottleneck(x3)
        x = self.decoder3(xb, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)
        return self.head(x)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

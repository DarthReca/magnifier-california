# Implementation of Simone Monaco from https://github.com/dbdmg/rescue

from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn

from .unet_parts import *


class UNet(nn.Module):
    def __init__(
        self,
        n_channels,
        n_classes,
    ):
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(n_channels)
        self.decoder = UNetDecoder(n_classes)

    def forward(self, x):
        x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x5, x1, x2, x3, x4)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, n_channels: int) -> None:
        super().__init__()

        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNetDecoder(nn.Module):
    def __init__(self, num_classes: int, n_channels: int) -> None:
        super().__init__()
        multiplier = n_channels // 1024

        self.up1 = up(1024 * multiplier, 512 * multiplier)
        self.up2 = up(512 * multiplier, 256 * multiplier)
        self.up3 = up(256 * multiplier, 128 * multiplier)
        self.up4 = up(128 * multiplier, 64 * multiplier)
        self.outc = outconv(64 * multiplier, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

import torch
import torch.nn as nn


# Difference module from ChangeFormer
class ConvDiff(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):
        super(ConvDiff, self).__init__()
        self.diff_module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.diff_module(x)

# Public model architecture definition
# Extracted from the exact training implementation used in the associated study.
# Training, evaluation, plotting, and data-processing code are not included.

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out = out + identity
        out = self.relu(out)

        return out


class CustomEncoder(nn.Module):
    """
    Lightweight encoder for DeepLabV3+.

    Input size:
        512 x 512

    Low-level feature:
        128 x 128

    High-level feature:
        32 x 32
    """

    def __init__(self, base_channels=32):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = nn.Sequential(
            ConvBNReLU(3, c1, kernel_size=3, stride=2, padding=1),    # 256
            ConvBNReLU(c1, c1, kernel_size=3, stride=1, padding=1),   # 256
        )

        self.layer1 = nn.Sequential(
            ResidualBlock(c1, c2, stride=2),  # 128
            ResidualBlock(c2, c2, stride=1),
        )

        self.layer2 = nn.Sequential(
            ResidualBlock(c2, c3, stride=2),  # 64
            ResidualBlock(c3, c3, stride=1),
        )

        self.layer3 = nn.Sequential(
            ResidualBlock(c3, c4, stride=2),  # 32
            ResidualBlock(c4, c4, stride=1),
        )

        self.layer4 = nn.Sequential(
            ResidualBlock(c4, c5, stride=1),  # 32
            ResidualBlock(c5, c5, stride=1),
        )

        self.low_level_channels = c2
        self.high_level_channels = c5

    def forward(self, x):
        x = self.stem(x)

        low_level = self.layer1(x)

        x = self.layer2(low_level)
        x = self.layer3(x)
        high_level = self.layer4(x)

        return low_level, high_level


class ASPPConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[-2:]

        x = self.pool(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            size=size,
            mode="bilinear",
            align_corners=True,
        )

        return x


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        self.branch2 = ASPPConv(in_channels, out_channels, dilation=6)
        self.branch3 = ASPPConv(in_channels, out_channels, dilation=12)
        self.branch4 = ASPPConv(in_channels, out_channels, dilation=18)
        self.branch5 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(
            nn.Conv2d(
                out_channels * 5,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        features = [
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            self.branch5(x),
        ]

        x = torch.cat(features, dim=1)
        x = self.project(x)

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32):
        super().__init__()

        self.encoder = CustomEncoder(base_channels=base_channels)

        self.aspp = ASPP(
            in_channels=self.encoder.high_level_channels,
            out_channels=256,
        )

        self.low_level_project = nn.Sequential(
            nn.Conv2d(
                self.encoder.low_level_channels,
                48,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            ConvBNReLU(256 + 48, 256, kernel_size=3, stride=1, padding=1),
            ConvBNReLU(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channels, kernel_size=1),
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        low_level, high_level = self.encoder(x)

        x = self.aspp(high_level)

        x = F.interpolate(
            x,
            size=low_level.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        low_level = self.low_level_project(low_level)

        x = torch.cat([x, low_level], dim=1)
        x = self.decoder(x)

        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=True,
        )

        return x


# ============================================================
# 7. Loss Functions
# ============================================================


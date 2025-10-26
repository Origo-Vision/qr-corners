from __future__ import annotations

import torch
from torch import nn


class CBR(nn.Module):
    def __init__(self: CBR, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.cbr = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self: CBR, x: torch.Tensor) -> torch.Tensor:
        return self.cbr(x)


class Down(nn.Module):
    def __init__(self: Down, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            CBR(in_channels=in_channels, out_channels=out_channels),
        )

    def forward(self: Down, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class Up(nn.Module):
    def __init__(self: Up, in_channels: int, out_channels) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2
        )
        self.cbr = CBR(in_channels=in_channels, out_channels=out_channels)

    def forward(self: Up, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.cbr(x)


class UNet(nn.Module):
    """
    Classic, full sized U-Net model.
    """

    def __init__(self: UNet, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.input = CBR(in_channels=in_channels, out_channels=64)
        self.down1 = Down(in_channels=64, out_channels=128)
        self.down2 = Down(in_channels=128, out_channels=256)
        self.down3 = Down(in_channels=256, out_channels=512)
        self.down4 = Down(in_channels=512, out_channels=1024)

        self.up1 = Up(in_channels=1024, out_channels=512)
        self.up2 = Up(in_channels=512, out_channels=256)
        self.up3 = Up(in_channels=256, out_channels=128)
        self.up4 = Up(in_channels=128, out_channels=64)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self: UNet, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.output(x)

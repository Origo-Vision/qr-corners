from __future__ import annotations

import pathlib

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


class ASPP(nn.Module):
    def __init__(self: ASPP, num_channels: int, dilations: list[int]) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=3,
                    padding=d,
                    dilation=d,
                    bias=False,
                )
                for d in dilations
            ]
        )

        self.conv = nn.Conv2d(
            in_channels=num_channels * len(dilations),
            out_channels=num_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(num_features=num_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self: ASPP, x: torch.Tensor) -> torch.Tensor:
        features = [conv(x) for conv in self.blocks]
        x = torch.cat(features, dim=1)
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


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


class LargeUNet(nn.Module):
    """
    Classic, full sized U-Net model.
    """

    def __init__(self: LargeUNet, in_channels: int, out_channels: int) -> None:
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

    def forward(self: LargeUNet, x: torch.Tensor) -> torch.Tensor:
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


class SmallUNet(nn.Module):
    def __init__(self: SmallUNet, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.input = CBR(in_channels=in_channels, out_channels=32)
        self.down1 = Down(in_channels=32, out_channels=64)
        self.down2 = Down(in_channels=64, out_channels=128)
        self.down3 = Down(in_channels=128, out_channels=256)

        self.up1 = Up(in_channels=256, out_channels=128)
        self.up2 = Up(in_channels=128, out_channels=64)
        self.up3 = Up(in_channels=64, out_channels=32)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self: SmallUNet, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.output(x)


class TinyUNet(nn.Module):
    def __init__(self: TinyUNet, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.input = CBR(in_channels=in_channels, out_channels=16)
        self.down1 = Down(in_channels=16, out_channels=32)
        self.down2 = Down(in_channels=32, out_channels=64)
        self.down3 = Down(in_channels=64, out_channels=128)

        self.aspp = ASPP(num_channels=128, dilations=[1, 2, 4, 8])

        self.up1 = Up(in_channels=128, out_channels=64)
        self.up2 = Up(in_channels=64, out_channels=32)
        self.up3 = Up(in_channels=32, out_channels=16)

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self: TinyUNet, x: torch.Tensor) -> torch.Tensor:
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x4 = self.aspp(x4)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        return self.output(x)


def kaiming_init(model: nn.Module) -> None:
    if isinstance(model, nn.Conv2d):
        nn.init.kaiming_normal_(model.weight, nonlinearity="relu")
        if not model.bias is None:
            nn.init.constant_(model.bias, 0.0)
    elif isinstance(model, nn.BatchNorm2d):
        nn.init.constant_(model.weight, 1.0)
        nn.init.constant_(model.bias, 0.0)


def empty(size: str) -> nn.Module:
    """
    Create an empty model with the given size.

    Parameters:
        size: The model size.

    Returns:
        The empty model.
    """
    model = None
    if size == "tiny":
        model = TinyUNet(in_channels=3, out_channels=4)
    elif size == "small":
        model = SmallUNet(in_channels=3, out_channels=4)
    elif size == "large":
        model = LargeUNet(in_channels=3, out_channels=4)
    else:
        assert False

    model.apply(kaiming_init)

    return model


def load(size: str, weights: pathlib.Path) -> nn.Module:
    """
    Load a model with weights.

    Parameters:
        size: The model size.
        weights: The path to the weights.

    Returns:
        The model.
    """
    model = empty(size)
    model.load_state_dict(torch.load(weights, weights_only=True))

    return model


def save(model: nn.Module, weights: pathlib.Path) -> None:
    """
    Save the model's weights.

    Parameters:
        model: The model.
        weights: The path to the weights.
    """
    torch.save(model.state_dict(), weights)

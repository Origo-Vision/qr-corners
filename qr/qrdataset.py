from __future__ import annotations

import glob
import pathlib

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class QRDataset(Dataset):
    def __init__(
        self: QRDataset, datadir: pathlib.Path, augmentations: Compose = Compose([])
    ) -> None:
        super().__init__()

        self.augmentations = augmentations

        images_dir = datadir / "images"
        self.images = list(
            map(pathlib.Path, sorted(glob.glob(str(images_dir / "*.png"))))
        )

        heatmaps_dir = datadir / "heatmaps"
        self.heatmaps = list(
            map(pathlib.Path, sorted(glob.glob(str(heatmaps_dir / "*.npy"))))
        )

        self.length = len(self.images)

        assert self.length == len(self.heatmaps)

    def __len__(self: QRDataset) -> int:
        return self.length

    def __getitem__(
        self: QRDataset, index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_COLOR_RGB)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        heatmap = np.load(self.heatmaps[index])
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        return self.augmentations(image), heatmap

    def multi_sample(
        self: QRDataset, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        images, heatmaps, points = [], [], []

        for index in indices:
            image, heatmap = self[index.item()]

            images.append(image.unsqueeze(0))
            heatmaps.append(heatmap.unsqueeze(0))

        return torch.cat(images, 0), torch.cat(heatmaps, 0)

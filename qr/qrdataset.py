from __future__ import annotations

import glob
import pathlib

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import Dataset


class QRDataset(Dataset):
    def __init__(self: QRDataset, datadir: pathlib.Path) -> None:
        super().__init__()

        images_dir = datadir / "images"
        self.images = list(
            map(pathlib.Path, sorted(glob.glob(str(images_dir / "*.png"))))
        )

        heatmaps_dir = datadir / "heatmaps"
        self.heatmaps = list(
            map(pathlib.Path, sorted(glob.glob(str(heatmaps_dir / "*.npy"))))
        )

        points_dir = datadir / "points"
        self.points = list(
            map(pathlib.Path, sorted(glob.glob(str(points_dir / "*.npy"))))
        )

        self.length = len(self.images)

        assert self.length == len(self.heatmaps)
        assert self.length == len(self.points)

    def __len__(self: QRDataset) -> int:
        return self.length

    def __getitem__(
        self: QRDataset, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = cv.imread(str(self.images[index]), cv.IMREAD_COLOR_RGB)
        image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0

        heatmap = np.load(self.heatmaps[index])
        heatmap = torch.tensor(heatmap, dtype=torch.float32)

        points = np.load(self.points[index])
        points = torch.tensor(points, dtype=torch.float32)

        return image, heatmap, points

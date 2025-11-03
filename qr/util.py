import random

import numpy as np
import torch
from torch import nn
from torchvision.transforms import Compose, v2


def set_seed(seed: int) -> None:
    """
    Set the random seed.

    Parameters:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_parameters(mod: nn.Module) -> int:
    """
    Count the number of parameters in a model.

    Parameters:
        mod: The model.

    Returns:
        The parameter count.
    """
    return sum(p.numel() for p in mod.parameters() if p.requires_grad)


def find_device(force_cpu: bool) -> torch.device:
    """
    Find the best the device for the execution.

    Parameters:
        force_cpu: If true, the device will be forced to CPU.

    Returns:
        The device.
    """
    if force_cpu:
        return torch.device("cpu")

    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def augmentations() -> Compose:
    """
    Create augmentation for the training data.
    """
    return Compose(
        [
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.0),
            v2.GaussianBlur(kernel_size=3, sigma=(0.1, 2)),
            v2.GaussianNoise(sigma=0.03),
            v2.RandomPosterize(bits=6),
        ]
    )


def batch_heatmap_points(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Pixel precision max location for batched heatmaps.

    Parameters:
        heatmap: The heatmap.

    Returns:
        The points matrix (B, 4, 2).
    """
    assert len(heatmap.shape) == 4

    points = []
    for i in range(heatmap.shape[0]):
        points.append(heatmap_points(heatmap[i]).unsqueeze(0))

    return torch.cat(points, 0).to(heatmap.device)


def heatmap_points(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Pixel precision max locations for the four channel heatmap.

    Parameters:
        heatmap: The heatmap.

    Returns:
        The points matrix (4, 2).
    """
    assert len(heatmap.shape) == 3
    assert heatmap.shape[0] == 4

    points = torch.zeros((4, 2), dtype=torch.float32).to(heatmap.device)
    for i in range(4):
        yx = torch.nonzero(heatmap[i] == torch.max(heatmap[i]))[0]
        points[i] = yx.flip(0)

    return subpixel_points(heatmap, points)

def subpixel_points(heatmap: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Subpixel precision max locations for the four channel heatmap.

    Parameters:
        heatmap: The heatmap.
        points: The discrete max locations.

    Returns:
        The points matrix (4, 2).
    """
    assert len(heatmap.shape) == 3
    assert heatmap.shape[0] == 4
    assert points.shape == (4, 2)

    h, w = heatmap.shape[1:]
    eps = 1e-6
    for i in range(4):
        x, y = map(int, points[i])

        if x > 0 and y > 0 and x < w - 1 and y < h - 1:
            mid = heatmap[i, y, x] + eps
            left = heatmap[i, y, x - 1]
            right = heatmap[i, y, x + 1]
            up = heatmap[i, y - 1, x]
            down = heatmap[i, y + 1, x]

            x_offset = (right / mid - left / mid) / 2.
            y_offset = (down / mid - up / mid) / 2.
            
            points[i, 0] += x_offset
            points[i, 1] += y_offset

    return points

def mean_point_accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Calculate the mean L2 accuracy between predicted points and ground truth points.

    Parameters:
        pred: The predicted points.
        target: The ground truth points.

    Returns:
        The accuracy.
    """
    diff = pred - target
    norm = torch.sum(diff**2, dim=-1) ** (1 / 2)
    return torch.mean(norm)

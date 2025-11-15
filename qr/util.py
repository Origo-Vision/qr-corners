import random

import numpy as np
from numpy.typing import ArrayLike, NDArray
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


def check_predicted_points(points: torch.Tensor) -> tuple[torch.Tensor, float] | None:
    assert points.shape == (5, 2)

    def hline(pt1: torch.Tensor, pt2: torch.Tensor) -> torch.Tensor:
        x1, y1 = pt1
        x2, y2 = pt2

        a = y1 - y2
        b = x2 - x1
        c = x1 * y2 - x2 * y1

        return torch.tensor([a, b, c])

    def hcross(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor | None:
        x, y, w = torch.linalg.cross(l1, l2)

        return torch.tensor([x / w, y / w]) if w > 1e-5 else None

    # LR UL
    l1 = hline(points[3], points[0])

    # LL UR
    l2 = hline(points[2], points[1])

    center = hcross(l1, l2)
    if not center is None:
        return center, torch.linalg.norm(points[4] - center).item()
    else:
        # Parallel lines happened.
        return None


def batch_heatmap_points(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Pixel precision max location for batched heatmaps.

    Parameters:
        heatmap: The heatmap.

    Returns:
        The points matrix (B, 5, 2).
    """
    assert len(heatmap.shape) == 4

    points = []
    for i in range(heatmap.shape[0]):
        points.append(heatmap_points(heatmap[i]).unsqueeze(0))

    return torch.cat(points, 0).to(heatmap.device)


def heatmap_points(heatmap: torch.Tensor) -> torch.Tensor:
    """
    Pixel precision max locations for the five channel heatmap.

    Parameters:
        heatmap: The heatmap.

    Returns:
        The points matrix (5, 2).
    """
    assert len(heatmap.shape) == 3
    assert heatmap.shape[0] == 5

    points = torch.zeros((5, 2), dtype=torch.float32).to(heatmap.device)
    for i in range(5):
        yx = torch.nonzero(heatmap[i] == torch.max(heatmap[i]))[0]
        points[i] = yx.flip(0)

    return subpixel_points(heatmap, points)


def subpixel_points(heatmap: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    Subpixel precision max locations for the five channel heatmap.

    Parameters:
        heatmap: The heatmap.
        points: The discrete max locations.

    Returns:
        The points matrix (5, 2).
    """
    assert len(heatmap.shape) == 3
    assert heatmap.shape[0] == 5
    assert points.shape == (5, 2)

    h, w = heatmap.shape[1:]
    eps = 1e-6
    for i in range(5):
        x, y = map(int, points[i])

        if x > 0 and y > 0 and x < w - 1 and y < h - 1:
            mid = heatmap[i, y, x] + eps
            left = heatmap[i, y, x - 1]
            right = heatmap[i, y, x + 1]
            up = heatmap[i, y - 1, x]
            down = heatmap[i, y + 1, x]

            x_offset = (right / mid - left / mid) / 2.0
            y_offset = (down / mid - up / mid) / 2.0

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


def transform_point(H: NDArray, point: ArrayLike) -> NDArray:
    """
    Transform a 2D point using a homography.

    Parameters:
        H: Homograpy.
        point: Point with x and y coordinates.

    Returns:
        The transformed point.
    """
    assert H.shape == (3, 3)
    assert len(point) == 2

    x, y = point
    point = H @ [x, y, 1.0]

    return point[:2] / point[2]

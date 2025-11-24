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


def rgb_to_tensor(rgb: NDArray) -> torch.Tensor:
    """
    Convert a RGB image (H, W, C) to tensor (1, C, H, W).

    Parameters:
        rgb: The image.

    Returns:
        The tensor.
    """
    Xb = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0
    return Xb.unsqueeze(0)


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

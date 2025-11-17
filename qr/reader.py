from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F

Peaks = namedtuple("Peaks", ["ul", "ur", "ll", "lr", "center"])


def nms(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """
    Non-maximum suppression, to make more distict heatmap peaks.

    Parameters:
        heatmap: Heatmap channels (expext B C H W).
        kernel_size: The kernel size for the filtering.

    Returns:
        Processed heatmap in same dimensions as input.
    """
    assert len(heatmap.shape) == 4

    # Max filtering to dilate the area around the peaks.
    max = F.max_pool2d(
        heatmap, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2
    )

    # Mask with the peaks.
    return heatmap * (max == heatmap)


def peak_coordinates(
    heatmap: torch.Tensor, k: int = 10, threshold: float = 0.4
) -> list[Peaks]:
    """
    Extract peak coordinates from a heatmap.

    Parameters:
        heatmap: Heatmap channels (expect B 5 H W).
        k: Extract k peaks at most.
        threshold: Value threshold for peaks.

    Returns:
        List of Peaks for ul, ur, ll, lr and center. One peak per batch instance.
        All peaks are sorted in strongest first order.
    """
    assert len(heatmap.shape) == 4
    assert heatmap.shape[1] == 5  # C

    # Non-maximum suppression to get crisp peaks.
    peakmap = nms(heatmap)

    # Get the top-k peaks.
    B, C, _, _ = peakmap.shape
    heatmap_peaks = torch.topk(peakmap.view(B, C, -1), k=k)

    # Generate one Peaks object per batch.
    peaks = []
    for b in range(B):
        points = []
        for c in range(C):
            points.append(
                _channel_peak_coordinates(
                    heatmap_peaks.values[b, c],
                    heatmap_peaks.indices[b, c],
                    heatmap[b, c],
                    threshold,
                )
            )
        peaks.append(Peaks(*points))

    return peaks


def _channel_peak_coordinates(
    peak_values: torch.Tensor,
    peak_indices: torch.Tensor,
    heatmap: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    assert peak_values.shape == peak_indices.shape

    H, W = heatmap.shape

    threshold = max(threshold, 1e-5)

    points = []
    for value, index in zip(peak_values, peak_indices):
        if value.item() > threshold:
            x, y = index.item() % W, index.item() // W

            if x > 0 and y > 0 and x < (W - 1) and y < (H - 1):
                up = heatmap[y - 1, x]
                left = heatmap[y, x - 1]
                center = heatmap[y, x]
                right = heatmap[y, x + 1]
                down = heatmap[y + 1, x]

                xoffset = right / center - left / center
                yoffset = down / center - up / center

                x += xoffset
                y += yoffset

            points.append(torch.tensor([x, y], dtype=torch.float32))

    return torch.stack(points)

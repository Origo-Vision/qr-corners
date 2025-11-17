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
) -> Peaks:
    """
    Extract peak coordinates from a heatmap.

    Parameters:
        heatmap: Heatmap channels (expect B 5 H W).
        k: Extract k peaks at most.
        threshold: Value threshold for peaks.

    Returns:
        Named tuple with Peaks for ul, ur, ll, lr and center.
    """
    assert len(heatmap.shape) == 4
    assert heatmap.shape[1] == 5

    # Non-maximum suppression to get crisp peaks.
    peakmap = nms(heatmap)

    # Get the top-k peaks.
    B, C, H, W = peakmap.shape
    peaks = torch.topk(peakmap.view(B, C, -1), k=k)

    # Only care about those with values above the threshold.
    peaks_mask = peaks.values > threshold
    indices = peaks.indices[peaks_mask].view(B, C, -1)

    # Convert the indices to px coordinates.
    def to_px(map_index: int, px_index: int) -> tuple[int, int]:
        return px_index % W, px_index // W

    px = []
    for i in range(5):
        func = torch.vmap(partial(to_px, i))
        xs, ys = func(indices[:, i])
        px.append(torch.hstack((xs.T, ys.T)))

    return Peaks(*px)

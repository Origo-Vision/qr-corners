from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F

Peaks = namedtuple("Peaks", ["ul", "ur", "ll", "lr", "center"])
"""
Named tuple representing sub-pixel peaks from a heatmap, grouped
into their corner classes.
"""


def localize_codes(heatmap: torch.Tensor) -> None:
    """
    Localize codes from a multibatch heatmap.
    """
    assert len(heatmap.shape) == 4

    list(map(_localize_codes, heatmap_peaks(heatmap)))


def _localize_codes(peaks: Peaks) -> list[int]:
    ul_indices = list(range(len(peaks.ul)))
    ur_indices = list(range(len(peaks.ur)))
    ll_indices = list(range(len(peaks.ll)))
    lr_indices = list(range(len(peaks.lr)))

    codes = []
    for center in peaks.center:
        print(f"testing center={center}")

        lr_ul_pairs = _find_opposing_pairs(
            center,
            indices1=lr_indices,
            points1=peaks.lr,
            indices2=ul_indices,
            points2=peaks.ul,
        )
        if lr_ul_pairs == []:
            print(" no lr/ul pairs found")
            continue

        ll_ur_pairs = _find_opposing_pairs(
            center,
            indices1=ll_indices,
            points1=peaks.ll,
            indices2=ur_indices,
            points2=peaks.ur,
        )
        if ll_ur_pairs == []:
            print(" no ll/ur pairs found")
            continue

        print(f"lr_ul_pairs={lr_ul_pairs}")
        print(f"ll_ur_pairs={ll_ur_pairs}")

        result = _validate_points(
            center, lr_ul_pairs=lr_ul_pairs, ll_ur_pairs=ll_ur_pairs, peaks=peaks
        )
        if not result is None:
            ul, ur, ll, lr, error = result

            print(f"Code found with error={error:.2f}")
            

            ul_indices.remove(ul)
            ur_indices.remove(ur)
            ll_indices.remove(ll)
            lr_indices.remove(lr)

    return codes


def _find_opposing_pairs(
    center: torch.Tensor,
    indices1: list[int],
    points1: torch.Tensor,
    indices2: list[int],
    points2: torch.Tensor,
    eps: float = 1e-2,
) -> list[tuple[int, int]]:
    index_map1 = {}
    for i in indices1:
        index_map1[i] = _normal_vector(points1[i] - center)

    index_map2 = {}
    for i in indices2:
        index_map2[i] = _normal_vector(points2[i] - center)

    pairs = []
    for i1, v1 in index_map1.items():
        for i2, v2 in index_map2.items():
            value = torch.dot(v1, v2).item()
            if value <= -1.0 + eps:
                pairs.append((i1, i2))

    return pairs


def _normal_vector(vec: torch.Tensor) -> torch.Tensor:
    return vec / torch.linalg.norm(vec)


def _validate_points(
    center: torch.Tensor,
    lr_ul_pairs: list[tuple[int, int]],
    ll_ur_pairs: list[tuple[int, int]],
    peaks: Peaks,
    threshold: float = 3.0,
) -> tuple[int, int, int, int, float] | None:
    result = None
    lowest_error = 1000.0

    for lr, ul in lr_ul_pairs:
        l1 = _hline(peaks.lr[lr], peaks.ul[ul])
        for ll, ur in ll_ur_pairs:
            l2 = _hline(peaks.ll[ll], peaks.ur[ur])

            est_center = _hcross(l1, l2)
            if not est_center is None:
                error = torch.linalg.norm(center - est_center).item()
                if error < threshold and error < lowest_error:
                    lowest_error = error
                    result = (ul, ur, ll, lr, error)

    return result


def _hline(pt1: torch.Tensor, pt2: torch.Tensor) -> torch.Tensor:
    x1, y1 = pt1
    x2, y2 = pt2

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return torch.tensor([a, b, c])


def _hcross(l1: torch.Tensor, l2: torch.Tensor) -> torch.Tensor | None:
    x, y, w = torch.linalg.cross(l1, l2)

    return torch.tensor([x / w, y / w]) if w > 1e-5 else None


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


def heatmap_peaks(
    heatmap: torch.Tensor, k: int = 10, threshold: float = 0.4
) -> list[Peaks]:
    """
    Extract peak coordinates from a multibatch heatmap.

    Parameters:
        heatmap: Heatmap (expect B 5 H W).
        k: Extract k peaks at most per batch instance
        threshold: Value threshold for peak suppression.

    Returns:
        List of peaks, one Peaks object per batch instance.
        Per Peaks object, the peaks are sorted in strongest first order.
    """
    assert len(heatmap.shape) == 4
    assert heatmap.shape[1] == 5  # C

    # Non-maximum suppression to get crisp peaks.
    peakmap = nms(heatmap)

    # Get the top-k peaks.
    B, C, _, _ = peakmap.shape
    heatmap_peaks = torch.topk(peakmap.view(B, C, -1), k=k)

    # Generate one Peaks object per batch instance.
    peaks = []
    for b in range(B):
        points = []
        for c in range(C):
            points.append(
                _per_channel_peaks(
                    heatmap_peaks.values[b, c],
                    heatmap_peaks.indices[b, c],
                    heatmap[b, c],
                    threshold,
                )
            )
        peaks.append(Peaks(*points))

    return peaks


def _per_channel_peaks(
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

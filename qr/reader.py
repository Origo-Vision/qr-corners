from __future__ import annotations
from collections import namedtuple
from functools import partial

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn.functional as F

import util

class Code:
    """
    Class representing a detected code.
    """

    def __init__(
        self: Code,
        ul: torch.Tensor,
        ur: torch.Tensor,
        ll: torch.Tensor,
        lr: torch.Tensor,
        center: torch.Tensor,
        error: float,
    ) -> None:
        """
        Create a code object.
        """
        self.points = torch.zeros((5, 2), dtype=torch.float32)
        self.points[0] = ul
        self.points[1] = ur
        self.points[2] = ll
        self.points[3] = lr
        self.points[4] = center

        self.error = error

    def corners(self: Code) -> torch.Tensor:
        """
        Return the cornes points in a (4, 2) tensor.
        """
        return self.points[:4]

    def find_homography(self: Code, size: int) -> NDArray:
        """
        Find a homograpy from the points to the given (square) image size.
        """
        dst_corners = np.array(
            [[0.0, 0.0], [size - 1, 0.0], [0.0, size - 1], [size - 1, size - 1]]
        )

        H, _ = cv.findHomography(self.corners().numpy(), dst_corners)

        return H
    
    def straight(self: Code, rgb: NDArray) -> NDArray:
        """
        Return an image with a straight, rectified and slightly
        preprocessed code.
        """
        H = self.find_homography(rgb.shape[0])

        gray = cv.cvtColor(rgb, cv.COLOR_RGB2GRAY)
        gray = cv.warpPerspective(gray, H, dsize=rgb.shape[:2])
        cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU, dst=gray)

        return cv.cvtColor(gray, cv.COLOR_GRAY2RGB)

    def __repr__(self: Code) -> str:
        return (
            f"Code(ul={self.points[0]},\n"
            f"     ur={self.points[1]},\n"
            f"     ll={self.points[2]},\n"
            f"     lr={self.points[3]},\n"
            f"     center={self.points[4]})\n"
        )



Peaks = namedtuple("Peaks", ["ul", "ur", "ll", "lr", "center"])
"""
Named tuple representing sub-pixel peaks from a heatmap, grouped
into their corner classes.
"""


def mean_code_accuracy(pred: list[list[Code]], target: list[list[Code]]) -> float:
    # Must be same number of batches.
    assert len(pred) == len(target)

    accuracy = 0.0
    for pbatch, tbatch in zip(pred, target):
        if len(pbatch) == len(tbatch):
            min_error = 100.
            for p in pbatch:
                for t in tbatch:
                    error = util.mean_point_accuracy(p.points, t.points).item()
                    if error < min_error:
                        min_error = error

            accuracy += min_error
        else:
            # Brute-force heuristic; each mismatch in number of codes yields 25 penalty.
            accuracy += abs(len(pbatch) - len(tbatch)) * 25.

    return accuracy

def localize_codes(heatmap: torch.Tensor) -> list[list[Code]]:
    """
    Localize codes from a multibatch heatmap.
    """
    assert len(heatmap.shape) == 4

    return list(map(_localize_codes, heatmap_peaks(heatmap)))


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
    """
    Extract peak coordinates from a single heatmap channel.

    Parameters:
        peak_values: Detected peak values.
        peak_indices: Detected peak linear indices.
        heatmap: Heatmap channel.
        threshold: Supression threshold for peak values.

    Returns:
        Tensor with sub-pixel peak coordinates.
    """
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


def _localize_codes(peaks: Peaks) -> list[Code]:
    """
    Localize codes from a Peaks object.

    Parameters:
        peaks: The peaks object.

    Returns:
        A list of detected codes.
    """
    ul_indices = list(range(len(peaks.ul)))
    ur_indices = list(range(len(peaks.ur)))
    ll_indices = list(range(len(peaks.ll)))
    lr_indices = list(range(len(peaks.lr)))

    codes = []
    for center in peaks.center:
        lr_ul_pairs = _find_diagonal_pairs(
            center,
            indices1=lr_indices,
            points1=peaks.lr,
            indices2=ul_indices,
            points2=peaks.ul,
        )
        if lr_ul_pairs == []:
            continue

        ll_ur_pairs = _find_diagonal_pairs(
            center,
            indices1=ll_indices,
            points1=peaks.ll,
            indices2=ur_indices,
            points2=peaks.ur,
        )
        if ll_ur_pairs == []:
            continue

        result = _validate_points(
            center, lr_ul_pairs=lr_ul_pairs, ll_ur_pairs=ll_ur_pairs, peaks=peaks
        )
        if not result is None:
            ul, ur, ll, lr, error = result

            code = Code(
                ul=peaks.ul[ul],
                ur=peaks.ur[ur],
                ll=peaks.ll[ll],
                lr=peaks.lr[lr],
                center=center,
                error=error,
            )
            codes.append(code)

            ul_indices.remove(ul)
            ur_indices.remove(ur)
            ll_indices.remove(ll)
            lr_indices.remove(lr)

    return codes


def _find_diagonal_pairs(
    center: torch.Tensor,
    indices1: list[int],
    points1: torch.Tensor,
    indices2: list[int],
    points2: torch.Tensor,
    error: float = 3.0,
) -> list[tuple[int, int]]:
    """
    Helper function to find diagonal pairs of corners on each side of the center.
    """
    # Sort indices to start searching in points closer to the center.
    def squared_distance(points: torch.Tensor, i: int) -> float:
        return torch.sum((center - points[i])**2).item()
    
    indices1 = sorted(indices1, key=partial(squared_distance, points1))
    indices2 = sorted(indices2, key=partial(squared_distance, points2))

    pairs = []
    for idx1 in indices1:
        pt1 = points1[idx1]
        for idx2 in indices2:
            pt2 = points2[idx2]

            # Project center on the line pt2 - pt1, must be 0 > t < 1.
            t = _proj_point(pt1, pt2, center)
            # Orthogonal distance from line must be < error.
            if t > 0.0 and t < 1.0 and _ortho_dist(pt1, pt2, center) < error:
                pairs.append((idx1, idx2))

    return pairs


def _validate_points(
    center: torch.Tensor,
    lr_ul_pairs: list[tuple[int, int]],
    ll_ur_pairs: list[tuple[int, int]],
    peaks: Peaks,
    threshold: float = 3.0,
) -> tuple[int, int, int, int, float] | None:
    """
    Validate corner points to see if any of their line intersections are close
    enough to the center coordinate.
    """
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


def _proj_point(a: torch.Tensor, b: torch.Tensor, pt: torch.Tensor) -> float:
    pa = pt - a
    ba = b - a

    return (torch.dot(pa, ba) / torch.dot(ba, ba)).item()


def _ortho_dist(a: torch.Tensor, b: torch.Tensor, pt: torch.Tensor) -> float:
    ax, ay = a
    bx, by = b
    px, py = pt

    nom = torch.abs((bx - ax) * (ay - py) - (ax - px) * (by - ay))
    den = torch.linalg.norm(b - a)

    return (nom / den).item()


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

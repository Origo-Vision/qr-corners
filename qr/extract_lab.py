import argparse
from collections import namedtuple
from functools import partial

import torch
import torch.nn.functional as F

import render
import util

from matplotlib import pyplot as plt

Peaks = namedtuple("Peaks", ["ul", "ur", "ll", "lr", "center"])


def nms(heatmap: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
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


def main(options: argparse.Namespace) -> None:
    _, heatmap, points = (
        render.make_random_multisample(options.sigma)
        if options.multi
        else render.make_random_sample(options.sigma)
    )

    heatmap = torch.tensor(heatmap).unsqueeze(0)
    peakmap = nms(heatmap)

    peaks = peak_coordinates(heatmap)
    print(peaks.center)

    plt.figure(figsize=(12, 8))
    labels = ("UL", "UR", "LL", "LR", "C")
    for i, label in enumerate(labels):
        ax1 = plt.subplot(2, 5, i + 1)
        plt.imshow(heatmap[0, i], cmap="gray")
        plt.title(label)
        plt.axis("off")

        ax2 = plt.subplot(2, 5, 5 + i + 1)
        plt.imshow(peakmap[0, i], cmap="gray")
        plt.axis("off")

        ax1.sharex(ax2)
        ax1.sharey(ax2)
        ax2.sharex(ax1)
        ax2.sharey(ax1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=7788, help="The random seed")
    parser.add_argument(
        "--sigma", type=float, default=3.0, help="Sigma for heatmap generation"
    )
    parser.add_argument("--multi", action="store_true", help="Multi-code mode")
    options = parser.parse_args()

    util.set_seed(options.seed)
    main(options)

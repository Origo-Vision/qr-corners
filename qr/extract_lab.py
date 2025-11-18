import argparse

import torch
from matplotlib import pyplot as plt

import reader
import render
import util


def main(options: argparse.Namespace) -> None:
    _, heatmap, points = (
        render.make_random_multisample(options.sigma)
        if options.multi
        else render.make_random_sample(options.sigma)
    )

    heatmap = torch.tensor(heatmap).unsqueeze(0)
    peakmap = reader.nms(heatmap)

    print(f"ground truth points=\n{points}")

    codes = reader.localize_codes(heatmap)
    print(codes)

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

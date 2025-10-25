import argparse
import pathlib

import cv2 as cv
import numpy as np

import render
import util


def generate(options: argparse.Namespace) -> None:
    """
    Generate a dataset.
    """
    images_dir = pathlib.Path(options.data_dir / "images")
    images_dir.mkdir(parents=True, exist_ok=True)

    heatmaps_dir = pathlib.Path(options.data_dir / "heatmaps")
    heatmaps_dir.mkdir(parents=True, exist_ok=True)

    points_dir = pathlib.Path(options.data_dir / "points")
    points_dir.mkdir(parents=True, exist_ok=True)

    for i in range(options.samples):
        print(f"Generate sample {i+1:5d}/{options.samples:5d}")
        image, heatmap, points = render.make_random_sample(sigma=options.sigma)

        image_path = images_dir / f"image_{i:05d}.png"
        cv.imwrite(str(image_path), image)

        heatmap_path = heatmaps_dir / f"heatmap_{i:05d}.npy"
        np.save(heatmap_path, heatmap)

        points_path = points_dir / f"points_{i:05d}.npy"
        np.save(points_path, points)


def play(options: argparse.Namespace) -> None:
    pass

    # while True:
    #     image, heatmap, pts = render.make_random_sample(sigma=options.sigma)
    #     display = render.display_sample(image, heatmap, pts)
    #     display = cv.cvtColor(display, cv.COLOR_RGB2BGR)
    #     cv.imshow("display", display)
    #     key = cv.waitKey(0)
    #     if key == 27:
    #         break

    #     cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("action", type=str, help="The data action (generate or play)")
    parser.add_argument(
        "--data-dir", type=pathlib.Path, required=True, help="The data directory"
    )
    parser.add_argument(
        "--samples", type=int, default=100, help="The number of samples to generate"
    )
    parser.add_argument("--seed", type=int, default=1598, help="The random seed")
    parser.add_argument(
        "--sigma", type=float, default=3.0, help="Sigma for heatmap generation"
    )
    options = parser.parse_args()

    util.set_seed(options.seed)

    if options.action == "generate":
        generate(options)
    elif options.action == "play":
        play(options)
    else:
        print("Not a valid action")

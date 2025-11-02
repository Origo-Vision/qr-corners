import argparse
import pathlib
import time

import cv2 as cv
import numpy as np
import torch

import models
import render
import util


def play(options: argparse.Namespace) -> None:
    device = util.find_device(options.force_cpu)
    print(f"Device={device}")

    model = models.load(options.weights).to(device)
    model.eval()

    augmentations = None
    if options.augment:
        augmentations = util.augmentations()

    cv.namedWindow("play")
    with torch.no_grad():
        while True:
            rgb, Yb, Pb = render.make_random_sample(3.0)

            Xb = torch.tensor(rgb.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            Xb = Xb.unsqueeze(0).to(device)
            if not augmentations is None:
                Xb = augmentations(Xb)

            start = time.time()
            Ypred = model(Xb)
            duration = time.time() - start

            Ypred = Ypred.squeeze(0).cpu()
            Ppred = util.heatmap_points(Ypred)
            Ypred = Ypred.numpy()

            accuracy = util.mean_point_accuracy(Ppred, torch.tensor(Pb)).item()

            print(f"Pb=\n{Pb}")
            print(f"Ppred=\n{Ppred}")

            rgb = (
                (Xb.cpu().squeeze(0).permute(1, 2, 0).numpy() * 255.0)
                .astype(np.uint8)
                .copy()
            )
            display = render.display_prediction(rgb, Ypred, Pb, Ppred.numpy())

            cv.imshow("play", cv.cvtColor(display, cv.COLOR_RGB2BGR))
            cv.setWindowTitle(
                "play",
                f"Inference time={duration*1000.0:.2f}ms, Avg accuracy={accuracy:.1f}px",
            )

            key = cv.waitKey(0)
            if key == 27 or chr(key) == "q":
                break

        cv.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--weights",
        type=pathlib.Path,
        required=True,
        help="The model weights used for inference",
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force execution on the CPU"
    )
    parser.add_argument("--seed", type=int, default=1598, help="The random seed")
    parser.add_argument(
        "--augment", action="store_true", help="Apply augmentations in play"
    )
    options = parser.parse_args()

    util.set_seed(options.seed)

    play(options)

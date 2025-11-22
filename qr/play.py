import argparse
import pathlib
import time

import cv2 as cv
import torch

import models
import reader
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

    total_samples = 0
    ok_samples = 0

    cv.namedWindow("play")
    with torch.no_grad():
        while True:
            rgb, heatmap = (
                render.make_random_multisample(3.0)
                if options.multi
                else render.make_random_sample(3.0)
            )
            heatmap = torch.tensor(heatmap).unsqueeze(0)

            total_samples += 1

            Xb = util.rgb_to_tensor(rgb).to(device)
            if not augmentations is None:
                Xb = augmentations(Xb)

            start = time.monotonic()
            Yb = model(Xb)
            duration = time.monotonic() - start

            prediction = Yb.cpu()

            target_codes = reader.localize_codes(heatmap=heatmap)
            predicted_codes = reader.localize_codes(heatmap=prediction)

            accuracy = reader.mean_code_accuracy(predicted_codes, target_codes)
            if accuracy < 3.0:
                ok_samples += 1

            display = render.display_prediction(rgb, target=heatmap, pred=prediction)

            cv.imshow("play", cv.cvtColor(display, cv.COLOR_RGB2BGR))
            cv.setWindowTitle(
                "play",
                f"Inference time={duration*1000.0:.2f}ms, point accuracy={accuracy:.1f}px, score={ok_samples / total_samples * 100:.1f}%",
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
    parser.add_argument("--multi", action="store_true", help="Generate multi-samples")
    options = parser.parse_args()

    util.set_seed(options.seed)

    play(options)

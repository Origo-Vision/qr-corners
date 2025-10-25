import argparse
import pathlib

import cv2 as cv
from matplotlib import pyplot as plt

import render
import util


def generate(options: argparse.Namespace) -> None:
    while True:
        image = render.make_random_sample()
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow("display", image)
        key = cv.waitKey(0)
        if key == 27:
            break

        cv.destroyAllWindows()


def play(options: argparse.Namespace) -> None:
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("action", type=str, help="The data action (generate or play)")
    parser.add_argument(
        "--data-dir", type=pathlib.Path, default="dataset", help="The data directory"
    )
    parser.add_argument("--seed", type=int, default=1598, help="The random seed")
    options = parser.parse_args()

    util.set_seed(options.seed)

    if options.action == "generate":
        generate(options)
    elif options.action == "play":
        play(options)
    else:
        print("Not a valid action")

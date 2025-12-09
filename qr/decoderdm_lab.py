import argparse

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ppf.datamatrix import datamatrix


def random_color() -> tuple[int, int, int]:
    r = int(np.random.uniform(10, 245))
    g = int(np.random.uniform(10, 245))
    b = int(np.random.uniform(10, 245))

    return r, g, b

def read_tile(
        data: NDArray, offsets: NDArray, traversal: NDArray, y: int, x: int
    ) -> int:
        color = random_color()

        h, w = data.shape

        byte = 0
        for i in range(8):
            yy, xx = (y, x) + offsets[i]

            if xx < 0:
                xx += w
                yy += 4 - ((w + 4) % 8)

            if yy < 0:
                yy += h
                xx += 4 - ((h + 4) % 8)

            byte |= int(data[yy, xx] << (7 - i))

            traversal[yy + 1, xx + 1] = color

        traversal[y + 1, x + 1] = (255, 0, 0)

        return byte

def read_data(symbol: NDArray) -> NDArray:
    data = symbol[1:-1, 1:-1]
    h, w = data.shape

    # Setting up traversal visualization.
    traversal = cv.cvtColor(1 - symbol, cv.COLOR_GRAY2RGB) * 255
    traversal[1:-1, 1:-1, :] = 255

    # Nominal bit offsets, relative to anchor, MSB first.
    nominal = np.array(
        [
            [-2, -2],
            [-2, -1],
            [-1, -2],
            [-1, -1],
            [-1, 0],
            [0, -2],
            [0, -1],
            [0, 0],
        ]
    )

    # Initial anchor read position.
    y = 4
    x = 0

    step = 2

    for i in range(3):
        byte = read_tile(data, nominal, traversal, y, x)

        print(f"byte={byte}, ascii={chr(byte - 1)}, bin={bin(byte)}")

        y -= step
        x += step

    return traversal


def main(options: argparse.Namespace) -> None:
    symbol = datamatrix.DataMatrix(msg=options.text, codecs=["ascii"])
    symbol = np.array(symbol.matrix, dtype=np.uint8)

    traversal = read_data(symbol)

    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(1 - symbol, cmap="gray")
    plt.title("Data Matrix Symbol (Inv)")

    plt.subplot(1, 2, 2)
    plt.imshow(traversal)
    plt.title("Read traversal order")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--text", type=str, default="abc", help="Text to encode in DM")
    options = parser.parse_args()

    np.random.seed(1598)
    main(options)

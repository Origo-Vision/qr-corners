import argparse

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from ppf.datamatrix import datamatrix


def read_data(symbol: NDArray) -> None:
    data = symbol[1:-1, 1:-1]
    h, w = data.shape

    # Nominal bit offsets, relative to anchor, MSB first.
    offsets = np.array(
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

    def read_tile(y: int, x: int) -> int:
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

        return byte

    # Initial anchor read position.
    y = 4
    x = 0

    step = 2

    for i in range(3):
        byte = read_tile(y, x)
        print(f"byte={byte}, ascii={chr(byte - 1)}, bin={bin(byte)}")

        y -= step
        x += step


def main(options: argparse.Namespace) -> None:
    symbol = datamatrix.DataMatrix(msg=options.text, codecs=["ascii"])
    symbol = np.array(symbol.matrix)

    read_data(symbol)

    # Visualization
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(1 - symbol, cmap="gray")
    plt.title("Data Matrix Symbol (Inv)")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--text", type=str, default="abc", help="Text to encode in DM")
    options = parser.parse_args()

    main(options)

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

        return byte

def read_corner(data: NDArray, indices: NDArray, traversal) -> int:
    color = random_color()

    byte = 0
    for i in range(8):
        y, x = indices[i]
        byte |= int(data[y, x] << (7 - i))

        traversal[y + 1, x + 1] = color

    return byte

def read_data(symbol: NDArray) -> NDArray:
    data = symbol[1:-1, 1:-1]
    h, w = data.shape

    # Setting up traversal visualization.
    traversal = cv.cvtColor(1 - symbol, cv.COLOR_GRAY2RGB) * 255
    traversal[1:-1, 1:-1, :] = 255

    # Nominal bit offsets, relative to anchor, MSB first.
    bit_offsets = np.array(
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

    # Corner indices, MSB first.
    corner_2 = np.array(
        [
            [h - 3, 0],
            [h - 2, 0],
            [h - 1, 0],
            [0, w - 4],
            [0, w - 3],
            [0, w - 2],
            [0, w - 1],
            [1, w - 1]
        ]
    )

    corner_4 = np.array(
        [
            [h - 3, 0],
            [h - 2, 0],
            [h - 1, 0],
            [0, w - 2],
            [0, w - 1],
            [1, w - 1],
            [2, w - 1],
            [3, w - 1]
        ]
    )

    # Initial anchor read position.
    y = 4
    x = 0

    step = 2

    bytes = []
    for i in range(25):
        if y == h - 3 and x == -1:
            print("Case A")
        elif y == h + 1 and x == 1 and (w % 8) == 0 and (h % 8) == 6:
            print("Case D")
        else:
            if y == 0 and x == w - 2 and (w % 4) != 0:
                print("Avoid corner 2")
                y -= step
                x += step
                continue

            if y < 0 or y >= h or x < 0 or x >= w:
                print("outside - turn around")
                step = -step
                y += 2 + step // 2
                x += 2 - step // 2

                while y < 0 or y >= h or x < 0 or x >= w:
                    y -= step
                    x += step

            if y == h - 2 and x == 0 and (w % 4) != 0:
                print("Corner case 2")
                byte = read_corner(data, corner_2, traversal)
                print(f"byte={byte}, ascii={chr(byte - 1)}, bin={bin(byte)}")
                bytes.append(byte)
            elif  y == h - 1 and x == 0 and (w % 8) == 4:
                print("Corner case 4")
                byte = read_corner(data, corner_4, traversal)
                print(f"byte={byte}, ascii={chr(byte - 1)}, bin={bin(byte)}")
                bytes.append(byte)
            else:
                print(y, x)
                byte = read_tile(data, bit_offsets, traversal, y, x)
                print(f"byte={byte}, ascii={chr(byte - 1)}, bin={bin(byte)}")
                bytes.append(byte)

        traversal[y + 1, x + 1] = (255, 0, 0)
        y -= step
        x += step

    print(len(bytes))

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

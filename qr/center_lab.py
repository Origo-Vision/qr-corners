import argparse

import numpy as np
from numpy.typing import NDArray

import render
import util


def hline(pt1: NDArray, pt2: NDArray) -> NDArray:
    x1, y1 = pt1
    x2, y2 = pt2

    a = y1 - y2
    b = x2 - x1
    c = x1 * y2 - x2 * y1

    return np.array([a, b, c])


def hcross(l1: NDArray, l2: NDArray) -> NDArray | None:
    x, y, w = np.cross(l1, l2)

    if w > 1e-5:
        return np.array([x / w, y / w])
    else:
        return None


def estimate_center(points: NDArray) -> NDArray | None:
    assert points.shape == (4, 2)

    l1 = hline(points[3], points[0])
    l2 = hline(points[2], points[1])

    return hcross(l1, l2)


def main() -> None:
    qr_size = 128

    H, dest = render.make_random_homography(qr_size=qr_size, image_size=256)
    true_center = H @ np.array([qr_size / 2, qr_size / 2, 1.0])
    true_center = true_center[:2] / true_center[2]

    est_center = estimate_center(dest)
    print(
        f"true center={true_center}, est center={est_center}, distance={np.linalg.norm(true_center - est_center):.2f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=1598, help="The random seed")
    options = parser.parse_args()

    util.set_seed(options.seed)
    main()

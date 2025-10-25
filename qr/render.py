import math

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from qrcode import QRCode


def make_random_sample() -> NDArray:
    """
    Generate a random sample with a QR projected on a random background.

    Returns:
        Numpy array RGB image.
    """
    qr_code = make_qr_code(random_string(10))
    assert qr_code.shape[0] == qr_code.shape[1]

    qr_size = qr_code.shape[0]
    image_size = 256
    assert qr_size < image_size

    background = make_random_background(size=image_size)
    H, dst = make_random_homography(qr_size, image_size)

    qr_code = cv.warpPerspective(qr_code, H, (image_size, image_size))
    qr_code = cv.cvtColor(qr_code, cv.COLOR_GRAY2RGB)

    qr_mask = make_qr_mask(qr_size, image_size, H)

    mask = qr_mask > 0
    background[mask] = qr_code[mask]

    return background


def random_string(length: int) -> str:
    """
    Utility to generate a random string of capital letters.

    Parameters:
        length: The length of the string.
    """
    assert length > 0
    return "".join(map(chr, np.random.randint(65, 91, (length,))))


def make_qr_code(text: str, version: int = 2) -> NDArray:
    """
    Generate a grayscale QR code image.

    Parameters:
        text: The text to be encoded in the code.
        version: The code version (default 2).

    Returns:
        Numpy array with code.
    """
    qr = QRCode(version=version, box_size=4, border=0)
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    return np.array(image, dtype=np.uint8) * 255


def make_random_background(size: int) -> NDArray:
    """
    Make a random colored, square, RGB image.

    Parameters:
        size: The size.

    Returns:
        Numpy array with random background.
    """
    bg = np.random.randint(100, 256, (size, size, 3), dtype=np.uint8)
    cv.randn(bg, (128, 128, 128), (20, 20, 20))

    return bg


def make_random_homography(qr_size: int, image_size: int) -> tuple[NDArray, NDArray]:
    """
    Make a random homography to transform the QR within the boundaries of the target image.

    Parameters:
        qr_size: Size of the QR code.
        image_size: Size of the image.

    Returns:
        Tuple (H, destination points: UL, UR, LL, LR).
    """
    assert image_size >= qr_size

    H = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    return H, make_corner_points(qr_size)


def make_qr_mask(qr_size: int, image_size: int, H: NDArray) -> NDArray:
    mask = np.ones((qr_size, qr_size), dtype=np.uint8)
    return cv.warpPerspective(mask, H, dsize=(image_size, image_size))


def make_corner_points(size: int) -> NDArray:
    """
    Make corner points for a square image: UL, UR, LL, LR.
    """
    return np.array(
        [[0.0, 0.0], [size - 1, 0.0], [0.0, size - 1], [size - 1, size - 1]]
    )

import argparse
from collections import Counter
import itertools
import math
import pathlib

import cv2 as cv
from matplotlib import pylab as plt
import numpy as np
from numpy.typing import NDArray
from qrcode import QRCode


def read_qr(path: pathlib.Path) -> NDArray:
    gray = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    h, w = gray.shape

    gray = cv.resize(gray, (min(h, w), min(h, w)))

    return preprocess_code(gray)


def gen_qr(module_size: int, version: int, text: str) -> NDArray:
    """
    Generate a test QR code.
    """
    qr = QRCode(version=version, box_size=module_size, border=0)
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    return np.array(image, dtype=np.uint8) * 255


def preprocess_code(code: NDArray) -> NDArray:
    """
    Preprocess a QR code before analysis.

    Parameters:
        code: Grayscale image.

    Returns:
        The preprocessed image.
    """
    blur = cv.GaussianBlur(code, (5, 5), 0)
    cv.adaptiveThreshold(
        blur,
        255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY,
        blockSize=21,
        C=2,
        dst=blur,
    )

    kernel = np.ones((3, 3), np.uint8)
    return cv.morphologyEx(blur, cv.MORPH_OPEN, kernel)


def run_lengths(seq: NDArray) -> set[int]:
    """
    Calculate run lengths (without values) along a vector.

    Parameters:
        seq: The sequence of pixels.

    Returns:
        A set with the observed lengths.
    """
    value = seq[0]
    length = 1

    lengths = set()
    for x in seq[1:]:
        if x == value:
            length += 1
        else:
            lengths.add(length)
            value = x
            length = 1

    lengths.add(length)

    return lengths


def estimate_seq_module_size(seq: NDArray) -> int | None:
    """
    Estimate the module size along a vector.

    Parameters:
        seq: The sequence of pixels (assuming binarized pixels).

    Returns:
        The estimated module size, or None.
    """
    lengths = run_lengths(seq)
    if len(lengths) == 0:
        return None

    # Test each element against each other as run / base. If close to an integer
    # the base is a candidate for the mudule size, as the run is a multiple of the module size.
    bases = []
    for base, run in itertools.permutations(lengths, 2):
        ratio = run / base
        score = abs(ratio - round(ratio))
        if score < 0.2:
            bases.append(base)

        # print(f"base={base}, run={run}, ratio={ratio}, score={score}")

    if len(bases) > 0:
        module_size, _ = Counter(bases).most_common(1)[0]
        return module_size
    else:
        return None


def estimate_module_size(code: NDArray, samples: int = 10) -> int | None:
    """
    Estimate the module size from sampling horizontally and vertically in the image.

    Parameters:
        code: Rectified, square and binarized code image.
        samples: The number of samples to be taken in each dimension.

    Returns:
        The estimated module size, or None.
    """
    h, w = code.shape
    assert h == w

    module_sizes = []
    for i in np.linspace(min(h, w) * 0.1, min(h, w) * 0.9, samples).astype(int):
        horizontal_estimate = estimate_seq_module_size(code[i, :])
        vertical_estimate = estimate_seq_module_size(code[:, i])

        if not horizontal_estimate is None:
            module_sizes.append(horizontal_estimate)

        if not vertical_estimate is None:
            module_sizes.append(vertical_estimate)

    return int(np.median(module_sizes)) if module_sizes != [] else None


def estimate_num_modules(code: NDArray, module_size: float) -> tuple[int, int]:
    """
    Estimate the number of modules in a code, and the code's version.

    Parameters:
        code: Rectified, square and binarized code image.
        module_size: The module size.

    Returns:
        Tuple (number of modules, version).
    """
    h, w = code.shape
    assert h == w

    N = w / module_size

    # k = v - 1
    # N = 21 + 4k
    # k = (N - 21) / 4, implying that (N - 21) % 4 == 0.
    # Round to the best fit, and recalculate N.
    k = round((N - 21) / 4)
    N = 21 + 4 * k

    return N, k + 1


def qr_matrix(code: NDArray, invert: bool = True) -> NDArray | None:
    """
    Sample a code into a NxN raster, where N is the number of modules.

    Parameters:
        code: Rectified, square and binarized code image.

    Returns:
        Binary image size NxN, or None.
    """
    h, w = code.shape
    if h != w:
        print("Code is not square")
        return None

    module_size = estimate_module_size(code)
    if module_size is None:
        return None

    num_modules, _ = estimate_num_modules(code, module_size)
    module_size = w / num_modules

    raster = np.zeros((num_modules, num_modules), dtype=np.uint8)
    size = max(1, math.ceil(module_size))
    patch_size = (size, size)

    for y in range(num_modules):
        for x in range(num_modules):
            px = (x + 0.5) * module_size
            py = (y + 0.5) * module_size
            patch = cv.getRectSubPix(code, patch_size, (px, py))
            value = np.median(patch) > 128
            raster[y, x] = 1 - value if invert else value

    return raster


def make_qr_masks(size: tuple[int, int]) -> NDArray:
    """
    Create the eight different QR masks.

    Parameters:
        size: Tuple (h, w).

    Returns:
        Masks with shape (h, w, 8).
    """
    h, w = size

    masks = np.zeros((h, w, 8), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            masks[y, x, 0] = (y + x) % 2 == 0
            masks[y, x, 1] = y % 2 == 0
            masks[y, x, 2] = x % 3 == 0
            masks[y, x, 3] = (y + x) % 3 == 0
            masks[y, x, 4] = (y // 2 + x // 3) % 2 == 0
            masks[y, x, 5] = (y * x) % 2 + (y * x) % 3 == 0
            masks[y, x, 6] = ((y * x) % 2 + (y * x) % 3) % 2 == 0
            masks[y, x, 7] = ((y + x) % 2 + (y * x) % 3) % 2 == 0

    return masks


def read_qr_format(raster: NDArray) -> tuple[int, int]:
    """
    Read the 15 two bit QR format strings.

    Parameters:
        raster: The QR matrix.

    Returns:
        Tuple (primary, secondary) of format bit strings.
    """
    primary = 0
    secondary = 0

    primary |= int(raster[8, 0]) << 14
    primary |= int(raster[8, 1]) << 13
    primary |= int(raster[8, 2]) << 12
    primary |= int(raster[8, 3]) << 11
    primary |= int(raster[8, 4]) << 10
    primary |= int(raster[8, 5]) << 9
    primary |= int(raster[8, 7]) << 8
    primary |= int(raster[8, 8]) << 7
    primary |= int(raster[7, 8]) << 6
    primary |= int(raster[5, 8]) << 5
    primary |= int(raster[4, 8]) << 4
    primary |= int(raster[3, 8]) << 3
    primary |= int(raster[2, 8]) << 2
    primary |= int(raster[1, 8]) << 1
    primary |= int(raster[0, 8])

    secondary |= int(raster[-1, 8]) << 14
    secondary |= int(raster[-2, 8]) << 13
    secondary |= int(raster[-3, 8]) << 12
    secondary |= int(raster[-4, 8]) << 11
    secondary |= int(raster[-5, 8]) << 10
    secondary |= int(raster[-6, 8]) << 9
    secondary |= int(raster[-7, 8]) << 8
    secondary |= int(raster[8, -8]) << 7
    secondary |= int(raster[8, -7]) << 6
    secondary |= int(raster[8, -6]) << 5
    secondary |= int(raster[8, -5]) << 4
    secondary |= int(raster[8, -4]) << 3
    secondary |= int(raster[8, -3]) << 2
    secondary |= int(raster[8, -2]) << 1
    secondary |= int(raster[8, -1])

    return primary, secondary

def hamming_distance(a: int, b: int) -> int:
    """
    Calculate the Hamming distance between two numbers.
    """
    return bin(a ^ b).count("1")

def show_masks() -> None:
    masks = make_qr_masks((128, 128))

    plt.figure(figsize=(12, 8))

    for i in range(8):
        mask = masks[:, :, i]

        plt.subplot(2, 4, i + 1)
        plt.imshow(mask, cmap="gray")
        plt.title(f"mask={i}")
        plt.axis("off")

    plt.show()


def main(options: argparse.Namespace) -> None:
    if options.show_masks:
        show_masks()
        return

    code = (
        gen_qr(
            module_size=options.module_size, version=options.version, text=options.text
        )
        if options.file is None
        else read_qr(options.file)
    )

    raster = qr_matrix(code, invert=True)
    if raster is None:
        print("Failed to construct QR matrix")
        return

    format = read_qr_format(raster)
    print(f"primary={bin(format[0])}, secondary={bin(format[1])}, hamming={hamming_distance(format[0], format[1])}")

    plt.figure(figsize=(12, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(code, cmap="gray")
    plt.title("binarized code")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(1 - raster, cmap="gray")
    plt.title("QR matrix")
    plt.axis("off")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--file", type=pathlib.Path, help="External, cropped, QR code file"
    )
    parser.add_argument(
        "--module-size",
        type=int,
        choices=[2, 3, 4, 5, 6],
        default=4,
        help="Module size (px)",
    )
    parser.add_argument("--show-masks", action="store_true", help="Show QR masks")
    parser.add_argument(
        "--version", type=int, choices=[1, 2, 3, 4, 5, 6], default=2, help="QR version"
    )
    parser.add_argument("--text", type=str, default="my code", help="Code text")
    options = parser.parse_args()

    main(options)

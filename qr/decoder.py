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

"""
Unmasked bit-string mapping to error level and masking.
"""
qr_format_table = {
    # 01=L, <7%
    0b111011111000100: (0b01, 0b000),
    0b111001011110011: (0b01, 0b001),
    0b111110110101010: (0b01, 0b010),
    0b111100010011101: (0b01, 0b011),
    0b110011000101111: (0b01, 0b100),
    0b110001100011000: (0b01, 0b101),
    0b110110001000001: (0b01, 0b110),
    0b110100101110110: (0b01, 0b111),
    # 00=M, <15%
    0b101010000010010: (0b00, 0b000),
    0b101000100100101: (0b00, 0b001),
    0b101111001111100: (0b00, 0b010),
    0b101101101001011: (0b00, 0b011),
    0b100010111111001: (0b00, 0b100),
    0b100000011001110: (0b00, 0b101),
    0b100111110010111: (0b00, 0b110),
    0b100101010100000: (0b00, 0b111),
    # 11=Q, <25%
    0b011010101011111: (0b11, 0b000),
    0b011000001101000: (0b11, 0b001),
    0b011111100110001: (0b11, 0b010),
    0b011101000000110: (0b11, 0b011),
    0b010010010110100: (0b11, 0b100),
    0b010000110000011: (0b11, 0b101),
    0b010111011011010: (0b11, 0b110),
    0b010101111101101: (0b11, 0b111),
    # 10=H, <30%
    0b001011010001001: (0b10, 0b000),
    0b001001110111110: (0b10, 0b001),
    0b001110011100111: (0b10, 0b010),
    0b001100111010000: (0b10, 0b011),
    0b000011101100010: (0b10, 0b100),
    0b000001001010101: (0b10, 0b101),
    0b000110100001100: (0b10, 0b110),
    0b000100000111011: (0b10, 0b111),
}


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


def decode_qr_format(bits: int) -> tuple[int, int, int]:
    """
    Decode the QR format using the format table.

    Parameters:
        bits: 15-bit bitstring with format code.

    Returns:
        Tuple (ECL, mask-id, Hamming distance).
    """
    # unmasked = bits ^ 0x5412

    selection = None
    lowest_error = 100
    for codeword, payload in qr_format_table.items():
        error = hamming_distance(codeword, bits)
        if error < lowest_error:
            selection = payload
            lowest_error = error

    print(lowest_error)

    return selection + (lowest_error,)


def qr_ecl_and_mask(raster: NDArray) -> tuple[int, int]:
    """
    Get ECL and mask from the QR matrix.

    Parameters:
        raster: The QR matrix.

    Returns:
        Tuple (ECL, mask-id).
    """
    primary, secondary = read_qr_format(raster)

    ecl1, mask1, err1 = decode_qr_format(primary)
    if primary == secondary:
        return ecl1, mask1

    ecl2, mask2, err2 = decode_qr_format(secondary)
    if err2 < err1:
        return ecl2, mask2
    else:
        return ecl1, mask1


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

    ecl, mask = qr_ecl_and_mask(raster)
    print(f"ECL={ecl}, mask={mask}")

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

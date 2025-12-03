import argparse
from collections import namedtuple

from matplotlib import pyplot as plt
import numpy as np
from numpy.typing import NDArray
from qrcode import QRCode
import reedsolo

FormatIndices = namedtuple(
    "FormatIndices", ["primary_xs", "primary_ys", "secondary_xs", "secondary_ys"]
)

# ECL 1 (L): 19, 7
# ECL 0 (M): 16, 10
# ECL 3 (Q): 13, 13
# ECL 2 (H): 9, 17

qr_format_table = {
    # 01=L, <7%
    0b111011111000100: ("L", 0),
    0b111001011110011: ("L", 1),
    0b111110110101010: ("L", 2),
    0b111100010011101: ("L", 3),
    0b110011000101111: ("L", 4),
    0b110001100011000: ("L", 5),
    0b110110001000001: ("L", 6),
    0b110100101110110: ("L", 7),
    # 00=M, <15%
    0b101010000010010: ("M", 0),
    0b101000100100101: ("M", 1),
    0b101111001111100: ("M", 2),
    0b101101101001011: ("M", 3),
    0b100010111111001: ("M", 4),
    0b100000011001110: ("M", 5),
    0b100111110010111: ("M", 6),
    0b100101010100000: ("M", 7),
    # 11=Q, <25%
    0b011010101011111: ("Q", 0),
    0b011000001101000: ("Q", 1),
    0b011111100110001: ("Q", 2),
    0b011101000000110: ("Q", 3),
    0b010010010110100: ("Q", 4),
    0b010000110000011: ("Q", 5),
    0b010111011011010: ("Q", 6),
    0b010101111101101: ("Q", 7),
    # 10=H, <30%
    0b001011010001001: ("H", 0),
    0b001001110111110: ("H", 1),
    0b001110011100111: ("H", 2),
    0b001100111010000: ("H", 3),
    0b000011101100010: ("H", 4),
    0b000001001010101: ("H", 5),
    0b000110100001100: ("H", 6),
    0b000100000111011: ("H", 7),
}


def generate_code(text: str, error_correction: int = 1) -> NDArray:
    qr = QRCode(version=1, box_size=1, border=0, error_correction=error_correction)
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    return 1 - np.array(image, dtype=np.uint8)


def data_module_mask() -> NDArray:
    mask = np.ones((21, 21), dtype=np.uint8)

    # Timing patterns.
    mask[6, :] = 0
    mask[:, 6] = 0

    # UL position marker + quiet area + format.
    mask[0:9, 0:9] = 0

    # UR position marker + quiet area + format copy.
    mask[0:9, 21 - 8 : 21] = 0

    # LL position marker + quite area + format copy + black module.
    mask[21 - 8 : 21, 0:9] = 0

    return mask


def qr_pattern_mask() -> NDArray:
    h, w = (21, 21)

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


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def make_format_indices() -> FormatIndices:
    primary_xs = [0, 1, 2, 3, 4, 5, 7, 8, 8, 8, 8, 8, 8, 8, 8]
    primary_ys = [8, 8, 8, 8, 8, 8, 8, 8, 7, 5, 4, 3, 2, 1, 0]

    secondary_xs = [8, 8, 8, 8, 8, 8, 8, -8, -7, -6, -5, -4, -3, -2, -1]
    secondary_ys = [-1, -2, -3, -4, -5, -6, -7, 8, 8, 8, 8, 8, 8, 8, 8]

    return FormatIndices(
        np.array(primary_xs),
        np.array(primary_ys),
        np.array(secondary_xs),
        np.array(secondary_ys),
    )


def read_format_bits(code: NDArray, xs: NDArray, ys: NDArray) -> int:
    assert len(xs) == 15
    assert len(ys) == 15
    bits = code[ys, xs]

    value = 0
    for bit in bits:
        value = (value << 1) + int(bit)

    return value


def read_format(code: NDArray, indices: FormatIndices) -> tuple[str, int] | None:
    primary = read_format_bits(code, indices.primary_xs, indices.primary_ys)
    secondary = read_format_bits(code, indices.secondary_xs, indices.secondary_ys)

    if primary in qr_format_table:
        return qr_format_table[primary]

    if secondary in qr_format_table:
        return qr_format_table[secondary]

    lowest_error = 15
    selection = None
    for key, spec in qr_format_table.items():
        primary_error = hamming_distance(key, primary)
        if primary_error < lowest_error:
            lowest_error = primary_error
            selection = spec

        secondary_error = hamming_distance(key, secondary)
        if secondary_error < lowest_error:
            lowest_error = secondary_error
            selection = spec

    return selection


def main(options: argparse.Namespace) -> None:
    code = generate_code("www.wikipedia.org")

    assert code.shape == (21, 21)

    data_mask = data_module_mask()
    qr_mask = qr_pattern_mask()

    format_indices = make_format_indices()
    result = read_format(code, format_indices)
    if result is None:
        print("Failed to read the format")

    ecl, mask_id = result
    print(f"ECL={ecl}, mask id={mask_id}")
    flip_mask = qr_mask[:, :, mask_id]

    unmasked = code.copy()
    toggle = (data_mask & flip_mask) > 0
    unmasked[toggle] ^= 1

    plt.figure(figsize=(12, 4))
    plt.subplot(2, 4, 1)
    plt.imshow(code, cmap="gray")
    plt.axis("off")
    plt.title("QR Matrix")

    plt.subplot(2, 4, 2)
    plt.imshow(1 - code, cmap="gray")
    plt.axis("off")
    plt.title("QR Matrix Viz")

    plt.subplot(2, 4, 3)
    plt.imshow(data_mask, cmap="gray")
    plt.axis("off")
    plt.title("Data Mask")

    plt.subplot(2, 4, 4)
    plt.imshow(flip_mask, cmap="gray")
    plt.axis("off")
    plt.title("Flip Mask")

    plt.subplot(2, 4, 5)
    plt.imshow(unmasked, cmap="gray")
    plt.axis("off")
    plt.title("Mask Flipped QR")

    plt.subplot(2, 4, 6)
    plt.imshow(1 - unmasked, cmap="gray")
    plt.axis("off")
    plt.title("Mask Flipped QR Viz")

    plt.subplot(2, 4, 7)
    plt.imshow((1 - unmasked) & data_mask, cmap="gray")
    plt.axis("off")
    plt.title("Data Bits")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    options = parser.parse_args()
    main(options)

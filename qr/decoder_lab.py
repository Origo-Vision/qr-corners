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

DataIndices = namedtuple("DataIndices", ["xs", "ys"])


ecl_table = {
    # Payload CW, EC CW. Note: effective payload is two bytes less.
    "L": (19, 7),
    "M": (16, 10),
    "Q": (13, 13),
    "H": (9, 17),
}

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


def make_data_indices(data_mask: NDArray) -> DataIndices:
    assert data_mask.shape == (21, 21)

    N, _ = data_mask.shape

    xs = []
    ys = []

    col = N - 1
    up = True
    down_rows = range(0, N)
    up_rows = range(N - 1, -1, -1)
    while col > 0:
        x0, x1 = col - 1, col

        for y in up_rows if up else down_rows:
            if data_mask[y, x1] > 0:
                xs.append(x1)
                ys.append(y)

            if data_mask[y, x0] > 0:
                xs.append(x0)
                ys.append(y)

        up = not up
        col -= 2
        if col == 6:
            # Just skip the column with the vertical timing pattern.
            col -= 1

    return DataIndices(np.array(xs), np.array(ys))


def read_data(unmasked: NDArray, indices: DataIndices) -> bytearray:
    bits = unmasked[indices.ys, indices.xs]
    assert len(bits) == 208, "Should be 208 bits => 26 bytes"

    bytes = bytearray(26)
    for i in range(26):
        seq = bits[i * 8 : i * 8 + 8]
        byte = (
            (seq[0] << 7)
            + (seq[1] << 6)
            + (seq[2] << 5)
            + (seq[3] << 4)
            + (seq[4] << 3)
            + (seq[5] << 2)
            + (seq[6] << 1)
            + seq[7]
        )
        bytes[i] = byte

    return bytes


def render_data_read_order(indices: DataIndices) -> NDArray:
    image = np.zeros((21, 21), dtype=np.uint8)

    color = 1
    for x, y in zip(indices.xs, indices.ys):
        image[y, x] = color
        color += 1

    return image


def parse_payload(decoded: bytearray) -> str | None:
    bits = []
    for byte in decoded:
        bits.append((byte >> 7) & 1)
        bits.append((byte >> 6) & 1)
        bits.append((byte >> 5) & 1)
        bits.append((byte >> 4) & 1)
        bits.append((byte >> 3) & 1)
        bits.append((byte >> 2) & 1)
        bits.append((byte >> 1) & 1)
        bits.append(byte & 1)

    current = 0

    def take_next(num: int) -> int:
        nonlocal bits
        nonlocal current

        chunk = bits[current : current + num]
        byte = 0
        for i in range(num):
            byte = (byte << 1) + chunk[i]

        current += num

        return byte

    mode = take_next(4)
    if mode != 0b0100:
        print(f"Only byte mode for payload parsning is supported. Got={bin(mode)}")
        return None

    size = take_next(8)
    if size > 17:
        print(f"Size exceeded the maximum of 17. Got={size}")
        return None

    payload = []
    for byte in range(size):
        payload.append(take_next(8))

    return bytes(payload).decode("utf-8")


def main(options: argparse.Namespace) -> None:
    # Create the code
    code = generate_code("www.wikipedia.org")

    assert code.shape == (21, 21)

    # Create the data mask for version 1 QR codes.
    data_mask = data_module_mask()

    # Create a deck of flip mask for version 1 QR codes.
    qr_mask = qr_pattern_mask()

    # Read the format (ECL and mask id) from pre-calculated indices.
    format_indices = make_format_indices()
    result = read_format(code, format_indices)
    if result is None:
        print("Failed to read the format")

    # Extract the correct flip mask from the deck.
    ecl, mask_id = result
    flip_mask = qr_mask[:, :, mask_id]

    # Unmask the code by flipping bits where the flip mask is true.
    unmasked = code.copy()
    toggle = (data_mask & flip_mask) > 0
    unmasked[toggle] ^= 1

    # Create data read indices for version 1 QR codes.
    data_indices = make_data_indices(data_mask)
    data_read_order = render_data_read_order(data_indices)

    # Read the data from the pre-calculated indices.
    bytes = read_data(unmasked, data_indices)
    _, ec_cw = ecl_table[ecl]

    # Setup the RS decoder.
    rs = reedsolo.RSCodec(nsym=ec_cw)

    # Try to decode.
    try:
        decoded, _, _ = rs.decode(bytes)
        payload = parse_payload(decoded)
        if not payload is None:
            print(f"Read the payload='{payload}' from the code")
        else:
            print("Failed to parse the decoded data")

    except Exception as e:
        print(f"Failed to RS decode the data. Error={e}")

    # Visualization.
    plt.figure(figsize=(12, 4))
    plt.subplot(2, 4, 1)
    plt.imshow(code, cmap="gray")
    plt.axis("off")
    plt.title("QR Matrix")

    plt.subplot(2, 4, 2)
    plt.imshow(1 - code, cmap="gray")
    plt.axis("off")
    plt.title("QR Matrix (Inv)")

    plt.subplot(2, 4, 3)
    plt.imshow(data_mask, cmap="gray")
    plt.axis("off")
    plt.title("Data Mask")

    plt.subplot(2, 4, 4)
    plt.imshow(flip_mask, cmap="gray")
    plt.axis("off")
    plt.title(f"Flip Mask ({mask_id})")

    plt.subplot(2, 4, 5)
    plt.imshow(unmasked, cmap="gray")
    plt.axis("off")
    plt.title("Mask Flipped QR")

    plt.subplot(2, 4, 6)
    plt.imshow(1 - unmasked, cmap="gray")
    plt.axis("off")
    plt.title("Mask Flipped QR (Inv)")

    plt.subplot(2, 4, 7)
    plt.imshow((1 - unmasked) & data_mask, cmap="gray")
    plt.axis("off")
    plt.title("Data Bits (Inv)")

    plt.subplot(2, 4, 8)
    plt.imshow(data_read_order, cmap="hot")
    plt.axis("off")
    plt.title("Data Read Order")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    options = parser.parse_args()
    main(options)

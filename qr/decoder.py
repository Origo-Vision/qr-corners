import argparse
from collections import Counter
import itertools

from matplotlib import pylab as plt
import numpy as np
from numpy.typing import NDArray
from qrcode import QRCode


def gen_qr(module_size: int, version: int, text: str) -> NDArray:
    """
    Generate a test QR code.
    """
    qr = QRCode(version=version, box_size=module_size, border=0)
    qr.add_data(text)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")

    return np.array(image, dtype=np.uint8) * 255


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
        code: Rectified and binariezed code image.
        samples: The number of samples to be taken in each dimension.

    Returns:
        The estimated module size, or None.
    """
    # Assume that h and w are equal, or at least very similar.
    h, w = code.shape

    module_sizes = []
    for i in np.linspace(min(h, w) * 0.1, min(h, w) * 0.9, samples).astype(int):
        horizontal_estimate = estimate_seq_module_size(code[i, :])
        vertical_estimate = estimate_seq_module_size(code[:, i])

        if not horizontal_estimate is None:
            module_sizes.append(horizontal_estimate)

        if not vertical_estimate is None:
            module_sizes.append(vertical_estimate)

    return int(np.median(module_sizes)) if module_sizes != [] else None


def main(options: argparse.Namespace) -> None:
    gray = gen_qr(
        module_size=options.module_size, version=options.version, text=options.text
    )

    result = estimate_module_size(gray)
    if not result is None:
        module_size = result

        print(f"module size={module_size}px")

    # return

    plt.figure(figsize=(12, 8))
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--module-size",
        type=int,
        choices=[2, 3, 4, 5, 6],
        default=4,
        help="Module size (px)",
    )
    parser.add_argument(
        "--version", type=int, choices=[1, 2, 3, 4, 5, 6], default=2, help="QR version"
    )
    parser.add_argument("--text", type=str, default="my code", help="Code text")
    options = parser.parse_args()

    main(options)

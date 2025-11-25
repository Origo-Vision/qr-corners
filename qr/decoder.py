import argparse

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


def main(options: argparse.Namespace) -> None:
    gray = gen_qr(
        module_size=options.module_size, version=options.version, text=options.text
    )

    plt.figure(figsize=(12, 8))
    plt.imshow(gray, cmap="gray")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--module-size", type=int, choices=[2, 3, 4, 5, 6], default=4, help="Module size (px)")
    parser.add_argument("--version", type=int, choices=[1, 2, 3, 4, 5, 6], default=2, help="QR version")
    parser.add_argument("--text", type=str, default="my code", help="Code text")
    options = parser.parse_args()

    main(options)

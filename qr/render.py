import math

import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from qrcode import QRCode


def make_random_sample(sigma: float) -> tuple[NDArray, NDArray, NDArray]:
    """
    Generate a random sample with a QR projected on a random background.

    Parameters:
        sigma: Standard deviation for heatmap generation.

    Returns:
        Tuple with RGB image, four channel heatmap and corner points.
    """
    # Make QR.
    qr_code = make_qr_code(random_string(10))
    assert qr_code.shape[0] == qr_code.shape[1]

    qr_size = qr_code.shape[0]
    image_size = 256
    assert qr_size < image_size

    H, dst = make_random_homography(qr_size, image_size)

    qr_code = cv.warpPerspective(qr_code, H, (image_size, image_size))
    qr_code = cv.cvtColor(qr_code, cv.COLOR_GRAY2RGB)

    qr_mask = make_qr_mask(qr_size, image_size, H)
    mask = qr_mask > 0

    # Make image.
    image = make_random_background(size=image_size)
    image[mask] = qr_code[mask]

    # Make heat maps.
    heatmap = np.zeros((4, image_size, image_size), dtype=np.float32)
    for i, (cx, cy) in enumerate(dst):
        y, x = np.ogrid[:image_size, :image_size]
        heatmap[i] = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))

    return image, heatmap, dst


def display_sample(image: NDArray, heatmap: NDArray, pts: NDArray) -> NDArray:
    """
    Make a sample triplet displayable.

    Parameters:
        image: The image to decode.
        heatmap: The heatmap channels.
        pts: The corner points for the code in the image.

    Returns:
        An RGB image for display.
    """
    hm = heatmap_to_rgb(heatmap)

    dst = make_corner_points(image.shape[0])
    H, _ = cv.findHomography(pts, dst)
    code = cv.warpPerspective(image, H, dsize=image.shape[:2])

    return np.hstack((image, hm, code))


def display_prediction(
    image: NDArray, heatmap: NDArray, pts_true: NDArray, pts_pred: NDArray
) -> NDArray:
    """
    Make a prediction displayable.

    Parameters:
        image: The image to decode.
        heatmap: The heatmap channels.
        pts_true: The true corner points for the code in the image.
        pts_pred: The predicted corner points for the code in the image.

    Returns:
        An RGB image for display.
    """
    assert pts_true.shape == (4, 2)
    assert pts_true.shape == pts_pred.shape

    # Warped code.
    dst = make_corner_points(image.shape[0])
    H, _ = cv.findHomography(pts_pred, dst)
    code = cv.warpPerspective(image, H, dsize=image.shape[:2])

    # Image with true and predicted corners points.
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 255)]
    for i in range(4):
        pt_true = tuple(map(int, pts_true[i]))
        pt_pred = tuple(map(int, pts_pred[i]))
        cv.drawMarker(image, pt_true, colors[i], thickness=2)
        cv.drawMarker(
            image, pt_pred, colors[i], markerType=cv.MARKER_TILTED_CROSS, thickness=2
        )

    # The 2x3 mosaic.
    h, w = image.shape[:2]
    display = np.zeros((h * 2, w * 3, 3), np.uint8)

    # Fill in the parts.
    display[0:h, 0:w, :] = image[:, :, :]
    display[0:h, w : w * 2, 0] = (heatmap[0, :, :] * 255.0).astype(np.uint8)
    display[0:h, w * 2 : w * 3, 1] = (heatmap[1, :, :] * 255.0).astype(np.uint8)

    display[h : h * 2, 0:w, :] = code[:, :, :]
    display[h : h * 2, w : w * 2, 2] = (heatmap[2, :, :] * 255.0).astype(np.uint8)
    display[h : h * 2, w * 2 : w * 3, 0] = (heatmap[3, :, :] * 255.0).astype(np.uint8)
    display[h : h * 2, w * 2 : w * 3, 1] = (heatmap[3, :, :] * 255.0).astype(np.uint8)
    display[h : h * 2, w * 2 : w * 3, 2] = (heatmap[3, :, :] * 255.0).astype(np.uint8)

    # Draw a thin gray line to mark the shape of the warped code in the heatmaps.
    for y in range(2):
        for x in range(1, 3):
            cv.line(
                display[y * h : (y + 1) * h, x * w : (x + 1) * w, :],
                tuple(map(int, pts_true[0])),
                tuple(map(int, pts_true[1])),
                (127, 127, 127),
            )
            cv.line(
                display[y * h : (y + 1) * h, x * w : (x + 1) * w, :],
                tuple(map(int, pts_true[1])),
                tuple(map(int, pts_true[3])),
                (127, 127, 127),
            )
            cv.line(
                display[y * h : (y + 1) * h, x * w : (x + 1) * w, :],
                tuple(map(int, pts_true[3])),
                tuple(map(int, pts_true[2])),
                (127, 127, 127),
            )
            cv.line(
                display[y * h : (y + 1) * h, x * w : (x + 1) * w, :],
                tuple(map(int, pts_true[2])),
                tuple(map(int, pts_true[0])),
                (127, 127, 127),
            )

    return display


def heatmap_to_rgb(heatmap: NDArray) -> NDArray:
    """
    Transform a heatmap to an RGB image, where UL=red, UR=green,
    LL=blue and LR=white.

    Parameters:
        heatmap: The heatmap.

    Returns:
        The RGB image.
    """
    rgb = np.zeros(heatmap.shape[1:] + (3,), dtype=np.uint8)
    for i in range(3):
        rgb[:, :, i] = ((heatmap[i] + heatmap[3]) * 255.0).astype(np.uint8)

    return rgb


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

    d_qr = qr_size / 2
    d_image = image_size / 2

    # Create corner points and translate those points to a zero centered system.
    src = make_corner_points(qr_size)
    dst = src - d_qr

    # Calculate norms and scale factors for each of the points.
    norm = np.linalg.norm(dst, axis=1).reshape(4, 1)
    scale = np.random.uniform(0.05 * d_image, d_image, (4, 1))

    # Scale the corners.
    dst = (dst / norm) * scale

    # Random rotate in Z.
    z = np.random.uniform(0.0, 2.0 * math.pi)
    z_rot = np.array(
        [
            [math.cos(z), -math.sin(z)],
            [math.sin(z), math.cos(z)],
        ]
    )

    dst = dst @ z_rot

    # Translate back to the image system.
    dst += d_qr

    # Translate to the center of the image.
    dst += d_image - d_qr

    # Find the bounding box, and add a final translation.
    min, max = bounding_box(dst)
    min_trans_x = -min[0]
    min_trans_y = -min[1]
    max_trans_x = image_size - 1 - max[0]
    max_trans_y = image_size - 1 - max[1]

    trans_x = np.random.uniform(min_trans_x, max_trans_x)
    trans_y = np.random.uniform(min_trans_y, max_trans_y)
    trans = np.array([[trans_x, trans_y]])

    dst += trans

    # Find homography.
    H, _ = cv.findHomography(src, dst)

    return H, dst


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


def bounding_box(pts: NDArray) -> tuple[NDArray, NDArray]:
    """
    Get the bounding box of four points.

    Parameters:
        pts: Points as an array with shape (4, 2).

    Returns:
        Tuple with arrays min of x, y and max of x, y.
    """
    assert pts.shape == (4, 2)

    return np.min(pts, axis=0), np.max(pts, axis=0)

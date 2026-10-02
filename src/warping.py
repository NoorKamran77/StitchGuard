import cv2
import numpy as np


def warp_pair_to_canvas(image1, image2, H):
    """Warp image2 into image1's coordinate system and create a common canvas."""
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Corners of image 2 in homogeneous coordinates.
    corners2 = np.float32([
        [0, 0],
        [w2 - 1, 0],
        [w2 - 1, h2 - 1],
        [0, h2 - 1]
    ]).reshape(-1, 1, 2)

    warped_corners2 = cv2.perspectiveTransform(corners2, H)

    # Corners of image 1.
    corners1 = np.float32([
        [0, 0],
        [w1 - 1, 0],
        [w1 - 1, h1 - 1],
        [0, h1 - 1]
    ]).reshape(-1, 1, 2)

    all_corners = np.concatenate(
        [corners1, warped_corners2],
        axis=0
    ).reshape(-1, 2)

    min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
    max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)

    # Translation makes negative coordinates valid canvas coordinates.
    tx = -min_x if min_x < 0 else 0
    ty = -min_y if min_y < 0 else 0

    T = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ], dtype=np.float64)

    canvas_w = max_x - min_x + 1
    canvas_h = max_y - min_y + 1

    warped2 = cv2.warpPerspective(
        image2,
        T @ H,
        (canvas_w, canvas_h)
    )

    # Put image1 onto the same canvas.
    canvas1 = np.zeros_like(warped2)

    y1, y2 = ty, ty + h1
    x1, x2 = tx, tx + w1

    canvas1[y1:y2, x1:x2] = image1

    # Binary masks for later blending.
    mask1 = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    mask1[y1:y2, x1:x2] = 255

    mask2 = cv2.warpPerspective(
        np.ones((h2, w2), dtype=np.uint8) * 255,
        T @ H,
        (canvas_w, canvas_h)
    )

    return canvas1, warped2, mask1, mask2

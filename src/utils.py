import cv2
import numpy as np


def draw_good_matches(image1, kp1, image2, kp2, matches, output_path):
    """Save a visualization of accepted feature matches."""
    vis = cv2.drawMatches(
        image1, kp1, image2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(output_path), vis)


def draw_inlier_matches(image1, kp1, image2, kp2, matches, inlier_mask,
                        output_path):
    """Save only RANSAC inlier matches."""
    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]

    vis = cv2.drawMatches(
        image1, kp1, image2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(output_path), vis)


def compose_no_blend(canvas1, warped2, mask1, mask2):
    """Hard overlay baseline: image 1 wins in the overlap."""
    result = canvas1.copy()
    only2 = (mask2 > 0) & (mask1 == 0)
    result[only2] = warped2[only2]
    return result


def compose_basic_blend(canvas1, warped2, mask1, mask2):
    """Simple 50/50 alpha blend in the overlap."""
    result = canvas1.copy()
    only2 = (mask2 > 0) & (mask1 == 0)
    overlap = (mask1 > 0) & (mask2 > 0)

    result[only2] = warped2[only2]
    result[overlap] = (
        0.5 * canvas1[overlap].astype(np.float32)
        + 0.5 * warped2[overlap].astype(np.float32)
    ).astype(np.uint8)

    return result


def compute_overlap_seam_error(canvas1, warped2, mask1, mask2):
    """Initial diagnostic: mean absolute grayscale difference in overlap."""
    overlap = (mask1 > 0) & (mask2 > 0)

    if not np.any(overlap):
        return float("nan")

    gray1 = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(warped2, cv2.COLOR_BGR2GRAY)

    return float(
        np.mean(
            np.abs(
                gray1[overlap].astype(np.float32)
                - gray2[overlap].astype(np.float32)
            )
        )
    )

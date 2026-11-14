import cv2
import numpy as np

# Re-export key functionality from specialized modules
from src.blending import (
    blend_no_blend,
    blend_basic_alpha,
    blend_distance_feather,
    blend_adaptive_seam,
    find_optimal_seam,
)
from src.exposure import (
    compensate_exposure_gain,
    compensate_exposure_blocks,
    compensate_exposure,
)
from src.metrics import (
    compute_photometric_metrics,
    compute_seam_discontinuity,
    compute_ssim_overlap,
    PerformanceTracker,
)


def draw_good_matches(image1, kp1, image2, kp2, matches, output_path):
    """Save a visualization of accepted feature matches."""
    vis = cv2.drawMatches(
        image1, kp1, image2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(output_path), vis)
    return vis


def draw_inlier_matches(image1, kp1, image2, kp2, matches, inlier_mask,
                        output_path):
    """Save only RANSAC inlier matches."""
    inlier_matches = [m for m, keep in zip(matches, inlier_mask) if keep]

    vis = cv2.drawMatches(
        image1, kp1, image2, kp2, inlier_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    cv2.imwrite(str(output_path), vis)
    return vis


def compose_no_blend(canvas1, warped2, mask1, mask2):
    """Hard overlay baseline: image 1 wins in the overlap."""
    return blend_no_blend(canvas1, warped2, mask1, mask2)


def compose_basic_blend(canvas1, warped2, mask1, mask2):
    """Simple 50/50 alpha blend in the overlap."""
    return blend_basic_alpha(canvas1, warped2, mask1, mask2, alpha=0.5)


def compose_feather_blend(canvas1, warped2, mask1, mask2):
    """Distance-transform linear feathering blend."""
    return blend_distance_feather(canvas1, warped2, mask1, mask2)


def compose_adaptive_seam_blend(canvas1, warped2, mask1, mask2, feather_radius=11, return_details=False):
    """Adaptive seam blending with optimal energy path and narrow-band feathering."""
    return blend_adaptive_seam(canvas1, warped2, mask1, mask2,
                               feather_radius=feather_radius,
                               return_details=return_details)


def compute_overlap_seam_error(canvas1, warped2, mask1, mask2):
    """Mean absolute grayscale difference in the overlap region."""
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

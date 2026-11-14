from pathlib import Path
import cv2
import numpy as np

from src.features import detect_sift
from src.matching import ratio_test_match
from src.homography import estimate_homography
from src.warping import warp_pair_to_canvas
from src.exposure import compensate_exposure
from src.blending import blend_adaptive_seam, blend_distance_feather, blend_basic_alpha


def stitch_pair(img1, img2, use_ransac=True, exposure_method="gain",
                blending_method="adaptive"):
    """Stitch two images with specified homography, exposure compensation, and blending.

    Returns:
        blended_panorama, metadata_dict
    """
    kp1, des1 = detect_sift(img1)
    kp2, des2 = detect_sift(img2)

    matches = ratio_test_match(des1, des2)
    if len(matches) < 4:
        raise RuntimeError(f"Insufficient matches between pair: {len(matches)}")

    H, inlier_mask, src_pts, dst_pts = estimate_homography(
        kp1, kp2, matches, use_ransac=use_ransac
    )

    canvas1, warped2, mask1, mask2 = warp_pair_to_canvas(img1, img2, H)

    # Exposure compensation
    if exposure_method and exposure_method != "none":
        warped2_comp, exp_meta = compensate_exposure(
            canvas1, warped2, mask1, mask2, method=exposure_method
        )
    else:
        warped2_comp = warped2
        exp_meta = {"method": "none"}

    # Blending
    if blending_method == "adaptive":
        pano = blend_adaptive_seam(canvas1, warped2_comp, mask1, mask2)
    elif blending_method == "feather":
        pano = blend_distance_feather(canvas1, warped2_comp, mask1, mask2)
    elif blending_method == "basic":
        pano = blend_basic_alpha(canvas1, warped2_comp, mask1, mask2, alpha=0.5)
    else:  # no blend
        pano = canvas1.copy()
        pano[(mask2 > 0) & (mask1 == 0)] = warped2_comp[(mask2 > 0) & (mask1 == 0)]

    metadata = {
        "kp1": len(kp1),
        "kp2": len(kp2),
        "matches": len(matches),
        "inliers": int(inlier_mask.sum()) if use_ransac else len(matches),
        "H": H,
        "exposure": exp_meta,
    }
    return pano, metadata


def stitch_image_sequence(image_list, use_ransac=True, exposure_method="gain",
                           blending_method="adaptive"):
    """Sequentially stitch a chain of 3 to 6 images into a full panorama."""
    if len(image_list) < 2:
        raise ValueError("At least 2 images required for stitching.")

    current_pano = image_list[0]
    all_metadata = []

    for i in range(1, len(image_list)):
        next_img = image_list[i]
        current_pano, meta = stitch_pair(
            current_pano,
            next_img,
            use_ransac=use_ransac,
            exposure_method=exposure_method,
            blending_method=blending_method
        )
        all_metadata.append(meta)

    return current_pano, all_metadata

import cv2
import numpy as np


def get_points(kp1, kp2, matches):
    """Return corresponding points.

    We estimate a transform from image 2 -> image 1.
    Therefore:
        src_pts = points from image 2
        dst_pts = points from image 1
    """
    src_pts = np.float32(
        [kp2[m.trainIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp1[m.queryIdx].pt for m in matches]
    ).reshape(-1, 1, 2)

    return src_pts, dst_pts


def estimate_homography(kp1, kp2, matches, use_ransac=True,
                        reproj_threshold=5.0):
    """Estimate H mapping image 2 coordinates to image 1 coordinates."""
    if len(matches) < 4:
        raise ValueError(
            f"Need at least 4 matches for homography; got {len(matches)}."
        )

    src_pts, dst_pts = get_points(kp1, kp2, matches)

    if use_ransac:
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=cv2.RANSAC,
            ransacReprojThreshold=reproj_threshold
        )
    else:
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            method=0
        )

    if H is None or mask is None:
        raise RuntimeError("Homography estimation failed.")

    return H, mask.ravel().astype(bool), src_pts, dst_pts


def compute_reprojection_errors(H, src_pts, dst_pts):
    """Compute Euclidean reprojection errors for correspondences under homography H.

    Formula:
        e_i = || dst_pts_i - pi(H * src_pts_i) ||_2

    Returns:
        errors (np.ndarray): 1D array of per-point reprojection errors.
        rmse (float): Root mean squared reprojection error.
        mae (float): Mean absolute reprojection error.
    """
    if H is None or len(src_pts) == 0:
        return np.array([]), float("nan"), float("nan")

    src = src_pts.reshape(-1, 1, 2)
    dst = dst_pts.reshape(-1, 2)

    # Project src_pts using H
    projected = cv2.perspectiveTransform(src, H).reshape(-1, 2)

    # Compute Euclidean distance
    errors = np.linalg.norm(dst - projected, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(errors))

    return errors, rmse, mae

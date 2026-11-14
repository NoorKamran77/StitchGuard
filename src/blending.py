import cv2
import numpy as np


def blend_no_blend(canvas1, warped2, mask1, mask2):
    """Hard overlay baseline: image 1 takes full precedence in overlap.

    Any pixel where mask1 > 0 uses canvas1.
    Remaining valid pixels where mask2 > 0 use warped2.
    """
    result = canvas1.copy()
    only2 = (mask2 > 0) & (mask1 == 0)
    result[only2] = warped2[only2]
    return result


def blend_basic_alpha(canvas1, warped2, mask1, mask2, alpha=0.5):
    """Simple uniform alpha blend across the entire overlap region.

    In overlap: result = alpha * image1 + (1 - alpha) * image2.
    Outside overlap: respective source image.
    """
    result = canvas1.copy()
    only2 = (mask2 > 0) & (mask1 == 0)
    overlap = (mask1 > 0) & (mask2 > 0)

    result[only2] = warped2[only2]
    if np.any(overlap):
        result[overlap] = np.clip(
            alpha * canvas1[overlap].astype(np.float32)
            + (1.0 - alpha) * warped2[overlap].astype(np.float32),
            0,
            255
        ).astype(np.uint8)

    return result


def blend_distance_feather(canvas1, warped2, mask1, mask2):
    """Smooth linear feathering using Euclidean distance transforms.

    Each pixel in the overlap is weighted proportionally to its distance
    to the boundary of its respective source mask:
        w1(p) = d1(p) / (d1(p) + d2(p))
        w2(p) = d2(p) / (d1(p) + d2(p))
    This produces a continuous transition without edge step jumps.
    """
    result = np.zeros_like(canvas1)
    only1 = (mask1 > 0) & (mask2 == 0)
    only2 = (mask2 > 0) & (mask1 == 0)
    overlap = (mask1 > 0) & (mask2 > 0)

    result[only1] = canvas1[only1]
    result[only2] = warped2[only2]

    if np.any(overlap):
        d1 = cv2.distanceTransform(mask1, cv2.DIST_L2, 5).astype(np.float32)
        d2 = cv2.distanceTransform(mask2, cv2.DIST_L2, 5).astype(np.float32)

        denom = d1 + d2 + 1e-6
        w1 = (d1 / denom)[:, :, None]
        w2 = (d2 / denom)[:, :, None]

        blended = canvas1.astype(np.float32) * w1 + warped2.astype(np.float32) * w2
        result[overlap] = np.clip(blended[overlap], 0, 255).astype(np.uint8)

    return result


def find_optimal_seam(canvas1, warped2, mask1, mask2, max_work_size=500):
    """Find the optimal seam through the overlap using dynamic programming.

    Energy Formulation:
        E(y, x) = ||I1(y, x) - I2(y, x)||^2 / (||grad(I1)|| + ||grad(I2)|| + eps)

    Pixels outside the overlap mask are assigned a massive penalty (1e8) to force
    the seam strictly through valid overlapping image areas.

    Returns:
        seam_mask1 (np.ndarray): Boolean mask of pixels assigned to Image 1 (canvas size).
        seam_mask2 (np.ndarray): Boolean mask of pixels assigned to Image 2 (canvas size).
        seam_coords (tuple): (y_coords, x_coords) in full canvas coordinates.
        energy_roi (np.ndarray): 2D energy map in the overlap bounding box.
        roi_bbox (tuple): (ymin, ymax, xmin, xmax).
    """
    overlap = (mask1 > 0) & (mask2 > 0)
    if not np.any(overlap):
        # Fallback if no overlap
        return (mask1 > 0), (mask2 > 0) & (mask1 == 0), ([], []), None, None

    ys, xs = np.where(overlap)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    roi1 = canvas1[ymin:ymax + 1, xmin:xmax + 1].astype(np.float32)
    roi2 = warped2[ymin:ymax + 1, xmin:xmax + 1].astype(np.float32)
    roi_ov = overlap[ymin:ymax + 1, xmin:xmax + 1]

    # 1. Color difference energy
    color_diff = np.sum((roi1 - roi2) ** 2, axis=2)

    # 2. Gradient magnitude (Sobel)
    gray1 = cv2.cvtColor(roi1.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(roi2.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx1 = cv2.Sobel(gray1, cv2.CV_32F, 1, 0, ksize=3)
    gy1 = cv2.Sobel(gray1, cv2.CV_32F, 0, 1, ksize=3)
    gx2 = cv2.Sobel(gray2, cv2.CV_32F, 1, 0, ksize=3)
    gy2 = cv2.Sobel(gray2, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx1 ** 2 + gy1 ** 2) + np.sqrt(gx2 ** 2 + gy2 ** 2) + 1.0

    energy = color_diff / grad_mag
    energy[~roi_ov] = 1e8

    # Determine orientation (vertical seam for left-right overlap, horizontal for top-bottom)
    h_roi, w_roi = energy.shape
    is_vertical_seam = (h_roi >= w_roi) or (h_roi > 100)

    # Check relative positioning of images
    c1_x = np.mean(np.where(mask1)[1]) if np.any(mask1) else 0
    c2_x = np.mean(np.where(mask2)[1]) if np.any(mask2) else 0
    c1_y = np.mean(np.where(mask1)[0]) if np.any(mask1) else 0
    c2_y = np.mean(np.where(mask2)[0]) if np.any(mask2) else 0

    img1_is_left = (c1_x < c2_x)
    img1_is_top = (c1_y < c2_y)

    # Scale down if ROI is very large for computational efficiency
    scale = min(1.0, float(max_work_size) / max(h_roi, w_roi))
    if scale < 1.0:
        energy_s = cv2.resize(energy, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        energy_s = energy.copy()

    hs, ws = energy_s.shape

    if is_vertical_seam:
        # Seam runs vertically from top row to bottom row
        dp = np.zeros_like(energy_s)
        backtrack = np.zeros((hs, ws), dtype=np.int32)
        dp[0] = energy_s[0]

        for y in range(1, hs):
            prev = dp[y - 1]
            left = np.pad(prev[:-1], (1, 0), constant_values=1e9)
            mid = prev
            right = np.pad(prev[1:], (0, 1), constant_values=1e9)
            choices = np.stack([left, mid, right], axis=0)
            min_idx = np.argmin(choices, axis=0)
            dp[y] = energy_s[y] + np.min(choices, axis=0)
            backtrack[y] = np.arange(ws) + (min_idx - 1)

        seam_x_s = np.zeros(hs, dtype=np.int32)
        seam_x_s[-1] = np.argmin(dp[-1])
        for y in range(hs - 2, -1, -1):
            seam_x_s[y] = backtrack[y + 1, seam_x_s[y + 1]]

        # Map back to full ROI resolution
        seam_x = np.interp(
            np.arange(h_roi),
            np.linspace(0, h_roi - 1, hs),
            seam_x_s / scale
        ).astype(np.int32)
        seam_x = np.clip(seam_x, 0, w_roi - 1)
        seam_y = np.arange(h_roi)

        # Build ROI assignment mask
        roi_mask1 = np.zeros((h_roi, w_roi), dtype=bool)
        for y in range(h_roi):
            sx = seam_x[y]
            if img1_is_left:
                roi_mask1[y, :sx] = True
            else:
                roi_mask1[y, sx:] = True

        full_seam_y = seam_y + ymin
        full_seam_x = seam_x + xmin

    else:
        # Horizontal seam
        dp = np.zeros_like(energy_s)
        backtrack = np.zeros((hs, ws), dtype=np.int32)
        dp[:, 0] = energy_s[:, 0]

        for x in range(1, ws):
            prev = dp[:, x - 1]
            up = np.pad(prev[:-1], (1, 0), constant_values=1e9)
            mid = prev
            down = np.pad(prev[1:], (0, 1), constant_values=1e9)
            choices = np.stack([up, mid, down], axis=0)
            min_idx = np.argmin(choices, axis=0)
            dp[:, x] = energy_s[:, x] + np.min(choices, axis=0)
            backtrack[:, x] = np.arange(hs) + (min_idx - 1)

        seam_y_s = np.zeros(ws, dtype=np.int32)
        seam_y_s[-1] = np.argmin(dp[:, -1])
        for x in range(ws - 2, -1, -1):
            seam_y_s[x] = backtrack[seam_y_s[x + 1], x + 1]

        seam_y = np.interp(
            np.arange(w_roi),
            np.linspace(0, w_roi - 1, ws),
            seam_y_s / scale
        ).astype(np.int32)
        seam_y = np.clip(seam_y, 0, h_roi - 1)
        seam_x = np.arange(w_roi)

        roi_mask1 = np.zeros((h_roi, w_roi), dtype=bool)
        for x in range(w_roi):
            sy = seam_y[x]
            if img1_is_top:
                roi_mask1[:sy, x] = True
            else:
                roi_mask1[sy:, x] = True

        full_seam_y = seam_y + ymin
        full_seam_x = seam_x + xmin

    # Full canvas masks
    seam_mask1 = (mask1 > 0).copy()
    # In overlap, enforce seam decision
    seam_mask1[ymin:ymax + 1, xmin:xmax + 1][roi_ov] = roi_mask1[roi_ov]

    seam_mask2 = (mask2 > 0) & (~seam_mask1)

    return seam_mask1, seam_mask2, (full_seam_y, full_seam_x), energy, (ymin, ymax, xmin, xmax)


def blend_adaptive_seam(canvas1, warped2, mask1, mask2, feather_radius=11, return_details=False):
    """Adaptive seam blending using energy-minimizing optimal cut.

    Finds the optimal seam line minimizing photometric differences and gradient
    transitions. Then applies narrow-band feathering across the seam line
    to eliminate any edge step jump while preventing ghosting artifacts.

    Args:
        canvas1: Canvas image 1.
        warped2: Warped image 2.
        mask1: Binary mask 1.
        mask2: Binary mask 2.
        feather_radius: Radius for narrow-band blur along the seam cut (0 = hard cut).
        return_details: If True, returns auxiliary seam metadata and visualizations.

    Returns:
        blended_panorama (np.ndarray)
        (Optional) details dictionary
    """
    seam_mask1, seam_mask2, (seam_y, seam_x), energy, bbox = find_optimal_seam(
        canvas1, warped2, mask1, mask2
    )

    if feather_radius > 0 and len(seam_y) > 0:
        # Narrow-band feathering along the seam
        k = feather_radius if (feather_radius % 2 == 1) else (feather_radius + 1)
        weight1 = cv2.GaussianBlur(seam_mask1.astype(np.float32), (k, k), 0)
        weight1 = weight1[:, :, None]

        valid_region = (mask1 > 0) | (mask2 > 0)
        blended = np.clip(
            canvas1.astype(np.float32) * weight1
            + warped2.astype(np.float32) * (1.0 - weight1),
            0,
            255
        ).astype(np.uint8)

        # Clear background outside valid masks
        blended[~valid_region] = 0
    else:
        # Pure hard cut along optimal seam
        blended = canvas1.copy()
        blended[seam_mask2] = warped2[seam_mask2]

    if not return_details:
        return blended

    # Generate seam visualization overlay
    vis_seam = blended.copy()
    if len(seam_y) > 0:
        for y, x in zip(seam_y, seam_x):
            cv2.circle(vis_seam, (int(x), int(y)), 2, (0, 255, 0), -1)

    details = {
        "seam_coords": (seam_y, seam_x),
        "seam_mask1": seam_mask1,
        "seam_mask2": seam_mask2,
        "energy_map": energy,
        "bbox": bbox,
        "seam_overlay": vis_seam,
    }
    return blended, details

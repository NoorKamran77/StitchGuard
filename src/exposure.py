import cv2
import numpy as np


def compensate_exposure_gain(canvas1, warped2, mask1, mask2):
    """Global channel-wise gain compensation based on overlap region statistics.

    Computes per-channel gain:
        g_c = mean(I1[overlap, c]) / mean(I2[overlap, c])
    and scales image 2 so that its mean luminance matches image 1.

    Returns:
        compensated_warped2 (np.ndarray): Exposure-compensated warped image.
        gains (list of float): Per-channel scale factor [g_B, g_G, g_R].
    """
    overlap = (mask1 > 0) & (mask2 > 0)

    if not np.any(overlap):
        return warped2.copy(), [1.0, 1.0, 1.0]

    gains = []
    compensated = warped2.astype(np.float32).copy()

    for c in range(3):
        m1 = float(np.mean(canvas1[overlap, c]))
        m2 = float(np.mean(warped2[overlap, c]))
        gain = m1 / max(m2, 1e-4)
        gain = float(np.clip(gain, 0.2, 5.0))
        gains.append(gain)
        compensated[:, :, c] = np.clip(compensated[:, :, c] * gain, 0, 255)

    return compensated.astype(np.uint8), gains


def compensate_exposure_blocks(canvas1, warped2, mask1, mask2, block_size=64):
    """Block-based local gain compensation for non-uniform lighting or vignetting.

    Partitions the overlap into block_size x block_size grids, computes local
    gains, and smoothly interpolates the gain surface across the canvas.

    Returns:
        compensated_warped2 (np.ndarray): Exposure-compensated warped image.
        gain_map (np.ndarray): 2D/3D interpolated gain map.
    """
    overlap = (mask1 > 0) & (mask2 > 0)
    if not np.any(overlap):
        return warped2.copy(), np.ones_like(canvas1, dtype=np.float32)

    h, w = canvas1.shape[:2]
    ny = int(np.ceil(h / block_size))
    nx = int(np.ceil(w / block_size))

    # Compute global gains as fallback
    _, global_gains = compensate_exposure_gain(canvas1, warped2, mask1, mask2)

    grid_gains = np.ones((ny, nx, 3), dtype=np.float32)
    grid_gains[:, :] = global_gains

    for by in range(ny):
        y0 = by * block_size
        y1 = min(h, y0 + block_size)
        for bx in range(nx):
            x0 = bx * block_size
            x1 = min(w, x0 + block_size)

            blk_ov = overlap[y0:y1, x0:x1]
            if np.count_nonzero(blk_ov) > (block_size * block_size * 0.05):
                for c in range(3):
                    m1 = float(np.mean(canvas1[y0:y1, x0:x1, c][blk_ov]))
                    m2 = float(np.mean(warped2[y0:y1, x0:x1, c][blk_ov]))
                    g = m1 / max(m2, 1e-4)
                    grid_gains[by, bx, c] = float(np.clip(g, 0.3, 3.5))

    # Resize grid_gains to full canvas size with bilinear interpolation
    gain_map = cv2.resize(grid_gains, (w, h), interpolation=cv2.INTER_LINEAR)
    gain_map = cv2.GaussianBlur(gain_map, (31, 31), 0)

    compensated = np.clip(warped2.astype(np.float32) * gain_map, 0, 255).astype(np.uint8)
    return compensated, gain_map


def compensate_exposure(canvas1, warped2, mask1, mask2, method="gain"):
    """Unified exposure compensation interface.

    Args:
        canvas1: Base canvas with image 1.
        warped2: Warped image 2.
        mask1: Binary mask of image 1.
        mask2: Binary mask of image 2.
        method: 'gain' (global per-channel), 'blocks' (local blocks), or 'none'.

    Returns:
        compensated_warped2, metadata
    """
    if method == "none" or method is None:
        return warped2.copy(), {"method": "none", "gains": [1.0, 1.0, 1.0]}

    if method == "blocks":
        comp, gmap = compensate_exposure_blocks(canvas1, warped2, mask1, mask2)
        return comp, {"method": "blocks", "gain_map": gmap}

    # Default: global gain
    comp, gains = compensate_exposure_gain(canvas1, warped2, mask1, mask2)
    return comp, {"method": "gain", "gains": gains}

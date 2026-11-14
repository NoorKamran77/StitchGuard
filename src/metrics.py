import time
import cv2
import numpy as np


class PerformanceTracker:
    """Timer utility to record latency of each pipeline stage."""

    def __init__(self):
        self.timings = {}
        self._start_times = {}

    def start(self, stage_name):
        self._start_times[stage_name] = time.perf_counter()

    def stop(self, stage_name):
        if stage_name in self._start_times:
            elapsed_ms = (time.perf_counter() - self._start_times[stage_name]) * 1000.0
            self.timings[stage_name] = elapsed_ms
            del self._start_times[stage_name]
            return elapsed_ms
        return 0.0

    def get_summary(self):
        """Return timings dictionary with total latency."""
        res = dict(self.timings)
        res["total_pipeline_ms"] = sum(self.timings.values())
        return res


def compute_ssim_roi(roi1, roi2, roi_mask):
    """Compute Structural Similarity Index (SSIM) within the overlap ROI."""
    if not np.any(roi_mask):
        return float("nan")

    C1 = (0.01 * 255.0) ** 2
    C2 = (0.03 * 255.0) ** 2

    ksize = 11
    sigma = 1.5

    mu1 = cv2.GaussianBlur(roi1, (ksize, ksize), sigma)
    mu2 = cv2.GaussianBlur(roi2, (ksize, ksize), sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(roi1 ** 2, (ksize, ksize), sigma) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(roi2 ** 2, (ksize, ksize), sigma) - mu2_sq
    sigma12 = cv2.GaussianBlur(roi1 * roi2, (ksize, ksize), sigma) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-7
    )

    # Valid interior pixels
    kernel = np.ones((5, 5), np.uint8)
    valid = cv2.erode(roi_mask.astype(np.uint8), kernel) > 0
    if not np.any(valid):
        valid = roi_mask

    return float(np.mean(ssim_map[valid]))


def compute_ssim_overlap(canvas1, warped2, overlap):
    """Compute SSIM on overlap region."""
    ys, xs = np.where(overlap)
    if len(ys) == 0:
        return float("nan")
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())
    roi1 = cv2.cvtColor(canvas1[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi2 = cv2.cvtColor(warped2[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi_ov = overlap[ymin:ymax + 1, xmin:xmax + 1]
    return compute_ssim_roi(roi1, roi2, roi_ov)


def compute_photometric_metrics(canvas1, warped2, mask1, mask2):
    """Compute photometric error metrics between aligned images in the overlap.

    Returns:
        metrics (dict):
            - overlap_pixels: Number of overlapping pixels
            - mae: Mean Absolute Error (grayscale [0-255])
            - rmse: Root Mean Square Error
            - psnr: Peak Signal-to-Noise Ratio (dB)
            - ssim: Structural Similarity Index [-1, 1]
    """
    overlap = (mask1 > 0) & (mask2 > 0)

    if not np.any(overlap):
        return {
            "overlap_pixels": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "psnr": float("nan"),
            "ssim": float("nan"),
        }

    ys, xs = np.where(overlap)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    roi1 = cv2.cvtColor(canvas1[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi2 = cv2.cvtColor(warped2[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi_ov = overlap[ymin:ymax + 1, xmin:xmax + 1]

    diff = roi1[roi_ov] - roi2[roi_ov]
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    psnr = float(20.0 * np.log10(255.0 / max(rmse, 1e-6)))
    ssim = compute_ssim_roi(roi1, roi2, roi_ov)

    return {
        "overlap_pixels": int(np.sum(overlap)),
        "mae": mae,
        "rmse": rmse,
        "psnr": psnr,
        "ssim": ssim,
    }


def compute_seam_discontinuity(blended, mask1, mask2):
    """Compute gradient discontinuity metric across the composite seam.

    Measures average gradient magnitude across the boundary region between
    image 1 and image 2 in the final blended composition.
    A lower value indicates a smoother, less visible transition.
    """
    overlap = (mask1 > 0) & (mask2 > 0)
    if not np.any(overlap):
        return 0.0

    ys, xs = np.where(overlap)
    ymin, ymax = int(ys.min()), int(ys.max())
    xmin, xmax = int(xs.min()), int(xs.max())

    # Crop to overlap bounding box
    roi_mask1 = mask1[ymin:ymax + 1, xmin:xmax + 1]
    roi_mask2 = mask2[ymin:ymax + 1, xmin:xmax + 1]
    roi_blended = cv2.cvtColor(blended[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)

    kernel = np.ones((5, 5), np.uint8)
    eroded1 = cv2.erode(roi_mask1.astype(np.uint8), kernel)
    eroded2 = cv2.erode(roi_mask2.astype(np.uint8), kernel)

    boundary1 = (roi_mask1 > 0) & (eroded1 == 0) & (roi_mask2 > 0)
    boundary2 = (roi_mask2 > 0) & (eroded2 == 0) & (roi_mask1 > 0)
    seam_zone = boundary1 | boundary2

    if not np.any(seam_zone):
        seam_zone = (roi_mask1 > 0) & (roi_mask2 > 0)

    gx = cv2.Sobel(roi_blended, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi_blended, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx ** 2 + gy ** 2)

    return float(np.mean(grad_mag[seam_zone]))

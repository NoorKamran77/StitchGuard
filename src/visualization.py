from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def bgr_to_rgb(img, max_dim=1600):
    """Convert BGR image to RGB and optionally scale down for fast matplotlib rendering."""
    if img is None:
        return None
    if len(img.shape) == 2:
        res = img
    else:
        res = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = res.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        nw, nh = int(round(w * scale)), int(round(h * scale))
        res = cv2.resize(res, (nw, nh), interpolation=cv2.INTER_AREA)

    return res


def plot_feature_matches_and_inliers(img1, kp1, img2, kp2, matches, inlier_mask, save_path):
    """Save side-by-side figure of all ratio-test matches vs RANSAC inliers."""
    inliers = [m for m, keep in zip(matches, inlier_mask) if keep]
    inlier_ratio = len(inliers) / max(len(matches), 1)

    vis_all = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    vis_inliers = cv2.drawMatches(
        img1, kp1, img2, kp2, inliers, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    axes[0].imshow(bgr_to_rgb(vis_all))
    axes[0].set_title(
        f"Lowe's Ratio Test Matches (Count: {len(matches)})",
        fontsize=13, fontweight="bold"
    )
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(vis_inliers))
    axes[1].set_title(
        f"RANSAC Inlier Matches (Inliers: {len(inliers)}, Inlier Ratio: {inlier_ratio:.2%})",
        fontsize=13, fontweight="bold", color="darkgreen"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_homography_ablation(no_ransac_pano, ransac_pano, metrics_no_ransac, metrics_ransac, save_path):
    """Save visual comparison between No-RANSAC and RANSAC homography stitching."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 9))

    axes[0].imshow(bgr_to_rgb(no_ransac_pano))
    axes[0].set_title(
        f"Without RANSAC (Matches: {metrics_no_ransac.get('matches', 'N/A')}, "
        f"Reproj RMSE: {metrics_no_ransac.get('reproj_rmse', float('nan')):.2f}px, "
        f"Overlap Error: {metrics_no_ransac.get('seam_error', float('nan')):.2f})",
        fontsize=12, fontweight="bold", color="darkred"
    )
    axes[0].axis("off")

    axes[1].imshow(bgr_to_rgb(ransac_pano))
    axes[1].set_title(
        f"With RANSAC (Inliers: {metrics_ransac.get('inliers', 'N/A')} "
        f"[{metrics_ransac.get('inlier_ratio', 0.0):.1%}], "
        f"Reproj RMSE: {metrics_ransac.get('reproj_rmse', float('nan')):.2f}px, "
        f"Overlap Error: {metrics_ransac.get('seam_error', float('nan')):.2f})",
        fontsize=12, fontweight="bold", color="darkgreen"
    )
    axes[1].axis("off")

    plt.suptitle("StitchGuard: Homography Ablation (No-RANSAC vs. RANSAC)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_blending_ablation(no_blend, basic_alpha, feather, adaptive, metrics_dict, save_path):
    """Save 4-panel visual comparison of all blending strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    disc = metrics_dict.get("seam_discontinuity", {})

    axes[0, 0].imshow(bgr_to_rgb(no_blend))
    axes[0, 0].set_title(
        f"(a) No Blending (Hard Overlay)\nSeam Discontinuity: {disc.get('no_blend', 0.0):.2f}",
        fontsize=12, fontweight="bold"
    )
    axes[0, 0].axis("off")

    axes[0, 1].imshow(bgr_to_rgb(basic_alpha))
    axes[0, 1].set_title(
        f"(b) Basic Blending (50/50 Uniform Alpha)\nSeam Discontinuity: {disc.get('basic_alpha', 0.0):.2f}",
        fontsize=12, fontweight="bold"
    )
    axes[0, 1].axis("off")

    axes[1, 0].imshow(bgr_to_rgb(feather))
    axes[1, 0].set_title(
        f"(c) Distance-Transform Feathering (Linear Ramp)\nSeam Discontinuity: {disc.get('feather', 0.0):.2f}",
        fontsize=12, fontweight="bold"
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(bgr_to_rgb(adaptive))
    axes[1, 1].set_title(
        f"(d) Adaptive Seam Blending (Optimal Energy Cut)\nSeam Discontinuity: {disc.get('adaptive', 0.0):.2f}",
        fontsize=12, fontweight="bold", color="darkgreen"
    )
    axes[1, 1].axis("off")

    plt.suptitle("StitchGuard: Blending Ablation Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_exposure_ablation(canvas1, warped2_raw, warped2_comp, mask1, mask2,
                           metrics_raw, metrics_comp, save_path):
    """Save comparative panel showing stitched panoramas and difference error heatmaps."""
    overlap = (mask1 > 0) & (mask2 > 0)

    # Crop to overlap bounding box for heatmaps
    ys, xs = np.where(overlap)
    ymin, ymax = ys.min(), ys.max()
    xmin, xmax = xs.min(), xs.max()

    gray1_roi = cv2.cvtColor(canvas1[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2_raw_roi = cv2.cvtColor(warped2_raw[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2_comp_roi = cv2.cvtColor(warped2_comp[ymin:ymax + 1, xmin:xmax + 1], cv2.COLOR_BGR2GRAY).astype(np.float32)
    roi_ov = overlap[ymin:ymax + 1, xmin:xmax + 1]

    diff_raw_roi = np.zeros_like(gray1_roi)
    diff_comp_roi = np.zeros_like(gray1_roi)

    diff_raw_roi[roi_ov] = np.abs(gray1_roi[roi_ov] - gray2_raw_roi[roi_ov])
    diff_comp_roi[roi_ov] = np.abs(gray1_roi[roi_ov] - gray2_comp_roi[roi_ov])

    # Simple blend of raw vs comp for panorama display
    pano_raw = canvas1.copy()
    pano_raw[(mask2 > 0) & (mask1 == 0)] = warped2_raw[(mask2 > 0) & (mask1 == 0)]
    pano_comp = canvas1.copy()
    pano_comp[(mask2 > 0) & (mask1 == 0)] = warped2_comp[(mask2 > 0) & (mask1 == 0)]

    fig = plt.figure(figsize=(15, 9))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(bgr_to_rgb(pano_raw))
    ax0.set_title(
        f"Without Exposure Compensation\n(Overlap MAE: {metrics_raw['mae']:.2f}, "
        f"RMSE: {metrics_raw['rmse']:.2f}, PSNR: {metrics_raw['psnr']:.2f}dB)",
        fontsize=11, fontweight="bold", color="darkred"
    )
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(bgr_to_rgb(pano_comp))
    ax1.set_title(
        f"With Exposure Compensation (Gain Matching)\n(Overlap MAE: {metrics_comp['mae']:.2f}, "
        f"RMSE: {metrics_comp['rmse']:.2f}, PSNR: {metrics_comp['psnr']:.2f}dB)",
        fontsize=11, fontweight="bold", color="darkgreen"
    )
    ax1.axis("off")

    v_max = max(50.0, float(np.percentile(diff_raw_roi[roi_ov], 98)))

    ax2 = fig.add_subplot(gs[1, 0])
    im2 = ax2.imshow(diff_raw_roi, cmap="inferno", vmin=0, vmax=v_max)
    ax2.set_title("Overlap Error Heatmap (Without Compensation)", fontsize=11, fontweight="bold")
    ax2.axis("off")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label="Error [0-255]")

    ax3 = fig.add_subplot(gs[1, 1])
    im3 = ax3.imshow(diff_comp_roi, cmap="inferno", vmin=0, vmax=v_max)
    ax3.set_title("Overlap Error Heatmap (With Compensation)", fontsize=11, fontweight="bold")
    ax3.axis("off")
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label="Error [0-255]")

    plt.suptitle("StitchGuard: Exposure Compensation Evaluation", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_seam_analysis(energy_roi, seam_coords, seam_overlay, save_path):
    """Save detailed analysis of adaptive seam energy and the optimal path."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    seam_y, seam_x = seam_coords

    if energy_roi is not None:
        energy_disp = np.log1p(np.clip(energy_roi, 0, 1e5))
        axes[0].imshow(energy_disp, cmap="viridis")
        axes[0].set_title("Seam Energy Cost Map E(y, x) [log scale]", fontsize=12, fontweight="bold")
        axes[0].axis("off")

    if seam_overlay is not None:
        axes[1].imshow(bgr_to_rgb(seam_overlay))
        axes[1].set_title(
            f"Panorama with Optimal Seam Cut Line (Green) [{len(seam_y)} points]",
            fontsize=12, fontweight="bold"
        )
        axes[1].axis("off")

    plt.suptitle("StitchGuard: Adaptive Seam Energy & Optimal Cut Trajectory", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_performance_matrix(metrics_df, timing_dict, save_path):
    """Save high-resolution charts representing the performance matrix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Seam Discontinuity across Blending Methods
    blend_methods = ["No Blend", "Basic 50/50", "Distance Feather", "Adaptive Seam"]
    discontinuities = []
    for m in blend_methods:
        match = metrics_df.loc[metrics_df["Method"] == m, "Seam Discontinuity"]
        discontinuities.append(float(match.values[0]) if len(match) > 0 else 0.0)

    colors_blend = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]
    bars = axes[0, 0].bar(blend_methods, discontinuities, color=colors_blend, edgecolor="black", width=0.55)
    axes[0, 0].set_title("Seam Discontinuity Metric (Lower = Better)", fontsize=11, fontweight="bold")
    axes[0, 0].set_ylabel("Gradient Discontinuity")
    axes[0, 0].grid(axis="y", linestyle="--", alpha=0.6)
    for bar in bars:
        h = bar.get_height()
        axes[0, 0].annotate(f"{h:.2f}",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Chart 2: Overlap Error: Without vs With Exposure Compensation
    exp_labels = ["Without Exp. Comp", "With Exp. Comp"]
    m_no = metrics_df.loc[metrics_df["Method"] == "No Comp (RANSAC)"]
    m_with = metrics_df.loc[metrics_df["Method"] == "With Exp Comp"]

    mae_vals = [
        float(m_no["Overlap MAE"].values[0]) if len(m_no) > 0 else 0.0,
        float(m_with["Overlap MAE"].values[0]) if len(m_with) > 0 else 0.0,
    ]
    rmse_vals = [
        float(m_no["Overlap RMSE"].values[0]) if len(m_no) > 0 else 0.0,
        float(m_with["Overlap RMSE"].values[0]) if len(m_with) > 0 else 0.0,
    ]

    x = np.arange(len(exp_labels))
    w = 0.35
    b1 = axes[0, 1].bar(x - w / 2, mae_vals, w, label="MAE", color="#337ab7", edgecolor="black")
    b2 = axes[0, 1].bar(x + w / 2, rmse_vals, w, label="RMSE", color="#e74c3c", edgecolor="black")
    axes[0, 1].set_title("Exposure Compensation Impact on Overlap Error", fontsize=11, fontweight="bold")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(exp_labels)
    axes[0, 1].set_ylabel("Intensity Error [0-255]")
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", linestyle="--", alpha=0.6)
    for b in list(b1) + list(b2):
        h = b.get_height()
        axes[0, 1].annotate(f"{h:.2f}",
                            xy=(b.get_x() + b.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8, fontweight="bold")

    # Chart 3: Geometric Quality (No RANSAC vs RANSAC)
    geo_labels = ["No RANSAC", "With RANSAC"]
    m_nr = metrics_df.loc[metrics_df["Method"] == "No RANSAC"]
    m_r = metrics_df.loc[metrics_df["Method"] == "RANSAC"]

    reproj_rmse = [
        float(m_nr["Reproj RMSE (px)"].values[0]) if len(m_nr) > 0 else 0.0,
        float(m_r["Reproj RMSE (px)"].values[0]) if len(m_r) > 0 else 0.0,
    ]
    inlier_ratios = [
        float(m_nr["Inlier Ratio"].values[0] * 100) if len(m_nr) > 0 else 0.0,
        float(m_r["Inlier Ratio"].values[0] * 100) if len(m_r) > 0 else 0.0,
    ]

    x2 = np.arange(len(geo_labels))
    b3 = axes[1, 0].bar(x2 - w / 2, reproj_rmse, w, label="Reproj RMSE (px)", color="#8e44ad", edgecolor="black")
    ax_twin = axes[1, 0].twinx()
    b4 = ax_twin.bar(x2 + w / 2, inlier_ratios, w, label="Inlier %", color="#27ae60", edgecolor="black")
    axes[1, 0].set_title("Homography Quality: No-RANSAC vs. RANSAC", fontsize=11, fontweight="bold")
    axes[1, 0].set_xticks(x2)
    axes[1, 0].set_xticklabels(geo_labels)
    axes[1, 0].set_ylabel("Reprojection RMSE (pixels)")
    ax_twin.set_ylabel("Inlier Ratio (%)")
    axes[1, 0].grid(axis="y", linestyle="--", alpha=0.6)

    # Chart 4: Latency Breakdown (Performance Matrix)
    stages = [k.replace("_", " ").title() for k in timing_dict.keys() if k != "total_pipeline_ms"]
    latencies = [float(timing_dict[k]) for k in timing_dict.keys() if k != "total_pipeline_ms"]

    y_pos = np.arange(len(stages))
    colors_time = plt.cm.viridis(np.linspace(0.2, 0.85, len(stages)))
    axes[1, 1].barh(y_pos, latencies, color=colors_time, edgecolor="black")
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(stages)
    axes[1, 1].set_xlabel("Time (milliseconds)")
    total_time = timing_dict.get("total_pipeline_ms", sum(latencies))
    axes[1, 1].set_title(f"Pipeline Latency Breakdown (Total: {total_time:.1f} ms)", fontsize=11, fontweight="bold")
    axes[1, 1].grid(axis="x", linestyle="--", alpha=0.6)
    for i, v in enumerate(latencies):
        axes[1, 1].text(v + 1, i, f" {v:.1f} ms", va="center", fontsize=8, fontweight="bold")

    plt.suptitle("StitchGuard: Comprehensive Performance Matrix Dashboard", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()


def plot_comprehensive_dashboard(img1, img2, final_panorama, df_summary, save_path):
    """Save an executive summary publication-ready figure of the pipeline and results."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.3])

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(bgr_to_rgb(img1))
    ax0.set_title("Input Image 1 (Reference)", fontsize=11, fontweight="bold")
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(bgr_to_rgb(img2))
    ax1.set_title("Input Image 2 (Target)", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, :])
    ax2.imshow(bgr_to_rgb(final_panorama))
    ax2.set_title(
        "StitchGuard Final Stitched Panorama (RANSAC + Exposure Comp + Adaptive Seam Blending)",
        fontsize=13, fontweight="bold", color="darkgreen"
    )
    ax2.axis("off")

    plt.suptitle("StitchGuard: Uncertainty-Aware Panorama Stitching System", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close()

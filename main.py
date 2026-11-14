import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd
from tabulate import tabulate

from src.features import detect_sift
from src.matching import ratio_test_match
from src.homography import estimate_homography, compute_reprojection_errors
from src.warping import warp_pair_to_canvas
from src.exposure import compensate_exposure
from src.blending import (
    blend_no_blend,
    blend_basic_alpha,
    blend_distance_feather,
    blend_adaptive_seam,
)
from src.metrics import (
    compute_photometric_metrics,
    compute_seam_discontinuity,
    PerformanceTracker,
)
from src.visualization import (
    plot_feature_matches_and_inliers,
    plot_homography_ablation,
    plot_blending_ablation,
    plot_exposure_ablation,
    plot_seam_analysis,
    plot_performance_matrix,
    plot_comprehensive_dashboard,
)
from src.utils import draw_good_matches, draw_inlier_matches


def run_stitching_pipeline(dataset_dir, output_dir, dataset_name="Set 1"):
    """Execute complete StitchGuard pipeline and ablations on an image pair."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    img1_path = dataset_dir / "img1.jpg"
    img2_path = dataset_dir / "img2.jpg"

    if not img1_path.exists() or not img2_path.exists():
        raise FileNotFoundError(
            f"Images not found in {dataset_dir}. Expected img1.jpg and img2.jpg."
        )

    image1 = cv2.imread(str(img1_path))
    image2 = cv2.imread(str(img2_path))

    if image1 is None or image2 is None:
        raise ValueError(f"Could not load images from {dataset_dir}.")

    tracker = PerformanceTracker()

    print(f"\n{'='*75}")
    print(f"  StitchGuard: Processing {dataset_name} ({dataset_dir})")
    print(f"{'='*75}")
    print(f"Image 1 dimensions: {image1.shape[1]}x{image1.shape[0]}")
    print(f"Image 2 dimensions: {image2.shape[1]}x{image2.shape[0]}")

    # -------------------------------------------------------------
    # Stage 1: Feature Detection (SIFT)
    # -------------------------------------------------------------
    tracker.start("sift_detection")
    kp1, des1 = detect_sift(image1)
    kp2, des2 = detect_sift(image2)
    t_detect = tracker.stop("sift_detection")
    print(f"[Stage 1] SIFT Keypoints: Image 1 = {len(kp1)}, Image 2 = {len(kp2)} ({t_detect:.1f} ms)")

    # -------------------------------------------------------------
    # Stage 2: Feature Matching (BFMatcher + Lowe's Ratio Test)
    # -------------------------------------------------------------
    tracker.start("feature_matching")
    matches = ratio_test_match(des1, des2, ratio=0.75)
    t_match = tracker.stop("feature_matching")
    print(f"[Stage 2] Good Matches after Lowe's Ratio Test: {len(matches)} ({t_match:.1f} ms)")

    if len(matches) < 4:
        raise RuntimeError("Fewer than 4 matches found. Stitching cannot proceed.")

    draw_good_matches(image1, kp1, image2, kp2, matches, output_dir / "good_matches.jpg")

    # -------------------------------------------------------------
    # Stage 3: Homography Estimation Ablation (No-RANSAC vs. RANSAC)
    # -------------------------------------------------------------
    print("\n--- Running Homography Ablation (No-RANSAC vs. RANSAC) ---")
    tracker.start("homography_no_ransac")
    H_no_ransac, mask_no_ransac, src_pts, dst_pts = estimate_homography(
        kp1, kp2, matches, use_ransac=False
    )
    t_h_no_ransac = tracker.stop("homography_no_ransac")
    _, no_ransac_rmse, no_ransac_mae = compute_reprojection_errors(H_no_ransac, src_pts, dst_pts)

    tracker.start("homography_ransac")
    H_ransac, inlier_mask, _, _ = estimate_homography(
        kp1, kp2, matches, use_ransac=True, reproj_threshold=5.0
    )
    t_h_ransac = tracker.stop("homography_ransac")
    inlier_src = src_pts[inlier_mask]
    inlier_dst = dst_pts[inlier_mask]
    _, ransac_rmse, ransac_mae = compute_reprojection_errors(H_ransac, inlier_src, inlier_dst)

    inlier_count = int(inlier_mask.sum())
    inlier_ratio = inlier_count / len(matches)

    draw_inlier_matches(
        image1, kp1, image2, kp2, matches, inlier_mask,
        output_dir / "ransac_inlier_matches.jpg"
    )

    plot_feature_matches_and_inliers(
        image1, kp1, image2, kp2, matches, inlier_mask,
        output_dir / "feature_matches_and_inliers.png"
    )

    # Warp both homographies to canvas
    canvas1_nr, warped2_nr, mask1_nr, mask2_nr = warp_pair_to_canvas(image1, image2, H_no_ransac)
    no_ransac_pano = blend_no_blend(canvas1_nr, warped2_nr, mask1_nr, mask2_nr)
    cv2.imwrite(str(output_dir / "no_ransac_panorama.jpg"), no_ransac_pano)

    tracker.start("warping_to_canvas")
    canvas1, warped2, mask1, mask2 = warp_pair_to_canvas(image1, image2, H_ransac)
    t_warp = tracker.stop("warping_to_canvas")

    ransac_pano_no_blend = blend_no_blend(canvas1, warped2, mask1, mask2)
    cv2.imwrite(str(output_dir / "ransac_panorama.jpg"), ransac_pano_no_blend)

    # Photometric error for homography comparison
    metrics_nr = compute_photometric_metrics(canvas1_nr, warped2_nr, mask1_nr, mask2_nr)
    metrics_r_raw = compute_photometric_metrics(canvas1, warped2, mask1, mask2)

    homography_summary = {
        "no_ransac": {
            "matches": len(matches),
            "inliers": len(matches),
            "inlier_ratio": 1.0,
            "reproj_rmse": no_ransac_rmse,
            "reproj_mae": no_ransac_mae,
            "seam_error": metrics_nr["mae"],
            "time_ms": t_h_no_ransac,
        },
        "ransac": {
            "matches": len(matches),
            "inliers": inlier_count,
            "inlier_ratio": inlier_ratio,
            "reproj_rmse": ransac_rmse,
            "reproj_mae": ransac_mae,
            "seam_error": metrics_r_raw["mae"],
            "time_ms": t_h_ransac,
        },
    }

    plot_homography_ablation(
        no_ransac_pano, ransac_pano_no_blend,
        homography_summary["no_ransac"], homography_summary["ransac"],
        output_dir / "homography_ablation.png"
    )

    # -------------------------------------------------------------
    # Stage 4: Exposure Compensation Ablation (Without vs. With)
    # -------------------------------------------------------------
    print("\n--- Running Exposure Compensation Ablation (Without vs. With) ---")
    tracker.start("exposure_compensation")
    warped2_comp, exp_meta = compensate_exposure(canvas1, warped2, mask1, mask2, method="gain")
    t_exp = tracker.stop("exposure_compensation")

    metrics_comp = compute_photometric_metrics(canvas1, warped2_comp, mask1, mask2)

    print(f"Exposure Gains applied [B, G, R]: {exp_meta.get('gains', [])}")
    print(f"Overlap MAE : Without = {metrics_r_raw['mae']:.2f}  -->  With = {metrics_comp['mae']:.2f}")
    print(f"Overlap RMSE: Without = {metrics_r_raw['rmse']:.2f} -->  With = {metrics_comp['rmse']:.2f}")
    print(f"Overlap PSNR: Without = {metrics_r_raw['psnr']:.2f}dB -->  With = {metrics_comp['psnr']:.2f}dB")
    print(f"Overlap SSIM: Without = {metrics_r_raw['ssim']:.4f} -->  With = {metrics_comp['ssim']:.4f}")

    plot_exposure_ablation(
        canvas1, warped2, warped2_comp, mask1, mask2,
        metrics_r_raw, metrics_comp,
        output_dir / "exposure_compensation_comparison.png"
    )

    # -------------------------------------------------------------
    # Stage 5: Blending Ablation (No Blend vs Basic 50/50 vs Feathering vs Adaptive)
    # -------------------------------------------------------------
    print("\n--- Running Blending Ablation (No Blend vs. Basic vs. Adaptive Seam) ---")

    # (A) No blending
    pano_no_blend = blend_no_blend(canvas1, warped2_comp, mask1, mask2)
    cv2.imwrite(str(output_dir / "ransac_no_blend.jpg"), pano_no_blend)
    disc_no_blend = compute_seam_discontinuity(pano_no_blend, mask1, mask2)

    # (B) Basic 50/50 blend
    pano_basic_alpha = blend_basic_alpha(canvas1, warped2_comp, mask1, mask2, alpha=0.5)
    cv2.imwrite(str(output_dir / "ransac_basic_blend.jpg"), pano_basic_alpha)
    disc_basic_alpha = compute_seam_discontinuity(pano_basic_alpha, mask1, mask2)

    # (C) Distance-transform feathering
    pano_feather = blend_distance_feather(canvas1, warped2_comp, mask1, mask2)
    cv2.imwrite(str(output_dir / "ransac_distance_feather.jpg"), pano_feather)
    disc_feather = compute_seam_discontinuity(pano_feather, mask1, mask2)

    # (D) Adaptive seam blending
    tracker.start("adaptive_seam_blending")
    pano_adaptive, seam_details = blend_adaptive_seam(
        canvas1, warped2_comp, mask1, mask2,
        feather_radius=11, return_details=True
    )
    t_blend = tracker.stop("adaptive_seam_blending")
    cv2.imwrite(str(output_dir / "ransac_adaptive_seam.jpg"), pano_adaptive)
    disc_adaptive = compute_seam_discontinuity(pano_adaptive, mask1, mask2)

    # Save seam visualization
    cv2.imwrite(str(output_dir / "adaptive_seam_overlay.jpg"), seam_details["seam_overlay"])
    plot_seam_analysis(
        seam_details["energy_map"],
        seam_details["seam_coords"],
        seam_details["seam_overlay"],
        output_dir / "seam_energy_and_path.png"
    )

    # Comparative blending panel
    blending_metrics = {
        "seam_discontinuity": {
            "no_blend": disc_no_blend,
            "basic_alpha": disc_basic_alpha,
            "feather": disc_feather,
            "adaptive": disc_adaptive,
        }
    }
    plot_blending_ablation(
        pano_no_blend, pano_basic_alpha, pano_feather, pano_adaptive,
        blending_metrics, output_dir / "blending_ablation.png"
    )

    # -------------------------------------------------------------
    # Stage 6: Final Recommended Panorama (StitchGuard Output)
    # -------------------------------------------------------------
    cv2.imwrite(str(output_dir / "stitchguard_final_panorama.jpg"), pano_adaptive)

    # -------------------------------------------------------------
    # Stage 7: Performance Matrix & Results Tabulation
    # -------------------------------------------------------------
    timing_summary = tracker.get_summary()

    # Create master ablation DataFrame
    rows = [
        {
            "Experiment": "Homography",
            "Method": "No RANSAC",
            "Matches / Inliers": f"{len(matches)} / {len(matches)}",
            "Inlier Ratio": 1.0,
            "Reproj RMSE (px)": no_ransac_rmse,
            "Reproj MAE (px)": no_ransac_mae,
            "Overlap MAE": metrics_nr["mae"],
            "Overlap RMSE": metrics_nr["rmse"],
            "PSNR (dB)": metrics_nr["psnr"],
            "SSIM": metrics_nr["ssim"],
            "Seam Discontinuity": compute_seam_discontinuity(no_ransac_pano, mask1_nr, mask2_nr),
        },
        {
            "Experiment": "Homography",
            "Method": "RANSAC",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_r_raw["mae"],
            "Overlap RMSE": metrics_r_raw["rmse"],
            "PSNR (dB)": metrics_r_raw["psnr"],
            "SSIM": metrics_r_raw["ssim"],
            "Seam Discontinuity": disc_no_blend,
        },
        {
            "Experiment": "Exposure",
            "Method": "No Comp (RANSAC)",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_r_raw["mae"],
            "Overlap RMSE": metrics_r_raw["rmse"],
            "PSNR (dB)": metrics_r_raw["psnr"],
            "SSIM": metrics_r_raw["ssim"],
            "Seam Discontinuity": disc_no_blend,
        },
        {
            "Experiment": "Exposure",
            "Method": "With Exp Comp",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_comp["mae"],
            "Overlap RMSE": metrics_comp["rmse"],
            "PSNR (dB)": metrics_comp["psnr"],
            "SSIM": metrics_comp["ssim"],
            "Seam Discontinuity": disc_no_blend,
        },
        {
            "Experiment": "Blending",
            "Method": "No Blend",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_comp["mae"],
            "Overlap RMSE": metrics_comp["rmse"],
            "PSNR (dB)": metrics_comp["psnr"],
            "SSIM": metrics_comp["ssim"],
            "Seam Discontinuity": disc_no_blend,
        },
        {
            "Experiment": "Blending",
            "Method": "Basic 50/50",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_comp["mae"],
            "Overlap RMSE": metrics_comp["rmse"],
            "PSNR (dB)": metrics_comp["psnr"],
            "SSIM": metrics_comp["ssim"],
            "Seam Discontinuity": disc_basic_alpha,
        },
        {
            "Experiment": "Blending",
            "Method": "Distance Feather",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_comp["mae"],
            "Overlap RMSE": metrics_comp["rmse"],
            "PSNR (dB)": metrics_comp["psnr"],
            "SSIM": metrics_comp["ssim"],
            "Seam Discontinuity": disc_feather,
        },
        {
            "Experiment": "Blending",
            "Method": "Adaptive Seam",
            "Matches / Inliers": f"{len(matches)} / {inlier_count}",
            "Inlier Ratio": inlier_ratio,
            "Reproj RMSE (px)": ransac_rmse,
            "Reproj MAE (px)": ransac_mae,
            "Overlap MAE": metrics_comp["mae"],
            "Overlap RMSE": metrics_comp["rmse"],
            "PSNR (dB)": metrics_comp["psnr"],
            "SSIM": metrics_comp["ssim"],
            "Seam Discontinuity": disc_adaptive,
        },
    ]

    df_results = pd.DataFrame(rows)

    # Save to CSV
    csv_path = output_dir / "ablation_study.csv"
    df_results.to_csv(str(csv_path), index=False)

    # Save timing matrix
    df_timing = pd.DataFrame([
        {"Stage": k.replace("_", " ").title(), "Latency (ms)": v}
        for k, v in timing_summary.items()
    ])
    df_timing.to_csv(str(output_dir / "performance_matrix.csv"), index=False)

    # Generate Performance Matrix Charts & Executive Dashboard
    plot_performance_matrix(df_results, timing_summary, output_dir / "performance_matrix.png")
    plot_comprehensive_dashboard(image1, image2, pano_adaptive, df_results, output_dir / "stitchguard_dashboard.png")

    # -------------------------------------------------------------
    # Print Beautiful Formatted Tables
    # -------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"            STITCHGUARD RESULTS & PERFORMANCE MATRIX: {dataset_name.upper()}")
    print("=" * 90)

    # Table 1: Homography Comparison
    df_h = df_results[df_results["Experiment"] == "Homography"][
        ["Method", "Matches / Inliers", "Inlier Ratio", "Reproj RMSE (px)", "Reproj MAE (px)", "Overlap MAE"]
    ]
    print("\n[TABLE 1] HOMOGRAPHY ESTIMATION: WITHOUT RANSAC vs. WITH RANSAC")
    print(tabulate(df_h, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4f"))

    # Table 2: Exposure Compensation Comparison
    df_e = df_results[df_results["Experiment"] == "Exposure"][
        ["Method", "Overlap MAE", "Overlap RMSE", "PSNR (dB)", "SSIM"]
    ]
    print("\n[TABLE 2] EXPOSURE COMPENSATION: WITHOUT vs. WITH")
    print(tabulate(df_e, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4f"))

    # Table 3: Blending Ablation Comparison
    df_b = df_results[df_results["Experiment"] == "Blending"][
        ["Method", "Seam Discontinuity", "Overlap MAE", "PSNR (dB)", "SSIM"]
    ]
    print("\n[TABLE 3] BLENDING ABLATION: NO BLEND vs. BASIC vs. ADAPTIVE SEAM")
    print(tabulate(df_b, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".4f"))

    # Table 4: Latency Performance Matrix
    print("\n[TABLE 4] PERFORMANCE MATRIX: EXECUTION LATENCY BREAKDOWN")
    print(tabulate(df_timing, headers="keys", tablefmt="fancy_grid", showindex=False, floatfmt=".2f"))

    print("\n" + "=" * 90)
    print(f"All results, CSVs, and visualization figures saved in: {output_dir.resolve()}")
    print("=" * 90 + "\n")

    return df_results, timing_summary


def main():
    parser = argparse.ArgumentParser(description="StitchGuard: Uncertainty-Aware Panorama Stitching")
    parser.add_argument(
        "--dataset", type=str, default="set1",
        help="Dataset name or path (e.g., 'set1', 'set2', 'set3', or full folder path)"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run evaluation on all available datasets (set1, set2, set3) and compare"
    )
    parser.add_argument(
        "--results_dir", type=str, default="results",
        help="Base directory to save output results"
    )

    args = parser.parse_args()
    base_results = Path(args.results_dir)

    if args.all:
        datasets = [
            ("data/set1", "Set 1: Normal Static Scene", base_results / "set1"),
            ("data/set2", "Set 2: Exposure Variation", base_results / "set2"),
            ("data/set3", "Set 3: Dynamic Object / Parallax", base_results / "set3"),
        ]

        all_summaries = []
        for dpath, dname, outdir in datasets:
            if Path(dpath).exists():
                df_res, _ = run_stitching_pipeline(dpath, outdir, dataset_name=dname)
                df_res["Dataset"] = dname
                all_summaries.append(df_res)

        if all_summaries:
            master_df = pd.concat(all_summaries, ignore_index=True)
            master_csv = base_results / "cross_dataset_ablation_summary.csv"
            master_df.to_csv(str(master_csv), index=False)
            print(f"\n[SUMMARY] Cross-dataset benchmark exported to: {master_csv.resolve()}")
    else:
        # Single dataset run
        dataset_name = args.dataset
        if (Path("data") / dataset_name).exists():
            data_path = Path("data") / dataset_name
            out_path = base_results if dataset_name == "set1" else base_results / dataset_name
        else:
            data_path = Path(dataset_name)
            out_path = base_results

        run_stitching_pipeline(data_path, out_path, dataset_name=f"Dataset ({data_path.name})")


if __name__ == "__main__":
    main()

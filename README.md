# StitchGuard — Uncertainty-Aware Panorama Stitching

Comprehensive panorama-stitching pipeline with RANSAC-based homography estimation, exposure compensation, adaptive seam blending, and quantitative performance benchmarking.

IISER Bhopal — Computer Vision (October 2026)  
Project Team: Noor Kamran (24233), Shaurya Awasthi (24308), Pragya Priya (24249)

---

## Complete Pipeline Architecture

```text
Input Images (Image 1 + Image 2 ... Image N)
   │
   ├─► SIFT Keypoint Detection & Feature Extraction
   │
   ├─► BFMatcher (L2 Norm) + Lowe's Ratio Test (tau = 0.75)
   │
   ├─► Homography Estimation & Ablation:
   │     ├─ Without RANSAC (Direct Least Squares)
   │     └─ With RANSAC (Reprojection Threshold = 5.0 px)
   │
   ├─► Common Canvas Calculation & Coordinate Translation
   │
   ├─► Exposure Compensation & Ablation:
   │     ├─ Without Compensation (Raw intensities)
   │     └─ With Exposure Compensation (Overlap Gain Matching & Block Compensation)
   │
   ├─► Blending Strategies & Ablation:
   │     ├─ (a) No Blending (Hard overlay baseline)
   │     ├─ (b) Basic 50/50 Alpha Blending
   │     ├─ (c) Distance-Transform Linear Feathering
   │     └─ (d) Adaptive Seam Blending (Optimal DP Energy Cut + Narrow-Band Feathering)
   │
   └─► Quantitative Evaluation & Performance Dashboard:
         ├─ RANSAC Inlier Ratio (%)
         ├─ Reprojection RMSE & MAE (pixels)
         ├─ Overlap Photometric MAE, RMSE, PSNR (dB), SSIM
         ├─ Seam Discontinuity Metric
         ├─ Pipeline Latency Matrix (ms)
         └─ Automated Visual Figure Generation & CSV Tabulation
```

---

## Directory Structure

```text
StitchGuard/
├── data/
│   ├── set1/                   # Set 1: Static / Normal scene
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   ├── set2/                   # Set 2: Exposure variation challenge
│   │   ├── img1.jpg
│   │   └── img2.jpg
│   └── set3/                   # Set 3: Parallax / dynamic object / texture
│       ├── img1.jpg
│       └── img2.jpg
├── results/                    # Generated outputs, figures, and CSV reports
│   ├── set1/                   # Benchmark figures for Set 1
│   ├── set2/                   # Benchmark figures for Set 2
│   ├── set3/                   # Benchmark figures for Set 3
│   ├── ablation_study.csv
│   ├── performance_matrix.csv
│   ├── cross_dataset_ablation_summary.csv
│   ├── feature_matches_and_inliers.png
│   ├── homography_ablation.png
│   ├── exposure_compensation_comparison.png
│   ├── blending_ablation.png
│   ├── seam_energy_and_path.png
│   ├── performance_matrix.png
│   ├── stitchguard_dashboard.png
│   └── stitchguard_final_panorama.jpg
├── src/
│   ├── features.py             # SIFT detector
│   ├── matching.py             # BFMatcher + Lowe's ratio test
│   ├── homography.py           # Homography (RANSAC & No-RANSAC) + Reprojection RMSE/MAE
│   ├── warping.py              # Common canvas calculation & perspective warp
│   ├── exposure.py             # Global & Block-based gain exposure compensation
│   ├── blending.py             # No-blend, 50/50, distance feather, adaptive seam
│   ├── metrics.py              # Geometric, photometric (MAE, RMSE, PSNR, SSIM) & seam metrics
│   ├── visualization.py        # Publication-ready figure plotting & charts
│   ├── multi_image.py          # Sequential multi-image stitching engine (3-6 images)
│   └── utils.py                # Helper utilities and backwards-compatible wrappers
├── main.py                     # Master execution and ablation CLI script
├── requirements.txt            # Python dependencies
└── summary.md                  # Comprehensive implementation and results report
```

---

## Setup & Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:
- `opencv-python`
- `numpy`
- `matplotlib`
- `pandas`
- `tabulate`

---

## Usage

### 1. Run standard evaluation on Set 1 (Normal Scene)
```bash
python main.py
```
Or specify dataset:
```bash
python main.py --dataset set1
```

### 2. Run on Exposure Variation Challenge (Set 2)
```bash
python main.py --dataset set2
```

### 3. Run on Dynamic Object / Parallax Challenge (Set 3)
```bash
python main.py --dataset set3
```

### 4. Run full cross-dataset benchmark across all sets
```bash
python main.py --all
```
This runs Set 1, Set 2, and Set 3, tabulates all metrics in the terminal, and exports `cross_dataset_ablation_summary.csv`.

---

## Generated Visualizations & Reports

All artifacts are automatically generated in `results/`:

| Output File | Description |
| :--- | :--- |
| `feature_matches_and_inliers.png` | Side-by-side comparison of SIFT matches vs RANSAC inliers with inlier ratio. |
| `homography_ablation.png` | Visual comparison of panorama using No-RANSAC vs RANSAC alignment. |
| `exposure_compensation_comparison.png` | Before/after exposure compensation with overlap error heatmaps. |
| `blending_ablation.png` | 4-panel comparison: (a) No Blend, (b) Basic 50/50, (c) Feathering, (d) Adaptive Seam. |
| `seam_energy_and_path.png` | Overlap cost energy map and optimal seam path cut line overlay. |
| `performance_matrix.png` | Multi-panel performance matrix charts (seam discontinuity, overlap error, RMSE, latencies). |
| `stitchguard_dashboard.png` | Publication-ready executive summary figure of the pipeline. |
| `ablation_study.csv` | Full numerical metrics across all tested configurations. |
| `performance_matrix.csv` | Millisecond execution time breakdown per pipeline stage. |

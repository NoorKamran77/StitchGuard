# StitchGuard — Initial Implementation

Initial pairwise panorama-stitching implementation for the Computer Vision project.

## Current pipeline

Input image 1 + image 2
→ SIFT feature detection
→ BFMatcher + Lowe ratio test
→ Homography without RANSAC
→ Homography with RANSAC
→ Common canvas calculation
→ Perspective warping
→ Pairwise panorama output
→ Match/inlier visualization
→ Basic quantitative statistics

## Folder structure

```text
StitchGuard_initial/
├── data/
│   └── set1/
│       ├── img1.jpg
│       └── img2.jpg
├── results/
├── src/
│   ├── features.py
│   ├── matching.py
│   ├── homography.py
│   ├── warping.py
│   └── utils.py
├── main.py
└── requirements.txt
```

## Setup

```bash
python -m pip install -r requirements.txt
```

Put two overlapping images in:

```text
data/set1/img1.jpg
data/set1/img2.jpg
```

Then run:

```bash
python main.py
```

Results are written to `results/`.

## Important

This is intentionally the INITIAL implementation. It does not yet include:
- multi-image stitching for 3–6 images
- exposure compensation
- adaptive seam blending
- final seam metrics
- full experimental CSV generation

Those are the next stages.


## Current outputs

After `python main.py`, you should get:

- `good_matches.jpg` — accepted SIFT matches after the ratio test.
- `ransac_inlier_matches.jpg` — matches classified as RANSAC inliers.
- `no_ransac_panorama.jpg` — hard/no-blending composition using the non-RANSAC homography.
- `ransac_panorama.jpg` — hard/no-blending composition using the RANSAC homography.
- `ransac_basic_blend.jpg` — simple 50/50 blend using the RANSAC alignment.

The printed seam-error value is an initial overlap-intensity diagnostic. It is not yet the final seam metric for the completed project.

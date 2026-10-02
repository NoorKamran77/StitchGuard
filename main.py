from pathlib import Path
import cv2

from src.features import detect_sift
from src.matching import ratio_test_match
from src.homography import estimate_homography
from src.warping import warp_pair_to_canvas
from src.utils import (
    draw_good_matches,
    draw_inlier_matches,
    compose_no_blend,
    compose_basic_blend,
    compute_overlap_seam_error,
)


DATA_DIR = Path("data/set1")
RESULTS_DIR = Path("results")
IMG1_PATH = DATA_DIR / "img1.jpg"
IMG2_PATH = DATA_DIR / "img2.jpg"


def run_experiment(image1, image2, kp1, kp2, matches, use_ransac, name):
    H, inlier_mask, _, _ = estimate_homography(
        kp1,
        kp2,
        matches,
        use_ransac=use_ransac,
        reproj_threshold=5.0
    )

    canvas1, warped2, mask1, mask2 = warp_pair_to_canvas(
        image1, image2, H
    )

    # True no-blending composition for the RANSAC ablation.
    panorama = compose_no_blend(
        canvas1, warped2, mask1, mask2
    )

    cv2.imwrite(
        str(RESULTS_DIR / f"{name}_panorama.jpg"),
        panorama
    )

    cv2.imwrite(
        str(RESULTS_DIR / f"{name}_warped.jpg"),
        warped2
    )

    # Save a simple blending result from the RANSAC alignment.
    if use_ransac:
        basic_blend = compose_basic_blend(
            canvas1, warped2, mask1, mask2
        )
        cv2.imwrite(
            str(RESULTS_DIR / "ransac_basic_blend.jpg"),
            basic_blend
        )

        draw_inlier_matches(
            image1, kp1, image2, kp2,
            matches, inlier_mask,
            RESULTS_DIR / "ransac_inlier_matches.jpg"
        )

    seam_error = compute_overlap_seam_error(
        canvas1, warped2, mask1, mask2
    )

    inliers = int(inlier_mask.sum()) if use_ransac else len(matches)
    inlier_ratio = inliers / len(matches)

    print(f"\n--- {name} ---")
    print("Total good matches :", len(matches))
    print("Inliers             :", inliers)
    print(f"Inlier ratio        : {inlier_ratio:.4f}")
    print(f"Overlap seam error  : {seam_error:.4f}")
    print("Homography:")
    print(H)

    return {
        "name": name,
        "matches": len(matches),
        "inliers": inliers,
        "inlier_ratio": inlier_ratio,
        "seam_error": seam_error,
        "H": H,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    image1 = cv2.imread(str(IMG1_PATH))
    image2 = cv2.imread(str(IMG2_PATH))

    if image1 is None or image2 is None:
        raise FileNotFoundError(
            "Put img1.jpg and img2.jpg inside data/set1/"
        )

    kp1, des1 = detect_sift(image1)
    kp2, des2 = detect_sift(image2)

    print("Image 1 keypoints:", len(kp1))
    print("Image 2 keypoints:", len(kp2))

    matches = ratio_test_match(des1, des2, ratio=0.75)

    print("Good matches after ratio test:", len(matches))

    if len(matches) < 4:
        raise RuntimeError(
            "Too few good matches. Try images with more overlap/texture."
        )

    draw_good_matches(
        image1, kp1, image2, kp2, matches,
        RESULTS_DIR / "good_matches.jpg"
    )

    no_ransac = run_experiment(
        image1, image2, kp1, kp2, matches,
        use_ransac=False,
        name="no_ransac"
    )

    ransac = run_experiment(
        image1, image2, kp1, kp2, matches,
        use_ransac=True,
        name="ransac"
    )

    print("\n=== INITIAL COMPARISON ===")
    print(
        f"No RANSAC: matches={no_ransac['matches']}, "
        f"seam_error={no_ransac['seam_error']:.4f}"
    )
    print(
        f"RANSAC   : inliers={ransac['inliers']}, "
        f"inlier_ratio={ransac['inlier_ratio']:.4f}, "
        f"seam_error={ransac['seam_error']:.4f}"
    )

    print("\nResults saved in:", RESULTS_DIR.resolve())


if __name__ == "__main__":
    main()

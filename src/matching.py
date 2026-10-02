import cv2


def ratio_test_match(des1, des2, ratio=0.75):
    """BFMatcher KNN matching followed by Lowe's ratio test."""
    if des1 is None or des2 is None:
        raise ValueError("SIFT descriptors could not be computed.")

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < ratio * n.distance:
            good_matches.append(m)

    return good_matches

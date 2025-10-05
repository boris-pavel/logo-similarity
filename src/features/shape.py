"""Shape-driven feature extraction helpers."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_MAX_FEATURES = 500
_LOWE_RATIO = 0.75


def orb_keypoints_match(img1: Image.Image, img2: Image.Image) -> float:
    """Return an ORB-based similarity score between two images in [0, 1]."""
    if not isinstance(img1, Image.Image) or not isinstance(img2, Image.Image):
        raise TypeError("orb_keypoints_match expects PIL.Image.Image inputs")

    gray1 = np.asarray(img1.convert("L"))
    gray2 = np.asarray(img2.convert("L"))

    if gray1.size == 0 or gray2.size == 0:
        return 0.0

    orb = cv2.ORB_create(nfeatures=_MAX_FEATURES)

    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)

    if not keypoints1 or not keypoints2 or descriptors1 is None or descriptors2 is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    try:
        raw_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
    except cv2.error:
        return 0.0

    good_matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        best, contender = pair
        if contender.distance == 0:  # pragma: no cover - guard division by zero
            continue
        if best.distance < _LOWE_RATIO * contender.distance:
            good_matches.append(best)

    denominator = max(1, min(len(keypoints1), len(keypoints2)))
    score = len(good_matches) / denominator
    return float(max(0.0, min(1.0, score)))

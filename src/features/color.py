"""Color histogram utilities."""

from __future__ import annotations


import cv2
import numpy as np
from PIL import Image

_DEFAULT_BINS = (12, 6, 6)
_HUE_BINS = 36


def hsv_histogram(
    img: Image.Image,
    bins: tuple[int, int, int] = _DEFAULT_BINS,
) -> list[float]:
    """Return a flattened HSV histogram normalised to sum to 1."""
    if len(bins) != 3:
        raise ValueError("bins must contain three integers for H, S, and V")

    rgb_image = img.convert("RGB") if img.mode != "RGB" else img
    np_pixels = np.asarray(rgb_image)
    if np_pixels.size == 0:
        return [0.0] * (bins[0] * bins[1] * bins[2])

    hsv = cv2.cvtColor(np_pixels, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        bins,
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist.flatten().astype(float).tolist()


def dominant_hues(img: Image.Image, k: int = 3) -> list[int]:
    """Return the *k* most dominant hue bin centres (degrees)."""
    if k <= 0:
        return []

    rgb_image = img.convert("RGB") if img.mode != "RGB" else img
    np_pixels = np.asarray(rgb_image)
    if np_pixels.size == 0:
        return []

    hsv = cv2.cvtColor(np_pixels, cv2.COLOR_RGB2HSV)
    hue_channel = hsv[:, :, 0]
    hist = cv2.calcHist(
        [hue_channel],
        [0],
        None,
        [_HUE_BINS],
        [0, 180],
    ).flatten()

    if not np.any(hist):
        return []

    top_indices = np.argsort(hist)[::-1][:k]
    bin_width = 180 / _HUE_BINS
    centres = [int(round((idx + 0.5) * bin_width)) for idx in top_indices]
    return centres


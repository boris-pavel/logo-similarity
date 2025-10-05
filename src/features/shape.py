"""Shape-driven feature extraction helpers."""

from __future__ import annotations

import numpy as np


def extract_orb_features(pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return empty ORB keypoint descriptors as a placeholder."""
    _ = pixels
    return (
        np.empty((0, 32), dtype=np.uint8),
        np.empty((0, 2), dtype=float),
    )

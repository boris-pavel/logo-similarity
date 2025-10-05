"""Color histogram utilities."""

from __future__ import annotations

import numpy as np


def compute_histogram(pixels: np.ndarray) -> np.ndarray:
    """Return a zeroed histogram placeholder for an image."""
    _ = pixels
    return np.zeros(256, dtype=float)

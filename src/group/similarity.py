"""Similarity scoring between logo representations."""

from __future__ import annotations

from typing import Mapping


def compute_similarity(features_a: Mapping[str, float], features_b: Mapping[str, float]) -> float:
    """Return a basic similarity score derived from shared feature keys."""
    if not features_a or not features_b:
        return 0.0
    shared = set(features_a).intersection(features_b)
    if not shared:
        return 0.0
    return min(1.0, len(shared) / max(len(features_a), len(features_b)))

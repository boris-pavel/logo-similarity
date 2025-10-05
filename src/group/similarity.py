"""Similarity scoring between logo representations."""

from __future__ import annotations

from typing import Dict, Iterator, Mapping, Sequence, Tuple

from PIL import Image

from ..features.perceptual import hamming_distance_hex
from ..features.shape import orb_keypoints_match
from ..io.models import LogoFeatures


WEIGHTS: Dict[str, float] = {
    "phash": 0.35,
    "dhash": 0.25,
    "ahash": 0.15,
    "hist": 0.25,
}

T_LINK: float = 0.72
T_CONFIRM: float = 0.86


def histogram_intersection(a: Sequence[float], b: Sequence[float]) -> float:
    """Return histogram intersection between *a* and *b* in the unit interval."""
    if not a or not b:
        return 0.0

    total = 0.0
    for aval, bval in zip(a, b):
        total += min(float(aval), float(bval))

    return float(max(0.0, min(1.0, total)))


def base_similarity(fA: LogoFeatures, fB: LogoFeatures) -> Dict[str, float]:
    """Return per-channel similarity components between *fA* and *fB*."""
    hashes_a = fA.perceptual or {}
    hashes_b = fB.perceptual or {}

    return {
        "ahash": _hash_similarity(hashes_a.get("ahash"), hashes_b.get("ahash")),
        "dhash": _hash_similarity(hashes_a.get("dhash"), hashes_b.get("dhash")),
        "phash": _hash_similarity(hashes_a.get("phash"), hashes_b.get("phash")),
        "hist": histogram_intersection(fA.hsv_histogram, fB.hsv_histogram),
    }


def combined_similarity(
    fA: LogoFeatures, fB: LogoFeatures, orb_score: float | None = None
) -> float:
    """Return the weighted similarity between two logo feature sets."""
    components = base_similarity(fA, fB)
    return combine_components(components, orb_score)


def shortlist_by_hash(
    fA: LogoFeatures,
    f_all: Mapping[str, LogoFeatures],
    max_candidates: int = 50,
    max_hamming: int = 16,
) -> list[str]:
    """Return candidate website keys close to *fA* by perceptual hash."""
    if max_candidates <= 0:
        return []

    anchor_hash = (fA.perceptual or {}).get("phash")
    if not anchor_hash:
        return []

    ranked: list[Tuple[int, str]] = []
    for website, feature in f_all.items():
        if feature is fA or feature.website == fA.website:
            continue
        other_hash = (feature.perceptual or {}).get("phash")
        if not other_hash:
            continue
        try:
            distance = hamming_distance_hex(anchor_hash, other_hash)
        except (TypeError, ValueError):
            continue
        if distance <= max_hamming:
            ranked.append((distance, website))

    ranked.sort(key=lambda item: item[0])
    return [website for _, website in ranked[:max_candidates]]


def pairwise_scores(
    features_map: Mapping[str, LogoFeatures],
    images_map: Mapping[str, Image.Image],
    max_candidates: int = 50,
    max_hamming: int = 16,
    t_link: float = T_LINK,
) -> Iterator[Tuple[str, str, float]]:
    """Yield pairwise similarity edges between websites exceeding *t_link*."""
    if not features_map:
        return

    websites = sorted(features_map)
    for website in websites:
        feature = features_map[website]
        shortlist = shortlist_by_hash(
            feature, features_map, max_candidates=max_candidates, max_hamming=max_hamming
        )
        for candidate in shortlist:
            if candidate not in features_map:
                continue
            if candidate <= website:
                continue

            other_feature = features_map[candidate]
            components = base_similarity(feature, other_feature)
            score = combine_components(components)
            orb_score = None
            if t_link - 0.05 <= score <= t_link + 0.1:
                img_a = images_map.get(website)
                img_b = images_map.get(candidate)
                if img_a is not None and img_b is not None:
                    orb_score = orb_keypoints_match(img_a, img_b)
                    score = combine_components(components, orb_score)

            if score >= t_link:
                yield (website, candidate, float(score))


def _hash_similarity(hash_a: str | None, hash_b: str | None) -> float:
    if not hash_a or not hash_b:
        return 0.0
    try:
        distance = hamming_distance_hex(hash_a, hash_b)
    except (TypeError, ValueError):
        return 0.0
    score = 1.0 - (distance / 64.0)
    return float(max(0.0, min(1.0, score)))


def combine_components(components: Mapping[str, float], orb_score: float | None = None) -> float:
    score = 0.0
    for key, weight in WEIGHTS.items():
        score += weight * float(components.get(key, 0.0))

    if orb_score is not None:
        score = (0.8 * score) + (0.2 * max(0.0, min(1.0, float(orb_score))))

    return float(max(0.0, min(1.0, score)))


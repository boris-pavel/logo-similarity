"""Perceptual feature computations."""

from __future__ import annotations


import imagehash
from PIL import Image

_HASH_SIZE = 8


def compute_hashes(img: Image.Image) -> dict[str, str]:
    """Return the perceptual hash variants for *img* as hex strings."""
    if not isinstance(img, Image.Image):
        raise TypeError("compute_hashes expects a PIL.Image.Image instance")

    work_img = img.convert("RGB") if img.mode not in {"RGB", "L"} else img

    return {
        "ahash": str(imagehash.average_hash(work_img, hash_size=_HASH_SIZE)),
        "phash": str(imagehash.phash(work_img, hash_size=_HASH_SIZE)),
        "dhash": str(imagehash.dhash(work_img, hash_size=_HASH_SIZE)),
    }


def hamming_distance_hex(h1: str, h2: str) -> int:
    """Return the Hamming distance between two hexadecimal hash strings."""
    normalized_1 = _normalise_hex(h1)
    normalized_2 = _normalise_hex(h2)

    max_len = max(len(normalized_1), len(normalized_2))
    if max_len == 0:
        return 0

    normalized_1 = normalized_1.zfill(max_len)
    normalized_2 = normalized_2.zfill(max_len)

    try:
        value_1 = int(normalized_1, 16)
        value_2 = int(normalized_2, 16)
    except ValueError as exc:  # pragma: no cover - defensive path
        raise ValueError("Inputs must be hexadecimal strings") from exc

    xor = value_1 ^ value_2
    return xor.bit_count()


def _normalise_hex(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Hash values must be provided as strings")
    stripped = value.strip().lower()
    return stripped[2:] if stripped.startswith("0x") else stripped




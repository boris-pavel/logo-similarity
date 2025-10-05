"""Perceptual feature computations."""

from __future__ import annotations


def compute_phash_signature(image_bytes: bytes) -> str:
    """Return a deterministic placeholder perceptual hash."""
    if not image_bytes:
        return "0" * 16
    checksum = sum(image_bytes) % 16
    return f"{checksum:01x}" * 16

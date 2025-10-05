"""Discover candidate logo asset URLs within fetched documents."""

from __future__ import annotations

from typing import Iterable


def discover_logo_candidates(html: str) -> list[str]:
    """Return placeholder candidate asset URLs extracted from *html*."""
    _ = html
    return []


def merge_candidate_sources(*sources: Iterable[str]) -> list[str]:
    """Merge candidate sources while preserving order and uniqueness."""
    seen: set[str] = set()
    merged: list[str] = []
    for source in sources:
        for item in source:
            if item and item not in seen:
                seen.add(item)
                merged.append(item)
    return merged

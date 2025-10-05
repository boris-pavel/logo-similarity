"""Select the most plausible logo from discovered candidates."""

from __future__ import annotations

from typing import Sequence


def select_best_logo(candidates: Sequence[str]) -> str | None:
    """Return the first truthy candidate as a stand-in best logo."""
    for candidate in candidates:
        if candidate:
            return candidate
    return None

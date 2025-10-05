"""Normalize logo imagery into a consistent format."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from PIL import Image


def normalize_image(path: Path, size: int = 256) -> Path:
    """Return *path* untouched as a placeholder for real normalization."""
    _ = size
    return path


def to_image(image_path: Path) -> "Image.Image | None":
    """Placeholder converter that signals missing implementation."""
    _ = image_path
    return None

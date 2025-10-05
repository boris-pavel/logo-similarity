"""Data models shared across the logo similarity pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping


@dataclass(slots=True)
class LogoAsset:
    """Metadata for a discovered logo asset."""

    website: str
    asset_url: str | None
    local_path: Path | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class LogoFeatures:
    """Aggregated features computed for a website's preferred logo."""

    website: str
    original_path: Path | None = None
    normalized_path: Path | None = None
    preview_path: Path | None = None
    perceptual: Dict[str, str] = field(default_factory=dict)
    hsv_histogram: List[float] = field(default_factory=list)
    dominant_hues: List[int] = field(default_factory=list)


@dataclass(slots=True)
class LogoGroup:
    """Grouping of visually similar logos."""

    representative: str
    members: List[str]
    score: float | None = None


@dataclass(slots=True)
class PipelineReport:
    """High-level summary of the pipeline run."""

    total_websites: int
    processed: int
    grouped: int
    coverage: float
    notes: Dict[str, str] = field(default_factory=dict)

"""Output helpers for persisting pipeline results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from .models import LogoGroup, PipelineReport


def write_groups(path: Path, groups: Sequence[LogoGroup]) -> Path:
    """Write *groups* to *path* as JSON and return the path."""
    serialised = [asdict(group) for group in groups]
    path.write_text(json.dumps(serialised, indent=2), encoding="utf-8")
    return path


def write_report(path: Path, report: PipelineReport) -> Path:
    """Write a pipeline report to *path* as JSON and return the path."""
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path

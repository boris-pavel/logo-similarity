"""HTTP fetching utilities for the logo similarity pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FetchResult:
    """Container for fetched website content."""

    url: str
    html: str
    status: int | None = None


def fetch_html(url: str, timeout: float = 10.0) -> FetchResult:
    """Return placeholder fetch results for *url*.

    Real fetching will be implemented later; for now we return an empty payload
    so that downstream modules can be exercised without network access.
    """
    _ = timeout
    return FetchResult(url=url, html="", status=None)

"""Select the most plausible logo from discovered candidates."""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any, Mapping, Sequence
from urllib.parse import unquote_to_bytes
import xml.etree.ElementTree as ET

import requests
from PIL import Image, UnidentifiedImageError
from PIL.Image import DecompressionBombError

from ..crawl.fetch import normalize_url

logger = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
_IMAGE_TIMEOUT = 10.0
_MAX_FETCH = 6

CandidateLike = Mapping[str, Any]
ImageInfo = dict[str, Any]


def fetch_image_bytes(url: str, referer: str | None = None) -> bytes | None:
    """Download and return the raw bytes for an image URL."""
    if not url:
        return None
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "image/*,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer
    try:
        response = requests.get(url, headers=headers, timeout=_IMAGE_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        logger.debug("Failed to fetch image bytes from %s", url, exc_info=True)
        return None
    return response.content


def sniff_image_info(image_bytes: bytes) -> ImageInfo | None:
    """Inspect *image_bytes* and return basic metadata if recognised."""
    if not image_bytes:
        return None
    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            image.load()  # ensure image is fully parsed
            width, height = image.size
            has_alpha = _has_alpha_channel(image)
            mime = Image.MIME.get(image.format)
            aspect_ratio = _compute_aspect_ratio(width, height)
            return {
                "width": width,
                "height": height,
                "has_alpha": has_alpha,
                "mime": mime,
                "aspect_ratio": aspect_ratio,
            }
    except (UnidentifiedImageError, DecompressionBombError, OSError):
        svg_info = _sniff_svg_metadata(image_bytes)
        if svg_info:
            return svg_info
        logger.debug("Unable to sniff image metadata", exc_info=True)
    return None


def score_candidate(candidate: CandidateLike, image_info: ImageInfo | None) -> float:
    """Return a heuristic score for *candidate* using optional image metadata."""
    base = float(candidate.get("confidence", 0.0) or 0.0)
    if not image_info:
        return base

    score = base
    has_alpha = bool(image_info.get("has_alpha"))
    width = _coerce_numeric(image_info.get("width"))
    height = _coerce_numeric(image_info.get("height"))
    aspect_ratio = _coerce_numeric(image_info.get("aspect_ratio"))

    if has_alpha:
        score += 0.05

    if aspect_ratio is not None and 0.8 <= aspect_ratio <= 5.0:
        score += 0.05

    if width is not None and height is not None:
        if min(width, height) < 48:
            score -= 0.10
        if (aspect_ratio is not None and aspect_ratio > 6.0) or (
            width > 1024 and height > 1024 and not has_alpha
        ):
            score -= 0.15

    return score


def select_best(
    candidates: Sequence[CandidateLike],
    base_url: str,
    lazy: bool = False,
) -> dict[str, Any] | None:
    """Return the best candidate enriched with image bytes and metadata."""
    if not candidates:
        return None

    ordered = sorted(
        candidates,
        key=lambda item: float(item.get("confidence", 0.0) or 0.0),
        reverse=True,
    )

    if lazy:
        top = dict(ordered[0])
        resolved = _resolve_candidate_src(top.get("src"), base_url)
        if resolved:
            top.setdefault("resolved_src", resolved)
        return top

    best_with_bytes: dict[str, Any] | None = None
    best_with_bytes_score = float("-inf")
    fallback_best: dict[str, Any] | None = None
    fallback_score = float("-inf")

    for index, candidate in enumerate(ordered):
        candidate_copy: dict[str, Any] = dict(candidate)
        resolved_src = _resolve_candidate_src(candidate_copy.get("src"), base_url)
        if resolved_src:
            candidate_copy.setdefault("resolved_src", resolved_src)

        image_bytes: bytes | None = None
        image_info: ImageInfo | None = None

        if index < _MAX_FETCH:
            image_bytes = _load_candidate_bytes(resolved_src, base_url)
            if image_bytes:
                image_info = sniff_image_info(image_bytes)
                candidate_copy["image_bytes"] = image_bytes
                if image_info:
                    candidate_copy["image_info"] = image_info
        # score using available info (may be None)
        score = score_candidate(candidate_copy, image_info)
        candidate_copy["_score"] = score

        if score > fallback_score:
            fallback_best = candidate_copy
            fallback_score = score

        if image_bytes and score > best_with_bytes_score:
            best_with_bytes = candidate_copy
            best_with_bytes_score = score

    return best_with_bytes or fallback_best


def _load_candidate_bytes(src: str | None, referer: str | None) -> bytes | None:
    if not src:
        return None
    if src.startswith("data:"):
        return _decode_data_uri(src)
    return fetch_image_bytes(src, referer)


def _resolve_candidate_src(src: Any, base_url: str) -> str | None:
    if not isinstance(src, str) or not src.strip():
        return None
    value = src.strip()
    if value.startswith("data:"):
        return value
    try:
        return normalize_url(value, base_url)
    except Exception:  # noqa: BLE001 - failure to normalise should not abort
        logger.debug("Failed to normalise candidate src %s", value, exc_info=True)
        return value


def _decode_data_uri(uri: str) -> bytes | None:
    if "data:" not in uri:
        return None
    try:
        header, data = uri.split(",", 1)
    except ValueError:
        return None
    if ";base64" in header:
        try:
            return base64.b64decode(data, validate=True)
        except (base64.binascii.Error, ValueError):
            return None
    try:
        return unquote_to_bytes(data)
    except Exception:  # noqa: BLE001 - conservative fallback
        return None


def _sniff_svg_metadata(image_bytes: bytes) -> ImageInfo | None:
    head = image_bytes.lstrip()[:512].lower()
    if not head.startswith(b"<") or b"<svg" not in head:
        return None
    try:
        root = ET.fromstring(image_bytes.decode("utf-8", errors="ignore"))
    except ET.ParseError:
        return None
    if not root.tag.lower().endswith("svg"):
        return None
    width = _extract_svg_dimension(root.get("width"))
    height = _extract_svg_dimension(root.get("height"))
    if (width is None or height is None) and root.get("viewBox"):
        view_box = root.get("viewBox")
        if view_box:
            parts = re.split(r"[\s,]+", view_box.strip())
            if len(parts) == 4:
                vb_width = _to_float(parts[2])
                vb_height = _to_float(parts[3])
                width = width if width is not None else vb_width
                height = height if height is not None else vb_height
    aspect_ratio = _compute_aspect_ratio(width, height)
    return {
        "width": width,
        "height": height,
        "has_alpha": True,
        "mime": "image/svg+xml",
        "aspect_ratio": aspect_ratio,
    }


def _extract_svg_dimension(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"([0-9]*\.?[0-9]+)", value)
    if not match:
        return None
    return float(match.group(1))


def _has_alpha_channel(image: Image.Image) -> bool:
    bands = image.getbands()
    if not bands:
        return False
    if "A" in bands:
        return True
    if image.mode in {"P", "L"}:
        return image.info.get("transparency") is not None
    return False


def _compute_aspect_ratio(width: float | None, height: float | None) -> float | None:
    if not width or not height:
        return None
    if height == 0:
        return None
    return float(width) / float(height)


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

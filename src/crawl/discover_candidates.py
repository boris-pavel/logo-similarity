"""Discover candidate logo asset URLs within fetched documents."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterable, Iterator
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from bs4.element import Tag

from .fetch import normalize_url

logger = logging.getLogger(__name__)

Candidate = dict[str, Any]

_CONFIDENCE_SCORES: dict[str, float] = {
    "org_logo": 0.95,
    "apple_touch": 0.7,
    "icon": 0.55,
    "og_image": 0.6,
    "twitter_image": 0.6,
    "header_img": 0.8,
    "common_path": 0.65,
    "css_bg": 0.6,
}

_LOGO_FILENAME_EXTS = (".svg", ".png", ".jpg", ".jpeg", ".webp", ".ico", ".gif")
_UNLIKELY_FILENAME_PATTERN = re.compile(
    r"(?:hero|banner|placeholder|header|cover|background|slider)", re.IGNORECASE
)
_LOGO_KEYWORDS = re.compile(r"(logo|brand|mark)", re.IGNORECASE)

_COMMON_PATH_EXTS = ("svg", "png", "jpg", "jpeg", "webp")
_COMMON_PATH_ROOTS = tuple(f"/logo.{ext}" for ext in _COMMON_PATH_EXTS)
_COMMON_PATH_PREFIXES = ("/assets/logo", "/static/logo")
_COMMON_EXTRA_PATHS = {"/favicon.svg"}


def discover_logo_candidates(html: str, base_url: str) -> list[Candidate]:
    """Return a ranked list of likely logo asset candidates discovered in *html*."""
    soup = BeautifulSoup(html or "", "lxml")

    results: list[Candidate] = []
    seen: set[str] = set()

    extractors = (
        _extract_jsonld_logos,
        _extract_link_icons,
        _extract_meta_social_images,
        _extract_logo_images,
        _extract_common_path_candidates,
        _extract_css_backgrounds_stub,
    )

    for extractor in extractors:
        try:
            for candidate in extractor(soup, base_url, html):
                src = candidate.get("src")
                if not src or src in seen:
                    continue
                seen.add(src)
                results.append(candidate)
        except Exception:  # noqa: BLE001 - best-effort aggregation
            logger.debug(
                "Candidate extractor %s failed", extractor.__name__, exc_info=True
            )

    return results


def is_plausible_logo_filename(path: str) -> bool:
    """Return ``True`` when *path* resembles a likely logo asset filename."""
    if not path:
        return True
    sanitized = path.split("?", 1)[0].split("#", 1)[0]
    if not sanitized:
        return True
    filename = sanitized.rsplit("/", 1)[-1]
    if not filename:
        return True
    lower = filename.lower()
    if _UNLIKELY_FILENAME_PATTERN.search(lower):
        return False
    if "." in lower:
        ext = "." + lower.rsplit(".", 1)[-1]
        if ext and ext not in _LOGO_FILENAME_EXTS:
            return False
    keywords = ("logo", "brand", "icon", "mark", "favicon")
    if any(key in lower for key in keywords):
        return True
    return any(lower.endswith(ext) for ext in _LOGO_FILENAME_EXTS)


def _extract_jsonld_logos(
    soup: BeautifulSoup, base_url: str, _: str
) -> Iterator[Candidate]:
    selector = lambda value: value and "ld+json" in value.lower()
    for index, script in enumerate(soup.find_all("script", attrs={"type": selector})):
        text = script.string or script.get_text()
        if not text:
            continue
        text = text.strip()
        if not text:
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Skipping invalid JSON-LD block", exc_info=True)
            continue
        yield from _collect_jsonld_logo_candidates(data, base_url, index)


def _collect_jsonld_logo_candidates(
    data: Any, base_url: str, script_index: int
) -> Iterator[Candidate]:
    for path, node in _iter_dicts(data):
        logos = node.get("logo")
        if not logos:
            continue
        types = _normalize_jsonld_types(node.get("@type"))
        for logo_value in _iter_logo_values(logos):
            absolute = _make_absolute(logo_value, base_url)
            if not absolute or not _is_valid_candidate_src(absolute):
                continue
            context = {
                "jsonld_path": "/".join(path + ("logo",)),
                "types": types or None,
                "script_index": script_index,
            }
            yield _build_candidate(
                absolute, "org_logo", _CONFIDENCE_SCORES["org_logo"], context
            )


def _extract_link_icons(
    soup: BeautifulSoup, base_url: str, _: str
) -> Iterator[Candidate]:
    for link in soup.find_all("link"):
        rel_values = _get_rel_values(link)
        if not rel_values:
            continue
        source: str | None = None
        if any("apple-touch-icon" in value for value in rel_values):
            source = "apple_touch"
        elif any("icon" == value or value.endswith("icon") for value in rel_values):
            source = "icon"
        elif any("mask-icon" in value for value in rel_values):
            source = "icon"
        if source is None:
            continue
        href = link.get("href")
        if not isinstance(href, str):
            continue
        absolute = _make_absolute(href, base_url)
        if not absolute or not _is_valid_candidate_src(absolute):
            continue
        context = {
            "tag": "link",
            "attr": _clean_attrs(link, ("rel", "sizes", "type", "href", "color")),
        }
        yield _build_candidate(absolute, source, _CONFIDENCE_SCORES[source], context)


def _extract_meta_social_images(
    soup: BeautifulSoup, base_url: str, _: str
) -> Iterator[Candidate]:
    for meta in soup.find_all("meta"):
        key_raw = meta.get("property") or meta.get("name")
        if not isinstance(key_raw, str):
            continue
        key = key_raw.lower()
        if key != "og:image" and key not in {"twitter:image", "twitter:image:src"}:
            continue
        content = meta.get("content")
        if not isinstance(content, str):
            continue
        absolute = _make_absolute(content, base_url)
        if not absolute or not _is_valid_candidate_src(absolute):
            continue
        source = "og_image" if key == "og:image" else "twitter_image"
        context = {
            "tag": "meta",
            "attr": _clean_attrs(meta, ("property", "name", "content")),
        }
        yield _build_candidate(absolute, source, _CONFIDENCE_SCORES[source], context)


def _extract_logo_images(
    soup: BeautifulSoup, base_url: str, _: str
) -> Iterator[Candidate]:
    for img in soup.find_all("img"):
        descriptors = _gather_img_descriptors(img)
        if not descriptors:
            continue
        if not _LOGO_KEYWORDS.search(descriptors):
            continue
        raw_src = _resolve_img_src(img)
        if not raw_src:
            continue
        absolute = _make_absolute(raw_src, base_url)
        if not absolute or not _is_valid_candidate_src(absolute):
            continue
        in_header = any(
            isinstance(parent, Tag) and parent.name in {"header", "nav"}
            for parent in img.parents
        )
        confidence = (
            _CONFIDENCE_SCORES["header_img"]
            if in_header
            else max(0.0, _CONFIDENCE_SCORES["header_img"] - 0.05)
        )
        context = {
            "tag": "img",
            "attr": _clean_attrs(img, ("id", "class", "alt", "src")),
            "in_header": in_header,
        }
        yield _build_candidate(absolute, "header_img", confidence, context)


def _extract_common_path_candidates(
    soup: BeautifulSoup, base_url: str, _: str
) -> Iterator[Candidate]:
    emitted: set[str] = set()
    for value in _collect_attribute_urls(soup):
        if not _matches_common_path(value):
            continue
        absolute = _make_absolute(value, base_url)
        if not absolute or not _is_valid_candidate_src(absolute):
            continue
        if absolute in emitted:
            continue
        emitted.add(absolute)
        context = {"detected_from": "attribute", "value": value}
        yield _build_candidate(
            absolute, "common_path", _CONFIDENCE_SCORES["common_path"], context
        )

    for generated in _generate_common_paths(base_url):
        if not generated or generated in emitted:
            continue
        emitted.add(generated)
        context = {"detected_from": "heuristic", "value": urlparse(generated).path}
        yield _build_candidate(
            generated, "common_path", _CONFIDENCE_SCORES["common_path"], context
        )


def _extract_css_backgrounds_stub(
    _soup: BeautifulSoup, _base_url: str, _html: str
) -> Iterator[Candidate]:
    """Placeholder for future CSS background discovery via rendered pages."""
    yield from ()


def _build_candidate(
    src: str, source: str, confidence: float, context: dict[str, Any]
) -> Candidate:
    return {"src": src, "source": source, "confidence": confidence, "context": context}


def _make_absolute(raw_url: str, base_url: str) -> str | None:
    if not raw_url:
        return None
    candidate = raw_url.strip()
    if not candidate:
        return None
    if candidate.startswith("data:"):
        return candidate
    try:
        return normalize_url(candidate, base_url)
    except Exception:  # pragma: no cover - defensive against malformed URLs
        logger.debug(
            "Failed normalising %s against %s", candidate, base_url, exc_info=True
        )
        return None


def _is_valid_candidate_src(src: str) -> bool:
    if src.startswith("data:"):
        return True
    parsed = urlparse(src)
    if parsed.scheme and parsed.scheme not in {"http", "https"}:
        return False
    return is_plausible_logo_filename(parsed.path)


def _iter_dicts(
    data: Any, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], dict[str, Any]]]:
    if isinstance(data, dict):
        yield path, data
        for key, value in data.items():
            yield from _iter_dicts(value, path + (str(key),))
    elif isinstance(data, list):
        for index, item in enumerate(data):
            yield from _iter_dicts(item, path + (str(index),))


def _normalize_jsonld_types(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, list):
        return [str(item) for item in raw if isinstance(item, str)]
    return []


def _iter_logo_values(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            yield stripped
    elif isinstance(value, dict):
        for key in ("@id", "url", "contentUrl", "href"):
            nested = value.get(key)
            if isinstance(nested, str):
                stripped = nested.strip()
                if stripped:
                    yield stripped
    elif isinstance(value, list):
        for item in value:
            yield from _iter_logo_values(item)


def _get_rel_values(tag: Tag) -> list[str]:
    rel = tag.get("rel")
    values: list[str] = []
    if isinstance(rel, str):
        values.append(rel.lower())
    elif isinstance(rel, (list, tuple)):
        for item in rel:
            if isinstance(item, str):
                values.append(item.lower())
    return values


def _clean_attrs(tag: Tag, keys: Iterable[str]) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key in keys:
        if key not in tag.attrs:
            continue
        attrs[key] = _coerce_attr_value(tag.attrs.get(key))
    return attrs


def _coerce_attr_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        coerced = []
        for item in value:
            if isinstance(item, (str, int, float, bool)) or item is None:
                coerced.append(item)
            else:
                coerced.append(str(item))
        return coerced
    return str(value)


def _gather_img_descriptors(img: Tag) -> str:
    descriptors: list[str] = []
    for key in ("id", "alt", "aria-label", "data-testid"):
        value = img.get(key)
        if isinstance(value, str):
            descriptors.append(value)
    classes = img.get("class")
    if isinstance(classes, list):
        descriptors.extend(cls for cls in classes if isinstance(cls, str))
    elif isinstance(classes, str):
        descriptors.append(classes)
    return " ".join(descriptors).strip()


def _resolve_img_src(img: Tag) -> str | None:
    for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-hires"):
        value = img.get(attr)
        if isinstance(value, str) and value.strip():
            return value.strip()
    srcset = img.get("srcset") or img.get("data-srcset")
    if isinstance(srcset, str):
        first = srcset.split(",")[0].strip()
        if first:
            url_part = first.split()[0]
            if url_part:
                return url_part
    return None


def _collect_attribute_urls(soup: BeautifulSoup) -> set[str]:
    values: set[str] = set()
    for tag in soup.find_all(True):
        for attr_value in tag.attrs.values():
            if isinstance(attr_value, str):
                values.add(attr_value)
            elif isinstance(attr_value, (list, tuple)):
                for item in attr_value:
                    if isinstance(item, str):
                        values.add(item)
    return values


def _matches_common_path(value: str) -> bool:
    parsed = urlparse(value)
    path = (parsed.path or value or "").split("?", 1)[0].split("#", 1)[0].lower()
    if not path:
        return False
    if path in _COMMON_EXTRA_PATHS:
        return True
    if path in _COMMON_PATH_ROOTS:
        return True
    for prefix in _COMMON_PATH_PREFIXES:
        for ext in _COMMON_PATH_EXTS:
            candidate = f"{prefix}.{ext}"
            if path == candidate:
                return True
    return False


def _generate_common_paths(base_url: str) -> Iterator[str]:
    emitted: set[str] = set()
    for root in _COMMON_PATH_ROOTS:
        absolute = _make_absolute(root, base_url)
        if absolute and absolute not in emitted:
            emitted.add(absolute)
            yield absolute
    for prefix in _COMMON_PATH_PREFIXES:
        for ext in _COMMON_PATH_EXTS:
            absolute = _make_absolute(f"{prefix}.{ext}", base_url)
            if absolute and absolute not in emitted:
                emitted.add(absolute)
                yield absolute
    for extra in _COMMON_EXTRA_PATHS:
        absolute = _make_absolute(extra, base_url)
        if absolute and absolute not in emitted:
            emitted.add(absolute)
            yield absolute

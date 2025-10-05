"""Command-line interface for the logo_similarity project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

import pandas as pd
from PIL import Image

from .crawl.discover_candidates import discover_logo_candidates
from .crawl.fetch import fetch_html
from .extract.normalize import normalize_logo
from .extract.select_logo import select_best
from .features.color import dominant_hues, hsv_histogram
from .features.perceptual import compute_hashes
from .features.shape import orb_keypoints_match
from .group.group_and_metrics import group_and_report
from .group.similarity import (
    T_LINK,
    base_similarity,
    combine_components,
    pairwise_scores,
)
from .io.models import LogoFeatures

DEFAULT_ASSET_DIR = Path("out") / "extracted"
PREVIEW_BG_RGBA = (0xF5, 0xF5, 0xF5, 255)

_MIME_EXTENSIONS = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/svg+xml": ".svg",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/x-icon": ".ico",
    "image/vnd.microsoft.icon": ".ico",
}

def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the logo similarity pipeline."""
    parser = argparse.ArgumentParser(
        description="Inspect a list of website URLs and report the count."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a text file containing website URLs, one per line.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Directory path where JSON outputs will be written.",
    )
    parser.add_argument(
        "--assets",
        required=False,
        default=None,
        help="Directory path where downloaded logo assets will be stored.",
    )
    parser.add_argument(
        "--debug-fetch",
        action="store_true",
        help="Fetch the first few URLs and print HTML length for diagnostics.",
    )
    parser.add_argument(
        "--debug-candidates",
        action="store_true",
        help="Extract logo candidates for the first two URLs and print the top ones.",
    )
    parser.add_argument(
        "--lazy-selection",
        action="store_true",
        help="Skip candidate image downloads and select by confidence only.",
    )

    parser.add_argument(
        "--debug-pairs",
        nargs="?",
        const=20,
        type=int,
        metavar="N",
        default=0,
        help="Show the top N pairwise similarity scores (default 20).",
    )
    return parser.parse_args(list(argv) if argv is not None else None)

def read_input(path: Path) -> list[str]:
    """Read newline separated entries from *path* and return non-empty lines."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines()]
    return [line for line in lines if line]

def _debug_fetch(urls: list[str]) -> None:
    """Perform lightweight fetches for the first few *urls* and print stats."""
    for url in urls[:3]:
        final_url, html = fetch_html(url)
        html_length = len(html) if html else 0
        print(f"[debug] {url} -> {final_url or 'None'} ({html_length} chars)")

def _debug_candidates(urls: list[str]) -> None:
    """Fetch pages and print top logo candidates for the first two *urls*."""
    for url in urls[:2]:
        final_url, html = fetch_html(url)
        base_url = final_url or url
        print(f"[candidates] {url} -> {base_url or 'None'}")
        if not html:
            print("  (no html)")
            continue
        candidates = discover_logo_candidates(html, base_url)
        if not candidates:
            print("  (no candidates)")
            continue
        top_candidates = sorted(
            candidates,
            key=lambda item: float(item.get("confidence", 0.0) or 0.0),
            reverse=True,
        )[:5]
        for index, candidate in enumerate(top_candidates, start=1):
            src = candidate.get("src", "")
            source = candidate.get("source", "unknown")
            confidence = float(candidate.get("confidence", 0.0) or 0.0)
            print(f"  {index}. {source} (conf={confidence:.2f}) -> {src}")

def _debug_pairs(features: list[LogoFeatures], limit: int) -> None:
    """Compute pairwise similarity scores and print the strongest matches."""
    if limit <= 0:
        return
    feature_map = _build_feature_map(features)
    if not feature_map:
        print("[pairs] no features available for pairwise debug")
        return
    images_map = _load_normalized_images(feature_map)
    try:
        edges = list(pairwise_scores(feature_map, images_map))
        if not edges:
            print(f"[pairs] no pairwise links above threshold {T_LINK:.2f}")
            return
        edges.sort(key=lambda item: item[2], reverse=True)
        top_edges = edges[:limit]
        print(
            f"[pairs] showing top {len(top_edges)} of {len(edges)} edges (>= {T_LINK:.2f})"
        )
        component_order = ("phash", "dhash", "ahash", "hist")
        for index, (left, right, score) in enumerate(top_edges, start=1):
            components = base_similarity(feature_map[left], feature_map[right])
            base_score = combine_components(components)
            orb_score = None
            if T_LINK - 0.05 <= base_score <= T_LINK + 0.1:
                image_a = images_map.get(left)
                image_b = images_map.get(right)
                if image_a is not None and image_b is not None:
                    orb_score = orb_keypoints_match(image_a, image_b)
            component_summary = ", ".join(
                f"{name}={components.get(name, 0.0):.3f}" for name in component_order
            )
            orb_fragment = f", orb={orb_score:.3f}" if orb_score is not None else ""
            print(
                f"  {index}. {left} <-> {right} score={score:.3f}, base={base_score:.3f}{orb_fragment}"
                f" ({component_summary})"
            )
    finally:
        for image in images_map.values():
            try:
                image.close()
            except Exception:
                pass

def _build_feature_map(features: Iterable[LogoFeatures]) -> dict[str, LogoFeatures]:
    """Return a map keyed by website identifier for quick lookups."""
    feature_map: dict[str, LogoFeatures] = {}
    for feature in features:
        website = feature.website
        if not website:
            continue
        feature_map[website] = feature
    return feature_map

def _load_normalized_images(
    feature_map: Mapping[str, LogoFeatures],
) -> dict[str, Image.Image]:
    """Load normalized logo images for ORB comparisons."""
    images: dict[str, Image.Image] = {}
    for website, feature in feature_map.items():
        path = feature.normalized_path
        if not path:
            continue
        try:
            image = Image.open(path)
        except Exception as exc:
            print(f"[pairs] {website}: failed to open normalized image ({exc})")
            continue
        images[website] = image
    return images


def _close_images(images: Mapping[str, Image.Image]) -> None:
    """Close all PIL image handles in *images*."""
    for image in images.values():
        try:
            image.close()
        except Exception:
            pass


def _process_sites(urls: list[str], assets_dir: Path, lazy: bool) -> list[LogoFeatures]:
    """Fetch pages, discover candidates, and persist the preferred logo."""
    feature_rows: list[LogoFeatures] = []
    if not urls:
        return feature_rows
    if not lazy:
        assets_dir.mkdir(parents=True, exist_ok=True)
    for url in urls:
        final_url, html = fetch_html(url)
        base_url = final_url or url
        host_label = _safe_host_label(base_url)
        if not html:
            print(f"[warn] {host_label}: no HTML content fetched")
            continue
        candidates = discover_logo_candidates(html, base_url)
        if not candidates:
            print(f"[warn] {host_label}: no logo candidates found")
            continue
        best = select_best(candidates, base_url, lazy=lazy)
        if not best:
            print(f"[warn] {host_label}: selection failed")
            continue
        image_bytes = best.get("image_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            print(
                f"[info] {host_label}: selection did not yield image bytes (lazy={lazy})"
            )
            continue
        raw_bytes = bytes(image_bytes)
        output_path = _determine_asset_path(assets_dir, host_label, best)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(raw_bytes)
        print(f"[saved] {host_label}: {output_path}")
        feature_entry = _persist_artifacts_and_features(
            raw_bytes, best, host_label, assets_dir, output_path
        )
        if feature_entry:
            feature_rows.append(feature_entry)
    return feature_rows

def _persist_artifacts_and_features(
    image_bytes: bytes,
    candidate: dict[str, Any],
    host_label: str,
    assets_dir: Path,
    original_path: Path,
) -> LogoFeatures | None:
    mime_hint = _candidate_mime_hint(candidate)
    try:
        normalized = normalize_logo(image_bytes, mime_hint)
    except Exception as exc:
        print(f"[warn] {host_label}: normalization failed ({exc})")
        return None

    try:
        perceptual = compute_hashes(normalized)
        histogram = hsv_histogram(normalized)
        hues = dominant_hues(normalized)
        normalized_path: Path | None = None
        preview_path: Path | None = None
        try:
            normalized_path, preview_path = _save_normalized_assets(
                normalized, host_label, assets_dir
            )
        except Exception as exc:
            print(f"[warn] {host_label}: failed to write normalized assets ({exc})")
        return LogoFeatures(
            website=host_label,
            original_path=original_path,
            normalized_path=normalized_path,
            preview_path=preview_path,
            perceptual=perceptual,
            hsv_histogram=histogram,
            dominant_hues=hues,
        )
    finally:
        normalized.close()

def _save_normalized_assets(
    normalized: Image.Image, host_label: str, assets_dir: Path
) -> tuple[Path, Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = assets_dir / f"{host_label}.png"
    normalized.save(normalized_path, format="PNG")

    preview_background = Image.new("RGBA", normalized.size, PREVIEW_BG_RGBA)
    preview = Image.alpha_composite(preview_background, normalized)
    preview_path = assets_dir / f"{host_label}.preview.png"
    preview_rgb = preview.convert("RGB")
    preview_rgb.save(preview_path, format="PNG")
    print(f"[normalized] {host_label}: {normalized_path} (preview {preview_path})")
    preview_rgb.close()
    preview.close()
    preview_background.close()
    return normalized_path, preview_path

def _candidate_mime_hint(candidate: dict[str, Any]) -> str | None:
    info = candidate.get("image_info")
    if isinstance(info, dict):
        mime_value = info.get("mime")
        if isinstance(mime_value, str):
            return mime_value
    return None

def _write_feature_table(features: list[LogoFeatures], out_dir: Path) -> None:
    if not features:
        print("[features] no feature rows to write")
        return

    rows: list[dict[str, Any]] = []
    for feature in features:
        perceptual = feature.perceptual or {}
        rows.append(
            {
                "website": feature.website,
                "original_path": str(feature.original_path)
                if feature.original_path
                else None,
                "normalized_path": str(feature.normalized_path)
                if feature.normalized_path
                else None,
                "preview_path": str(feature.preview_path)
                if feature.preview_path
                else None,
                "ahash": perceptual.get("ahash"),
                "phash": perceptual.get("phash"),
                "dhash": perceptual.get("dhash"),
                "hsv_histogram": feature.hsv_histogram,
                "dominant_hues": feature.dominant_hues,
            }
        )

    df = pd.DataFrame(rows)
    features_path = out_dir / "features.parquet"
    features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(features_path, index=False, engine="pyarrow")
    print(f"[features] wrote {len(df)} rows to {features_path}")

def _determine_asset_path(assets_dir: Path, host_label: str, candidate: dict[str, Any]) -> Path:
    extension = _extension_from_info(candidate)
    suffix = extension or ""
    base_name = f"{host_label}.orig{suffix}"
    path = assets_dir / base_name
    counter = 2
    while path.exists():
        if suffix:
            base_name = f"{host_label}.orig_{counter}{suffix}"
        else:
            base_name = f"{host_label}.orig_{counter}"
        path = assets_dir / base_name
        counter += 1
    return path

def _extension_from_info(candidate: dict[str, Any]) -> str | None:
    info = candidate.get("image_info")
    if isinstance(info, dict):
        mime = info.get("mime")
        if isinstance(mime, str):
            ext = _MIME_EXTENSIONS.get(mime.lower())
            if ext:
                return ext
    for key in ("resolved_src", "src"):
        value = candidate.get(key)
        if isinstance(value, str):
            ext = _extension_from_url(value)
            if ext:
                return ext
    return None

def _extension_from_url(value: str) -> str | None:
    parsed = urlparse(value)
    path = parsed.path or ""
    if not path:
        return None
    filename = path.rsplit("/", 1)[-1]
    if not filename or "." not in filename:
        return None
    ext = filename.rsplit(".", 1)[-1].lower()
    if not ext or len(ext) > 6:
        return None
    return f".{ext}"

def _safe_host_label(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path or "site"
    sanitized = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in host)
    sanitized = sanitized.strip("._-")
    return sanitized or "site"

def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the CLI."""
    args = parse_args(argv)
    input_path = Path(args.input)
    entries = read_input(input_path)
    print(len(entries))
    if args.debug_fetch:
        _debug_fetch(entries)
    if args.debug_candidates:
        _debug_candidates(entries)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = Path(args.assets) if args.assets else DEFAULT_ASSET_DIR
    features = _process_sites(entries, assets_dir, lazy=args.lazy_selection)
    _write_feature_table(features, out_dir)

    feature_map = _build_feature_map(features)
    images_map = _load_normalized_images(feature_map)
    try:
        group_and_report(
            feature_map,
            images_map,
            total_sites=len(entries),
            out_dir=out_dir,
            t_link=T_LINK,
        )
    finally:
        _close_images(images_map)

    if args.debug_pairs:
        _debug_pairs(features, args.debug_pairs)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

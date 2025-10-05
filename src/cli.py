"""Command-line interface for the logo_similarity project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .crawl.discover_candidates import discover_logo_candidates
from .crawl.fetch import fetch_html
from .extract.select_logo import select_best

DEFAULT_ASSET_DIR = Path("out") / "extracted"

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


def _process_sites(urls: list[str], assets_dir: Path, lazy: bool) -> None:
    """Fetch pages, discover candidates, and persist the preferred logo."""
    if not urls:
        return
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
        output_path = _determine_asset_path(assets_dir, host_label, best)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(bytes(image_bytes))
        print(f"[saved] {host_label}: {output_path}")


def _determine_asset_path(assets_dir: Path, host_label: str, candidate: dict[str, Any]) -> Path:
    extension = _extension_from_info(candidate) or ".orig"
    path = assets_dir / f"{host_label}{extension}"
    counter = 2
    while path.exists():
        path = assets_dir / f"{host_label}_{counter}{extension}"
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
    _process_sites(entries, assets_dir, lazy=args.lazy_selection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

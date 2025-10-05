"""Command-line interface for the logo_similarity project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from .crawl.fetch import fetch_html


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
    return parser.parse_args(list(argv) if argv is not None else None)


def read_input(path: Path) -> list[str]:
    """Read newline separated entries from *path* and return non-empty lines."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def _debug_fetch(urls: list[str]) -> None:
    """Perform lightweight fetches for the first few *urls* and print stats."""
    for url in urls[:3]:
        final_url, html = fetch_html(url)
        html_length = len(html) if html else 0
        print(f"[debug] {url} -> {final_url or 'None'} ({html_length} chars)")


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the CLI."""
    args = parse_args(argv)
    input_path = Path(args.input)
    entries = read_input(input_path)
    print(len(entries))
    if args.debug_fetch:
        _debug_fetch(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

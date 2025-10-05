"""Command-line interface for the logo_similarity project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


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
    return parser.parse_args(list(argv) if argv is not None else None)


def read_input(path: Path) -> list[str]:
    """Read newline separated entries from *path* and return non-empty lines."""
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


def main(argv: Iterable[str] | None = None) -> int:
    """Entry point for the CLI."""
    args = parse_args(argv)
    input_path = Path(args.input)
    entries = read_input(input_path)
    print(len(entries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

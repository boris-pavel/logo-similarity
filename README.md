# logo-similarity

A modular toolkit for discovering and grouping visually similar company logos.

## Prerequisites

- Python 3.10 or later
- Recommended: virtual environment (via `venv` or similar)

## Installation

```
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

Provide a newline separated list of website URLs and run the CLI to inspect the dataset size.

```
python -m src.cli --input data\logos_list.txt --out out --assets out\assets
```

The current implementation only parses the input file and prints the number of entries. Subsequent steps will add crawling, extraction, feature computation, and grouping.

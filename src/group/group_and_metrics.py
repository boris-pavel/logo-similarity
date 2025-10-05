"""Grouping and reporting utilities for the logo similarity pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

from .similarity import T_LINK as DEFAULT_T_LINK, pairwise_scores
from .unionfind import UnionFind
from ..io.models import LogoFeatures

Edge = Tuple[str, str, float]
FeaturesMap = Mapping[str, LogoFeatures]
ImagesMap = Mapping[str, Image.Image]


def build_similarity_edges(
    features: FeaturesMap,
    images: ImagesMap,
    t_link: float = DEFAULT_T_LINK,
) -> list[Edge]:
    """Return all pairwise similarity edges on or above *t_link*."""
    if not features:
        return []

    edges: list[Edge] = []
    iterator = pairwise_scores(features, images, t_link=t_link)
    for left, right, score in tqdm(
        iterator, desc="Scoring similarities", unit="pair", leave=False
    ):
        edges.append((left, right, float(score)))
    return edges


def group_and_report(
    features_map: FeaturesMap,
    images_map: ImagesMap,
    total_sites: int,
    out_dir: str | Path,
    t_link: float = DEFAULT_T_LINK,
) -> Dict[str, Any]:
    """Compute similarity groups, persist reports, and print a console summary."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    edges = build_similarity_edges(features_map, images_map, t_link=t_link)

    uf = UnionFind()
    uf.add_all(features_map.keys())

    for left, right, _ in tqdm(edges, desc="Linking groups", unit="edge", leave=False):
        uf.union(left, right)

    groups_map = uf.groups()
    sorted_groups: Sequence[tuple[str, list[str]]] = sorted(
        ((root, sorted(members)) for root, members in groups_map.items()),
        key=lambda item: (-len(item[1]), item[0]),
    )

    groups_payload = [
        {"group_id": root, "members": members} for root, members in sorted_groups
    ]

    extracted = len(features_map)
    coverage = (extracted / total_sites) if total_sites else 0.0
    largest_group = max((len(m["members"]) for m in groups_payload), default=0)
    metrics: Dict[str, Any] = {
        "total": int(total_sites),
        "extracted": int(extracted),
        "coverage": coverage,
        "pairs": len(edges),
        "groups": len(groups_payload),
        "largest_group": int(largest_group),
        "threshold": float(t_link),
    }

    groups_path = out_path / "groups.json"
    try:
        groups_path.write_text(json.dumps(groups_payload, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[group] failed to write {groups_path}: {exc}")

    pairs_path = out_path / "pairs_sample.csv"
    top_edges = sorted(edges, key=lambda item: item[2], reverse=True)[:500]
    try:
        with pairs_path.open("w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["left", "right", "score"])
            for left, right, score in top_edges:
                writer.writerow([left, right, f"{score:.6f}"])
    except OSError as exc:
        print(f"[group] failed to write {pairs_path}: {exc}")

    metrics_path = out_path / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    except OSError as exc:
        print(f"[group] failed to write {metrics_path}: {exc}")

    coverage_pct = coverage * 100.0
    print(f"Total sites: {total_sites}")
    print(f"Extracted: {extracted} (coverage {coverage_pct:.1f}%)")
    print(f"Groups: {len(groups_payload)}")
    print(f"Largest group: {largest_group} logos")
    print(f"Threshold: {t_link:.2f}")

    return {"edges": edges, "groups": groups_payload, "metrics": metrics}

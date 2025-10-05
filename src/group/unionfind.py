"""Simple union-find data structure for grouping logos."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


class UnionFind:
    """Disjoint set union structure with path compression."""

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, item: str) -> str:
        """Return the canonical representative for *item*."""
        if item not in self._parent:
            self._parent[item] = item
            self._rank[item] = 0
            return item
        if self._parent[item] != item:
            self._parent[item] = self.find(self._parent[item])
        return self._parent[item]

    def union(self, a: str, b: str) -> None:
        """Merge the sets containing *a* and *b*."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        rank_a = self._rank[root_a]
        rank_b = self._rank[root_b]
        if rank_a < rank_b:
            self._parent[root_a] = root_b
        elif rank_a > rank_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1

    def add_all(self, items: Iterable[str]) -> None:
        """Ensure that all *items* exist in the data structure."""
        for item in items:
            self.find(item)

    def groups(self) -> Dict[str, List[str]]:
        """Return the current partitioning as a mapping of roots to members."""
        buckets: Dict[str, List[str]] = defaultdict(list)
        for item in self._parent:
            root = self.find(item)
            buckets[root].append(item)
        return {root: members for root, members in buckets.items()}

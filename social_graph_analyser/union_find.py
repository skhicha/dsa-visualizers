"""
Disjoint Set Union (Union-Find) with path compression and union-by-rank.

Used for O(alpha(n)) (effectively constant amortized) community detection:
after unioning every edge of a social graph, nodes sharing a root belong to
the same community/connected component.
"""

from __future__ import annotations

from typing import Dict, Hashable, List


class UnionFind:
    def __init__(self, nodes: List[Hashable] | None = None) -> None:
        self._parent: Dict[Hashable, Hashable] = {}
        self._rank: Dict[Hashable, int] = {}
        if nodes:
            for node in nodes:
                self.make_set(node)

    def make_set(self, node: Hashable) -> None:
        if node not in self._parent:
            self._parent[node] = node
            self._rank[node] = 0

    def find(self, node: Hashable) -> Hashable:
        """Find the representative (root) of node's set.

        Uses path compression: every node visited on the way to the root
        is re-pointed directly at the root, flattening the tree for future
        lookups.
        """
        self.make_set(node)
        root = node
        path: List[Hashable] = []
        while self._parent[root] != root:
            path.append(root)
            root = self._parent[root]
        for visited in path:
            self._parent[visited] = root
        return root

    def union(self, a: Hashable, b: Hashable) -> None:
        """Union by rank: attach the shorter tree under the taller one."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return

        rank_a, rank_b = self._rank[root_a], self._rank[root_b]
        if rank_a < rank_b:
            self._parent[root_a] = root_b
        elif rank_a > rank_b:
            self._parent[root_b] = root_a
        else:
            self._parent[root_b] = root_a
            self._rank[root_a] += 1

    def connected(self, a: Hashable, b: Hashable) -> bool:
        return self.find(a) == self.find(b)

    def components(self) -> Dict[Hashable, List[Hashable]]:
        """Group every known node by its root representative."""
        groups: Dict[Hashable, List[Hashable]] = {}
        for node in self._parent:
            root = self.find(node)
            groups.setdefault(root, []).append(node)
        return groups

    def tree_height(self, node: Hashable) -> int:
        """Number of hops from node up to its root (for diagnostics/tests)."""
        height = 0
        current = node
        while self._parent[current] != current:
            current = self._parent[current]
            height += 1
        return height

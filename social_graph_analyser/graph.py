"""
Weighted undirected graph implemented from scratch using adjacency lists.

No graph-algorithm library is used here - BFS, DFS and Dijkstra are all
hand-implemented. NetworkX/Matplotlib are only used downstream for drawing
the graph (see visualize.py), never for computing traversals or shortest
paths.
"""

from __future__ import annotations

import heapq
from collections import deque
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple


class Graph:
    """Weighted undirected graph backed by an adjacency-list dict.

    Internal representation: self._adj[u] = {v: weight, ...}
    Edges are stored symmetrically (u->v and v->u) since the graph is
    undirected.
    """

    def __init__(self) -> None:
        self._adj: Dict[Hashable, Dict[Hashable, float]] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def add_node(self, node: Hashable) -> None:
        self._adj.setdefault(node, {})

    def add_edge(self, u: Hashable, v: Hashable, weight: float = 1.0) -> None:
        if u == v:
            raise ValueError("self-loops are not supported")
        self.add_node(u)
        self.add_node(v)
        self._adj[u][v] = weight
        self._adj[v][u] = weight

    def has_edge(self, u: Hashable, v: Hashable) -> bool:
        return u in self._adj and v in self._adj[u]

    def neighbors(self, node: Hashable) -> Iterable[Hashable]:
        return self._adj.get(node, {}).keys()

    def weight(self, u: Hashable, v: Hashable) -> float:
        return self._adj[u][v]

    @property
    def nodes(self) -> List[Hashable]:
        return list(self._adj.keys())

    @property
    def edges(self) -> List[Tuple[Hashable, Hashable, float]]:
        seen: Set[frozenset] = set()
        result = []
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                key = frozenset((u, v))
                if key not in seen:
                    seen.add(key)
                    result.append((u, v, w))
        return result

    def degree(self, node: Hashable) -> int:
        return len(self._adj.get(node, {}))

    def __len__(self) -> int:
        return len(self._adj)

    # ------------------------------------------------------------------
    # BFS - iterative, O(V + E)
    # ------------------------------------------------------------------
    def bfs(self, source: Hashable) -> List[Hashable]:
        """Return nodes in breadth-first visitation order starting at source.

        Uses a deque as a FIFO queue. Neighbors are visited in sorted order
        where possible so traversal order is deterministic (falls back to
        insertion order if nodes aren't orderable amongst themselves).
        """
        if source not in self._adj:
            raise KeyError(f"node {source!r} not in graph")

        visited: Set[Hashable] = {source}
        order: List[Hashable] = []
        queue = deque([source])

        while queue:
            node = queue.popleft()
            order.append(node)
            for nbr in self._sorted_neighbors(node):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return order

    def bfs_reachable(self, source: Hashable) -> Set[Hashable]:
        """Set of all nodes reachable from source (including source)."""
        return set(self.bfs(source))

    # ------------------------------------------------------------------
    # DFS - iterative using an explicit stack (no recursion)
    # ------------------------------------------------------------------
    def dfs(self, source: Hashable) -> List[Hashable]:
        """Return nodes in depth-first visitation order starting at source.

        Implemented iteratively with an explicit stack to avoid Python's
        recursion limit on large/deep graphs.
        """
        if source not in self._adj:
            raise KeyError(f"node {source!r} not in graph")

        visited: Set[Hashable] = set()
        order: List[Hashable] = []
        stack: List[Hashable] = [source]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            # Push neighbors in reverse sorted order so that, after the
            # stack pops them off, they are visited in ascending order -
            # this keeps traversal order deterministic.
            for nbr in reversed(self._sorted_neighbors(node)):
                if nbr not in visited:
                    stack.append(nbr)
        return order

    # ------------------------------------------------------------------
    # Dijkstra's shortest path - min-heap based, O((V + E) log V)
    # ------------------------------------------------------------------
    def dijkstra(
        self, source: Hashable
    ) -> Tuple[Dict[Hashable, float], Dict[Hashable, Optional[Hashable]]]:
        """Single-source shortest paths from source.

        Returns (dist, prev) where dist[node] is the shortest distance from
        source and prev[node] is the predecessor of node on that shortest
        path (None for source or unreachable nodes).
        """
        if source not in self._adj:
            raise KeyError(f"node {source!r} not in graph")

        dist: Dict[Hashable, float] = {node: float("inf") for node in self._adj}
        prev: Dict[Hashable, Optional[Hashable]] = {node: None for node in self._adj}
        dist[source] = 0.0

        # (distance, tie-breaking counter, node) - the counter avoids
        # comparing nodes of possibly-unorderable types directly.
        counter = 0
        heap: List[Tuple[float, int, Hashable]] = [(0.0, counter, source)]
        visited: Set[Hashable] = set()

        while heap:
            d, _, node = heapq.heappop(heap)
            if node in visited:
                continue
            visited.add(node)

            for nbr, w in self._adj[node].items():
                if w < 0:
                    raise ValueError("Dijkstra's algorithm requires non-negative weights")
                candidate = d + w
                if candidate < dist[nbr]:
                    dist[nbr] = candidate
                    prev[nbr] = node
                    counter += 1
                    heapq.heappush(heap, (candidate, counter, nbr))

        return dist, prev

    def shortest_path(
        self, source: Hashable, target: Hashable
    ) -> Tuple[float, List[Hashable]]:
        """Shortest path distance and node sequence from source to target."""
        dist, prev = self.dijkstra(source)
        if target not in dist or dist[target] == float("inf"):
            return float("inf"), []

        path = [target]
        while path[-1] != source:
            path.append(prev[path[-1]])
        path.reverse()
        return dist[target], path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sorted_neighbors(self, node: Hashable) -> List[Hashable]:
        neighbors = list(self._adj[node].keys())
        try:
            return sorted(neighbors)
        except TypeError:
            return neighbors

"""Degree centrality - a simple influence/importance measure per node."""

from __future__ import annotations

from typing import Dict, Hashable, List, Tuple

from .graph import Graph


def degree_centrality(graph: Graph) -> Dict[Hashable, float]:
    """Normalized degree centrality: degree(v) / (n - 1) for each node v.

    This is the standard definition used by NetworkX as well, so results
    are directly comparable, even though the degree counting here is done
    against our own from-scratch Graph adjacency structure.
    """
    n = len(graph)
    if n <= 1:
        return {node: 0.0 for node in graph.nodes}
    return {node: graph.degree(node) / (n - 1) for node in graph.nodes}


def top_influencers(graph: Graph, k: int = 10) -> List[Tuple[Hashable, float]]:
    """Top-k nodes by degree centrality, highest first."""
    scores = degree_centrality(graph)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:k]

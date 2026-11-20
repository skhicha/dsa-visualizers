"""
Synthetic social-network dataset generator.

Builds a graph with realistic community clustering: nodes are split into
several "friend groups" (communities) that are densely connected inside,
plus a sparse set of random cross-community edges (the "weak ties" that
hold the whole network together, à la Granovetter). This gives Union-Find
real, non-trivial communities to discover and gives degree centrality a
meaningful spread of influencers vs. peripheral nodes.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .graph import Graph


def generate_social_network(
    num_nodes: int = 100,
    num_communities: int = 6,
    intra_community_density: float = 0.35,
    inter_community_edges: int = 40,
    seed: int = 42,
) -> Graph:
    """Generate a synthetic weighted social graph with community structure.

    Args:
        num_nodes: total number of people (nodes) in the network.
        num_communities: number of distinct friend groups to carve nodes into.
        intra_community_density: probability that any two people in the same
            community are friends (edge exists).
        inter_community_edges: number of extra random edges connecting
            different communities ("weak ties" bridging groups).
        seed: RNG seed for reproducibility.

    Returns:
        A Graph where node ids are "person_0".."person_{n-1}" and edge
        weights represent "closeness" (1 = acquaintance, 5 = best friend).
    """
    rng = random.Random(seed)
    graph = Graph()

    nodes = [f"person_{i}" for i in range(num_nodes)]
    for node in nodes:
        graph.add_node(node)

    # Assign every node to a community (roughly evenly sized).
    communities: Dict[int, List[str]] = {c: [] for c in range(num_communities)}
    for i, node in enumerate(nodes):
        communities[i % num_communities].append(node)

    # Dense intra-community edges.
    for members in communities.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                if rng.random() < intra_community_density:
                    weight = rng.randint(1, 5)
                    graph.add_edge(members[i], members[j], weight)

    # Make sure every community member has at least one edge within their
    # community (ring fallback), so communities never fragment into
    # singleton islands purely by bad luck.
    for members in communities.values():
        for i in range(len(members)):
            a, b = members[i], members[(i + 1) % len(members)]
            if len(members) > 1 and not graph.has_edge(a, b):
                graph.add_edge(a, b, rng.randint(1, 5))

    # Sparse inter-community "weak tie" edges bridging different groups.
    community_ids = list(communities.keys())
    attempts = 0
    added = 0
    while added < inter_community_edges and attempts < inter_community_edges * 20:
        attempts += 1
        c1, c2 = rng.sample(community_ids, 2)
        if not communities[c1] or not communities[c2]:
            continue
        a = rng.choice(communities[c1])
        b = rng.choice(communities[c2])
        if not graph.has_edge(a, b):
            graph.add_edge(a, b, rng.randint(1, 3))
            added += 1

    return graph, communities


def community_label_for(node: str, communities: Dict[int, List[str]]) -> int:
    """Ground-truth community id a node was generated into (for validation)."""
    for cid, members in communities.items():
        if node in members:
            return cid
    raise KeyError(node)

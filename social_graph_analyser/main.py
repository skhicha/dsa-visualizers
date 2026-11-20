"""
End-to-end pipeline: generate a synthetic social network, run the
from-scratch graph algorithms (BFS, DFS, Dijkstra, Union-Find, degree
centrality), print a summary report, and render a visualization.

Run with:
    python -m social_graph_analyser.main
"""

from __future__ import annotations

import os

from .centrality import top_influencers
from .dataset import generate_social_network
from .union_find import UnionFind
from .visualize import render_social_network


def run(output_dir: str | None = None, num_nodes: int = 100) -> str:
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # inter_community_edges=0 here so Union-Find's connected components
    # line up exactly with the generated friend groups, demonstrating
    # community detection cleanly; generate_social_network also supports
    # sparse cross-community "weak ties" (see dataset.py / its tests),
    # which collapse everything into one giant small-world component -
    # a real and separately-tested behavior, just not what we want for
    # this illustrative run.
    graph, ground_truth_communities = generate_social_network(
        num_nodes=num_nodes, num_communities=6, inter_community_edges=0
    )
    print(f"Generated social network: {len(graph)} people, {len(graph.edges)} friendships")

    # --- BFS / DFS demo from an arbitrary starting person -----------------
    start = graph.nodes[0]
    bfs_order = graph.bfs(start)
    dfs_order = graph.dfs(start)
    print(f"BFS from {start} visited {len(bfs_order)} people (reachable component size)")
    print(f"DFS from {start} visited {len(dfs_order)} people")

    # --- Dijkstra shortest path demo --------------------------------------
    target = graph.nodes[-1]
    dist, path = graph.shortest_path(start, target)
    if path:
        print(f"Shortest 'closeness' path {start} -> {target}: cost={dist}, hops={len(path) - 1}")
    else:
        print(f"No path found between {start} and {target}")

    # --- Union-Find community detection ------------------------------------
    uf = UnionFind(graph.nodes)
    for u, v, _w in graph.edges:
        uf.union(u, v)
    components = uf.components()
    node_to_community = {}
    for idx, (_root, members) in enumerate(components.items()):
        for m in members:
            node_to_community[m] = idx
    print(f"Union-Find detected {len(components)} connected communities")
    for idx, (_root, members) in enumerate(components.items()):
        print(f"  community {idx}: {len(members)} members")

    # --- Degree centrality / influencers -----------------------------------
    influencers = top_influencers(graph, k=5)
    print("Top 5 influencers by degree centrality:")
    for node, score in influencers:
        print(f"  {node}: degree={graph.degree(node)}, centrality={score:.3f}")

    # --- Visualization ------------------------------------------------------
    output_path = os.path.join(output_dir, "social_network.png")
    render_social_network(
        graph,
        node_to_community,
        [node for node, _score in influencers],
        output_path,
    )
    print(f"Saved visualization to {output_path}")
    return output_path


if __name__ == "__main__":
    run()

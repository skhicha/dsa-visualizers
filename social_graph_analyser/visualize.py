"""
Visualization layer: NetworkX is used only for graph layout, and Matplotlib
only for rendering. All the actual graph algorithms (BFS/DFS/Dijkstra,
Union-Find, degree centrality) are computed entirely by our own from-scratch
implementations in graph.py / union_find.py / centrality.py - this module
never calls nx.shortest_path, nx.connected_components, etc.
"""

from __future__ import annotations

from typing import Dict, Hashable, List

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, no display needed

import matplotlib.pyplot as plt
import networkx as nx

from .graph import Graph


def _build_nx_graph_for_layout(graph: Graph) -> nx.Graph:
    """Build a plain NetworkX graph solely so nx can compute a layout.

    No NetworkX algorithm (shortest_path, connected_components, etc.) is
    ever invoked on this object - it exists purely to feed spring_layout.
    """
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(graph.nodes)
    for u, v, w in graph.edges:
        nx_graph.add_edge(u, v, weight=w)
    return nx_graph


def render_social_network(
    graph: Graph,
    communities: Dict[Hashable, int],
    influencers: List[Hashable],
    output_path: str,
    seed: int = 7,
) -> str:
    """Draw the graph colored by community, sized by degree, with
    top influencers highlighted, and save it to output_path.
    """
    nx_graph = _build_nx_graph_for_layout(graph)
    pos = nx.spring_layout(nx_graph, seed=seed, k=0.5)

    node_colors = [communities.get(node, 0) for node in graph.nodes]
    node_sizes = [60 + graph.degree(node) * 25 for node in graph.nodes]

    fig, ax = plt.subplots(figsize=(14, 10))

    nx.draw_networkx_edges(nx_graph, pos, ax=ax, alpha=0.25, width=0.8)
    nodes_drawn = nx.draw_networkx_nodes(
        nx_graph,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        cmap=plt.cm.tab10,
        linewidths=0.5,
        edgecolors="black",
    )

    influencer_set = set(influencers)
    influencer_pos = {n: p for n, p in pos.items() if n in influencer_set}
    if influencer_pos:
        xs = [p[0] for p in influencer_pos.values()]
        ys = [p[1] for p in influencer_pos.values()]
        ax.scatter(xs, ys, s=350, facecolors="none", edgecolors="red", linewidths=2.2,
                   label="Top influencers (degree centrality)")

    ax.set_title(
        f"Social Network Graph — {len(graph)} people, "
        f"{len(set(communities.values()))} communities (Union-Find), "
        f"{len(graph.edges)} edges",
        fontsize=13,
    )
    ax.legend(loc="lower left")
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

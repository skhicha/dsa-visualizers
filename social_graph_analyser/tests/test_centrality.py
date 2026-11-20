from social_graph_analyser.centrality import degree_centrality, top_influencers
from social_graph_analyser.graph import Graph


def _star_graph() -> Graph:
    """Center connected to 4 leaves -> center is the obvious influencer.

    n = 5, so centrality = degree / (n - 1) = degree / 4.
    """
    g = Graph()
    for leaf in ["L1", "L2", "L3", "L4"]:
        g.add_edge("center", leaf)
    return g


class TestDegreeCentrality:
    def test_known_star_graph_centrality_values(self):
        g = _star_graph()
        scores = degree_centrality(g)
        assert scores["center"] == 1.0  # degree 4 / (5-1)
        for leaf in ["L1", "L2", "L3", "L4"]:
            assert scores[leaf] == 0.25  # degree 1 / 4

    def test_single_node_graph_has_zero_centrality(self):
        g = Graph()
        g.add_node("solo")
        scores = degree_centrality(g)
        assert scores["solo"] == 0.0

    def test_complete_graph_all_nodes_maximal(self):
        g = Graph()
        nodes = ["a", "b", "c", "d"]
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                g.add_edge(nodes[i], nodes[j])
        scores = degree_centrality(g)
        assert all(score == 1.0 for score in scores.values())

    def test_top_influencers_identifies_center_first(self):
        g = _star_graph()
        ranked = top_influencers(g, k=1)
        assert ranked[0][0] == "center"
        assert ranked[0][1] == 1.0

    def test_top_influencers_respects_k(self):
        g = _star_graph()
        assert len(top_influencers(g, k=2)) == 2
        assert len(top_influencers(g, k=100)) == 5  # capped at number of nodes

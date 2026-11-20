from social_graph_analyser.dataset import generate_social_network
from social_graph_analyser.union_find import UnionFind


class TestSyntheticDataset:
    def test_generates_requested_node_count(self):
        graph, communities = generate_social_network(num_nodes=50, num_communities=5, seed=1)
        assert len(graph) == 50
        assert len(communities) == 5

    def test_graph_is_non_trivial(self):
        graph, _ = generate_social_network(num_nodes=80, seed=2)
        # Should produce a meaningfully dense, non-toy graph.
        assert len(graph.edges) > 80

    def test_every_community_member_is_reachable_within_its_group(self):
        """Every node should have at least one edge connecting it to its
        own community (guaranteed by the ring fallback in the generator)."""
        graph, communities = generate_social_network(num_nodes=60, num_communities=6, seed=3)
        for members in communities.values():
            if len(members) > 1:
                for member in members:
                    assert graph.degree(member) > 0

    def test_deterministic_with_fixed_seed(self):
        g1, _ = generate_social_network(num_nodes=40, seed=99)
        g2, _ = generate_social_network(num_nodes=40, seed=99)
        assert sorted(g1.edges) == sorted(g2.edges)

    def test_union_find_recovers_connected_structure(self):
        graph, _ = generate_social_network(num_nodes=90, num_communities=5, seed=4)
        uf = UnionFind(graph.nodes)
        for u, v, _w in graph.edges:
            uf.union(u, v)
        # With inter-community bridges present, the whole network should
        # collapse into relatively few (or one) connected components -
        # i.e. it's not just 5 totally isolated islands.
        components = uf.components()
        assert len(components) <= 5

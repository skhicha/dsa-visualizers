import pytest

from social_graph_analyser.graph import Graph


def _small_unweighted_graph() -> Graph:
    """
    A - B
    |   |
    C - D - E - F

    Edges: A-B, A-C, B-D, C-D, D-E, E-F
    """
    g = Graph()
    g.add_edge("A", "B")
    g.add_edge("A", "C")
    g.add_edge("B", "D")
    g.add_edge("C", "D")
    g.add_edge("D", "E")
    g.add_edge("E", "F")
    return g


def _weighted_graph() -> Graph:
    """Classic hand-worked weighted graph for Dijkstra verification.

    A-B:4  A-C:1  C-B:2  B-D:5  C-D:8  C-E:10  D-E:2

    Hand-computed shortest distances from A:
        A=0, C=1, B=3 (A-C-B), D=8 (A-C-B-D), E=10 (A-C-B-D-E)
    """
    g = Graph()
    g.add_edge("A", "B", 4)
    g.add_edge("A", "C", 1)
    g.add_edge("C", "B", 2)
    g.add_edge("B", "D", 5)
    g.add_edge("C", "D", 8)
    g.add_edge("C", "E", 10)
    g.add_edge("D", "E", 2)
    return g


class TestGraphBasics:
    def test_add_edge_is_undirected(self):
        g = Graph()
        g.add_edge("A", "B", 2.5)
        assert g.has_edge("A", "B")
        assert g.has_edge("B", "A")
        assert g.weight("A", "B") == 2.5
        assert g.weight("B", "A") == 2.5

    def test_degree(self):
        g = _small_unweighted_graph()
        assert g.degree("D") == 3  # B, C, E
        assert g.degree("A") == 2
        assert g.degree("F") == 1

    def test_rejects_self_loop(self):
        g = Graph()
        with pytest.raises(ValueError):
            g.add_edge("A", "A")

    def test_edges_listed_once(self):
        g = _small_unweighted_graph()
        assert len(g.edges) == 6


class TestBFS:
    def test_bfs_order_hand_verified(self):
        g = _small_unweighted_graph()
        # Hand-verified BFS order starting at A (sorted-neighbor expansion).
        assert g.bfs("A") == ["A", "B", "C", "D", "E", "F"]

    def test_bfs_reachable_set(self):
        g = _small_unweighted_graph()
        assert g.bfs_reachable("A") == {"A", "B", "C", "D", "E", "F"}

    def test_bfs_disconnected_component(self):
        g = _small_unweighted_graph()
        g.add_edge("X", "Y")
        assert g.bfs_reachable("X") == {"X", "Y"}
        assert "A" not in g.bfs_reachable("X")

    def test_bfs_unknown_source_raises(self):
        g = _small_unweighted_graph()
        with pytest.raises(KeyError):
            g.bfs("nonexistent")

    def test_bfs_visits_every_node_exactly_once(self):
        g = _small_unweighted_graph()
        order = g.bfs("A")
        assert len(order) == len(set(order)) == len(g)


class TestDFS:
    def test_dfs_order_hand_verified(self):
        g = _small_unweighted_graph()
        # Hand-verified iterative-stack DFS order starting at A.
        assert g.dfs("A") == ["A", "B", "D", "C", "E", "F"]

    def test_dfs_visits_every_node_exactly_once(self):
        g = _small_unweighted_graph()
        order = g.dfs("A")
        assert len(order) == len(set(order)) == len(g)

    def test_dfs_same_reachable_set_as_bfs(self):
        g = _small_unweighted_graph()
        assert set(g.dfs("A")) == g.bfs_reachable("A")

    def test_dfs_unknown_source_raises(self):
        g = _small_unweighted_graph()
        with pytest.raises(KeyError):
            g.dfs("nonexistent")


class TestDijkstra:
    def test_dijkstra_hand_verified_distances(self):
        g = _weighted_graph()
        dist, _prev = g.dijkstra("A")
        assert dist["A"] == 0
        assert dist["C"] == 1
        assert dist["B"] == 3
        assert dist["D"] == 8
        assert dist["E"] == 10

    def test_shortest_path_cost_and_route(self):
        g = _weighted_graph()
        cost, path = g.shortest_path("A", "E")
        assert cost == 10
        assert path == ["A", "C", "B", "D", "E"]

    def test_shortest_path_to_self_is_zero(self):
        g = _weighted_graph()
        cost, path = g.shortest_path("A", "A")
        assert cost == 0
        assert path == ["A"]

    def test_unreachable_target_returns_infinity_and_empty_path(self):
        g = _weighted_graph()
        g.add_edge("Z", "Y", 1)  # disconnected component
        cost, path = g.shortest_path("A", "Z")
        assert cost == float("inf")
        assert path == []

    def test_negative_weight_rejected(self):
        g = Graph()
        g.add_edge("A", "B", -3)
        with pytest.raises(ValueError):
            g.dijkstra("A")

    def test_dijkstra_matches_brute_force_on_random_graph(self):
        """Cross-check against a brute-force all-paths shortest-path search
        on a slightly bigger random graph, to catch bugs the hand-worked
        example might not expose."""
        import itertools
        import random

        rng = random.Random(123)
        g = Graph()
        nodes = [f"n{i}" for i in range(8)]
        for node in nodes:
            g.add_node(node)
        edges = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if rng.random() < 0.4:
                    w = rng.randint(1, 10)
                    g.add_edge(nodes[i], nodes[j], w)
                    edges.append((nodes[i], nodes[j], w))

        def brute_force_shortest(source, target):
            adj = {}
            for u, v, w in edges:
                adj.setdefault(u, []).append((v, w))
                adj.setdefault(v, []).append((u, w))

            best = [float("inf")]

            def dfs(node, visited, cost):
                if node == target:
                    best[0] = min(best[0], cost)
                    return
                if cost >= best[0]:
                    return
                for nbr, w in adj.get(node, []):
                    if nbr not in visited:
                        dfs(nbr, visited | {nbr}, cost + w)

            dfs(source, {source}, 0)
            return best[0]

        source = nodes[0]
        for target in nodes[1:]:
            expected = brute_force_shortest(source, target)
            dist, _ = g.dijkstra(source)
            assert dist[target] == expected

import math

from social_graph_analyser.union_find import UnionFind


class TestBasicUnionFind:
    def test_initially_each_node_is_its_own_component(self):
        uf = UnionFind(["a", "b", "c"])
        assert uf.find("a") == "a"
        assert not uf.connected("a", "b")

    def test_union_merges_components(self):
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        assert uf.connected("a", "b")
        assert not uf.connected("a", "c")
        uf.union("b", "c")
        assert uf.connected("a", "c")  # transitive
        assert not uf.connected("a", "d")

    def test_union_of_already_connected_is_noop(self):
        uf = UnionFind(["a", "b"])
        uf.union("a", "b")
        root_before = uf.find("a")
        uf.union("a", "b")
        assert uf.find("a") == root_before

    def test_components_grouping(self):
        uf = UnionFind(["a", "b", "c", "d", "e"])
        uf.union("a", "b")
        uf.union("c", "d")
        groups = uf.components()
        sizes = sorted(len(members) for members in groups.values())
        assert sizes == [1, 2, 2]  # {e}, {a,b}, {c,d}

    def test_community_detection_on_social_graph(self):
        """Two disjoint friend groups should resolve to exactly two
        communities, with a correct membership partition."""
        uf = UnionFind()
        group1 = ["alice", "bob", "carol"]
        group2 = ["dave", "erin"]
        for a, b in zip(group1, group1[1:]):
            uf.union(a, b)
        for a, b in zip(group2, group2[1:]):
            uf.union(a, b)

        assert uf.connected("alice", "carol")
        assert uf.connected("dave", "erin")
        assert not uf.connected("alice", "dave")

        groups = uf.components()
        membership = {frozenset(members) for members in groups.values()}
        assert frozenset(group1) in membership
        assert frozenset(group2) in membership


class TestPathCompression:
    def test_find_flattens_a_manually_built_deep_chain(self):
        """Build a deep chain a->b->c->d->e->f (f is root) by directly
        wiring the internal parent pointers (bypassing union-by-rank),
        then confirm a single find() call on the deepest node compresses
        every visited node directly onto the root.
        """
        uf = UnionFind(["a", "b", "c", "d", "e", "f"])
        uf._parent = {"a": "b", "b": "c", "c": "d", "d": "e", "e": "f", "f": "f"}

        root = uf.find("a")
        assert root == "f"
        # Path compression must re-point every node visited directly at
        # the root, flattening the chain to height 1 for all of them.
        for node in ["a", "b", "c", "d", "e"]:
            assert uf._parent[node] == "f"
            assert uf.tree_height(node) == 1

    def test_repeated_find_keeps_tree_flat(self):
        uf = UnionFind()
        chain = list(range(20))
        for a, b in zip(chain, chain[1:]):
            uf.union(a, b)
        for node in chain:
            uf.find(node)
        heights = [uf.tree_height(node) for node in chain]
        assert max(heights) <= 2


class TestUnionByRank:
    def test_rank_keeps_tree_height_logarithmic(self):
        """Union n singleton sets together pairwise ('tournament style')
        so union-by-rank is actually exercised, and verify the resulting
        tree height stays within the O(log n) bound guaranteed by
        union-by-rank (without path compression skewing the result,
        this is checked by inspecting the rank-driven structure directly).
        """
        n = 64
        uf = UnionFind(range(n))
        current = list(range(n))
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current) - 1, 2):
                uf.union(current[i], current[i + 1])
                next_level.append(current[i])
            if len(current) % 2 == 1:
                next_level.append(current[-1])
            current = next_level

        # With union-by-rank, merging n singletons pairwise in a balanced
        # tournament bounds every node's height by log2(n).
        max_height = max(uf.tree_height(node) for node in range(n))
        assert max_height <= math.ceil(math.log2(n))

    def test_rank_bound_holds_without_relying_on_compression(self):
        """Union by rank alone (independent of path compression) should
        prevent a long chain forming when merging in rank order."""
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")  # rank(a)=1 (or b), both rank 0 -> one becomes rank 1
        uf.union("c", "d")  # same, independent pair
        uf.union("a", "c")  # two rank-1 trees merge -> rank 2, height stays 2
        for node in ["a", "b", "c", "d"]:
            assert uf.tree_height(node) <= 2

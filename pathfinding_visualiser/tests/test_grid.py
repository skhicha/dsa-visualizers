import pytest

from pathfinding_visualiser.grid import Grid


class TestGrid:
    def test_in_bounds(self):
        g = Grid(5, 5)
        assert g.in_bounds((0, 0))
        assert g.in_bounds((4, 4))
        assert not g.in_bounds((5, 0))
        assert not g.in_bounds((-1, 0))

    def test_wall_toggle(self):
        g = Grid(3, 3)
        assert not g.is_wall((1, 1))
        g.set_wall((1, 1))
        assert g.is_wall((1, 1))
        g.set_wall((1, 1), False)
        assert not g.is_wall((1, 1))

    def test_set_wall_out_of_bounds_raises(self):
        g = Grid(3, 3)
        with pytest.raises(ValueError):
            g.set_wall((10, 10))

    def test_neighbors_excludes_walls_and_out_of_bounds(self):
        g = Grid(3, 3)
        g.set_wall((0, 1))
        neighbors = g.neighbors((0, 0))
        assert (0, 1) not in neighbors  # wall
        assert (1, 0) in neighbors
        assert len(neighbors) == 1  # corner cell with one wall neighbor

    def test_manhattan_distance(self):
        assert Grid.manhattan_distance((0, 0), (3, 4)) == 7
        assert Grid.manhattan_distance((2, 2), (2, 2)) == 0

    def test_rejects_non_positive_dimensions(self):
        with pytest.raises(ValueError):
            Grid(0, 5)
        with pytest.raises(ValueError):
            Grid(5, -1)

    def test_1x1_grid(self):
        g = Grid(1, 1)
        assert g.all_cells() == [(0, 0)]
        assert g.neighbors((0, 0)) == []

import pytest

from pathfinding_visualiser import maze
from pathfinding_visualiser.grid import Grid
from pathfinding_visualiser.solvers import a_star, dijkstra, is_valid_path


def _test_grids():
    """A representative set of maze/wall configurations to run the
    correctness + efficiency comparisons over."""
    rows, cols = 20, 20
    return [
        ("open grid", (Grid(rows, cols), (0, 0), (rows - 1, cols - 1))),
        ("random walls 20%", maze.random_walls(rows, cols, wall_prob=0.20, seed=1)),
        ("random walls 30%", maze.random_walls(rows, cols, wall_prob=0.30, seed=2)),
        ("random walls 40%", maze.random_walls(rows, cols, wall_prob=0.40, seed=3)),
        ("zigzag corridors", maze.zigzag_corridors(rows, cols)),
        ("spiral maze", maze.spiral_maze(rows, cols)),
        ("diagonal barrier", maze.diagonal_barrier(rows, cols)),
    ]


class TestCorrectnessParity:
    @pytest.mark.parametrize("name,config", _test_grids())
    def test_a_star_and_dijkstra_agree_on_cost(self, name, config):
        grid, start, end = config
        a_result = a_star(grid, start, end)
        d_result = dijkstra(grid, start, end)
        assert a_result.found == d_result.found, f"{name}: found mismatch"
        if a_result.found:
            assert a_result.cost == d_result.cost, f"{name}: cost mismatch"


class TestEfficiencyClaim:
    @pytest.mark.parametrize("name,config", _test_grids())
    def test_a_star_visits_fewer_or_equal_nodes(self, name, config):
        grid, start, end = config
        a_result = a_star(grid, start, end)
        d_result = dijkstra(grid, start, end)
        assert a_result.visited_count <= d_result.visited_count, (
            f"{name}: A* visited {a_result.visited_count}, "
            f"Dijkstra visited {d_result.visited_count}"
        )

    def test_a_star_strictly_fewer_with_obstacles_forcing_a_detour(self):
        """A plain open grid from corner to corner is a known degenerate
        case for the Manhattan heuristic: every cell inside the bounding
        rectangle lies on *some* shortest monotone path, so it ties with
        the optimal f-score and A* ends up exploring the same footprint
        as Dijkstra (see test_a_star_matches_dijkstra_on_open_grid below).
        A*'s advantage shows up once obstacles force real detours, which
        is exactly what the maze-based tests above already demonstrate;
        here we isolate a minimal obstacle to show the effect directly."""
        grid = Grid(30, 30)
        # A wall spanning most of the middle row forces a detour around
        # one end, breaking the heuristic-tie degeneracy of an open grid.
        for c in range(0, 25):
            grid.set_wall((15, c))
        a_result = a_star(grid, (0, 0), (29, 29))
        d_result = dijkstra(grid, (0, 0), (29, 29))
        assert a_result.found and d_result.found
        assert a_result.cost == d_result.cost
        assert a_result.visited_count < d_result.visited_count

    def test_a_star_matches_dijkstra_on_fully_open_grid(self):
        """Documents the degenerate case: with no obstacles at all and
        start/end at opposite corners, the Manhattan heuristic is exact
        along every monotone path, so A* and Dijkstra expand the same
        number of cells. This is expected A* behavior, not a bug."""
        grid = Grid(30, 30)
        a_result = a_star(grid, (0, 0), (29, 29))
        d_result = dijkstra(grid, (0, 0), (29, 29))
        assert a_result.visited_count == d_result.visited_count


class TestPathReconstruction:
    @pytest.mark.parametrize("name,config", _test_grids())
    def test_a_star_path_is_valid(self, name, config):
        grid, start, end = config
        result = a_star(grid, start, end)
        if result.found:
            assert is_valid_path(grid, result.path, start, end), f"{name}: invalid A* path"
            assert len(result.path) - 1 == result.cost

    @pytest.mark.parametrize("name,config", _test_grids())
    def test_dijkstra_path_is_valid(self, name, config):
        grid, start, end = config
        result = dijkstra(grid, start, end)
        if result.found:
            assert is_valid_path(grid, result.path, start, end), f"{name}: invalid Dijkstra path"
            assert len(result.path) - 1 == result.cost


class TestTieBreakingDeterminism:
    def test_repeated_runs_produce_identical_path(self):
        grid = Grid(15, 15)  # many equal-cost shortest paths exist
        start, end = (0, 0), (14, 14)
        paths = [a_star(grid, start, end).path for _ in range(5)]
        assert all(p == paths[0] for p in paths)

    def test_repeated_runs_produce_identical_visited_order(self):
        grid = Grid(15, 15)
        start, end = (0, 0), (14, 14)
        orders = [tuple(dijkstra(grid, start, end).visited_order) for _ in range(5)]
        assert all(o == orders[0] for o in orders)


class TestEdgeCases:
    def test_start_equals_end(self):
        grid = Grid(10, 10)
        result = a_star(grid, (3, 3), (3, 3))
        assert result.found
        assert result.path == [(3, 3)]
        assert result.cost == 0

    def test_fully_walled_off_target_reports_no_path_not_crash(self):
        grid = Grid(10, 10)
        end = (5, 5)
        # Wall off every 4-directional neighbor of end so it is unreachable.
        for r, c in [(4, 5), (6, 5), (5, 4), (5, 6)]:
            grid.set_wall((r, c))
        result = a_star(grid, (0, 0), end)
        assert result.found is False
        assert result.path == []
        assert result.cost == float("inf")

        result_d = dijkstra(grid, (0, 0), end)
        assert result_d.found is False

    def test_1x1_grid_start_equals_end(self):
        grid = Grid(1, 1)
        result = a_star(grid, (0, 0), (0, 0))
        assert result.found
        assert result.path == [(0, 0)]

    def test_start_is_a_wall_handled_gracefully(self):
        grid = Grid(5, 5)
        grid.set_wall((0, 0))
        result = a_star(grid, (0, 0), (4, 4))
        assert result.found is False  # no crash, just reported as unsolvable

    def test_end_is_a_wall_handled_gracefully(self):
        grid = Grid(5, 5)
        grid.set_wall((4, 4))
        result = dijkstra(grid, (0, 0), (4, 4))
        assert result.found is False

    def test_out_of_bounds_endpoints_handled_gracefully(self):
        grid = Grid(5, 5)
        result = a_star(grid, (0, 0), (99, 99))
        assert result.found is False

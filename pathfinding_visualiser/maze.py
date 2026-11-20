"""Maze/wall-configuration generators used both by tests and by the
empirical A* vs Dijkstra comparison script.

Each generator returns (grid, start, end) and is guaranteed to produce a
*solvable* configuration (start and end connected) - random generators
retry with a different seed internally if a particular draw happens to
seal the target off, so callers never have to special-case "no path"
just because of unlucky randomness.
"""

from __future__ import annotations

import random
from typing import Tuple

from .grid import Cell, Grid


def _bfs_reachable(grid: Grid, start: Cell) -> set:
    """Plain BFS reachability check used only to validate that a generated
    maze is solvable - not part of the A*/Dijkstra comparison itself."""
    from collections import deque

    seen = {start}
    queue = deque([start])
    while queue:
        cur = queue.popleft()
        for nbr in grid.neighbors(cur):
            if nbr not in seen:
                seen.add(nbr)
                queue.append(nbr)
    return seen


def random_walls(rows: int, cols: int, wall_prob: float = 0.28, seed: int = 0) -> Tuple[Grid, Cell, Cell]:
    """A grid with randomly scattered walls. Start/end are kept open.
    Retries with incrementing seeds if the draw happens to disconnect
    start from end, so this always returns a solvable maze."""
    start, end = (0, 0), (rows - 1, cols - 1)
    for attempt in range(200):
        rng = random.Random(seed + attempt)
        grid = Grid(rows, cols)
        for cell in grid.all_cells():
            if cell in (start, end):
                continue
            if rng.random() < wall_prob:
                grid.set_wall(cell)
        if end in _bfs_reachable(grid, start):
            return grid, start, end
    raise RuntimeError("could not generate a solvable random maze")


def zigzag_corridors(rows: int, cols: int) -> Tuple[Grid, Cell, Cell]:
    """Horizontal wall 'combs' with a gap each, alternating sides and
    leaving a wider open buffer between combs, forcing any solver to
    snake back and forth while still leaving enough open space around
    each comb for the two algorithms' exploration patterns to diverge."""
    grid = Grid(rows, cols)
    start, end = (0, 0), (rows - 1, cols - 1)
    gap_width = max(2, cols // 6)
    wall_rows = range(3, rows - 3, 5)
    for i, r in enumerate(wall_rows):
        if i % 2 == 0:
            gap_start = cols - gap_width
        else:
            gap_start = 0
        gap_cols = set(range(gap_start, gap_start + gap_width))
        for c in range(cols):
            if c not in gap_cols:
                grid.set_wall((r, c))
    grid.set_wall(start, False)
    grid.set_wall(end, False)
    return grid, start, end


def spiral_maze(rows: int, cols: int) -> Tuple[Grid, Cell, Cell]:
    """Concentric square 'walls' with single gaps, forming a spiral
    corridor from the outside in toward the center."""
    grid = Grid(rows, cols)
    start = (0, 0)
    end = (rows // 2, cols // 2)

    layer = 0
    while layer * 2 < min(rows, cols) - 2:
        top, bottom = layer, rows - 1 - layer
        left, right = layer, cols - 1 - layer
        if top >= bottom or left >= right:
            break
        for c in range(left, right + 1):
            grid.set_wall((top + 1, c))
        # leave a gap for the spiral corridor, alternating side per layer
        gap_col = right if layer % 2 == 0 else left
        grid.set_wall((top + 1, gap_col), False)
        layer += 1

    grid.set_wall(start, False)
    grid.set_wall(end, False)
    return grid, start, end


def diagonal_barrier(rows: int, cols: int) -> Tuple[Grid, Cell, Cell]:
    """A full-width wall across the middle of the grid with one narrow
    gap offset toward the far side (away from the direct start-end
    line), forcing a real detour: Dijkstra fans out evenly in all
    directions to find the gap, while A*'s heuristic keeps steering the
    search toward the end, visiting far fewer of the "wrong side" cells.
    """
    grid = Grid(rows, cols)
    start, end = (0, 0), (rows - 1, cols - 1)
    wall_row = rows // 2
    gap_width = max(2, cols // 10)
    gap_start = cols - gap_width - 2  # offset toward the end's side, not centered

    for c in range(cols):
        if not (gap_start <= c < gap_start + gap_width):
            grid.set_wall((wall_row, c))

    grid.set_wall(start, False)
    grid.set_wall(end, False)

    if end not in _bfs_reachable(grid, start):
        # Extremely defensive fallback - widen the gap if it somehow
        # still seals the two halves off for a given grid size.
        for c in range(cols):
            grid.set_wall((wall_row, c), False)
    return grid, start, end

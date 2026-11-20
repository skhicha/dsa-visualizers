"""Grid data structure shared by the A* and Dijkstra solvers and the
Pygame UI. A cell is a plain (row, col) tuple; the grid tracks which
cells are walls, and exposes Manhattan-adjacent (4-directional) neighbors.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple

Cell = Tuple[int, int]


class Grid:
    def __init__(self, rows: int, cols: int) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive")
        self.rows = rows
        self.cols = cols
        self.walls: Set[Cell] = set()

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_wall(self, cell: Cell) -> bool:
        return cell in self.walls

    def set_wall(self, cell: Cell, is_wall: bool = True) -> None:
        if not self.in_bounds(cell):
            raise ValueError(f"cell {cell} out of bounds")
        if is_wall:
            self.walls.add(cell)
        else:
            self.walls.discard(cell)

    def neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [
            n for n in candidates
            if self.in_bounds(n) and not self.is_wall(n)
        ]

    @staticmethod
    def manhattan_distance(a: Cell, b: Cell) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def all_cells(self) -> List[Cell]:
        return [(r, c) for r in range(self.rows) for c in range(self.cols)]

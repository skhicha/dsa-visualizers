"""
A* and Dijkstra pathfinders implemented from scratch for the grid
visualiser, written to be directly comparable:

- Both use a binary heap (heapq) as the priority queue.
- Both push (priority, tie_breaker, cell) tuples, where tie_breaker is a
  strictly increasing counter. This makes heap ordering fully
  deterministic when two entries have equal priority (equal f-scores for
  A*, equal distances for Dijkstra) - without it, heapq would fall back
  to comparing the `cell` tuples themselves, which happens to also be
  deterministic for (row, col) cells but is not guaranteed for arbitrary
  payloads, and which obscures *why* ties are broken a specific way.
- Both maintain a parallel `open_set` (a plain Python set) alongside the
  heap purely so "is this cell currently on the frontier" is an O(1)
  membership test, instead of an O(n) scan of heap contents.
- A* additionally tracks g-score (cost from start) and uses
  f(n) = g(n) + h(n) with the Manhattan heuristic (admissible: it never
  overestimates the true remaining 4-directional grid distance).
- Dijkstra is the same algorithm with h(n) = 0 everywhere, i.e. it ranks
  purely by accumulated cost g(n) - this is the standard relationship
  between the two algorithms and is exactly what lets us compare them
  fairly: identical heap/open-set/tie-break machinery, differing only in
  whether a heuristic steers the search toward the goal.

Grid edges all have uniform weight 1 (4-directional moves).
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .grid import Cell, Grid


@dataclass
class SolveResult:
    found: bool
    path: List[Cell] = field(default_factory=list)
    cost: float = float("inf")
    visited_order: List[Cell] = field(default_factory=list)  # cells expanded (closed), in order
    visited_count: int = 0


def _reconstruct_path(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def _validate_endpoints(grid: Grid, start: Cell, end: Cell) -> Optional[SolveResult]:
    """Shared edge-case handling for both solvers.

    Returns a SolveResult if the search should short-circuit (invalid
    endpoints, trivial start==end case), or None if the caller should
    proceed with a normal search.
    """
    if not grid.in_bounds(start) or not grid.in_bounds(end):
        return SolveResult(found=False)
    if grid.is_wall(start) or grid.is_wall(end):
        # A wall can never be a valid start/end - report "no path" rather
        # than silently searching through a cell that shouldn't be
        # traversable, and never crash.
        return SolveResult(found=False)
    if start == end:
        return SolveResult(found=True, path=[start], cost=0, visited_order=[start], visited_count=1)
    return None


def a_star(grid: Grid, start: Cell, end: Cell) -> SolveResult:
    shortcut = _validate_endpoints(grid, start, end)
    if shortcut is not None:
        return shortcut

    counter = itertools.count()
    open_set: set = {start}
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0}

    heap: List[tuple] = []
    h_start = Grid.manhattan_distance(start, end)
    heapq.heappush(heap, (h_start, next(counter), start))

    closed: set = set()
    visited_order: List[Cell] = []

    while heap:
        _f, _tie, current = heapq.heappop(heap)
        open_set.discard(current)

        if current in closed:
            continue
        closed.add(current)
        visited_order.append(current)

        if current == end:
            path = _reconstruct_path(came_from, current)
            return SolveResult(
                found=True,
                path=path,
                cost=g_score[current],
                visited_order=visited_order,
                visited_count=len(visited_order),
            )

        for neighbor in grid.neighbors(current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + Grid.manhattan_distance(neighbor, end)
                heapq.heappush(heap, (f_score, next(counter), neighbor))
                open_set.add(neighbor)

    return SolveResult(found=False, visited_order=visited_order, visited_count=len(visited_order))


def dijkstra(grid: Grid, start: Cell, end: Cell) -> SolveResult:
    shortcut = _validate_endpoints(grid, start, end)
    if shortcut is not None:
        return shortcut

    counter = itertools.count()
    open_set: set = {start}
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0}

    heap: List[tuple] = []
    heapq.heappush(heap, (0, next(counter), start))

    closed: set = set()
    visited_order: List[Cell] = []

    while heap:
        _dist, _tie, current = heapq.heappop(heap)
        open_set.discard(current)

        if current in closed:
            continue
        closed.add(current)
        visited_order.append(current)

        if current == end:
            path = _reconstruct_path(came_from, current)
            return SolveResult(
                found=True,
                path=path,
                cost=g_score[current],
                visited_order=visited_order,
                visited_count=len(visited_order),
            )

        for neighbor in grid.neighbors(current):
            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                heapq.heappush(heap, (tentative_g, next(counter), neighbor))
                open_set.add(neighbor)

    return SolveResult(found=False, visited_order=visited_order, visited_count=len(visited_order))


def is_valid_path(grid: Grid, path: List[Cell], start: Cell, end: Cell) -> bool:
    """Validate that `path` is a genuine connected start-to-end route:
    starts at start, ends at end, every consecutive pair is 4-adjacent,
    and no cell in the path is a wall."""
    if not path:
        return False
    if path[0] != start or path[-1] != end:
        return False
    for cell in path:
        if grid.is_wall(cell):
            return False
    for a, b in zip(path, path[1:]):
        if Grid.manhattan_distance(a, b) != 1:
            return False
    return True

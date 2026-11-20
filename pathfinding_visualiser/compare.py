"""
Empirical A* vs Dijkstra comparison: runs both solvers on several maze
configurations and reports path cost (should match) and nodes visited
(A* should visit fewer-or-equal, demonstrating the heuristic's benefit).

Run with:
    python -m pathfinding_visualiser.compare
"""

from __future__ import annotations

from . import maze
from .solvers import a_star, dijkstra


def run_comparison(rows: int = 30, cols: int = 30) -> list[dict]:
    configs = [
        ("Random walls (28% density)", maze.random_walls(rows, cols, wall_prob=0.28, seed=1)),
        ("Random walls (35% density)", maze.random_walls(rows, cols, wall_prob=0.35, seed=7)),
        ("Zigzag corridors", maze.zigzag_corridors(rows, cols)),
        ("Spiral maze", maze.spiral_maze(rows, cols)),
        ("Diagonal barrier w/ gap", maze.diagonal_barrier(rows, cols)),
    ]

    results = []
    for name, (grid, start, end) in configs:
        a_result = a_star(grid, start, end)
        d_result = dijkstra(grid, start, end)
        row = {
            "name": name,
            "found": a_result.found,
            "cost": a_result.cost,
            "a_star_visited": a_result.visited_count,
            "dijkstra_visited": d_result.visited_count,
        }
        results.append(row)
    return results


def print_report(results: list[dict]) -> None:
    header = f"{'Maze':<30} {'Path cost':>10} {'A* visited':>12} {'Dijkstra visited':>18} {'A* saves':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if not r["found"]:
            print(f"{r['name']:<30} {'NO PATH':>10}")
            continue
        saved = r["dijkstra_visited"] - r["a_star_visited"]
        pct = (saved / r["dijkstra_visited"] * 100) if r["dijkstra_visited"] else 0
        print(
            f"{r['name']:<30} {r['cost']:>10} {r['a_star_visited']:>12} "
            f"{r['dijkstra_visited']:>18} {saved:>6} ({pct:4.1f}%)"
        )


if __name__ == "__main__":
    print_report(run_comparison())

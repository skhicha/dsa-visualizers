"""
Interactive Pygame grid visualiser: draw walls, set start/end, press SPACE
to watch A* (left panel) and Dijkstra (right panel) explore the same grid
side by side, then see each one's reconstructed path drawn in green.

Controls:
    Left click + drag   - toggle walls
    S + left click      - set start cell
    E + left click      - set end cell
    SPACE               - run A* and Dijkstra on the current grid
    C                   - clear all walls
    R                   - reset entire grid (new empty grid)
    ESC / window close  - quit

Run with:
    python -m pathfinding_visualiser.app
"""

from __future__ import annotations

import sys
from typing import Optional, Tuple

import pygame

from .grid import Cell, Grid
from .solvers import SolveResult, a_star, dijkstra

CELL_SIZE = 18
GRID_MARGIN = 30
PANEL_GAP = 40
HEADER_HEIGHT = 40

COLOR_BG = (30, 30, 30)
COLOR_GRID_LINE = (60, 60, 60)
COLOR_WALL = (20, 20, 20)
COLOR_START = (0, 200, 0)
COLOR_END = (220, 30, 30)
COLOR_VISITED_ASTAR = (90, 140, 255)
COLOR_VISITED_DIJKSTRA = (255, 170, 90)
COLOR_PATH = (255, 255, 0)
COLOR_EMPTY = (245, 245, 245)
COLOR_TEXT = (230, 230, 230)


class GridPanel:
    """Renders one grid (walls + start/end + a solve result) at a given
    pixel origin on the shared screen surface."""

    def __init__(self, origin: Tuple[int, int], rows: int, cols: int, label: str,
                 visited_color: Tuple[int, int, int]):
        self.origin = origin
        self.rows = rows
        self.cols = cols
        self.label = label
        self.visited_color = visited_color

    def pixel_width(self) -> int:
        return self.cols * CELL_SIZE

    def pixel_height(self) -> int:
        return self.rows * CELL_SIZE

    def cell_rect(self, cell: Cell) -> pygame.Rect:
        r, c = cell
        ox, oy = self.origin
        return pygame.Rect(ox + c * CELL_SIZE, oy + r * CELL_SIZE, CELL_SIZE, CELL_SIZE)

    def cell_at_pixel(self, pos: Tuple[int, int]) -> Optional[Cell]:
        x, y = pos
        ox, oy = self.origin
        if not (ox <= x < ox + self.pixel_width() and oy <= y < oy + self.pixel_height()):
            return None
        col = (x - ox) // CELL_SIZE
        row = (y - oy) // CELL_SIZE
        return (int(row), int(col))

    def draw(self, surface: pygame.Surface, font: pygame.font.Font, grid: Grid,
              start: Cell, end: Cell, result: Optional[SolveResult]) -> None:
        ox, oy = self.origin

        for r in range(self.rows):
            for c in range(self.cols):
                cell = (r, c)
                color = COLOR_EMPTY
                if grid.is_wall(cell):
                    color = COLOR_WALL
                pygame.draw.rect(surface, color, self.cell_rect(cell))

        if result is not None:
            for cell in result.visited_order:
                if cell not in (start, end):
                    pygame.draw.rect(surface, self.visited_color, self.cell_rect(cell))
            if result.found:
                for cell in result.path:
                    if cell not in (start, end):
                        pygame.draw.rect(surface, COLOR_PATH, self.cell_rect(cell))

        pygame.draw.rect(surface, COLOR_START, self.cell_rect(start))
        pygame.draw.rect(surface, COLOR_END, self.cell_rect(end))

        for r in range(self.rows + 1):
            y = oy + r * CELL_SIZE
            pygame.draw.line(surface, COLOR_GRID_LINE, (ox, y), (ox + self.pixel_width(), y))
        for c in range(self.cols + 1):
            x = ox + c * CELL_SIZE
            pygame.draw.line(surface, COLOR_GRID_LINE, (x, oy), (x, oy + self.pixel_height()))

        label_text = self.label
        if result is not None and result.found:
            label_text += f"  (cost={result.cost:g}, visited={result.visited_count})"
        elif result is not None:
            label_text += "  (no path found)"
        text_surf = font.render(label_text, True, COLOR_TEXT)
        surface.blit(text_surf, (ox, oy - HEADER_HEIGHT + 8))


class App:
    def __init__(self, rows: int = 25, cols: int = 25) -> None:
        pygame.init()
        pygame.font.init()

        self.rows = rows
        self.cols = cols
        self.grid = Grid(rows, cols)
        self.start: Cell = (0, 0)
        self.end: Cell = (rows - 1, cols - 1)

        panel_width = cols * CELL_SIZE
        panel_height = rows * CELL_SIZE
        total_width = GRID_MARGIN * 2 + panel_width * 2 + PANEL_GAP
        total_height = GRID_MARGIN * 2 + HEADER_HEIGHT + panel_height

        self.screen = pygame.display.set_mode((total_width, total_height))
        pygame.display.set_caption("A* vs Dijkstra Pathfinding Visualiser")
        self.font = pygame.font.SysFont("monospace", 16)
        self.clock = pygame.time.Clock()

        left_origin = (GRID_MARGIN, GRID_MARGIN + HEADER_HEIGHT)
        right_origin = (GRID_MARGIN + panel_width + PANEL_GAP, GRID_MARGIN + HEADER_HEIGHT)
        self.left_panel = GridPanel(left_origin, rows, cols, "A* search", COLOR_VISITED_ASTAR)
        self.right_panel = GridPanel(right_origin, rows, cols, "Dijkstra", COLOR_VISITED_DIJKSTRA)

        self.result_a: Optional[SolveResult] = None
        self.result_d: Optional[SolveResult] = None

    def solve(self) -> None:
        self.result_a = a_star(self.grid, self.start, self.end)
        self.result_d = dijkstra(self.grid, self.start, self.end)

    def clear_walls(self) -> None:
        self.grid.walls.clear()
        self.result_a = None
        self.result_d = None

    def reset(self) -> None:
        self.grid = Grid(self.rows, self.cols)
        self.start = (0, 0)
        self.end = (self.rows - 1, self.cols - 1)
        self.result_a = None
        self.result_d = None

    def _cell_from_mouse(self, pos: Tuple[int, int]) -> Optional[Cell]:
        return self.left_panel.cell_at_pixel(pos) or self.right_panel.cell_at_pixel(pos)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            if event.key == pygame.K_SPACE:
                self.solve()
            elif event.key == pygame.K_c:
                self.clear_walls()
            elif event.key == pygame.K_r:
                self.reset()
        if event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            buttons = pygame.mouse.get_pressed()
            if buttons[0]:
                cell = self._cell_from_mouse(pygame.mouse.get_pos())
                if cell is not None and self.grid.in_bounds(cell):
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_SHIFT:
                        self.start = cell
                    elif mods & pygame.KMOD_CTRL:
                        self.end = cell
                    elif cell not in (self.start, self.end):
                        self.grid.set_wall(cell, True)
        return True

    def draw(self) -> None:
        self.screen.fill(COLOR_BG)
        self.left_panel.draw(self.screen, self.font, self.grid, self.start, self.end, self.result_a)
        self.right_panel.draw(self.screen, self.font, self.grid, self.start, self.end, self.result_d)
        pygame.display.flip()

    def run(self) -> None:
        running = True
        while running:
            for event in pygame.event.get():
                running = self.handle_event(event)
                if not running:
                    break
            self.draw()
            self.clock.tick(30)
        pygame.quit()


def run_headless_demo(rows: int = 20, cols: int = 20, maze_fn=None, frames: int = 5):
    """Build the full Pygame app (using whatever SDL video driver the
    calling environment has configured, e.g. SDL_VIDEODRIVER=dummy for
    headless CI), load an optional maze, run a full A*-vs-Dijkstra solve,
    and render a few frames - to prove the real Pygame rendering pipeline
    initializes and completes a solve loop without crashing.

    Returns (result_a, result_d).
    """
    app = App(rows, cols)
    if maze_fn is not None:
        grid, start, end = maze_fn(rows, cols)
        app.grid, app.start, app.end = grid, start, end

    app.solve()
    for _ in range(frames):
        app.draw()
        app.clock.tick(30)
    pygame.quit()
    return app.result_a, app.result_d


if __name__ == "__main__":
    App().run()
    sys.exit(0)

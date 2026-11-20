# DSA Visualizers

Two from-scratch graph/pathfinding algorithm implementations with live visualizations:

1. **`social_graph_analyser/`** — weighted graph, BFS/DFS/Dijkstra, and Union-Find community detection on a synthetic social network, rendered with NetworkX + Matplotlib.
2. **`pathfinding_visualiser/`** — an interactive Pygame grid visualiser comparing A* against Dijkstra, with an empirical node-visit comparison across several mazes.

Both modules implement every graph/search algorithm by hand (adjacency lists, binary heaps, disjoint-set forests) — libraries like NetworkX, Matplotlib and Pygame are used only for layout, rendering and the UI loop, never for the algorithms themselves.

## 1. Social Graph Analyser

`social_graph_analyser/graph.py` is a weighted undirected graph built on a plain adjacency-list dict (`{node: {neighbor: weight}}`), with:

- **BFS** — iterative, queue-based, `O(V + E)`.
- **DFS** — iterative using an explicit stack (no recursion), `O(V + E)`.
- **Dijkstra's shortest path** — binary-heap based, `O((V + E) log V)`, rejects negative weights.

`social_graph_analyser/union_find.py` implements **Union-Find (DSU)** with:
- **Path compression** — every node visited during `find()` is re-pointed directly at the root.
- **Union-by-rank** — the shorter tree is always attached under the taller one.
- Combined, this gives `O(α(n))` amortized find/union, used here for **community detection**.

`social_graph_analyser/centrality.py` computes **degree centrality** (`degree(v) / (n-1)`) to rank "influencers."

`social_graph_analyser/dataset.py` generates a synthetic social network (default: 100 people, 6 communities) with realistic clustering — dense intra-community edges plus optional sparse inter-community "weak tie" edges.

### Running it

```bash
pip install -r requirements.txt
python -m social_graph_analyser.main
```

This generates the dataset, runs BFS/DFS/Dijkstra/Union-Find/centrality, prints a report, and writes `social_graph_analyser/output/social_network.png` (NetworkX spring layout + Matplotlib rendering, nodes colored by Union-Find community, sized by degree, top 5 influencers circled in red).

Sample output from an actual run (100 nodes, 6 communities, no cross-community bridges so Union-Find's components line up exactly with the generated groups):

```
Generated social network: 100 people, 340 friendships
BFS from person_0 visited 17 people (reachable component size)
DFS from person_0 visited 17 people
Union-Find detected 6 connected communities
  community 0: 17 members
  community 1: 17 members
  community 2: 17 members
  community 3: 17 members
  community 4: 16 members
  community 5: 16 members
Top 5 influencers by degree centrality:
  person_36: degree=10, centrality=0.101
  ...
Saved visualization to social_graph_analyser/output/social_network.png
```

## 2. Pathfinding Visualiser

`pathfinding_visualiser/solvers.py` implements **A\*** and **Dijkstra** side by side, built to be directly comparable:

- Both use a **binary heap** (`heapq`) as the priority queue.
- Both push `(priority, tie_breaker, cell)` tuples, where `tie_breaker` is a strictly increasing counter — this makes heap ordering **fully deterministic** whenever two entries have equal priority.
- Both maintain a **parallel hash set** (`open_set`) alongside the heap, giving **O(1) open-set membership testing** instead of scanning heap contents.
- A* ranks cells by `f(n) = g(n) + h(n)` with the **Manhattan heuristic** `h` (admissible — it never overestimates true 4-directional grid distance). Dijkstra is the same algorithm with `h(n) = 0`, i.e. it ranks purely by accumulated cost.
- `came_from` dict reconstructs the path once the end cell is popped.

`pathfinding_visualiser/grid.py` is the grid model (walls, 4-directional neighbors, Manhattan distance). `pathfinding_visualiser/maze.py` generates several wall configurations (random walls, zigzag corridors, a spiral, and a funnel-gap barrier), all internally verified solvable via BFS before being handed to the solvers.

### Running the interactive visualiser

```bash
pip install -r requirements.txt
python -m pathfinding_visualiser.app
```

Controls: left-click + drag to draw walls, `Shift` + click to move the start, `Ctrl` + click to move the end, `SPACE` to run A* (left panel) and Dijkstra (right panel) side by side, `C` to clear walls, `R` to reset, `Esc` to quit.

### Empirical A* vs Dijkstra comparison

```bash
python -m pathfinding_visualiser.compare
```

Real numbers from an actual run on 30x30 grids (`path cost` is identical between the two algorithms on every maze — confirming correctness parity — while `visited` shows how many cells each algorithm actually expanded before finding the end):

| Maze                        | Path cost | A* visited | Dijkstra visited | A* saves        |
|------------------------------|----------:|-----------:|------------------:|------------------|
| Random walls (28% density)   | 58        | 398         | 624               | 226 (36.2%)      |
| Random walls (35% density)   | 60        | 276         | 558               | 282 (50.5%)      |
| Zigzag corridors             | 142       | 589         | 659               | 70 (10.6%)       |
| Spiral maze                  | 58        | 143         | 248               | 105 (42.3%)      |
| Funnel-gap barrier           | 58        | 523         | 657               | 134 (20.4%)      |

A* consistently reaches the same optimal path cost while expanding 10-50% fewer cells than Dijkstra, because the Manhattan heuristic steers it toward the goal instead of fanning out uniformly in every direction.

One deliberate exception is documented and covered by a test (`test_a_star_matches_dijkstra_on_fully_open_grid`): on a *fully open* grid with start/end at opposite corners, the Manhattan heuristic is exact along every monotone path, so every reachable cell ties on `f`-score and A* ends up expanding exactly as many cells as Dijkstra — the heuristic only earns its keep once obstacles exist to steer around.

## Tests

```bash
pip install -r requirements.txt
python -m pytest -v
```

83 tests, all passing, covering:

- **Graph module**: hand-verified BFS/DFS traversal order on a known graph, Dijkstra distances cross-checked against both a hand-worked example and a brute-force all-paths search, Union-Find correctness (union/find, path-compression flattening verified by inspecting the internal parent pointers, union-by-rank height bounds), degree centrality on known graphs (star graph, complete graph), and dataset generation sanity checks.
- **Pathfinding module**: A*/Dijkstra cost parity and node-visit-count comparison across 7 maze configurations, path-reconstruction validity, tie-breaking determinism across repeated runs, and edge cases (`start == end`, fully walled-off target, 1x1 grid, start/end on a wall, out-of-bounds endpoints) — all handled without crashing.

The Pygame visualiser was also run end-to-end under `SDL_VIDEODRIVER=dummy` (headless, no real display) across 5 different maze configurations, each completing a full A*-and-Dijkstra solve-and-render loop without error. The social graph pipeline was run end-to-end (generate → BFS/DFS/Dijkstra/Union-Find/centrality → render), producing a non-trivial 2100x1500 PNG.

## Project layout

```
social_graph_analyser/
  graph.py         weighted undirected graph: BFS, DFS, Dijkstra (from scratch)
  union_find.py     Union-Find with path compression + union-by-rank
  centrality.py     degree centrality / top influencers
  dataset.py        synthetic social network generator
  visualize.py       NetworkX layout + Matplotlib rendering
  main.py            end-to-end pipeline
  tests/

pathfinding_visualiser/
  grid.py            grid model (walls, neighbors, Manhattan distance)
  solvers.py         A* and Dijkstra (from scratch)
  maze.py            maze/wall-configuration generators
  app.py             interactive Pygame visualiser
  compare.py         empirical A* vs Dijkstra comparison script
  tests/
```

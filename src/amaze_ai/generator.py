from collections import deque
import random
from typing import List, Tuple, Optional

from amaze_ai.env import AmazeEnv
from amaze_ai.solver import solve_bfs

Grid = List[List[int]]
Position = Tuple[int, int]

def make_test_level() -> Tuple[Grid, Tuple[int, int]]:
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 1, 0, 0],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
    ]
    start = (0, 0)
    return grid, start

def _neighbors(r: int, c: int, rows: int, cols: int):
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc

def _random_open_grid(rows: int, cols: int, wall_prob: float) -> Grid:
    grid: Grid = []

    for _ in range(rows):
        row = []
        for _ in range(cols):
            cell = 0 if random.random() < wall_prob else 1
            row.append(cell)
        grid.append(row)
    
    return grid

def _find_first_open_cell(grid: Grid) -> Position | None:
    for r in range(len(grid)):
        for c in range(len(grid[0])):
            if grid[r][c] == 1:
                return (r, c)
    return None

def _connected_open_cells(grid: Grid, start: Position) -> set[Position]:
    rows = len(grid)
    cols = len(grid[0])
    visited: set[Position] = set()
    queue = deque([start])
    visited.add(start)

    while queue:
        r, c = queue.popleft()

        for nr, nc in _neighbors(r, c, rows, cols):
            if grid[nr][nc] == 1 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return visited

def _count_open_cells(grid: Grid) -> int:
    total = 0
    for row in grid:
        for cell in row:
            if cell == 1:
                total += 1
    return total

def _is_connected(grid: Grid) -> bool:
    start = _find_first_open_cell(grid)
    if start is None:
        return False
    
    reachable = _connected_open_cells(grid, start)
    return len(reachable) == _count_open_cells(grid)

def make_random_level(
        rows: int = 5,
        cols: int = 5,
        wall_prob: float = 0.2,
        min_open_cells: int = 8,
) -> Tuple[Grid, Position]:
    while True:
        grid = _random_open_grid(rows, cols, wall_prob)

        open_cells = _count_open_cells(grid)
        if open_cells < min_open_cells:
            continue

        if not _is_connected(grid):
            continue

        open_positions = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if grid[r][c] == 1
        ]

        start = random.choice(open_positions)
        return grid, start

def make_random_solvable_level(
    rows: int = 5,
    cols: int = 5,
    wall_prob: float = 0.2,
    min_open_cells: int = 8,
    max_attempts: int = 500,
) -> Tuple[Grid, Position]:
    for attempt in range(max_attempts):
        grid, start = make_random_level(
            rows=rows,
            cols=cols,
            wall_prob=wall_prob,
            min_open_cells=min_open_cells,
        )

        env = AmazeEnv(grid, start)
        solution = solve_bfs(env)

        if solution is not None:
            print(f"Solvable level found on attempt {attempt + 1}")
            return grid, start

    print("Generator exhausted all attempts.")
    raise RuntimeError("Failed to generate a solvable level within max_attempts.")

def make_random_solvable_level_with_length(
        rows: int = 7,
        cols: int = 7,
        wall_prob: float = 0.2,
        min_open_cells: int = 8,
        min_solution_len: int = 1,
        max_solution_len: Optional[int] = None,
        max_attempts: int = 500,
):
    for attempt in range(max_attempts):
        grid, start = make_random_level(
            rows=rows,
            cols=cols,
            wall_prob=wall_prob,
            min_open_cells=min_open_cells,
        )

        env = AmazeEnv(grid, start)
        solution = solve_bfs(env)

        if solution is None:
            continue

        solution_len = len(solution)

        if solution_len < min_solution_len:
            continue

        if max_solution_len is not None and solution_len > max_solution_len:
            continue

        print(f"Solvable level found on attempt {attempt + 1} with solution length {solution_len}")
        return grid, start, solution
    raise RuntimeError("Failed to generate a solvable level in the requested difficulty range.")

def make_easy_level():
    return make_random_solvable_level_with_length(
        rows=5,
        cols=5,
        wall_prob=0.18,
        min_open_cells=8,
        min_solution_len=3,
        max_solution_len=10,
        max_attempts=1200,
    )


def make_medium_level():
    return make_random_solvable_level_with_length(
        rows=7,
        cols=7,
        wall_prob=0.20,
        min_open_cells=8,
        min_solution_len=8,
        max_solution_len=18,
        max_attempts=1500,
    )


def make_hard_level():
    return make_random_solvable_level_with_length(
        rows=9,
        cols=9,
        wall_prob=0.22,
        min_open_cells=8,
        min_solution_len=15,
        max_solution_len=None,
        max_attempts=2000,
    )
from amaze_ai.env import AmazeEnv
from amaze_ai.generator import make_random_solvable_level
from amaze_ai.solver import solve_bfs

def test_random_level_has_valid_start():
    grid, start = make_random_solvable_level()
    r, c = start
    assert grid[r][c] == 1

def test_random_level_has_at_least_one_open_cell():
    grid, _ = make_random_solvable_level()
    assert any(cell == 1 for row in grid for cell in row)

def test_random_solvable_level_is_solvable():
    grid, start = make_random_solvable_level(rows=5, cols=5, wall_prob=0.2)
    env = AmazeEnv(grid, start)
    solution = solve_bfs(env)
    assert solution is not None
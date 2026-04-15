from amaze_ai.env import AmazeEnv
from amaze_ai.generator import make_test_level
from amaze_ai.solver import solve_bfs

def test_solver_finds_solution():
    grid, start = make_test_level()
    env = AmazeEnv(grid, start)

    solution = solve_bfs(env)
    assert solution is not None